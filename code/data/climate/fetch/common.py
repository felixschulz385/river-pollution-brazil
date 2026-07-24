from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from itertools import product
import json
import logging
import os
from pathlib import Path
import random
import re
from time import monotonic, sleep

ERA5_YEARS = [str(year) for year in range(1985, 2025)]
ERA5_MONTHS = [f"{month:02d}" for month in range(1, 13)]
ERA5_DAYS = [f"{day:02d}" for day in range(1, 32)]
ERA5_HOURS = [f"{hour:02d}:00" for hour in range(24)]
ERA5_AREA = [5.27, -73.99, -33.75, -34.78]
MAX_ACTIVE_REMOTE_REQUESTS = 150
WORKER_RECHECK_SECONDS = 120
ENABLE_PERIODIC_RECHECKS = True
FILE_LOCK_POLL_SECONDS = 2
FILE_LOCK_TIMEOUT_SECONDS = 900
REMOTE_RECHECK_ACCEPTED_SECONDS = 900
REMOTE_RECHECK_RUNNING_SECONDS = 300
DATASET_RUNNING_REMOTE_REQUEST_LIMITS = {
    "reanalysis-era5-land": 1,
    "derived-era5-land-daily-statistics": 1,
}

logger = logging.getLogger(__name__)


class ClimateCredentialsError(RuntimeError):
    """Raised when CDS credentials cannot be loaded from the project secret file."""


class ClimateFileLockTimeoutError(RuntimeError):
    """Raised when a climate file lock cannot be acquired within the timeout."""


def climate_raw_dir(root_dir="."):
    return Path(root_dir) / "data" / "climate" / "raw"


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _decency_wait(min_seconds=0.4, max_seconds=1.0):
    sleep(random.uniform(min_seconds, max_seconds))


def _worker_wait(seconds=WORKER_RECHECK_SECONDS):
    sleep(seconds)


def manifest_path_for(target_path: Path):
    return target_path.with_suffix(target_path.suffix + ".manifest.json")


def lock_path_for(target_path: Path):
    return target_path.with_suffix(target_path.suffix + ".lock")


def load_download_manifest(target_path: Path):
    manifest_path = manifest_path_for(target_path)
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _manifest_download_status(manifest):
    if manifest is None:
        return None
    return manifest.get("download_status", manifest.get("status"))


def _manifest_preprocess_status(manifest):
    if manifest is None:
        return None
    return manifest.get("preprocess_status")


def _parse_timestamp(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def write_download_manifest(
    target_path: Path,
    *,
    dataset,
    request,
    status,
    error=None,
    **extra_fields,
):
    manifest_path = manifest_path_for(target_path)
    payload = {
        "target_path": str(target_path),
        "dataset": dataset,
        "request": request,
        "status": status,
        "updated_at": _timestamp(),
        "error": error,
    }
    payload.update(extra_fields)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def _read_lock_metadata(lock_path: Path):
    if not lock_path.exists():
        return None
    try:
        return json.loads(lock_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _wait_for_lock_release(target_path: Path, *, timeout_seconds=FILE_LOCK_TIMEOUT_SECONDS):
    lock_path = lock_path_for(target_path)
    deadline = monotonic() + timeout_seconds
    while lock_path.exists():
        if monotonic() >= deadline:
            owner = _read_lock_metadata(lock_path)
            raise ClimateFileLockTimeoutError(
                f"Timed out waiting for lock {lock_path} to be released. Metadata={owner!r}"
            )
        logger.info(
            "Waiting for climate file lock on %s before continuing.",
            target_path.name,
        )
        sleep(FILE_LOCK_POLL_SECONDS)


@contextmanager
def climate_file_lock(
    target_path: Path,
    *,
    owner,
    timeout_seconds=FILE_LOCK_TIMEOUT_SECONDS,
):
    lock_path = lock_path_for(target_path)
    deadline = monotonic() + timeout_seconds
    payload = {
        "target_path": str(target_path),
        "lock_path": str(lock_path),
        "owner": owner,
        "pid": os.getpid(),
        "created_at": _timestamp(),
    }

    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if monotonic() >= deadline:
                owner_metadata = _read_lock_metadata(lock_path)
                raise ClimateFileLockTimeoutError(
                    f"Timed out acquiring lock for {target_path}. Metadata={owner_metadata!r}"
                )
            logger.info(
                "Climate file %s is locked by another process; waiting.",
                target_path.name,
            )
            sleep(FILE_LOCK_POLL_SECONDS)

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        yield lock_path
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def should_skip_download(target_path: Path):
    manifest = load_download_manifest(target_path)
    if manifest is None:
        return False
    if _manifest_preprocess_status(manifest) == "processed":
        return True
    if lock_path_for(target_path).exists():
        return False
    download_status = _manifest_download_status(manifest)
    return download_status == "downloaded" and target_path.exists()


def load_cds_credentials(root_dir="."):
    credentials_path = Path(root_dir) / "setup" / "secrets" / ".cdsapi"
    if not credentials_path.exists():
        raise ClimateCredentialsError(
            f"Missing CDS credentials file at {credentials_path}."
        )

    values = {}
    line_pattern = re.compile(r"^(?P<key>[A-Za-z_][A-Za-z0-9_-]*)\s*:\s*(?P<value>.+)$")
    for raw_line in credentials_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = line_pattern.match(line)
        if match is None:
            raise ClimateCredentialsError(
                f"Malformed CDS credentials file at {credentials_path}: {raw_line!r}"
            )
        values[match.group("key")] = match.group("value").strip()

    if not values.get("url") or not values.get("key"):
        raise ClimateCredentialsError(
            f"CDS credentials file at {credentials_path} must define both 'url' and 'key'."
        )
    return values


def create_cds_client(root_dir="."):
    import cdsapi

    credentials = load_cds_credentials(root_dir=root_dir)
    return cdsapi.Client(
        url=credentials["url"],
        key=credentials["key"],
        quiet=False,
    )


def create_datastores_client(root_dir="."):
    from ecmwf.datastores import Client

    credentials = load_cds_credentials(root_dir=root_dir)
    logging.getLogger("ecmwf.datastores.processing").setLevel(logging.WARNING)
    return Client(url=credentials["url"], key=credentials["key"])


def _remote_is_finished(remote):
    return getattr(remote, "results_ready", False) or getattr(remote, "status", None) == "successful"


def _remote_has_failed(remote):
    return getattr(remote, "status", None) in {"failed", "deleted", "cancelled", "unavailable", "rejected"}


def _remote_is_active(remote):
    return getattr(remote, "status", None) in {"accepted", "running"}


def _remote_is_running(remote):
    return getattr(remote, "status", None) == "running"


def _manifest_remote_is_running(manifest):
    if manifest is None:
        return False
    return manifest.get("remote_status") == "running"


def _manifest_remote_is_active(manifest):
    if manifest is None:
        return False
    download_status = _manifest_download_status(manifest)
    return download_status in {"submitted", "downloading"}


def _manifest_check_cooldown_seconds(manifest):
    if _manifest_remote_is_running(manifest):
        return REMOTE_RECHECK_RUNNING_SECONDS
    return REMOTE_RECHECK_ACCEPTED_SECONDS


def _manifest_is_due_for_remote_check(manifest):
    if manifest is None or not _manifest_remote_is_active(manifest):
        return True
    checked_at = _parse_timestamp(manifest.get("remote_checked_at"))
    if checked_at is None:
        return True
    elapsed = (datetime.now(timezone.utc) - checked_at).total_seconds()
    return elapsed >= _manifest_check_cooldown_seconds(manifest)


def _manifest_activity_state(manifest):
    return {
        "is_active": _manifest_remote_is_active(manifest),
        "is_running": _manifest_remote_is_running(manifest),
    }


def _manifest_activity_state_for_output(target_path: Path):
    manifest = load_download_manifest(target_path)
    return manifest, _manifest_activity_state(manifest)


def _is_due_manifest_candidate(output_path: Path, manifest_states) -> bool:
    manifest = manifest_states[output_path]["manifest"]
    if manifest is None:
        return False
    if should_skip_download(output_path):
        return False
    return _manifest_is_due_for_remote_check(manifest)


def _receipt_or_none(remote):
    try:
        return remote.get_receipt()
    except Exception:
        return None


def _is_remote_not_found_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "404" in text
        or "not found" in text
        or "job not found" in text
    )


def _check_existing_request(
    *,
    client,
    dataset,
    request,
    output_path: Path,
):
    manifest = load_download_manifest(output_path)
    if should_skip_download(output_path):
        logger.info("Skipping completed climate batch %s.", output_path.name)
        return {"is_active": False, "is_running": False}

    if manifest is None or "request_id" not in manifest:
        return {"is_active": False, "is_running": False}

    logger.info(
        "Checking climate batch %s with request_id=%s.",
        output_path.name,
        manifest["request_id"],
    )
    try:
        remote = client.get_remote(manifest["request_id"])
    except Exception as exc:
        if not _is_remote_not_found_error(exc):
            raise
        write_download_manifest(
            output_path,
            dataset=dataset,
            request=request,
            status="rejected",
            request_id=manifest["request_id"],
            remote_status="missing",
            remote_checked_at=_timestamp(),
            error=str(exc),
        )
        logger.warning(
            "Climate batch %s references a missing remote job %s; marking it for resubmission.",
            output_path.name,
            manifest["request_id"],
        )
        _decency_wait()
        return {"is_active": False, "is_running": False}
    remote_status = getattr(remote, "status", None)
    if _remote_has_failed(remote):
        write_download_manifest(
            output_path,
            dataset=dataset,
            request=request,
            status="rejected" if remote_status == "rejected" else "failed",
            request_id=remote.request_id,
            remote_status=remote_status,
            remote_checked_at=_timestamp(),
            receipt=_receipt_or_none(remote),
            error=f"Remote job finished with status {remote_status}.",
        )
        logger.warning(
            "Climate batch %s failed remotely with status=%s.",
            output_path.name,
            remote_status,
        )
        _decency_wait()
        return {"is_active": False, "is_running": False}

    if _remote_is_active(remote):
        write_download_manifest(
            output_path,
            dataset=dataset,
            request=request,
            status="submitted",
            request_id=remote.request_id,
            remote_status=remote_status,
            remote_checked_at=_timestamp(),
        )
        _decency_wait()
        return {
            "is_active": True,
            "is_running": _remote_is_running(remote),
        }

    if not _remote_is_finished(remote):
        write_download_manifest(
            output_path,
            dataset=dataset,
            request=request,
            status="submitted",
            request_id=remote.request_id,
            remote_status=remote_status,
            remote_checked_at=_timestamp(),
        )
        _decency_wait()
        return {"is_active": False, "is_running": False}

    write_download_manifest(
        output_path,
        dataset=dataset,
        request=request,
        status="downloading",
        request_id=remote.request_id,
        remote_status=remote_status,
        remote_checked_at=_timestamp(),
    )
    logger.info(
        "Downloading completed climate batch %s from request_id=%s.",
        output_path.name,
        remote.request_id,
    )
    try:
        with climate_file_lock(output_path, owner="climate_fetch_download"):
            results = remote.get_results()
            results.download(str(output_path))
    except Exception as exc:
        write_download_manifest(
            output_path,
            dataset=dataset,
            request=request,
            status="failed",
            request_id=remote.request_id,
            remote_status=remote_status,
            receipt=_receipt_or_none(remote),
            error=str(exc),
        )
        raise

    write_download_manifest(
        output_path,
        dataset=dataset,
        request=request,
        status="downloaded",
        request_id=remote.request_id,
        remote_status=remote_status,
        remote_checked_at=_timestamp(),
        receipt=_receipt_or_none(remote),
    )
    logger.info(
        "Finished climate batch %s and wrote %s.",
        output_path.name,
        output_path,
    )
    _decency_wait()
    return {"is_active": False, "is_running": False}


def _can_submit_from_manifest(target_path: Path):
    manifest = load_download_manifest(target_path)
    if manifest is None:
        return True
    if _manifest_preprocess_status(manifest) == "processed":
        return False
    download_status = _manifest_download_status(manifest)
    return download_status in {"failed", "rejected"}


def _submit_request(
    *,
    client,
    dataset,
    request,
    output_path: Path,
):
    logger.info("Submitting climate batch %s for dataset %s.", output_path.name, dataset)
    remote = client.submit(dataset, request)
    write_download_manifest(
        output_path,
        dataset=dataset,
        request=request,
        status="submitted",
        request_id=remote.request_id,
        remote_status=getattr(remote, "status", None),
        remote_checked_at=_timestamp(),
    )
    logger.info(
        "Submitted climate batch %s with request_id=%s and remote_status=%s.",
        output_path.name,
        remote.request_id,
        getattr(remote, "status", None),
    )
    _decency_wait()
    return True


def _manifest_status_counts(batch_payloads):
    counts = {
        "downloaded": 0,
        "submitted": 0,
        "downloading": 0,
        "failed": 0,
        "rejected": 0,
        "not_started": 0,
    }
    for _, output_path, _ in batch_payloads:
        manifest = load_download_manifest(output_path)
        if manifest is None:
            counts["not_started"] += 1
            continue
        if _manifest_preprocess_status(manifest) == "processed":
            counts["downloaded"] += 1
            continue
        status = _manifest_download_status(manifest)
        if status in counts:
            counts[status] += 1
        else:
            counts["not_started"] += 1
    return counts


def _manifest_is_locally_active(target_path: Path):
    manifest = load_download_manifest(target_path)
    if manifest is None:
        return False
    download_status = _manifest_download_status(manifest)
    return download_status in {"submitted", "downloading"}


def _log_worker_summary_box(
    *,
    dataset,
    cycle_number,
    total_batches,
    active_requests,
    running_requests,
    running_request_limit,
    submitted_this_cycle,
    counts,
):
    lines = [
        f"Dataset           : {dataset}",
        f"Worker cycle      : {cycle_number}",
        f"Total batches     : {total_batches}",
        f"Outstanding remote: {active_requests}/{MAX_ACTIVE_REMOTE_REQUESTS}",
        f"Running remote    : {running_requests}/{running_request_limit}",
        f"Submitted cycle   : {submitted_this_cycle}",
        f"Downloaded        : {counts['downloaded']}",
        f"Submitted         : {counts['submitted']}",
        f"Downloading       : {counts['downloading']}",
        f"Rejected          : {counts['rejected']}",
        f"Failed            : {counts['failed']}",
        f"Not started       : {counts['not_started']}",
    ]
    content_width = max(len(line) for line in lines + ["Climate Worker"])
    top_border = "+" + "-" * (content_width + 2) + "+"
    box_lines = [top_border, f"| {'Climate Worker'.ljust(content_width)} |"]
    box_lines.append("|" + " " * (content_width + 2) + "|")
    box_lines.extend(f"| {line.ljust(content_width)} |" for line in lines)
    box_lines.append(top_border)
    logger.info("\n%s", "\n".join(box_lines))


def retrieve_yearly_dataset(
    *,
    root_dir=".",
    dataset,
    request_factory,
    output_subdir,
    file_prefix,
    years=None,
    max_running_remote_requests=None,
):
    years = years or ERA5_YEARS
    return retrieve_batched_dataset(
        root_dir=root_dir,
        dataset=dataset,
        request_factory=request_factory,
        output_subdir=output_subdir,
        file_prefix=file_prefix,
        batches=[{"year": year} for year in years],
        output_name_factory=lambda batch: f"{file_prefix}_{batch['year']}.grib",
        max_running_remote_requests=max_running_remote_requests,
    )


def retrieve_batched_dataset(
    *,
    root_dir=".",
    dataset,
    request_factory,
    output_subdir,
    file_prefix,
    batches,
    output_name_factory,
    max_running_remote_requests=None,
):
    output_dir = climate_raw_dir(root_dir) / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    client = create_datastores_client(root_dir=root_dir)
    batch_list = list(batches)
    running_request_limit = (
        max_running_remote_requests
        if max_running_remote_requests is not None
        else DATASET_RUNNING_REMOTE_REQUEST_LIMITS.get(dataset, MAX_ACTIVE_REMOTE_REQUESTS)
    )
    batch_payloads = []
    total_batches = len(batch_list)
    for idx, batch in enumerate(batch_list, start=1):
        output_path = output_dir / output_name_factory(batch)
        request = request_factory(**batch)
        batch_payloads.append((idx, output_path, request))

    output_paths = [output_path for _, output_path, _ in batch_payloads]
    cycle_number = 0
    while True:
        cycle_number += 1
        active_requests = 0
        running_requests = 0
        submitted_this_cycle = 0

        manifest_states = {}
        for _, output_path, _ in batch_payloads:
            manifest, state = _manifest_activity_state_for_output(output_path)
            manifest_states[output_path] = {
                "manifest": manifest,
                "state": state,
            }
            active_requests += int(state["is_active"])
            running_requests += int(state["is_running"])

        priority_running_checks = []
        for idx, output_path, request in batch_payloads:
            manifest_state = manifest_states[output_path]["state"]
            if not manifest_state["is_running"]:
                continue
            if not _is_due_manifest_candidate(output_path, manifest_states):
                continue
            priority_running_checks.append((idx, output_path, request))

        checked_output_paths = set()
        for idx, output_path, request in priority_running_checks:
            manifest_state = manifest_states[output_path]["state"]
            status = _check_existing_request(
                client=client,
                dataset=dataset,
                request=request,
                output_path=output_path,
            )
            active_requests += int(status["is_active"]) - int(manifest_state["is_active"])
            running_requests += int(status["is_running"]) - int(manifest_state["is_running"])
            manifest_states[output_path] = {
                "manifest": load_download_manifest(output_path),
                "state": status,
            }
            checked_output_paths.add(output_path)

        running_requests = sum(
            int(state_bundle["state"]["is_running"])
            for state_bundle in manifest_states.values()
        )
        active_requests = sum(
            int(state_bundle["state"]["is_active"])
            for state_bundle in manifest_states.values()
        )

        for idx, output_path, request in batch_payloads:
            manifest_state = manifest_states[output_path]["state"]

            if should_skip_download(output_path):
                continue
            if output_path in checked_output_paths:
                continue

            if manifest_state["is_running"]:
                continue

            if running_requests >= running_request_limit and manifest_state["is_active"]:
                logger.debug(
                    "Skipping non-running remote status checks after satisfying the running limit (%s/%s).",
                    running_requests,
                    running_request_limit,
                )
                continue

            if not _is_due_manifest_candidate(output_path, manifest_states):
                logger.debug(
                    "Skipping remote recheck for %s because the last check is still fresh.",
                    output_path.name,
                )
                continue

            status = _check_existing_request(
                client=client,
                dataset=dataset,
                request=request,
                output_path=output_path,
            )
            active_requests += int(status["is_active"]) - int(manifest_state["is_active"])
            running_requests += int(status["is_running"]) - int(manifest_state["is_running"])
            manifest_states[output_path] = {
                "manifest": load_download_manifest(output_path),
                "state": status,
            }
            running_requests = sum(
                int(state_bundle["state"]["is_running"])
                for state_bundle in manifest_states.values()
            )
            active_requests = sum(
                int(state_bundle["state"]["is_active"])
                for state_bundle in manifest_states.values()
            )

        logger.info(
            "Climate queue occupancy after status checks: %s/%s outstanding remote requests, %s/%s running remote requests.",
            active_requests,
            MAX_ACTIVE_REMOTE_REQUESTS,
            running_requests,
            running_request_limit,
        )

        for _, output_path, request in batch_payloads:
            if should_skip_download(output_path):
                continue
            if not _can_submit_from_manifest(output_path):
                continue
            if active_requests >= MAX_ACTIVE_REMOTE_REQUESTS:
                continue
            submitted = _submit_request(
                client=client,
                dataset=dataset,
                request=request,
                output_path=output_path,
            )
            active_requests += int(submitted)
            submitted_this_cycle += int(submitted)

        counts = _manifest_status_counts(batch_payloads)
        _log_worker_summary_box(
            dataset=dataset,
            cycle_number=cycle_number,
            total_batches=total_batches,
            active_requests=active_requests,
            running_requests=running_requests,
            running_request_limit=running_request_limit,
            submitted_this_cycle=submitted_this_cycle,
            counts=counts,
        )

        if not ENABLE_PERIODIC_RECHECKS:
            break
        if counts["downloaded"] == total_batches:
            logger.info("Climate worker finished: all batches are downloaded.")
            break
        if active_requests == 0:
            logger.info("Climate worker stopping: no active remote requests remain.")
            break

        logger.info(
            "Climate worker sleeping for %s seconds before the next status re-check cycle.",
            WORKER_RECHECK_SECONDS,
        )
        _worker_wait(WORKER_RECHECK_SECONDS)
    return output_paths


def retrieve_yearly_dataset_in_monthly_batches(
    *,
    root_dir=".",
    dataset,
    request_factory,
    output_subdir,
    file_prefix,
    years=None,
    months=None,
    max_running_remote_requests=None,
):
    years = years or ERA5_YEARS
    months = months or ERA5_MONTHS
    batches = [
        {"year": year, "month": month}
        for year, month in product(years, months)
    ]
    return retrieve_batched_dataset(
        root_dir=root_dir,
        dataset=dataset,
        request_factory=request_factory,
        output_subdir=output_subdir,
        file_prefix=file_prefix,
        batches=batches,
        output_name_factory=lambda batch: f"{file_prefix}_{batch['year']}_{batch['month']}.grib",
        max_running_remote_requests=max_running_remote_requests,
    )
