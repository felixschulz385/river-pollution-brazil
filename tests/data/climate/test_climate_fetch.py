from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from src.cli import main as data_cli_main
from src.data.climate.core import Climate
from src.data.climate.fetch.common import ClimateCredentialsError, ERA5_YEARS, load_cds_credentials
from src.data.climate.fetch.common import (
    load_download_manifest,
    manifest_path_for,
    retrieve_batched_dataset,
)
from src.data.climate.fetch.era5_land_daily import (
    DATASET as DAILY_DATASET,
    build_era5_land_daily_request,
    fetch_era5_land_daily,
)
from src.data.climate.fetch.era5_land_hourly import (
    DATASET as HOURLY_DATASET,
    VARIABLES as HOURLY_VARIABLES,
    build_era5_land_hourly_request,
    fetch_era5_land_hourly,
)
from src.data.climate.fetch.verify import ERA5L_VALUE_RANGES, VerificationResult, verify_era5_grib_batch


def _write_cdsapi(root: Path, contents: str) -> None:
    credentials_path = root / "setup" / "secrets" / ".cdsapi"
    credentials_path.parent.mkdir(parents=True, exist_ok=True)
    credentials_path.write_text(contents, encoding="utf-8")


FIRST_YEAR = ERA5_YEARS[0]
SECOND_YEAR = ERA5_YEARS[1]
LAST_YEAR = ERA5_YEARS[-1]
TOTAL_MONTHLY_BATCHES = len(ERA5_YEARS) * 12
HOURLY_RUNNING_LIMIT = 1
DAILY_RUNNING_LIMIT = 40


@pytest.fixture(autouse=True)
def disable_climate_decency_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.data.climate.fetch.common._decency_wait", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.data.climate.fetch.common._worker_wait", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.data.climate.fetch.common.ENABLE_PERIODIC_RECHECKS", False)
    monkeypatch.setattr("src.data.climate.fetch.common.MAX_ACTIVE_REMOTE_REQUESTS", TOTAL_MONTHLY_BATCHES + 10)


def test_load_cds_credentials_reads_project_secret(tmp_path: Path) -> None:
    _write_cdsapi(
        tmp_path,
        "url: https://cds.climate.copernicus.eu/api\nkey: user:secret\n",
    )

    credentials = load_cds_credentials(root_dir=tmp_path)

    assert credentials == {
        "url": "https://cds.climate.copernicus.eu/api",
        "key": "user:secret",
    }


def test_load_cds_credentials_requires_file(tmp_path: Path) -> None:
    with pytest.raises(ClimateCredentialsError, match="Missing CDS credentials file"):
        load_cds_credentials(root_dir=tmp_path)


def test_load_cds_credentials_rejects_malformed_file(tmp_path: Path) -> None:
    _write_cdsapi(tmp_path, "url=https://cds.climate.copernicus.eu/api\n")

    with pytest.raises(ClimateCredentialsError, match="Malformed CDS credentials file"):
        load_cds_credentials(root_dir=tmp_path)


def test_fetch_era5_land_hourly_builds_expected_yearly_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_cdsapi(
        tmp_path,
        "url: https://cds.climate.copernicus.eu/api\nkey: user:secret\n",
    )
    submit_calls = []

    class DummyRemote:
        def __init__(self, request_id, status="accepted", results_ready=False):
            self.request_id = request_id
            self.status = status
            self.results_ready = results_ready

    class DummyClient:
        def submit(self, dataset, request):
            submit_calls.append((dataset, request))
            return DummyRemote(f"req-{len(submit_calls)}")

    monkeypatch.setattr(
        "src.data.climate.fetch.common.create_datastores_client",
        lambda root_dir=".": DummyClient(),
    )

    output_paths = fetch_era5_land_hourly(root_dir=tmp_path)

    assert output_paths[0].name == f"era5_land_hourly_{FIRST_YEAR}_01.grib"
    assert output_paths[-1].name == f"era5_land_hourly_{LAST_YEAR}_12.grib"
    assert len(output_paths) == TOTAL_MONTHLY_BATCHES
    assert len(submit_calls) == TOTAL_MONTHLY_BATCHES
    assert [call[0] for call in submit_calls[:3]] == [HOURLY_DATASET, HOURLY_DATASET, HOURLY_DATASET]
    assert submit_calls[0][1]["year"] == [FIRST_YEAR]
    assert submit_calls[0][1]["month"] == ["01"]
    assert submit_calls[0][1] == build_era5_land_hourly_request(FIRST_YEAR, "01")
    manifest = load_download_manifest(output_paths[0])
    assert manifest is not None
    assert manifest["status"] == "submitted"
    assert manifest["dataset"] == HOURLY_DATASET
    assert manifest["request_id"] == "req-1"
    assert not output_paths[0].exists()


def test_fetch_era5_land_daily_skips_existing_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_cdsapi(
        tmp_path,
        "url: https://cds.climate.copernicus.eu/api\nkey: user:secret\n",
    )
    output_dir = tmp_path / "data" / "climate" / "raw" / "era5_land_daily"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"era5_land_daily_{FIRST_YEAR}_01.grib").write_text("cached", encoding="utf-8")
    manifest_path_for(output_dir / f"era5_land_daily_{FIRST_YEAR}_01.grib").write_text(
        '{\n  "status": "downloaded"\n}',
        encoding="utf-8",
    )
    submit_calls = []

    class DummyRemote:
        def __init__(self, request_id, status="accepted", results_ready=False):
            self.request_id = request_id
            self.status = status
            self.results_ready = results_ready

    class DummyClient:
        def submit(self, dataset, request):
            submit_calls.append((dataset, request))
            return DummyRemote(f"req-{len(submit_calls)}")

    monkeypatch.setattr(
        "src.data.climate.fetch.common.create_datastores_client",
        lambda root_dir=".": DummyClient(),
    )

    output_paths = fetch_era5_land_daily(root_dir=tmp_path)

    assert output_paths[0].name == f"era5_land_daily_{FIRST_YEAR}_01.grib"
    assert output_paths[-1].name == f"era5_land_daily_{LAST_YEAR}_12.grib"
    assert len(output_paths) == TOTAL_MONTHLY_BATCHES
    assert len(submit_calls) == TOTAL_MONTHLY_BATCHES - 1
    assert submit_calls[0][1]["year"] == FIRST_YEAR
    assert submit_calls[0][1]["month"] == "02"
    assert submit_calls[10][1]["month"] == "12"
    assert submit_calls[11][1]["year"] == SECOND_YEAR
    assert submit_calls[0][1] == build_era5_land_daily_request(FIRST_YEAR, "02")
    manifest = load_download_manifest(output_paths[1])
    assert manifest is not None
    assert manifest["status"] == "submitted"


def test_fetch_era5_land_daily_does_not_skip_without_success_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_cdsapi(
        tmp_path,
        "url: https://cds.climate.copernicus.eu/api\nkey: user:secret\n",
    )
    output_dir = tmp_path / "data" / "climate" / "raw" / "era5_land_daily"
    output_dir.mkdir(parents=True, exist_ok=True)
    cached_file = output_dir / f"era5_land_daily_{FIRST_YEAR}_01.grib"
    cached_file.write_text("cached", encoding="utf-8")
    manifest_path_for(cached_file).write_text(
        '{\n  "status": "failed"\n}',
        encoding="utf-8",
    )
    submit_calls = []

    class DummyRemote:
        def __init__(self, request_id, status="accepted", results_ready=False):
            self.request_id = request_id
            self.status = status
            self.results_ready = results_ready

    class DummyClient:
        def submit(self, dataset, request):
            submit_calls.append((dataset, request))
            return DummyRemote(f"req-{len(submit_calls)}")

    monkeypatch.setattr(
        "src.data.climate.fetch.common.create_datastores_client",
        lambda root_dir=".": DummyClient(),
    )

    fetch_era5_land_daily(root_dir=tmp_path)

    assert submit_calls[0][1]["year"] == FIRST_YEAR
    assert submit_calls[0][1]["month"] == "01"


def test_fetch_era5_land_daily_resumes_and_downloads_successful_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_cdsapi(
        tmp_path,
        "url: https://cds.climate.copernicus.eu/api\nkey: user:secret\n",
    )
    output_path = (
        tmp_path
        / "data"
        / "climate"
        / "raw"
        / "era5_land_daily"
        / f"era5_land_daily_{FIRST_YEAR}_01.grib"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path_for(output_path).write_text(
        f'{{\n  "dataset": "derived-era5-land-daily-statistics",\n  "request": {{"year": "{FIRST_YEAR}", "month": "01"}},\n  "request_id": "req-1",\n  "status": "submitted"\n}}',
        encoding="utf-8",
    )
    downloaded_targets = []

    class DummyResults:
        def download(self, target):
            downloaded_targets.append(Path(target))
            Path(target).write_text("grib", encoding="utf-8")
            return target

    class DummyRemote:
        request_id = "req-1"
        status = "successful"
        results_ready = True

        def get_results(self):
            return DummyResults()

        def get_receipt(self):
            return {"request_id": self.request_id}

    submitted = []

    class DummyClient:
        def get_remote(self, request_id):
            assert request_id == "req-1"
            return DummyRemote()

        def submit(self, dataset, request):
            submitted.append((dataset, request))
            request_id = f"req-new-{len(submitted)}"
            return type(
                "SubmittedRemote",
                (),
                {
                    "request_id": request_id,
                    "status": "accepted",
                    "results_ready": False,
                },
            )()

    monkeypatch.setattr(
        "src.data.climate.fetch.common.create_datastores_client",
        lambda root_dir=".": DummyClient(),
    )
    monkeypatch.setattr(
        "src.data.climate.fetch.era5_land_daily.verify_era5_grib_batch",
        lambda path, bands: VerificationResult(ok=True),
    )

    fetch_era5_land_daily(root_dir=tmp_path)

    manifest = load_download_manifest(output_path)
    assert manifest is not None
    assert manifest["status"] == "downloaded"
    assert manifest["request_id"] == "req-1"
    assert output_path.exists()
    assert len(downloaded_targets) == 1
    assert len(submitted) == TOTAL_MONTHLY_BATCHES - 1


def test_fetch_era5_land_daily_survives_expired_results_and_resubmits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_cdsapi(
        tmp_path,
        "url: https://cds.climate.copernicus.eu/api\nkey: user:secret\n",
    )
    output_path = (
        tmp_path
        / "data"
        / "climate"
        / "raw"
        / "era5_land_daily"
        / f"era5_land_daily_{FIRST_YEAR}_01.grib"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path_for(output_path).write_text(
        f'{{\n  "dataset": "{DAILY_DATASET}",\n  "request": {{"year": "{FIRST_YEAR}", "month": "01"}},\n  "request_id": "req-1",\n  "status": "submitted"\n}}',
        encoding="utf-8",
    )

    class ExpiredResults:
        def download(self, target):
            raise RuntimeError(
                "404 Client Error: Not Found for url: "
                "https://cds.climate.copernicus.eu/api/retrieve/v1/jobs/req-1/results\nresults expired"
            )

    class DummyRemote:
        request_id = "req-1"
        status = "successful"
        results_ready = True

        def get_results(self):
            return ExpiredResults()

        def get_receipt(self):
            return {"request_id": self.request_id}

    submitted = []

    class DummyClient:
        def get_remote(self, request_id):
            assert request_id == "req-1"
            return DummyRemote()

        def submit(self, dataset, request):
            submitted.append((dataset, request))
            return type(
                "SubmittedRemote",
                (),
                {"request_id": f"req-new-{len(submitted)}", "status": "accepted", "results_ready": False},
            )()

    monkeypatch.setattr(
        "src.data.climate.fetch.common.create_datastores_client",
        lambda root_dir=".": DummyClient(),
    )

    # Must not raise: an individual batch's expired/failed download should be
    # recorded and rescheduled, not crash the whole fetch run.
    fetch_era5_land_daily(root_dir=tmp_path)

    manifest = load_download_manifest(output_path)
    assert manifest is not None
    # The same cycle's submit loop picks the now-"failed" batch back up and
    # resubmits it with a fresh request_id, mirroring how other download
    # failures are already rescheduled.
    assert manifest["status"] == "submitted"
    assert manifest["request_id"] == "req-new-1"
    assert not output_path.exists()
    assert submitted[0][1]["year"] == FIRST_YEAR
    assert submitted[0][1]["month"] == "01"


def test_fetch_era5_land_daily_marks_rejected_and_retries_when_queue_allows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_cdsapi(
        tmp_path,
        "url: https://cds.climate.copernicus.eu/api\nkey: user:secret\n",
    )
    output_path = (
        tmp_path
        / "data"
        / "climate"
        / "raw"
        / "era5_land_daily"
        / f"era5_land_daily_{FIRST_YEAR}_01.grib"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path_for(output_path).write_text(
        f'{{\n  "dataset": "derived-era5-land-daily-statistics",\n  "request": {{"year": "{FIRST_YEAR}", "month": "01"}},\n  "request_id": "req-old",\n  "status": "submitted"\n}}',
        encoding="utf-8",
    )
    submitted = []

    class RejectedRemote:
        request_id = "req-old"
        status = "rejected"
        results_ready = False

        def get_receipt(self):
            return {"request_id": self.request_id}

    class DummyClient:
        def get_remote(self, request_id):
            if request_id == "req-old":
                return RejectedRemote()
            raise AssertionError(f"Unexpected request_id: {request_id}")

        def submit(self, dataset, request):
            submitted.append((dataset, request))
            return type(
                "SubmittedRemote",
                (),
                {
                    "request_id": "req-new",
                    "status": "accepted",
                    "results_ready": False,
                },
            )()

    monkeypatch.setattr(
        "src.data.climate.fetch.common.create_datastores_client",
        lambda root_dir=".": DummyClient(),
    )

    fetch_era5_land_daily(root_dir=tmp_path)

    manifest = load_download_manifest(output_path)
    assert manifest is not None
    assert manifest["status"] == "submitted"
    assert manifest["request_id"] == "req-new"
    assert len(submitted) == TOTAL_MONTHLY_BATCHES
    assert submitted[0][1]["year"] == FIRST_YEAR
    assert submitted[0][1]["month"] == "01"


def test_fetch_era5_land_daily_marks_missing_remote_job_rejected_and_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_cdsapi(
        tmp_path,
        "url: https://cds.climate.copernicus.eu/api\nkey: user:secret\n",
    )
    output_path = (
        tmp_path
        / "data"
        / "climate"
        / "raw"
        / "era5_land_daily"
        / f"era5_land_daily_{FIRST_YEAR}_01.grib"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path_for(output_path).write_text(
        f'{{\n  "dataset": "{DAILY_DATASET}",\n  "request": {{"year": "{FIRST_YEAR}", "month": "01"}},\n  "request_id": "req-missing",\n  "status": "submitted"\n}}',
        encoding="utf-8",
    )
    submitted = []

    class DummyClient:
        def get_remote(self, request_id):
            raise RuntimeError("404 Client Error: Not Found for url: https://example.invalid/jobs/req-missing\njob not found")

        def submit(self, dataset, request):
            submitted.append((dataset, request))
            return type(
                "SubmittedRemote",
                (),
                {
                    "request_id": "req-new",
                    "status": "accepted",
                    "results_ready": False,
                },
            )()

    monkeypatch.setattr(
        "src.data.climate.fetch.common.create_datastores_client",
        lambda root_dir=".": DummyClient(),
    )

    fetch_era5_land_daily(root_dir=tmp_path)

    manifest = load_download_manifest(output_path)
    assert manifest is not None
    assert manifest["status"] == "submitted"
    assert manifest["request_id"] == "req-new"
    assert manifest["remote_status"] == "accepted"
    assert len(submitted) == TOTAL_MONTHLY_BATCHES


def test_fetch_era5_land_daily_skips_additional_remote_checks_after_running_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_cdsapi(
        tmp_path,
        "url: https://cds.climate.copernicus.eu/api\nkey: user:secret\n",
    )
    output_dir = tmp_path / "data" / "climate" / "raw" / "era5_land_daily"
    output_dir.mkdir(parents=True, exist_ok=True)
    for month in ("01", "02", "03"):
        target = output_dir / f"era5_land_daily_{FIRST_YEAR}_{month}.grib"
        manifest_path_for(target).write_text(
            f'{{\n  "dataset": "{DAILY_DATASET}",\n  "request": {{"year": "{FIRST_YEAR}", "month": "{month}"}},\n  "request_id": "req-{month}",\n  "status": "submitted"\n}}',
            encoding="utf-8",
        )

    checked_request_ids = []

    class RunningRemote:
        def __init__(self, request_id):
            self.request_id = request_id
            self.status = "running"
            self.results_ready = False

    class DummyClient:
        def get_remote(self, request_id):
            checked_request_ids.append(request_id)
            return RunningRemote(request_id)

        def submit(self, dataset, request):
            raise AssertionError("submit should not be called when 150-slot logic is not under test")

    monkeypatch.setattr(
        "src.data.climate.fetch.common.create_datastores_client",
        lambda root_dir=".": DummyClient(),
    )
    monkeypatch.setattr("src.data.climate.fetch.common.MAX_ACTIVE_REMOTE_REQUESTS", 3)
    monkeypatch.setitem(
        __import__("src.data.climate.fetch.common", fromlist=["DATASET_RUNNING_REMOTE_REQUEST_LIMITS"]).DATASET_RUNNING_REMOTE_REQUEST_LIMITS,
        DAILY_DATASET,
        2,
    )
    monkeypatch.setattr("src.data.climate.fetch.common.ENABLE_PERIODIC_RECHECKS", False)

    fetch_era5_land_daily(root_dir=tmp_path)

    assert checked_request_ids == ["req-01", "req-02"]


def test_fetch_era5_land_hourly_stops_remote_checks_after_running_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_cdsapi(
        tmp_path,
        "url: https://cds.climate.copernicus.eu/api\nkey: user:secret\n",
    )
    output_dir = tmp_path / "data" / "climate" / "raw" / "era5_land_hourly"
    output_dir.mkdir(parents=True, exist_ok=True)
    for month in ("01", "02", "03"):
        target = output_dir / f"era5_land_hourly_{FIRST_YEAR}_{month}.grib"
        manifest_path_for(target).write_text(
            f'{{\n  "dataset": "{HOURLY_DATASET}",\n  "request": {{"year": ["{FIRST_YEAR}"], "month": ["{month}"]}},\n  "request_id": "req-{month}",\n  "status": "submitted"\n}}',
            encoding="utf-8",
        )

    checked_request_ids = []

    class RunningRemote:
        def __init__(self, request_id):
            self.request_id = request_id
            self.status = "running"
            self.results_ready = False

    class DummyClient:
        def get_remote(self, request_id):
            checked_request_ids.append(request_id)
            return RunningRemote(request_id)

        def submit(self, dataset, request):
            raise AssertionError("submit should not be called when remote checks are being short-circuited")

    monkeypatch.setattr(
        "src.data.climate.fetch.common.create_datastores_client",
        lambda root_dir=".": DummyClient(),
    )
    monkeypatch.setattr("src.data.climate.fetch.common.MAX_ACTIVE_REMOTE_REQUESTS", 3)
    monkeypatch.setattr("src.data.climate.fetch.common.ENABLE_PERIODIC_RECHECKS", False)

    fetch_era5_land_hourly(root_dir=tmp_path)

    assert checked_request_ids == ["req-01"]


def test_stale_remote_status_running_does_not_starve_other_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_cdsapi(
        tmp_path,
        "url: https://cds.climate.copernicus.eu/api\nkey: user:secret\n",
    )
    output_dir = tmp_path / "data" / "climate" / "raw" / "era5_land_hourly"
    output_dir.mkdir(parents=True, exist_ok=True)

    # A batch that finished downloading and preprocessing long ago, but whose
    # manifest still carries a leftover remote_status: "running" from before
    # it completed -- mirrors a real ghost manifest found in production.
    ghost_target = output_dir / f"era5_land_hourly_{FIRST_YEAR}_01.grib"
    manifest_path_for(ghost_target).write_text(
        f'{{\n'
        f'  "dataset": "{HOURLY_DATASET}",\n'
        f'  "request": {{"year": ["{FIRST_YEAR}"], "month": ["01"]}},\n'
        f'  "request_id": "req-ghost",\n'
        f'  "status": "processed",\n'
        f'  "download_status": "downloaded",\n'
        f'  "preprocess_status": "processed",\n'
        f'  "remote_status": "running"\n'
        f'}}',
        encoding="utf-8",
    )

    # A genuinely still-submitted batch that should still get rechecked.
    submitted_target = output_dir / f"era5_land_hourly_{FIRST_YEAR}_02.grib"
    manifest_path_for(submitted_target).write_text(
        f'{{\n'
        f'  "dataset": "{HOURLY_DATASET}",\n'
        f'  "request": {{"year": ["{FIRST_YEAR}"], "month": ["02"]}},\n'
        f'  "request_id": "req-submitted",\n'
        f'  "status": "submitted",\n'
        f'  "remote_status": "accepted"\n'
        f'}}',
        encoding="utf-8",
    )

    checked_request_ids = []

    class DummyRemote:
        def __init__(self, request_id):
            self.request_id = request_id
            self.status = "accepted"
            self.results_ready = False

    class DummyClient:
        def get_remote(self, request_id):
            checked_request_ids.append(request_id)
            return DummyRemote(request_id)

        def submit(self, dataset, request):
            raise AssertionError("submit should not be called in this scenario")

    monkeypatch.setattr(
        "src.data.climate.fetch.common.create_datastores_client",
        lambda root_dir=".": DummyClient(),
    )
    # Only the genuinely-submitted batch is active, so capping the active
    # budget at 1 keeps the submit loop from touching the other ~478 batches
    # without manifests.
    monkeypatch.setattr("src.data.climate.fetch.common.MAX_ACTIVE_REMOTE_REQUESTS", 1)
    monkeypatch.setattr("src.data.climate.fetch.common.ENABLE_PERIODIC_RECHECKS", False)

    fetch_era5_land_hourly(root_dir=tmp_path)

    assert "req-ghost" not in checked_request_ids
    assert "req-submitted" in checked_request_ids


def test_fetch_era5_land_hourly_skips_fresh_remote_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_cdsapi(
        tmp_path,
        "url: https://cds.climate.copernicus.eu/api\nkey: user:secret\n",
    )
    output_dir = tmp_path / "data" / "climate" / "raw" / "era5_land_hourly"
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"era5_land_hourly_{FIRST_YEAR}_01.grib"
    manifest_path_for(target).write_text(
        '{\n'
        f'  "dataset": "{HOURLY_DATASET}",\n'
        f'  "request": {{"year": ["{FIRST_YEAR}"], "month": ["01"]}},\n'
        '  "request_id": "req-01",\n'
        '  "status": "submitted",\n'
        '  "remote_status": "accepted",\n'
        '  "remote_checked_at": "2026-04-24T10:00:00+00:00"\n'
        '}',
        encoding="utf-8",
    )

    class DummyClient:
        def get_remote(self, request_id):
            raise AssertionError("get_remote should not be called for freshly checked requests")

        def submit(self, dataset, request):
            raise AssertionError("submit should not be called for active requests")

    monkeypatch.setattr(
        "src.data.climate.fetch.common.create_datastores_client",
        lambda root_dir=".": DummyClient(),
    )
    monkeypatch.setattr("src.data.climate.fetch.common.ENABLE_PERIODIC_RECHECKS", False)
    monkeypatch.setattr(
        "src.data.climate.fetch.common.datetime",
        type(
            "FixedDateTime",
            (),
            {
                "now": staticmethod(lambda tz=None: __import__("datetime").datetime(2026, 4, 24, 10, 5, 0, tzinfo=tz)),
                "fromisoformat": staticmethod(__import__("datetime").datetime.fromisoformat),
            },
        ),
    )

    fetch_era5_land_hourly(root_dir=tmp_path)


def test_fetch_era5_land_daily_defers_new_submissions_when_queue_is_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_cdsapi(
        tmp_path,
        "url: https://cds.climate.copernicus.eu/api\nkey: user:secret\n",
    )
    active_target = (
        tmp_path
        / "data"
        / "climate"
        / "raw"
        / "era5_land_daily"
        / f"era5_land_daily_{FIRST_YEAR}_01.grib"
    )
    active_target.parent.mkdir(parents=True, exist_ok=True)
    manifest_path_for(active_target).write_text(
        f'{{\n  "dataset": "derived-era5-land-daily-statistics",\n  "request": {{"year": "{FIRST_YEAR}", "month": "01"}},\n  "request_id": "req-active",\n  "status": "submitted"\n}}',
        encoding="utf-8",
    )
    submit_calls = []

    class ActiveRemote:
        request_id = "req-active"
        status = "accepted"
        results_ready = False

    class DummyClient:
        def get_remote(self, request_id):
            assert request_id == "req-active"
            return ActiveRemote()

        def submit(self, dataset, request):
            submit_calls.append((dataset, request))
            raise AssertionError("submit should not be called when the queue is full")

    monkeypatch.setattr(
        "src.data.climate.fetch.common.create_datastores_client",
        lambda root_dir=".": DummyClient(),
    )
    monkeypatch.setattr("src.data.climate.fetch.common.MAX_ACTIVE_REMOTE_REQUESTS", 1)

    fetch_era5_land_daily(root_dir=tmp_path)

    assert submit_calls == []
    deferred_manifest = load_download_manifest(
        tmp_path
        / "data"
        / "climate"
        / "raw"
        / "era5_land_daily"
        / f"era5_land_daily_{FIRST_YEAR}_02.grib"
    )
    assert deferred_manifest is None


def test_climate_fetch_routes_supported_subtypes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Sentinel(list):
        pass

    hourly_output = Sentinel(["hourly"])
    daily_output = Sentinel(["daily"])
    arco_output = Sentinel(["arco"])

    monkeypatch.setattr(
        "src.data.climate.fetch.era5_land_hourly.fetch_era5_land_hourly",
        lambda root_dir=".": hourly_output,
    )
    monkeypatch.setattr(
        "src.data.climate.fetch.era5_land_daily.fetch_era5_land_daily",
        lambda root_dir=".": daily_output,
    )
    monkeypatch.setattr(
        "src.data.climate.fetch.era5_land_arco.fetch_era5_land_arco",
        lambda root_dir=".": arco_output,
    )

    agent = Climate(root_dir=tmp_path)

    assert agent.fetch(subtype="era5_land_hourly") is hourly_output
    assert agent.fetch(subtype="era5_land_daily") is daily_output
    assert agent.fetch(subtype="era5_land_arco") is arco_output
    with pytest.raises(ValueError, match="Unsupported climate fetch subtype"):
        agent.fetch(subtype="unknown")


def test_climate_preprocess_routes_supported_subtypes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hourly_output = object()
    daily_output = object()

    monkeypatch.setattr(
        "src.data.climate.preprocess.era5_land.preprocess_era5_land_worker",
        lambda root_dir=".", subtype="era5_land_hourly", stage="all", n_jobs=None: (
            hourly_output if subtype == "era5_land_hourly" else daily_output
        ),
    )

    agent = Climate(root_dir=tmp_path)

    assert agent.preprocess(subtype="era5_land_hourly") is hourly_output
    assert agent.preprocess(subtype="era5_land_daily") is daily_output
    with pytest.raises(ValueError, match="Unsupported climate preprocess subtype"):
        agent.preprocess(subtype="unknown")
    # era5_land_arco has no separate preprocess stage - opening the ARCO
    # store, aggregating, and writing all happen under `fetch` instead, since
    # (unlike the GRIB path) there's no distinct raw-download step.
    with pytest.raises(ValueError, match="Unsupported climate preprocess subtype"):
        agent.preprocess(subtype="era5_land_arco")


def test_climate_fetch_era5_land_arco_does_the_real_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arco_output = object()

    monkeypatch.setattr(
        "src.data.climate.preprocess.era5_land_arco.preprocess_era5_land_arco",
        lambda root_dir=".": arco_output,
    )

    agent = Climate(root_dir=tmp_path)

    assert agent.fetch(subtype="era5_land_arco") is arco_output


def test_era5_land_hourly_variables_exclude_arco_covered_vars() -> None:
    assert set(HOURLY_VARIABLES) == {
        "surface_runoff",
        "sub_surface_runoff",
        "potential_evaporation",
    }


def _era5_dataset(bands: dict) -> xr.Dataset:
    return xr.Dataset(
        {
            name: (("time", "latitude", "longitude"), np.asarray(values))
            for name, values in bands.items()
        }
    )


def test_verify_era5_grib_batch_passes_for_reasonable_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _era5_dataset({"2t": np.full((1, 2, 2), 290.0), "2d": np.full((1, 2, 2), 280.0)})
    monkeypatch.setattr("src.data.climate.preprocess.era5_land._open_era5_dataset", lambda path: dataset)

    result = verify_era5_grib_batch(tmp_path / "fake.grib", bands=["2t", "2d"])

    assert result.ok
    assert result.errors == []


def test_verify_era5_grib_batch_flags_out_of_range_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lo, hi = ERA5L_VALUE_RANGES["2t"]
    dataset = _era5_dataset({"2t": np.full((1, 2, 2), hi + 50.0)})
    monkeypatch.setattr("src.data.climate.preprocess.era5_land._open_era5_dataset", lambda path: dataset)

    result = verify_era5_grib_batch(tmp_path / "fake.grib", bands=["2t"])

    assert not result.ok
    assert "2t" in result.errors[0]
    assert lo is not None  # sanity: bounds are configured for this band


def test_verify_era5_grib_batch_tolerates_ocean_masked_nulls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ERA5_AREA includes a substantial stretch of ocean, which ERA5-Land
    # (a land-only dataset) legitimately reports as null every batch.
    values = np.full((1, 2, 5), 290.0)
    values[0, 0, :3] = np.nan  # 30% null -- comfortably below MAX_NULL_FRACTION
    dataset = _era5_dataset({"2t": values})
    monkeypatch.setattr("src.data.climate.preprocess.era5_land._open_era5_dataset", lambda path: dataset)

    result = verify_era5_grib_batch(tmp_path / "fake.grib", bands=["2t"])

    assert result.ok


def test_verify_era5_grib_batch_flags_null_values_beyond_ocean_margin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = np.full((1, 2, 5), 290.0)
    values[0, :, :] = np.nan
    values[0, 0, 0] = 290.0  # 90% null -- well beyond plausible ocean coverage
    dataset = _era5_dataset({"2t": values})
    monkeypatch.setattr("src.data.climate.preprocess.era5_land._open_era5_dataset", lambda path: dataset)

    result = verify_era5_grib_batch(tmp_path / "fake.grib", bands=["2t"])

    assert not result.ok
    assert "null" in result.errors[0]


def test_verify_era5_grib_batch_flags_missing_band(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _era5_dataset({"2t": np.full((1, 2, 2), 290.0)})
    monkeypatch.setattr("src.data.climate.preprocess.era5_land._open_era5_dataset", lambda path: dataset)

    result = verify_era5_grib_batch(tmp_path / "fake.grib", bands=["2t", "swvl1"])

    assert not result.ok
    assert "swvl1" in result.errors[0]


def test_retrieve_batched_dataset_reschedules_then_caps_verification_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_cdsapi(
        tmp_path,
        "url: https://cds.climate.copernicus.eu/api\nkey: user:secret\n",
    )
    output_path = tmp_path / "data" / "climate" / "raw" / "verify_test" / "verify_test_only.grib"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path_for(output_path).write_text(
        '{\n  "dataset": "test-dataset",\n  "request": {"batch": "only"},\n  "request_id": "req-1",\n  "status": "submitted"\n}',
        encoding="utf-8",
    )

    get_remote_calls = []

    class DummyResults:
        def download(self, target):
            Path(target).write_text("grib", encoding="utf-8")
            return target

    class DummyRemote:
        request_id = "req-1"
        status = "successful"
        results_ready = True

        def get_results(self):
            return DummyResults()

        def get_receipt(self):
            return {"request_id": self.request_id}

    class DummyClient:
        def get_remote(self, request_id):
            get_remote_calls.append(request_id)
            return DummyRemote()

        def submit(self, dataset, request):
            return type(
                "SubmittedRemote",
                (),
                {"request_id": "req-other", "status": "accepted", "results_ready": False},
            )()

    monkeypatch.setattr(
        "src.data.climate.fetch.common.create_datastores_client",
        lambda root_dir=".": DummyClient(),
    )
    # Bypass the remote-recheck cooldown so each run below re-checks the
    # batch immediately, simulating successive worker cycles.
    monkeypatch.setattr("src.data.climate.fetch.common._manifest_is_due_for_remote_check", lambda manifest: True)

    def run_once():
        retrieve_batched_dataset(
            root_dir=tmp_path,
            dataset="test-dataset",
            request_factory=lambda **batch: {"batch": batch["batch"]},
            output_subdir="verify_test",
            file_prefix="verify_test",
            batches=[{"batch": "only"}],
            output_name_factory=lambda batch: "verify_test_only.grib",
            verify_batch=lambda path: VerificationResult(ok=False, errors=["2t: out of range"]),
        )

    # A verification failure marks the batch "failed", which the same cycle's
    # submit loop immediately picks up and resubmits (mirroring how any other
    # download failure is already rescheduled) -- so the batch ends each cycle
    # back at "submitted", carrying its verification_attempts count forward.
    run_once()
    manifest = load_download_manifest(output_path)
    assert manifest["status"] == "submitted"
    assert manifest["verification_attempts"] == 1
    assert not output_path.exists()

    run_once()
    manifest = load_download_manifest(output_path)
    assert manifest["status"] == "submitted"
    assert manifest["verification_attempts"] == 2

    run_once()
    manifest = load_download_manifest(output_path)
    assert manifest["status"] == "verification_failed"
    assert manifest["verification_attempts"] == 3

    calls_before_final_check = len(get_remote_calls)
    run_once()
    manifest = load_download_manifest(output_path)
    assert manifest["status"] == "verification_failed"
    assert len(get_remote_calls) == calls_before_final_check


def test_cli_climate_help_mentions_subtype(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        data_cli_main(["climate", "--help"])

    assert excinfo.value.code == 0
    assert "--subtype" in capsys.readouterr().out


def test_cli_climate_fetch_routes_subtype(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = {}

    def fake_fetch(self, subtype="cloud_cover"):
        seen["subtype"] = subtype
        seen["root_dir"] = self.root_dir
        return []

    monkeypatch.setattr(Climate, "fetch", fake_fetch)

    exit_code = data_cli_main(
        [
            "climate",
            "fetch",
            "--subtype",
            "era5_land_hourly",
            "--root-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert seen == {"subtype": "era5_land_hourly", "root_dir": str(tmp_path)}


def test_cli_climate_preprocess_routes_subtype(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = {}

    def fake_preprocess(self, subtype="cloud_cover", stage="all", n_jobs=None):
        seen["subtype"] = subtype
        seen["root_dir"] = self.root_dir
        return []

    monkeypatch.setattr(Climate, "preprocess", fake_preprocess)

    exit_code = data_cli_main(
        [
            "climate",
            "preprocess",
            "--subtype",
            "era5_land_hourly",
            "--root-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert seen == {"subtype": "era5_land_hourly", "root_dir": str(tmp_path)}


def test_cli_climate_fetch_routes_arco_subtype(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = {}

    def fake_fetch(self, subtype="cloud_cover"):
        seen["subtype"] = subtype
        seen["root_dir"] = self.root_dir
        return []

    monkeypatch.setattr(Climate, "fetch", fake_fetch)

    exit_code = data_cli_main(
        [
            "climate",
            "fetch",
            "--subtype",
            "era5_land_arco",
            "--root-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert seen == {"subtype": "era5_land_arco", "root_dir": str(tmp_path)}
