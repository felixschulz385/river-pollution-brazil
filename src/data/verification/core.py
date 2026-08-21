"""Orchestrates per-source verification: adapter registry + fingerprint + sidecar cache."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .checks import CheckResult
from .constants import SOURCES, sidecar_path
from .fingerprint import compute_fingerprint
from .sources import SOURCE_ADAPTERS, FetchListing, OutputArtifactCheck

logger = logging.getLogger(__name__)


@dataclass
class SourceReport:
    """One source's verification outcome, cached or freshly computed."""

    source: str
    status: str  # "verified" | "failed" | "outstanding" | "not_present_locally"
    fingerprint: str
    verified_at: str
    checks: list[dict] = field(default_factory=list)
    fetch_completeness: dict | None = None
    outputs_present: bool = False
    from_cache: bool = False


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_sidecar(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_sidecar(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file and atomically rename it into place (matching
    # `shared.batches.write_manifest`), so a crash/SIGKILL mid-write (e.g. a
    # Slurm job hitting its time/mem limit) can never leave a torn sidecar
    # for a concurrent reader to see.
    temp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temp_path, path)


class Verification:
    """Run and cache per-source verification/summary against local data outputs."""

    def __init__(self, root_dir="."):
        self.root_dir = root_dir

    def _run_source(self, source: str, *, force: bool = False) -> SourceReport:
        if source not in SOURCE_ADAPTERS:
            raise ValueError(f"Unknown verification source: {source!r}. Expected one of {SOURCES}.")

        adapter = SOURCE_ADAPTERS[source]
        sidecar = sidecar_path(self.root_dir, source)

        fingerprint_reliable = True
        try:
            fingerprint_paths = adapter.fingerprint_paths(self.root_dir)
        except Exception:
            logger.exception(
                "Source '%s': fingerprint_paths() raised unexpectedly; skipping the cache "
                "for this run and re-checking from scratch.",
                source,
            )
            fingerprint_paths = []
            fingerprint_reliable = False
        fingerprint = compute_fingerprint(fingerprint_paths)
        if not fingerprint_reliable:
            # Never persist a fingerprint computed from a crash fallback: a
            # constant "empty" fingerprint would otherwise match itself on
            # every subsequent run and serve this run's result from cache
            # forever, even after the underlying crash is fixed.
            fingerprint = f"error:{_timestamp()}"

        existing = None if (force or not fingerprint_reliable) else _read_sidecar(sidecar)
        if (
            existing is not None
            and existing.get("fingerprint") == fingerprint
            and existing.get("status") != "outstanding"
        ):
            logger.debug("Source '%s' unchanged (fingerprint=%s); using cached sidecar.", source, fingerprint)
            return SourceReport(
                source=source,
                status=existing["status"],
                fingerprint=fingerprint,
                verified_at=existing["verified_at"],
                checks=existing.get("checks", []),
                fetch_completeness=existing.get("fetch_completeness"),
                outputs_present=existing.get("outputs_present", False),
                from_cache=True,
            )

        try:
            fetch_listing = adapter.list_fetched(self.root_dir, force=force)
        except Exception as exc:
            logger.exception("Source '%s': list_fetched() raised unexpectedly.", source)
            fetch_listing = FetchListing(
                present=0,
                expected=None,
                detail=f"list_fetched() crashed: {exc.__class__.__name__}: {exc}",
            )

        try:
            output_artifacts = adapter.check_outputs(self.root_dir)
        except Exception as exc:
            logger.exception("Source '%s': check_outputs() raised unexpectedly.", source)
            output_artifacts = [
                OutputArtifactCheck(
                    label="check_outputs_crashed",
                    path=Path(self.root_dir),
                    exists=True,
                    checks=[
                        CheckResult(
                            name="check_outputs_crashed",
                            ok=False,
                            message=f"check_outputs() raised {exc.__class__.__name__}: {exc}",
                        )
                    ],
                )
            ]

        outputs_present = any(artifact.exists for artifact in output_artifacts)
        any_present = fetch_listing.present > 0 or outputs_present
        all_checks: list[CheckResult] = [check for artifact in output_artifacts for check in artifact.checks]

        if not any_present:
            status = "not_present_locally"
        elif all_checks and not all(check.ok for check in all_checks):
            status = "failed"
        elif not output_artifacts:
            # An adapter that returns no artifacts at all has checked
            # nothing; report as outstanding rather than falling through to
            # "verified" with zero checks having actually run.
            logger.warning(
                "Source '%s': check_outputs() returned no artifacts despite data being "
                "present; reporting as outstanding.",
                source,
            )
            status = "outstanding"
        elif not all(artifact.exists for artifact in output_artifacts):
            status = "outstanding"
        else:
            status = "verified"

        checks_payload = [asdict(check) for check in all_checks]
        fetch_completeness = {
            "present": fetch_listing.present,
            "expected": fetch_listing.expected,
            "detail": fetch_listing.detail,
        }
        payload = {
            "source": source,
            "fingerprint": fingerprint,
            "verified_at": _timestamp(),
            "status": status,
            "checks": checks_payload,
            "fetch_completeness": fetch_completeness,
            "outputs_present": outputs_present,
        }
        _write_sidecar(sidecar, payload)
        logger.info("Verified source '%s': status=%s", source, status)
        return SourceReport(
            source=source,
            status=status,
            fingerprint=fingerprint,
            verified_at=payload["verified_at"],
            checks=checks_payload,
            fetch_completeness=fetch_completeness,
            outputs_present=outputs_present,
            from_cache=False,
        )

    def verify(self, source: str | None = None, force: bool = False) -> dict[str, SourceReport]:
        """Run verification for `source` (or all sources) and cache results.

        Each source is isolated: an adapter that raises unexpectedly (a bug in
        that source's checks, a malformed output file, etc.) is reported as a
        failed source rather than aborting the whole run and losing every
        other source's already-computed result.
        """
        names = [source] if source else list(SOURCES)
        reports: dict[str, SourceReport] = {}
        for name in names:
            try:
                reports[name] = self._run_source(name, force=force)
            except ValueError:
                raise  # programming error (unknown source) -- not a data issue to swallow
            except Exception as exc:
                logger.exception(
                    "Source '%s': verification crashed unexpectedly; continuing with remaining sources.",
                    name,
                )
                reports[name] = SourceReport(
                    source=name,
                    status="failed",
                    fingerprint="",
                    verified_at=_timestamp(),
                    checks=[
                        asdict(
                            CheckResult(
                                name="verification_crashed",
                                ok=False,
                                message=f"{exc.__class__.__name__}: {exc}",
                            )
                        )
                    ],
                    fetch_completeness=None,
                    from_cache=False,
                )
        return reports

    def summary(self, source: str | None = None, force: bool = False) -> dict[str, SourceReport]:
        """Return per-source reports, reusing verification's fingerprint short-circuit."""
        return self.verify(source=source, force=force)


__all__ = ["SourceReport", "Verification"]
