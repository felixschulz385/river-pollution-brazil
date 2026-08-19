"""Orchestrates per-source verification: adapter registry + fingerprint + sidecar cache."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .checks import CheckResult
from .constants import SOURCES, sidecar_path
from .fingerprint import compute_fingerprint
from .sources import SOURCE_ADAPTERS

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
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


class Verification:
    """Run and cache per-source verification/summary against local data outputs."""

    def __init__(self, root_dir="."):
        self.root_dir = root_dir

    def _run_source(self, source: str, *, force: bool = False) -> SourceReport:
        if source not in SOURCE_ADAPTERS:
            raise ValueError(f"Unknown verification source: {source!r}. Expected one of {SOURCES}.")

        adapter = SOURCE_ADAPTERS[source]
        sidecar = sidecar_path(self.root_dir, source)

        fingerprint_paths = adapter.fingerprint_paths(self.root_dir)
        fingerprint = compute_fingerprint(fingerprint_paths)

        existing = None if force else _read_sidecar(sidecar)
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
                from_cache=True,
            )

        fetch_listing = adapter.list_fetched(self.root_dir, force=force)
        output_artifacts = adapter.check_outputs(self.root_dir)

        any_present = fetch_listing.present > 0 or any(artifact.exists for artifact in output_artifacts)
        all_checks: list[CheckResult] = [check for artifact in output_artifacts for check in artifact.checks]

        if not any_present:
            status = "not_present_locally"
        elif all_checks and not all(check.ok for check in all_checks):
            status = "failed"
        elif output_artifacts and not all(artifact.exists for artifact in output_artifacts):
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
            from_cache=False,
        )

    def verify(self, source: str | None = None, force: bool = False) -> dict[str, SourceReport]:
        """Run verification for `source` (or all sources) and cache results."""
        names = [source] if source else list(SOURCES)
        return {name: self._run_source(name, force=force) for name in names}

    def summary(self, source: str | None = None, force: bool = False) -> dict[str, SourceReport]:
        """Return per-source reports, reusing verification's fingerprint short-circuit."""
        return self.verify(source=source, force=force)


__all__ = ["SourceReport", "Verification"]
