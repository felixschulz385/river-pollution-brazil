from __future__ import annotations

import json

from src.data.verification.checks import CheckResult
from src.data.verification.core import Verification
from src.data.verification.sources import FetchListing, OutputArtifactCheck, SourceAdapter


def _crashing_adapter(*, crash_in: str) -> SourceAdapter:
    def list_fetched(root_dir, force=False):
        if crash_in == "list_fetched":
            raise RuntimeError("boom in list_fetched")
        return FetchListing(present=0, expected=1, detail="ok")

    def check_outputs(root_dir):
        if crash_in == "check_outputs":
            raise RuntimeError("boom in check_outputs")
        return []

    def check_fetched(root_dir):
        if crash_in == "check_fetched":
            raise RuntimeError("boom in check_fetched")
        return []

    def fingerprint_paths(root_dir):
        if crash_in == "fingerprint_paths":
            raise RuntimeError("boom in fingerprint_paths")
        return []

    return SourceAdapter(
        name="fake",
        list_fetched=list_fetched,
        check_outputs=check_outputs,
        fingerprint_paths=fingerprint_paths,
        check_fetched=check_fetched,
    )


def test_check_outputs_crash_is_isolated_and_reported(tmp_path, monkeypatch):
    """A source whose check_outputs() raises must not crash the whole run --
    it should come back as a 'failed' report with the error surfaced in its
    checks, and other sources must still be verified normally."""
    from src.data.verification import core as core_module

    monkeypatch.setitem(core_module.SOURCE_ADAPTERS, "biomes", _crashing_adapter(crash_in="check_outputs"))

    reports = Verification(root_dir=tmp_path).verify()

    assert reports["biomes"].status == "failed"
    messages = [check["message"] for check in reports["biomes"].checks]
    assert any("check_outputs" in message and "boom in check_outputs" in message for message in messages)
    # every other source still ran without being taken down by biomes' crash
    for name, report in reports.items():
        if name != "biomes":
            assert report.status in {"not_present_locally", "outstanding", "verified", "failed"}


def test_list_fetched_crash_is_isolated_and_reported(tmp_path, monkeypatch):
    from src.data.verification import core as core_module

    monkeypatch.setitem(core_module.SOURCE_ADAPTERS, "biomes", _crashing_adapter(crash_in="list_fetched"))

    reports = Verification(root_dir=tmp_path).verify(source="biomes")

    assert reports["biomes"].status == "not_present_locally"
    assert "boom in list_fetched" in reports["biomes"].fetch_completeness["detail"]


def test_fingerprint_paths_crash_falls_back_to_fresh_check(tmp_path, monkeypatch):
    from src.data.verification import core as core_module

    monkeypatch.setitem(core_module.SOURCE_ADAPTERS, "biomes", _crashing_adapter(crash_in="fingerprint_paths"))

    reports = Verification(root_dir=tmp_path).verify(source="biomes")

    assert reports["biomes"].status == "not_present_locally"


def test_check_fetched_crash_is_isolated_and_reported(tmp_path, monkeypatch):
    """Mirrors test_check_outputs_crash_is_isolated_and_reported: a crashing
    check_fetched() must not take down the whole run, and must be reported
    via fetch_status rather than the output-side status."""
    from src.data.verification import core as core_module

    monkeypatch.setitem(core_module.SOURCE_ADAPTERS, "biomes", _crashing_adapter(crash_in="check_fetched"))

    reports = Verification(root_dir=tmp_path).verify()

    assert reports["biomes"].fetch_status == "failed"
    messages = [check["message"] for check in reports["biomes"].fetched_checks]
    assert any("check_fetched" in message and "boom in check_fetched" in message for message in messages)
    for name, report in reports.items():
        if name != "biomes":
            assert report.status in {"not_present_locally", "outstanding", "verified", "failed"}


def _adapter_with_fetched_checks(*, fetched_ok: bool) -> SourceAdapter:
    def list_fetched(root_dir, force=False):
        return FetchListing(present=1, expected=1, detail="ok")

    def check_outputs(root_dir):
        return [
            OutputArtifactCheck(
                label="output", path=root_dir, exists=True, checks=[CheckResult(name="ok", ok=True)]
            )
        ]

    def check_fetched(root_dir):
        return [
            OutputArtifactCheck(
                label="raw",
                path=root_dir,
                exists=True,
                checks=[CheckResult(name="raw_check", ok=fetched_ok, message="raw check result")],
            )
        ]

    def fingerprint_paths(root_dir):
        return []

    return SourceAdapter(
        name="fake",
        list_fetched=list_fetched,
        check_outputs=check_outputs,
        fingerprint_paths=fingerprint_paths,
        check_fetched=check_fetched,
    )


def test_fetched_check_failure_marks_fetch_status_failed_without_affecting_status(tmp_path, monkeypatch):
    """A failing raw-fetched-artifact check must not overshadow a passing
    output check: `status` stays driven purely by check_outputs(), while the
    new `fetch_status` field independently reflects the raw-check failure."""
    from src.data.verification import core as core_module

    monkeypatch.setitem(
        core_module.SOURCE_ADAPTERS, "biomes", _adapter_with_fetched_checks(fetched_ok=False)
    )

    reports = Verification(root_dir=tmp_path).verify(source="biomes")

    assert reports["biomes"].status == "verified"
    assert reports["biomes"].fetch_status == "failed"
    assert reports["biomes"].fetched_checks[0]["ok"] is False


def test_fetched_check_success_marks_fetch_status_verified(tmp_path, monkeypatch):
    from src.data.verification import core as core_module

    monkeypatch.setitem(
        core_module.SOURCE_ADAPTERS, "biomes", _adapter_with_fetched_checks(fetched_ok=True)
    )

    reports = Verification(root_dir=tmp_path).verify(source="biomes")

    assert reports["biomes"].status == "verified"
    assert reports["biomes"].fetch_status == "verified"


def test_default_check_fetched_reports_not_applicable(tmp_path):
    """assembly (and any source with no separate raw-artifact concept) uses
    SourceAdapter's default no-op check_fetched, and must be reported as
    'not_applicable' rather than 'outstanding' -- there's nothing to fetch,
    not something outstanding that hasn't been fetched yet."""
    reports = Verification(root_dir=tmp_path).verify(source="assembly")

    assert reports["assembly"].fetch_status == "not_applicable"
    assert reports["assembly"].fetched_checks == []


def test_sidecar_roundtrip_includes_fetched_checks(tmp_path, monkeypatch):
    from src.data.verification import core as core_module

    monkeypatch.setitem(
        core_module.SOURCE_ADAPTERS, "biomes", _adapter_with_fetched_checks(fetched_ok=False)
    )

    Verification(root_dir=tmp_path).verify(source="biomes")

    sidecar_path = tmp_path / "data" / "biomes" / ".verification.json"
    payload = json.loads(sidecar_path.read_text())
    assert payload["fetch_status"] == "failed"
    assert payload["fetched_checks"][0]["name"] == "raw_check"

    # Unchanged fingerprint -> served from cache, still carrying both fields.
    cached_reports = Verification(root_dir=tmp_path).verify(source="biomes")
    assert cached_reports["biomes"].from_cache is True
    assert cached_reports["biomes"].fetch_status == "failed"
    assert cached_reports["biomes"].fetched_checks[0]["name"] == "raw_check"


def _adapter_with_outputs(artifacts: list[OutputArtifactCheck]) -> SourceAdapter:
    def list_fetched(root_dir, force=False):
        return FetchListing(present=1, expected=1, detail="ok")

    def check_outputs(root_dir):
        return artifacts

    def fingerprint_paths(root_dir):
        return []

    return SourceAdapter(
        name="fake",
        list_fetched=list_fetched,
        check_outputs=check_outputs,
        fingerprint_paths=fingerprint_paths,
    )


def test_preprocess_complete_true_when_all_output_artifacts_exist_even_if_checks_fail(tmp_path, monkeypatch):
    """preprocess_complete is content-agnostic on purpose: a source with
    every declared file present but a failing value-range/schema check still
    counts as "preprocessed" for gating -- only presence matters, not
    correctness (which is a separate, already-reported concern)."""
    from src.data.verification import core as core_module

    artifacts = [
        OutputArtifactCheck(
            label="a", path=tmp_path, exists=True, checks=[CheckResult(name="check", ok=False, message="bad")]
        )
    ]
    monkeypatch.setitem(core_module.SOURCE_ADAPTERS, "biomes", _adapter_with_outputs(artifacts))

    reports = Verification(root_dir=tmp_path).verify(source="biomes")

    assert reports["biomes"].status == "failed"
    assert reports["biomes"].preprocess_complete is True


def test_preprocess_complete_false_when_any_output_artifact_missing(tmp_path, monkeypatch):
    from src.data.verification import core as core_module

    artifacts = [
        OutputArtifactCheck(label="a", path=tmp_path, exists=True, checks=[CheckResult(name="check", ok=True)]),
        OutputArtifactCheck(label="b", path=tmp_path, exists=False, checks=[]),
    ]
    monkeypatch.setitem(core_module.SOURCE_ADAPTERS, "biomes", _adapter_with_outputs(artifacts))

    reports = Verification(root_dir=tmp_path).verify(source="biomes")

    assert reports["biomes"].preprocess_complete is False


def test_preprocess_complete_false_when_no_output_artifacts_declared(tmp_path, monkeypatch):
    from src.data.verification import core as core_module

    monkeypatch.setitem(core_module.SOURCE_ADAPTERS, "biomes", _adapter_with_outputs([]))

    reports = Verification(root_dir=tmp_path).verify(source="biomes")

    assert reports["biomes"].preprocess_complete is False


def test_preprocess_complete_roundtrips_through_sidecar_cache(tmp_path, monkeypatch):
    from src.data.verification import core as core_module

    artifacts = [OutputArtifactCheck(label="a", path=tmp_path, exists=True, checks=[CheckResult(name="c", ok=True)])]
    monkeypatch.setitem(core_module.SOURCE_ADAPTERS, "biomes", _adapter_with_outputs(artifacts))

    Verification(root_dir=tmp_path).verify(source="biomes")
    sidecar_path = tmp_path / "data" / "biomes" / ".verification.json"
    assert json.loads(sidecar_path.read_text())["preprocess_complete"] is True

    cached_reports = Verification(root_dir=tmp_path).verify(source="biomes")
    assert cached_reports["biomes"].from_cache is True
    assert cached_reports["biomes"].preprocess_complete is True


def test_stale_sidecar_missing_new_fields_is_not_trusted_as_cache_hit(tmp_path, monkeypatch):
    """A sidecar written before fetch_status/preprocess_complete existed has
    neither key at all. Serving it as a cache hit would silently default
    fetch_status to "not_applicable" and preprocess_complete to False
    forever (via the .get() fallbacks) for any source whose fingerprint
    hasn't changed since -- it must instead be recomputed fresh once."""
    from src.data.verification import core as core_module
    from src.data.verification.fingerprint import compute_fingerprint

    # A real check_fetched (fetched_ok=True -> fetch_status="verified") so a
    # cache hit that wrongly defaulted to "not_applicable" is unambiguously
    # distinguishable from a genuine recompute.
    monkeypatch.setitem(core_module.SOURCE_ADAPTERS, "biomes", _adapter_with_fetched_checks(fetched_ok=True))

    sidecar_dir = tmp_path / "data" / "biomes"
    sidecar_dir.mkdir(parents=True)
    stale_payload = {
        "source": "biomes",
        "fingerprint": compute_fingerprint([]),
        "verified_at": "2020-01-01T00:00:00+00:00",
        "status": "verified",
        "checks": [],
        "fetch_completeness": {"present": 0, "expected": 1, "detail": "ok"},
        "outputs_present": True,
        # deliberately no "fetch_status" / "preprocess_complete" keys, as a
        # sidecar written before this feature existed would have.
    }
    (sidecar_dir / ".verification.json").write_text(json.dumps(stale_payload))

    reports = Verification(root_dir=tmp_path).verify(source="biomes")

    assert reports["biomes"].from_cache is False
    assert reports["biomes"].preprocess_complete is True
    assert reports["biomes"].fetch_status == "verified"
