from __future__ import annotations

from src.data.verification.core import Verification
from src.data.verification.sources import FetchListing, SourceAdapter


def _crashing_adapter(*, crash_in: str) -> SourceAdapter:
    def list_fetched(root_dir, force=False):
        if crash_in == "list_fetched":
            raise RuntimeError("boom in list_fetched")
        return FetchListing(present=0, expected=1, detail="ok")

    def check_outputs(root_dir):
        if crash_in == "check_outputs":
            raise RuntimeError("boom in check_outputs")
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
