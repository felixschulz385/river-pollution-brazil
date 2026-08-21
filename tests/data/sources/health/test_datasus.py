from __future__ import annotations

import pytest

from src.data.sources.health.fetch import datasus as datasus_module


class _FakeDriver:
    def __init__(self):
        self.window_handles = [1, 2]


class _FakeForm:
    def __init__(self):
        self.driver = _FakeDriver()

    def open(self, url):
        return None

    def reset_query(self):
        raise RuntimeError("cleanup: could not find .limpa button")


def test_execute_sih_manifest_entries_does_not_mask_original_exception_with_reset_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """A batch failure (e.g. a network error mid-query) must propagate as
    itself, not get replaced by an unrelated `reset_query()` cleanup failure
    in the `finally` block."""
    monkeypatch.setattr(datasus_module, "_list_source_years", lambda form, source_key: ([2020], {}))
    monkeypatch.setattr(
        datasus_module, "_filter_sih_period_values_for_year", lambda period_values, year: []
    )

    def _raise_query_error(*args, **kwargs):
        raise ValueError("original query failure")

    monkeypatch.setattr(datasus_module, "_run_sih_residence_query", _raise_query_error)

    manifest_entries = [
        {
            "batch_id": "batch-1",
            "status": "pending",
            "raw_path": str(tmp_path / "batch-1.csv"),
            "source_key": "legacy",
            "export_year": 2020,
            "metric_key": "hospitalizations",
            "row_value": "row",
            "column_value": "column",
            "chapter_filter_text": None,
            "morbidity_filter_text": None,
        }
    ]

    with pytest.raises(ValueError, match="original query failure"):
        datasus_module._execute_sih_manifest_entries(
            str(tmp_path), _FakeForm(), "sih_total", manifest_entries
        )


def test_parse_sih_period_value_decodes_year_and_month():
    assert datasus_module._parse_sih_period_value("nrbr2401.dbf") == (2024, 1)
    assert datasus_module._parse_sih_period_value("mrbr9512.dbf") == (1995, 12)


def test_parse_sih_period_value_rejects_implausible_year():
    # "94" would decode to 2094 under the raw `yy >= 95` rule, well outside
    # SIH's real 1995-present coverage -- must raise instead of silently
    # producing a nonsense future year.
    with pytest.raises(ValueError, match="Implausible year"):
        datasus_module._parse_sih_period_value("mrbr9412.dbf")
