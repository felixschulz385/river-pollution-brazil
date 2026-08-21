from __future__ import annotations

import pandas as pd
import pytest

from src.data.shared.batches import load_manifest
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


class _FakeYearLoopForm:
    """Stands in for `DatasusTabnetForm` in the mortality/birth-outcome
    per-year loops: every step succeeds except `reset_query()`."""

    def __init__(self, headless=False, download_dir=None):
        pass

    def open(self, url):
        return None

    def select_column(self, value):
        return None

    def select_option_value(self, value):
        return None

    def select_output_format_prn(self):
        return None

    def submit_query(self):
        return None

    def read_result_table(self):
        return pd.DataFrame({"value": [1]})

    def reset_query(self):
        raise RuntimeError("cleanup: could not find .limpa button")

    def close(self):
        return None


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


def test_fetch_mortality_age_tables_does_not_abort_on_reset_query_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """A `reset_query()` cleanup failure after a successful year's fetch must
    not abort the whole multi-year scrape and discard already-collected
    years' data."""
    monkeypatch.setattr(datasus_module, "DatasusTabnetForm", _FakeYearLoopForm)
    monkeypatch.setattr(
        datasus_module,
        "_mortality_age_year_codes",
        lambda: {"pre_1996": ["79"], "post_1995": ["96"]},
    )

    output_paths = datasus_module.fetch_mortality_age_tables(root_dir=str(tmp_path))

    assert set(output_paths) == {"pre_1996", "post_1995"}
    for path in output_paths.values():
        assert pd.read_parquet(path)["value"].tolist() == [1]


def test_fetch_birth_outcome_tables_does_not_abort_on_reset_query_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """Same guard as the mortality-age loop: a `reset_query()` failure must
    not discard the outcome's already-collected years."""
    monkeypatch.setattr(datasus_module, "DatasusTabnetForm", _FakeYearLoopForm)

    output_paths = datasus_module.fetch_birth_outcome_tables(
        root_dir=str(tmp_path), outcome_names=["gestational_duration"]
    )

    assert set(output_paths) == {"gestational_duration"}
    assert pd.read_parquet(output_paths["gestational_duration"])["value"].tolist() == [1] * 30


class _FakeYearLoopFormFailingOnce:
    """Like `_FakeYearLoopForm`, but `submit_query()` raises the first time
    a chosen year code is requested, then succeeds on any later call --
    simulating a transient failure partway through a multi-year scrape."""

    def __init__(self, headless=False, download_dir=None):
        pass

    def open(self, url):
        return None

    def select_column(self, value):
        return None

    def select_option_value(self, value):
        self._last_value = value

    def select_output_format_prn(self):
        return None

    def submit_query(self):
        if self._last_value == _FakeYearLoopFormFailingOnce.failing_value and not _FakeYearLoopFormFailingOnce.already_failed:
            _FakeYearLoopFormFailingOnce.already_failed = True
            raise RuntimeError("transient DATASUS failure")

    def read_result_table(self):
        return pd.DataFrame({"value": [1]})

    def reset_query(self):
        return None

    def close(self):
        return None


def test_fetch_mortality_age_tables_resumes_without_refetching_completed_years(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """A failure partway through the scrape must not force a full restart:
    a resumed call should only re-fetch the year that failed, leaving the
    already-completed year's checkpoint untouched."""
    monkeypatch.setattr(
        datasus_module, "DatasusTabnetForm", _FakeYearLoopFormFailingOnce
    )
    monkeypatch.setattr(
        datasus_module,
        "_mortality_age_year_codes",
        lambda: {"pre_1996": ["79", "80"], "post_1995": ["96"]},
    )
    _FakeYearLoopFormFailingOnce.failing_value = "obtbr80.dbf"
    _FakeYearLoopFormFailingOnce.already_failed = False

    with pytest.raises(RuntimeError, match="transient DATASUS failure"):
        datasus_module.fetch_mortality_age_tables(root_dir=str(tmp_path))

    # Year 79 must already be checkpointed as completed even though the
    # overall call raised on year 80.
    table_name = "mortality_age_pre_1996"
    manifest_entries = load_manifest(str(tmp_path), datasus_module.HEALTH_DATASET_NAME, table_name)
    statuses = {entry["batch_id"]: entry["status"] for entry in manifest_entries}
    assert statuses["79"] == "completed"
    assert statuses["80"] == "failed"

    # Resumed call: year 80 now succeeds (the fake only fails once), and
    # year 79 is not re-requested.
    output_paths = datasus_module.fetch_mortality_age_tables(root_dir=str(tmp_path))
    combined = pd.read_parquet(output_paths["pre_1996"])
    assert sorted(combined["year_code"].tolist()) == ["79", "80"]


def test_parse_sih_period_value_decodes_year_and_month():
    assert datasus_module._parse_sih_period_value("nrbr2401.dbf") == (2024, 1)
    assert datasus_module._parse_sih_period_value("mrbr9512.dbf") == (1995, 12)


def test_parse_sih_period_value_rejects_implausible_year():
    # "94" would decode to 2094 under the raw `yy >= 95` rule, well outside
    # SIH's real 1995-present coverage -- must raise instead of silently
    # producing a nonsense future year.
    with pytest.raises(ValueError, match="Implausible year"):
        datasus_module._parse_sih_period_value("mrbr9412.dbf")


def test_mortality_age_year_codes_cover_1979_to_2021_with_no_gaps_or_overlap():
    year_codes = datasus_module._mortality_age_year_codes()

    pre_1996 = year_codes["pre_1996"]
    post_1995 = year_codes["post_1995"]

    # "95" (year 1995) must be fetched from exactly one of the two periods,
    # not both (double-counting) and not neither (missing year).
    assert pre_1996.count("95") + post_1995.count("95") == 1

    expected_pre_1996 = [str(year).zfill(2) for year in range(79, 96)]
    expected_post_1995 = [str(year).zfill(2) for year in list(range(96, 100)) + list(range(0, 22))]
    assert pre_1996 == expected_pre_1996
    assert post_1995 == expected_post_1995

    # No code appears in both lists, and together they form the full,
    # contiguous 1979-2021 span.
    assert set(pre_1996).isdisjoint(post_1995)
    assert len(pre_1996) + len(post_1995) == (95 - 79 + 1) + (2021 - 1996 + 1)
