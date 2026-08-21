from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.sources.health.preprocess.preprocess import (
    _coerce_tabnet_numeric,
    _preprocess_sih_icd10_chapter_request,
    _read_datasus_csv,
)


def test_coerce_tabnet_numeric_treats_bare_dash_as_missing():
    series = pd.Series(["-", "10", "5,3"])

    result = _coerce_tabnet_numeric(series)

    assert result.tolist() == [0.0, 10.0, 5.3]


def test_coerce_tabnet_numeric_preserves_negative_values():
    series = pd.Series(["-5,3", "-12"])

    result = _coerce_tabnet_numeric(series)

    assert result.tolist() == [-5.3, -12.0]


def test_coerce_tabnet_numeric_strips_thousands_separator_without_decimal_comma():
    # A pure Brazilian-formatted integer like "1.234" (meaning 1234) has no
    # decimal comma at all -- the "." must still be treated as a thousands
    # separator and stripped, not parsed as a decimal point.
    series = pd.Series(["1.234", "12.345,67", "-", "-5,3", "999"])

    result = _coerce_tabnet_numeric(series)

    assert result.tolist() == [1234.0, 12345.67, 0.0, -5.3, 999.0]


def test_preprocess_sih_icd10_chapter_request_warns_on_unmapped_chapter_column(tmp_path, caplog):
    # A DATASUS chapter header that doesn't match any key in
    # ICD10_CHAPTER_LABELS (e.g. wording drift, a new "Ignorado" bucket) must
    # not be silently dropped with no trace -- it should at least log a
    # warning naming the dropped column(s).
    frame = pd.DataFrame(
        {
            "request_id": ["r1"],
            "export_year": [2020],
            "metric_key": ["hospitalizations"],
            "Município": ["350000 Some City"],
            "source_key": ["sih"],
            "Cap 01": ["10"],
            "Cap 99 Unrecognized": ["5"],
            "Total": ["15"],
        }
    )

    with caplog.at_level("WARNING"):
        _preprocess_sih_icd10_chapter_request(frame, tmp_path / "out.parquet")

    assert any(
        "Cap 99 Unrecognized" in message and "Dropping" in message
        for message in caplog.messages
    )

    result = pd.read_parquet(tmp_path / "out.parquet")
    matched = result.loc[
        (result["icd10_chapter_code"] == "01")
        & (result["metric_name"] == "hospitalizations_count")
    ]
    assert matched["metric_value"].iloc[0] == 10.0
    # No chapter code corresponds to the unmapped column -- it never made it
    # into the panel at all, mapped or otherwise.
    assert 5.0 not in set(result["metric_value"])


def test_read_datasus_csv_drops_all_trailing_total_rows(tmp_path: Path):
    csv_path = tmp_path / "export.csv"
    csv_path.write_text(
        '"Município";"Valor"\n'
        '"110001 Alta Floresta D\'Oeste";"10"\n'
        '"110002 Ariquemes";"20"\n'
        '"Total";"15"\n'
        '"Total";"30"\n',
        encoding="latin1",
    )

    frame = _read_datasus_csv(str(csv_path))

    assert "Total" not in frame.iloc[:, 0].tolist()
    assert len(frame) == 2
