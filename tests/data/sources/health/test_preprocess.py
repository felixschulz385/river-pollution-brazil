from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.sources.health.preprocess.preprocess import (
    LOW_BIRTH_WEIGHT_BINS,
    MORTALITY_AGE_COLUMNS,
    _clean_infant_death_frame,
    _clean_mortality_age_frame,
    _coerce_tabnet_numeric,
    _preprocess_sih_icd10_chapter_request,
    _read_datasus_csv,
    build_birth_summary,
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


def test_build_birth_summary_counts_low_weight_and_preterm_across_bin_layouts():
    birth_weight = pd.DataFrame(
        {
            "mun_id": ["123456", "654321"],
            "year": [2005, 2015],
            "Menos de 500g": [0.0, 1.0],
            "500 a 999g": [1.0, 0.0],
            "1000 a 1499 g": [0.0, 0.0],
            "1500 a 2499 g": [2.0, 3.0],
            "2500 a 2999 g": [3.0, 4.0],
            "3000 a 3999 g": [10.0, 10.0],
            "4000g e mais": [1.0, 1.0],
            "Ignorado": [1.0, 0.0],
            "Total": [18.0, 19.0],
        }
    )
    # Before ~2011 28-36 weeks is one bin; afterwards it is split 28-31/32-36.
    gestational_duration = pd.DataFrame(
        {
            "mun_id": ["123456", "654321"],
            "year": [2005, 2015],
            "Menos de 22 semanas": [0.0, 0.0],
            "De 22 a 27 semanas": [1.0, 0.0],
            "De 28 a 36 semanas, não especificado": [2.0, None],
            "De 37 a 41 semanas": [14.0, 15.0],
            "42 semanas ou mais": [0.0, 0.0],
            "Ignorado": [1.0, 0.0],
            "De 28 a 31 semanas": [None, 1.0],
            "De 32 a 36 semanas": [None, 3.0],
            "Total": [18.0, 19.0],
        }
    )

    summary = build_birth_summary(birth_weight, gestational_duration).set_index("mun_id")

    assert summary.loc["123456", "births_low_weight"] == 3.0
    assert summary.loc["123456", "births_weight_known"] == 17.0
    assert summary.loc["123456", "births_preterm"] == 3.0
    assert summary.loc["654321", "births_preterm"] == 4.0
    assert summary.loc["654321", "births_gestation_known"] == 19.0


def test_clean_mortality_age_frame_keeps_municipality_rows_and_decodes_years():
    age_columns = [column for column in MORTALITY_AGE_COLUMNS if column != "Total"]
    rows = [
        ("96", " 110001 ALTA FLORESTA D'OESTE", ["12", "2"] + ["-"] * 11, 14.0),
        ("96", " MUNICIPIO IGNORADO - RO", ["9"] + ["-"] * 12, 9.0),
        ("96", "Total", ["1.021", "2"] + ["-"] * 11, 1023.0),
        ("01", "110001 ALTA FLORESTA D'OESTE", ["3"] + ["-"] * 12, 3.0),
    ]
    frame = pd.DataFrame(
        [{"year_code": code, "Município": name, **dict(zip(age_columns, ages)), "Total": total} for code, name, ages, total in rows]
    )

    cleaned = _clean_mortality_age_frame(frame)

    assert cleaned[["mun_id", "year"]].values.tolist() == [["110001", 1996], ["110001", 2001]]
    assert cleaned["mun_name"].iloc[0] == "ALTA FLORESTA D'OESTE"
    assert cleaned["deaths_under_1"].tolist() == [12.0, 3.0]
    assert cleaned["deaths_1_to_4"].tolist() == [2.0, 0.0]
    assert cleaned["deaths_total"].tolist() == [14.0, 3.0]


def test_clean_mortality_age_frame_rejects_missing_age_columns():
    frame = pd.DataFrame({"year_code": ["96"], "Município": ["110001 X"], "Total": [1.0]})

    with pytest.raises(ValueError, match="Menor 1 ano"):
        _clean_mortality_age_frame(frame)


def test_build_birth_summary_adds_anomaly_and_apgar_counts_when_given():
    keys = {"mun_id": ["123456", "654321"], "year": [2015, 2015]}
    birth_weight = pd.DataFrame({**keys, **{bin_: [0.0, 0.0] for bin_ in LOW_BIRTH_WEIGHT_BINS},
                                 "Ignorado": [0.0, 0.0], "Total": [20.0, 10.0]})
    gestational_duration = pd.DataFrame({**keys, "Ignorado": [0.0, 0.0], "Total": [20.0, 10.0]})
    congenital_anomaly = pd.DataFrame({**keys, "Sim": [1.0, 0.0], "Não": [18.0, 10.0], "Ignorado": [1.0, 0.0],
                                       "Total": [20.0, 10.0]})
    # The second municipality's table lacks the empty low bins, as TabNet omits all-zero columns.
    apgar_1 = pd.DataFrame({**keys, "0 a 2": [1.0, None], "3 a 5": [2.0, None], "6 a 7": [3.0, 1.0],
                            "8 a 10": [13.0, 9.0], "Ignorado": [1.0, 0.0], "Total": [20.0, 10.0]})
    apgar_5 = apgar_1.drop(columns=["0 a 2"])

    summary = build_birth_summary(
        birth_weight, gestational_duration, congenital_anomaly=congenital_anomaly, apgar_1=apgar_1, apgar_5=apgar_5
    ).set_index("mun_id")

    assert summary.loc["123456", "births_anomaly"] == 1.0
    assert summary.loc["123456", "births_anomaly_ignored"] == 1.0
    assert summary.loc["123456", "births_apgar1_below_6"] == 3.0
    assert summary.loc["123456", "births_apgar1_below_8"] == 6.0
    assert summary.loc["123456", "births_apgar1_ignored"] == 1.0
    assert summary.loc["123456", "births_apgar5_below_6"] == 2.0
    assert summary.loc["654321", "births_apgar1_below_6"] == 0.0
    assert summary.loc["654321", "births_apgar1_below_8"] == 1.0


def test_build_birth_summary_without_optional_tables_keeps_core_columns():
    keys = {"mun_id": ["123456"], "year": [2015]}
    birth_weight = pd.DataFrame({**keys, **{bin_: [0.0] for bin_ in LOW_BIRTH_WEIGHT_BINS}, "Ignorado": [0.0], "Total": [5.0]})
    gestational_duration = pd.DataFrame({**keys, "Ignorado": [0.0], "Total": [5.0]})

    summary = build_birth_summary(birth_weight, gestational_duration)

    assert not any(column.startswith(("births_anomaly", "births_apgar")) for column in summary.columns)


def test_clean_infant_death_frame_splits_external_causes_and_zero_fills_absent_chapters():
    frame = pd.DataFrame(
        {
            "year_code": ["96", "96", "05"],
            "Município": ["MUNICIPIO IGNORADO - RO", "110002 ARIQUEMES", "110002 ARIQUEMES"],
            "Cap XVI": ["6", "37", "20"],
            "Cap XVII": ["-", "2", "3"],
            "Cap XX": ["-", "1", "-"],
            "Total": [9.0, 69.0, 30.0],
        }
    )

    cleaned = _clean_infant_death_frame(frame)

    assert cleaned[["mun_id", "year"]].values.tolist() == [["110002", 1996], ["110002", 2005]]
    assert cleaned["infant_deaths"].tolist() == [69.0, 30.0]
    assert cleaned["infant_deaths_external"].tolist() == [1.0, 0.0]
    assert cleaned["infant_deaths_nonexternal"].tolist() == [68.0, 30.0]
    assert cleaned["infant_deaths_congenital"].tolist() == [2.0, 3.0]
    assert cleaned["infant_deaths_ill_defined"].tolist() == [0.0, 0.0]
