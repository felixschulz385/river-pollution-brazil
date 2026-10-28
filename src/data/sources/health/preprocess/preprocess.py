import logging
import os
import csv
import unicodedata

import pandas as pd
from pandas.api.types import is_numeric_dtype
from src.data.shared.batches import load_manifest
from src.data.sources.health.fetch.datasus import SIH_CHANNEL_METRICS, SIH_SELECTED_MORBIDITY_CHANNELS
from src.data.sources.health.fetch.datasus import SIH_ALL_METRICS_KEY, tabnet_year_from_code
from ..constants import (
    BIRTHS_FILENAME,
    HEALTH_DATASET_NAME,
    HOSPITALIZATIONS_FILENAME,
    HOSPITALIZATIONS_ICD10_CHAPTER_FILENAME,
    HOSPITALIZATIONS_SELECTED_MORBIDITY_LIST_FILENAME,
    INFANT_DEATHS_FILENAME,
    MORTALITY_FILENAME,
    birth_outcome_filename,
)
from ..constants import health_dir as _shared_health_dir
from ..constants import processed_dir as _shared_processed_dir
from ..constants import raw_dir as _shared_raw_dir
from ..csv_utils import normalize_rows_to_header_width

logger = logging.getLogger(__name__)

SIH_METRIC_NAMES = {
    "hospitalizations": "hospitalizations_count",
    "aih_paid": "aih_paid_count",
    "aih_approved": "aih_approved_count",
    "total_approved_value": "approved_amount_brl",
    "hospital_services_value_brl": "hospital_services_amount_brl",
    "hospital_services_federal_complement_value_brl": "hospital_services_federal_complement_amount_brl",
    "hospital_services_manager_complement_value_brl": "hospital_services_manager_complement_amount_brl",
    "professional_services_value_brl": "professional_services_amount_brl",
    "professional_services_federal_complement_value_brl": "professional_services_federal_complement_amount_brl",
    "professional_services_manager_complement_value_brl": "professional_services_manager_complement_amount_brl",
    "average_aih_value_brl": "average_aih_value_brl",
    "average_hospitalization_value_brl": "average_hospitalization_value_brl",
    "days_of_stay": "inpatient_days_count",
    "average_length_of_stay": "average_length_of_stay_days",
    "in_hospital_deaths": "in_hospital_deaths_count",
    "hospital_mortality_rate": "hospital_mortality_rate_pct",
    "Internações": "hospitalizations_count",
    "AIH Pagas": "aih_paid_count",
    "AIH pagas": "aih_paid_count",
    "AIH aprovadas": "aih_approved_count",
    "Valor total": "approved_amount_brl",
    "Valor Total": "approved_amount_brl",
    "Valor serviços hospitalares": "hospital_services_amount_brl",
    "Val serv hosp - compl federal": "hospital_services_federal_complement_amount_brl",
    "Val serv hosp - compl gestor": "hospital_services_manager_complement_amount_brl",
    "Valor serviços profissionais": "professional_services_amount_brl",
    "Val serv prof - compl federal": "professional_services_federal_complement_amount_brl",
    "Val serv prof - compl gestor": "professional_services_manager_complement_amount_brl",
    "Valor Médio AIH": "average_aih_value_brl",
    "Valor médio AIH": "average_aih_value_brl",
    "Valor Médio Int": "average_hospitalization_value_brl",
    "Valor médio Int": "average_hospitalization_value_brl",
    "Valor médio intern": "average_hospitalization_value_brl",
    "Dias permanência": "inpatient_days_count",
    "Dias Permanência": "inpatient_days_count",
    "Média permanência": "average_length_of_stay_days",
    "Média Permanência": "average_length_of_stay_days",
    "Óbitos": "in_hospital_deaths_count",
    "Taxa mortalidade": "hospital_mortality_rate_pct",
    "Taxa Mortalidade": "hospital_mortality_rate_pct",
}
ICD10_CHAPTER_LABELS = {
    "Cap 01": ("01", "Certain infectious and parasitic diseases"),
    "Cap 02": ("02", "Neoplasms"),
    "Cap 03": ("03", "Diseases of the blood and blood-forming organs and certain immune disorders"),
    "Cap 04": ("04", "Endocrine, nutritional and metabolic diseases"),
    "Cap 05": ("05", "Mental and behavioural disorders"),
    "Cap 06": ("06", "Diseases of the nervous system"),
    "Cap 07": ("07", "Diseases of the eye and adnexa"),
    "Cap 08": ("08", "Diseases of the ear and mastoid process"),
    "Cap 09": ("09", "Diseases of the circulatory system"),
    "Cap 10": ("10", "Diseases of the respiratory system"),
    "Cap 11": ("11", "Diseases of the digestive system"),
    "Cap 12": ("12", "Diseases of the skin and subcutaneous tissue"),
    "Cap 13": ("13", "Diseases of the musculoskeletal system and connective tissue"),
    "Cap 14": ("14", "Diseases of the genitourinary system"),
    "Cap 15": ("15", "Pregnancy, childbirth and the puerperium"),
    "Cap 16": ("16", "Certain conditions originating in the perinatal period"),
    "Cap 17": ("17", "Congenital malformations, deformations and chromosomal abnormalities"),
    "Cap 18": ("18", "Symptoms, signs and abnormal clinical and laboratory findings"),
    "Cap 19": ("19", "Injury, poisoning and certain other consequences of external causes"),
    "Cap 20": ("20", "External causes of morbidity and mortality"),
    "Cap 21": ("21", "Factors influencing health status and contact with health services"),
    "Não disponível": ("99", "Not available"),
}


def _normalize_metric_label(value):
    normalized = unicodedata.normalize("NFKD", str(value))
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.lower().split())
    return normalized


SIH_METRIC_NAMES_NORMALIZED = {
    _normalize_metric_label(key): value for key, value in SIH_METRIC_NAMES.items()
}

def _health_dir(root_dir):
    path = str(_shared_health_dir(root_dir))
    os.makedirs(path, exist_ok=True)
    return path


def _raw_dir(root_dir):
    _health_dir(root_dir)
    return str(_shared_raw_dir(root_dir))


def _processed_dir(root_dir):
    path = str(_shared_processed_dir(root_dir))
    os.makedirs(path, exist_ok=True)
    return path


def _load_completed_batch_frame(root_dir, table_name, legacy_raw_filename=None):
    completed_entries = [
        entry
        for entry in load_manifest(root_dir, HEALTH_DATASET_NAME, table_name)
        if entry.get("status") == "completed"
        and entry.get("raw_path")
        and os.path.exists(entry["raw_path"])
    ]
    if completed_entries:
        frames = []
        for entry in completed_entries:
            raw_path = entry["raw_path"]
            if raw_path.endswith(".csv"):
                frame = _read_datasus_csv(raw_path)
                frame.insert(0, "request_id", entry["request_id"])
                frame.insert(1, "source_key", entry["source_key"])
                frame.insert(2, "export_year", entry["export_year"])
                frame.insert(3, "metric_key", entry["metric_key"])
                insert_at = 4
                if entry.get("morbidity_channel") is not None:
                    frame.insert(insert_at, "morbidity_channel", entry["morbidity_channel"])
                    insert_at += 1
                if entry.get("morbidity_filter_values") is not None:
                    frame.insert(insert_at, "morbidity_filter_values", ",".join(entry["morbidity_filter_values"]))
                    insert_at += 1
                if entry.get("morbidity_filter_value") is not None:
                    frame.insert(insert_at, "morbidity_list_cid10_value", entry["morbidity_filter_value"])
                    insert_at += 1
                if entry.get("morbidity_filter_text") is not None:
                    frame.insert(insert_at, "morbidity_list_cid10", entry["morbidity_filter_text"])
            else:
                frame = pd.read_parquet(raw_path)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    if legacy_raw_filename is not None:
        legacy_path = os.path.join(_raw_dir(root_dir), legacy_raw_filename)
        if os.path.exists(legacy_path):
            return pd.read_parquet(legacy_path)

    raise FileNotFoundError(f"No completed raw batches found for {table_name}")


def _read_datasus_csv(path):
    with open(path, "r", encoding="latin1", errors="ignore", newline="") as handle:
        lines = [line.rstrip("\n\r") for line in handle]

    header_index = next(
        (
            idx
            for idx, line in enumerate(lines)
            if ";" in line and '"Munic' in line
        ),
        None,
    )
    if header_index is None:
        raise ValueError(f"Unable to locate DATASUS CSV header in {path}")

    data_lines = []
    for line in lines[header_index:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Fonte:") or stripped.startswith("Fonte :"):
            break
        data_lines.append(line)

    rows = list(csv.reader(data_lines, delimiter=";", quotechar='"'))
    if not rows:
        return pd.DataFrame()

    header = rows[0]
    body = rows[1:]
    # An export can carry more than one trailing summary/subtotal line
    # depending on the request type, not just a single grand-total row; strip
    # all of them, not just `body[-1]`.
    while body and body[-1] and body[-1][0].strip('"') == "Total":
        logger.debug(
            "Dropping DATASUS grand-total row from %s: %r", path, body[-1]
        )
        body = body[:-1]
    return pd.DataFrame(normalize_rows_to_header_width(header, body), columns=header)


def _empty_total_hospitalization_frame():
    return pd.DataFrame(
        columns=[
            "municipality_code",
            "municipality_name",
            "year",
            "source_system",
            "metric_name",
            "metric_value",
        ]
    )


def _empty_icd10_hospitalization_frame():
    return pd.DataFrame(
        columns=[
            "municipality_code",
            "municipality_name",
            "year",
            "source_system",
            "icd10_chapter_code",
            "icd10_chapter_name",
            "metric_name",
            "metric_value",
        ]
    )


def _empty_morbidity_hospitalization_frame():
    return pd.DataFrame(
        columns=[
            "municipality_code",
            "municipality_name",
            "year",
            "source_system",
            "morbidity_channel",
            "morbidity_list_value",
            "morbidity_list_slug",
            "morbidity_list_name",
            "metric_name",
            "metric_value",
        ]
    )


# SIM TABNET age-band columns -> processed death-count columns.
MORTALITY_AGE_COLUMNS = {
    "Menor 1 ano": "deaths_under_1",
    "1 a 4 anos": "deaths_1_to_4",
    "5 a 9 anos": "deaths_5_to_9",
    "10 a 14 anos": "deaths_10_to_14",
    "15 a 19 anos": "deaths_15_to_19",
    "20 a 29 anos": "deaths_20_to_29",
    "30 a 39 anos": "deaths_30_to_39",
    "40 a 49 anos": "deaths_40_to_49",
    "50 a 59 anos": "deaths_50_to_59",
    "60 a 69 anos": "deaths_60_to_69",
    "70 a 79 anos": "deaths_70_to_79",
    "80 anos e mais": "deaths_80_and_more",
    "Idade ignorada": "deaths_age_unknown",
    "Total": "deaths_total",
}
# The pre-1996 form (obt09br) codes causes in ICD-9, the post-1995 one (obt10br) in ICD-10.
MORTALITY_SOURCE_SYSTEMS = {"pre_1996": "sim_icd9", "post_1995": "sim_icd10"}


def _clean_mortality_age_frame(frame):
    """One row per municipality and year with `deaths_*` counts by age band.

    Rows without a 6-digit municipality code (the national "Total" row, the
    per-state "MUNICIPIO IGNORADO" rows, footnotes) are dropped. Municipalities
    a year's table does not list stay absent rather than zero-filled: SIM lists
    every municipality with at least one death, so a missing row mostly means
    the municipality did not exist yet.
    """
    frame = frame.rename(columns={"total": "Total"})
    missing = sorted(set(MORTALITY_AGE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Mortality table is missing age columns: {missing}")
    municipality = frame["Município"].astype(str).str.strip()
    parts = municipality.str.extract(r"^(\d{6})\s*(.*)$")
    out = pd.DataFrame(
        {
            "mun_id": parts[0],
            "mun_name": parts[1].str.strip(),
            "year": frame["year_code"].map(tabnet_year_from_code),
        }
    )
    for raw_column, column in MORTALITY_AGE_COLUMNS.items():
        out[column] = _coerce_tabnet_numeric(frame[raw_column]).fillna(0).astype("float32")
    out = out.dropna(subset=["mun_id"])
    if out.duplicated(["mun_id", "year"]).any():
        raise ValueError("Mortality table has duplicate (mun_id, year) rows.")
    return out.sort_values(["mun_id", "year"], ignore_index=True)


def preprocess_mortality_age_tables(root_dir="."):
    """Clean the pre-1996 and post-1995 SIM tables and stack them into `health_mortality.parquet`."""
    raw_dir = _raw_dir(root_dir)
    frames = []
    for period, source_system in MORTALITY_SOURCE_SYSTEMS.items():
        raw_path = os.path.join(raw_dir, f"mortality_age_counts_{period}_raw.parquet")
        cleaned = _clean_mortality_age_frame(pd.read_parquet(raw_path))
        cleaned.insert(3, "source_system", source_system)
        frames.append(cleaned)
    mortality = pd.concat(frames, ignore_index=True)
    if mortality.duplicated(["mun_id", "year"]).any():
        raise ValueError("Pre-1996 and post-1995 SIM tables overlap in a year.")
    output_path = os.path.join(_processed_dir(root_dir), MORTALITY_FILENAME)
    mortality.to_parquet(output_path, index=False)
    logger.info("Wrote %s (%d rows)", output_path, len(mortality))
    return output_path


# SIM infant-death ICD-10 chapter columns -> processed columns; `Total` covers all chapters.
INFANT_DEATH_CHAPTER_COLUMNS = {
    "Total": "infant_deaths",
    "Cap XVII": "infant_deaths_congenital",
    "Cap XVIII": "infant_deaths_ill_defined",
    "Cap XX": "infant_deaths_external",
}


def _clean_infant_death_frame(frame):
    """One row per municipality and year with infant deaths in total and for the chapters of interest.

    As for the age table, rows without a 6-digit municipality code are dropped
    and unlisted municipalities stay absent. A chapter with no deaths in a year
    is absent from that year's table and counts as zero.
    """
    if "Total" not in frame.columns:
        raise ValueError("Infant-death table has no 'Total' column.")
    parts = frame["Município"].astype(str).str.strip().str.extract(r"^(\d{6})\s*(.*)$")
    out = pd.DataFrame({"mun_id": parts[0], "year": frame["year_code"].map(tabnet_year_from_code)})
    for raw_column, column in INFANT_DEATH_CHAPTER_COLUMNS.items():
        values = frame[raw_column] if raw_column in frame.columns else pd.Series(0.0, index=frame.index)
        out[column] = _coerce_tabnet_numeric(values).fillna(0).astype("float32")
    out = out.dropna(subset=["mun_id"])
    if out.duplicated(["mun_id", "year"]).any():
        raise ValueError("Infant-death table has duplicate (mun_id, year) rows.")
    out["infant_deaths_nonexternal"] = out["infant_deaths"] - out["infant_deaths_external"]
    return out.sort_values(["mun_id", "year"], ignore_index=True)


def preprocess_infant_death_tables(root_dir="."):
    """Clean the SIM infant-death table by ICD-10 chapter into `health_infant_deaths.parquet`."""
    raw_path = os.path.join(_raw_dir(root_dir), "infant_deaths_icd10_raw.parquet")
    infant = _clean_infant_death_frame(pd.read_parquet(raw_path))
    output_path = os.path.join(_processed_dir(root_dir), INFANT_DEATHS_FILENAME)
    infant.to_parquet(output_path, index=False)
    logger.info("Wrote %s (%d rows)", output_path, len(infant))
    return output_path


def _coerce_tabnet_numeric(series):
    if is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    normalized = series.astype(str).str.strip()
    # DATASUS uses a bare "-" as its no-data placeholder for a whole cell.
    # Only replace that exact placeholder, not every "-" substring, or a
    # genuine negative value like "-5,3" gets its sign stripped ("-5,3" ->
    # "05,3" -> 5.3) instead of being parsed as -5.3.
    normalized = normalized.where(normalized != "-", "0")
    # DATASUS-formatted numbers use "." exclusively as a thousands separator
    # (never a decimal point) and "," as the decimal separator, so it's safe
    # to strip every "." unconditionally before converting "," to ".": e.g.
    # "1.234" (a plain integer, no decimal comma) must become 1234, not the
    # float 1.234, and "12.345,67" must become 12345.67.
    normalized = normalized.str.replace(".", "", regex=False)
    normalized = normalized.str.replace(",", ".", regex=False)

    return pd.to_numeric(
        normalized,
        errors="coerce",
    )


def _extract_municipality_fields(frame):
    frame = frame.copy()
    frame["municipality_code"] = frame["Município"].str.extract(r"(\d{6})")[0].str.zfill(6)
    frame["municipality_name"] = frame["Município"].str.extract(r"\d{6}(.*)")[0].str.strip()
    cleaned = frame.dropna(subset=["municipality_code"])
    n_dropped = len(frame) - len(cleaned)
    if n_dropped:
        logger.warning(
            "Dropped %d row(s) with no parseable 6-digit municipality code out of %d.",
            n_dropped,
            len(frame),
        )
    return cleaned


def _metric_name(metric_value):
    if metric_value in SIH_METRIC_NAMES.values():
        return metric_value
    if metric_value in SIH_METRIC_NAMES:
        return SIH_METRIC_NAMES[metric_value]

    normalized_metric_value = _normalize_metric_label(metric_value)
    if normalized_metric_value in SIH_METRIC_NAMES_NORMALIZED:
        return SIH_METRIC_NAMES_NORMALIZED[normalized_metric_value]

    raise KeyError(metric_value)


def _slugify_label(value):
    return "".join(char.lower() if char.isalnum() else "_" for char in str(value)).strip("_")


def _single_value_column(frame, excluded_columns):
    value_columns = [column for column in frame.columns if column not in excluded_columns]
    if len(value_columns) != 1:
        raise ValueError(f"Expected exactly one value column, found {value_columns}")
    return value_columns[0]


def _write_parquet(frame, output_path, columns):
    frame = frame.reindex(columns=columns)
    frame.to_parquet(output_path, index=False)
    return output_path


def _complete_panel(frame, category_columns, category_frame, metric_names):
    """Zero-fill municipalities a fetched table omits (TABNET drops all-zero rows).

    Only (year, source_system, category) cells with at least one fetched row
    are completed: a batch that was never fetched (e.g. a morbidity channel
    missing from the manifest) must stay missing rather than become zeros.
    `category_frame` restricts the categories considered.
    """
    municipality_frame = (
        frame[["municipality_code", "municipality_name"]]
        .drop_duplicates()
        .sort_values(["municipality_code", "municipality_name"], ignore_index=True)
    )
    batch_columns = ["year", "source_system", *category_columns]
    fetched_batches = (
        frame[batch_columns]
        .drop_duplicates()
        .merge(category_frame, on=list(category_columns), how="inner")
        .sort_values(batch_columns, ignore_index=True)
    )
    metric_frame = pd.DataFrame({"metric_name": metric_names})
    complete_index = municipality_frame.merge(fetched_batches, how="cross")
    complete_index = complete_index.merge(metric_frame, how="cross")
    completed = complete_index.merge(
        frame,
        on=[
            "municipality_code",
            "municipality_name",
            "year",
            "source_system",
            *category_columns,
            "metric_name",
        ],
        how="left",
    )
    completed["metric_value"] = completed["metric_value"].fillna(0)
    return completed


def _preprocess_sih_total_request(raw_path, output_path):
    total_columns = list(_empty_total_hospitalization_frame().columns)
    frame = pd.read_parquet(raw_path) if isinstance(raw_path, str) else raw_path
    if frame.empty:
        return _write_parquet(_empty_total_hospitalization_frame(), output_path, total_columns)

    frame = _extract_municipality_fields(frame)
    if {"export_year", "metric_key", "Total"}.issubset(frame.columns):
        long_frame = frame[
            [
                "municipality_code",
                "municipality_name",
                "source_key",
                "export_year",
                "metric_key",
                "Total",
            ]
        ].rename(
            columns={
                "source_key": "source_system",
                "export_year": "year",
                "metric_key": "metric_name",
                "Total": "metric_value",
            }
        )
        long_frame["metric_name"] = long_frame["metric_name"].map(_metric_name)
        long_frame["metric_value"] = _coerce_tabnet_numeric(long_frame["metric_value"])
    elif {"export_year", "metric_key", "source_key"}.issubset(frame.columns):
        value_columns = [
            column
            for column in frame.columns
            if column
            not in {
                "request_id",
                "source_key",
                "export_year",
                "metric_key",
                "Município",
                "municipality_code",
                "municipality_name",
            }
        ]
        long_frame = frame.melt(
            id_vars=[
                "municipality_code",
                "municipality_name",
                "source_key",
                "export_year",
                "metric_key",
            ],
            value_vars=value_columns,
            var_name="raw_metric_name",
            value_name="metric_value",
        )
        long_frame = long_frame.dropna(subset=["metric_value"]).copy()
        long_frame["source_system"] = long_frame["source_key"]
        long_frame["year"] = long_frame["export_year"].astype(int)
        if long_frame["metric_key"].eq(SIH_ALL_METRICS_KEY).all():
            long_frame["metric_name"] = long_frame["raw_metric_name"].map(_metric_name)
        else:
            long_frame["metric_name"] = long_frame["metric_key"].map(_metric_name)
            long_frame["raw_metric_name_normalized"] = long_frame["raw_metric_name"].map(_metric_name)
            long_frame = long_frame[
                long_frame["metric_name"].eq(long_frame["raw_metric_name_normalized"])
            ].copy()
        long_frame = long_frame[
            [
                "municipality_code",
                "municipality_name",
                "year",
                "source_system",
                "metric_name",
                "metric_value",
            ]
        ]
        long_frame["metric_value"] = _coerce_tabnet_numeric(long_frame["metric_value"])
    else:
        year_columns = [column for column in frame.columns if str(column).isdigit()]
        long_frame = frame.melt(
            id_vars=["Município", "municipality_code", "municipality_name", "content_metric"],
            value_vars=year_columns,
            var_name="year",
            value_name="metric_value",
        )
        long_frame["year"] = long_frame["year"].astype(int)
        long_frame["metric_value"] = _coerce_tabnet_numeric(long_frame["metric_value"])
        long_frame["source_system"] = None
        long_frame["metric_name"] = long_frame["content_metric"].map(_metric_name)
        long_frame = long_frame[
            [
                "municipality_code",
                "municipality_name",
                "year",
                "source_system",
                "metric_name",
                "metric_value",
            ]
        ]

    long_frame["year"] = long_frame["year"].astype(int)
    tidy_frame = (
        long_frame.dropna(subset=["metric_value", "metric_name"])
        .sort_values(["municipality_code", "year", "metric_name"], ignore_index=True)
    )
    return _write_parquet(tidy_frame, output_path, total_columns)


def _preprocess_sih_icd10_chapter_request(raw_path, output_path):
    output_columns = list(_empty_icd10_hospitalization_frame().columns)
    frame = pd.read_parquet(raw_path) if isinstance(raw_path, str) else raw_path
    if frame.empty:
        return _write_parquet(_empty_icd10_hospitalization_frame(), output_path, output_columns)

    frame = _extract_municipality_fields(frame)
    base_id_columns = {
        "request_id",
        "export_year",
        "metric_key",
        "Município",
        "municipality_code",
        "municipality_name",
        "source_key",
    }
    chapter_columns = [column for column in frame.columns if column not in base_id_columns and column != "Total"]
    long_frame = frame.melt(
        id_vars=[
            "source_key",
            "export_year",
            "metric_key",
            "municipality_code",
            "municipality_name",
        ],
        value_vars=chapter_columns,
        var_name="raw_icd10_chapter",
        value_name="metric_value",
    )
    long_frame["metric_value"] = _coerce_tabnet_numeric(long_frame["metric_value"])
    long_frame = long_frame.dropna(subset=["metric_value"]).copy()
    unmapped_chapters = sorted(
        set(long_frame["raw_icd10_chapter"]) - set(ICD10_CHAPTER_LABELS)
    )
    if unmapped_chapters:
        logger.warning(
            "Dropping %d ICD-10 chapter column(s) with no entry in "
            "ICD10_CHAPTER_LABELS (DATASUS header text may have changed): %s",
            len(unmapped_chapters),
            unmapped_chapters,
        )
    long_frame = long_frame[long_frame["raw_icd10_chapter"].isin(ICD10_CHAPTER_LABELS)].copy()
    long_frame["year"] = long_frame["export_year"].astype(int)
    long_frame["source_system"] = long_frame["source_key"]
    long_frame["metric_name"] = long_frame["metric_key"].map(_metric_name)
    long_frame["icd10_chapter_code"] = long_frame["raw_icd10_chapter"].map(
        lambda value: ICD10_CHAPTER_LABELS[value][0]
    )
    long_frame["icd10_chapter_name"] = long_frame["raw_icd10_chapter"].map(
        lambda value: ICD10_CHAPTER_LABELS[value][1]
    )
    tidy_frame = long_frame[
        [
            "municipality_code",
            "municipality_name",
            "year",
            "source_system",
            "icd10_chapter_code",
            "icd10_chapter_name",
            "metric_name",
            "metric_value",
        ]
    ]
    category_frame = pd.DataFrame(
        [
            {
                "icd10_chapter_code": code,
                "icd10_chapter_name": name,
            }
            for code, name in ICD10_CHAPTER_LABELS.values()
        ]
    )
    tidy_frame = _complete_panel(
        tidy_frame,
        ["icd10_chapter_code", "icd10_chapter_name"],
        category_frame,
        [_metric_name(metric_name) for metric_name in SIH_CHANNEL_METRICS],
    ).sort_values(
        ["municipality_code", "year", "icd10_chapter_code", "metric_name"],
        ignore_index=True,
    )
    return _write_parquet(tidy_frame, output_path, output_columns)


def _preprocess_sih_morbidity_request(raw_path, output_path):
    output_columns = list(_empty_morbidity_hospitalization_frame().columns)
    frame = pd.read_parquet(raw_path) if isinstance(raw_path, str) else raw_path
    if frame.empty:
        return _write_parquet(_empty_morbidity_hospitalization_frame(), output_path, output_columns)

    frame = _extract_municipality_fields(frame)
    if not {"source_key", "export_year", "metric_key"}.issubset(frame.columns):
        raise ValueError("Unexpected SIH morbidity raw schema.")

    if "morbidity_channel" in frame.columns:
        id_columns = [
            "municipality_code",
            "municipality_name",
            "source_key",
            "export_year",
            "metric_key",
            "morbidity_channel",
        ]
        value_columns = [
            column
            for column in frame.columns
            if column not in set(id_columns + ["request_id", "morbidity_filter_values", "Município"])
        ]
        is_single_metric_batch = (
            len(value_columns) == 1
            and frame["metric_key"].nunique() == 1
            and frame["metric_key"].iloc[0] != "all_metrics"
        )
        if is_single_metric_batch:
            tidy_frame = frame[id_columns + value_columns].rename(
                columns={
                    "source_key": "source_system",
                    "export_year": "year",
                    "metric_key": "metric_name",
                    value_columns[0]: "metric_value",
                }
            )
            tidy_frame["metric_name"] = tidy_frame["metric_name"].map(_metric_name)
        else:
            tidy_frame = frame.melt(
                id_vars=id_columns,
                value_vars=value_columns,
                var_name="raw_metric_name",
                value_name="metric_value",
            )
            tidy_frame = tidy_frame.dropna(subset=["metric_value"]).rename(
                columns={
                    "source_key": "source_system",
                    "export_year": "year",
                }
            )
            tidy_frame["metric_name"] = tidy_frame["raw_metric_name"].map(_metric_name)
            tidy_frame = tidy_frame.drop(columns=["raw_metric_name", "metric_key"])
        tidy_frame["morbidity_list_value"] = tidy_frame["morbidity_channel"]
        tidy_frame["morbidity_list_name"] = tidy_frame["morbidity_channel"]
        tidy_frame["morbidity_list_slug"] = tidy_frame["morbidity_channel"]
    else:
        if "Total" in frame.columns:
            value_column = "Total"
        else:
            value_column = _single_value_column(
                frame,
                {
                    "request_id",
                    "source_key",
                    "export_year",
                    "metric_key",
                    "morbidity_list_cid10_value",
                    "morbidity_list_cid10",
                    "Município",
                    "municipality_code",
                    "municipality_name",
                },
            )
        channel_by_value = {
            value: channel
            for channel, values in SIH_SELECTED_MORBIDITY_CHANNELS.items()
            for value in values
        }
        tidy_frame = frame[
            [
                "municipality_code",
                "municipality_name",
                "source_key",
                "export_year",
                "metric_key",
                "morbidity_list_cid10_value",
                "morbidity_list_cid10",
                value_column,
            ]
        ].rename(
            columns={
                "source_key": "source_system",
                "export_year": "year",
                "metric_key": "metric_name",
                "morbidity_list_cid10_value": "morbidity_list_value",
                "morbidity_list_cid10": "morbidity_list_name",
                value_column: "metric_value",
            }
        )
        tidy_frame["morbidity_list_value"] = tidy_frame["morbidity_list_value"].astype(str)
        tidy_frame["morbidity_channel"] = tidy_frame["morbidity_list_value"].map(channel_by_value)
        tidy_frame["morbidity_list_slug"] = tidy_frame["morbidity_list_name"].map(_slugify_label)

    tidy_frame["year"] = tidy_frame["year"].astype(int)
    tidy_frame["metric_name"] = tidy_frame["metric_name"].map(_metric_name)
    tidy_frame["metric_value"] = _coerce_tabnet_numeric(tidy_frame["metric_value"])
    tidy_frame = tidy_frame[
        [
            "municipality_code",
            "municipality_name",
            "year",
            "source_system",
            "morbidity_channel",
            "morbidity_list_value",
            "morbidity_list_slug",
            "morbidity_list_name",
            "metric_name",
            "metric_value",
        ]
    ].dropna(subset=["metric_value", "metric_name"])
    category_columns = [
        "morbidity_channel",
        "morbidity_list_value",
        "morbidity_list_slug",
        "morbidity_list_name",
    ]
    category_frame = tidy_frame[category_columns].drop_duplicates().sort_values(
        category_columns,
        ignore_index=True,
    )
    is_channel_level_frame = (
        not category_frame.empty
        and category_frame["morbidity_channel"].eq(category_frame["morbidity_list_value"]).all()
        and category_frame["morbidity_channel"].eq(category_frame["morbidity_list_slug"]).all()
        and category_frame["morbidity_channel"].eq(category_frame["morbidity_list_name"]).all()
    )
    if is_channel_level_frame:
        category_frame = pd.DataFrame(
            [
                {
                    "morbidity_channel": channel,
                    "morbidity_list_value": channel,
                    "morbidity_list_slug": channel,
                    "morbidity_list_name": channel,
                }
                for channel in SIH_SELECTED_MORBIDITY_CHANNELS
            ]
        )
    tidy_frame = _complete_panel(
        tidy_frame,
        category_columns,
        category_frame,
        [_metric_name(metric_name) for metric_name in SIH_CHANNEL_METRICS],
    ).sort_values(
        ["municipality_code", "year", "morbidity_channel", "morbidity_list_value", "metric_name"],
        ignore_index=True,
    )
    return _write_parquet(tidy_frame, output_path, output_columns)


def preprocess_hospitalization_tables(root_dir="."):
    """Clean SIH residence hospitalization requests and write final parquet outputs."""
    processed_dir = _processed_dir(root_dir)
    outputs = {}
    try:
        total_frame = _load_completed_batch_frame(
            root_dir,
            "SIH_RESIDENCE_TOTAL_MUNICIPALITY_YEAR",
            legacy_raw_filename="sih_residence_total_municipality_year_raw.parquet",
        )
    except FileNotFoundError:
        total_frame = _empty_total_hospitalization_frame()
    outputs["SIH_RESIDENCE_TOTAL_MUNICIPALITY_YEAR"] = _preprocess_sih_total_request(
        total_frame,
        os.path.join(processed_dir, HOSPITALIZATIONS_FILENAME),
    )

    try:
        icd10_frame = _load_completed_batch_frame(
            root_dir,
            "SIH_RESIDENCE_ICD10_CHAPTER_MUNICIPALITY_YEAR",
            legacy_raw_filename="sih_residence_icd10_chapter_municipality_year_raw.parquet",
        )
    except FileNotFoundError:
        icd10_frame = _empty_icd10_hospitalization_frame()
    outputs["SIH_RESIDENCE_ICD10_CHAPTER_MUNICIPALITY_YEAR"] = _preprocess_sih_icd10_chapter_request(
        icd10_frame,
        os.path.join(processed_dir, HOSPITALIZATIONS_ICD10_CHAPTER_FILENAME),
    )

    try:
        morbidity_frame = _load_completed_batch_frame(
            root_dir,
            "SIH_RESIDENCE_SELECTED_MORBIDITY_LIST_MUNICIPALITY_YEAR",
            legacy_raw_filename="sih_residence_selected_morbidity_list_municipality_year_raw.parquet",
        )
    except FileNotFoundError:
        morbidity_frame = _empty_morbidity_hospitalization_frame()
    outputs["SIH_RESIDENCE_SELECTED_MORBIDITY_LIST_MUNICIPALITY_YEAR"] = _preprocess_sih_morbidity_request(
        morbidity_frame,
        os.path.join(processed_dir, HOSPITALIZATIONS_SELECTED_MORBIDITY_LIST_FILENAME),
    )
    return outputs


def _clean_birth_outcome_frame(frame):
    frame = frame.copy()
    frame["mun_id"] = frame["Município"].str.extract(r"(\d{6})")[0].str.zfill(6)
    frame["mun_name"] = frame["Município"].str.extract(r"\d{6}(.*)")[0].str.strip()
    frame = frame.drop(columns=["Município"])
    value_columns = [col for col in frame.columns if col not in {"mun_id", "mun_name", "year", "Total"}]
    frame[value_columns] = frame[value_columns].apply(_coerce_tabnet_numeric, axis=0).astype("float32")
    frame["Total"] = _coerce_tabnet_numeric(frame["Total"]).astype("float32")
    frame = frame[["mun_id", "mun_name", "year"] + value_columns + ["Total"]]
    return frame.dropna(subset=["mun_id"])


# Required for the births summary table; the others are added to it when fetched.
CORE_BIRTH_OUTCOMES = ("gestational_duration", "birth_weight")
OPTIONAL_BIRTH_OUTCOMES = ("congenital_anomaly", "apgar_1", "apgar_5")


def preprocess_birth_outcome_tables(root_dir=".", outcome_names=None):
    """Clean raw birth tables and write final parquet outputs.

    Without `outcome_names`, cleans the core outcomes and every optional one
    whose raw table has been fetched.
    """
    raw_dir = _raw_dir(root_dir)
    processed_dir = _processed_dir(root_dir)
    outputs = {}
    selected_outcomes = outcome_names or [
        *CORE_BIRTH_OUTCOMES,
        *(name for name in OPTIONAL_BIRTH_OUTCOMES if os.path.exists(os.path.join(raw_dir, f"{name}_raw.parquet"))),
    ]

    for outcome_name in selected_outcomes:
        raw_path = os.path.join(raw_dir, f"{outcome_name}_raw.parquet")
        cleaned = _clean_birth_outcome_frame(pd.read_parquet(raw_path))
        output_path = os.path.join(processed_dir, birth_outcome_filename(outcome_name))
        cleaned.to_parquet(output_path, index=False)
        outputs[outcome_name] = output_path

    if set(CORE_BIRTH_OUTCOMES).issubset(selected_outcomes):
        outputs["births"] = write_birth_summary_table(root_dir=root_dir)
    return outputs


LOW_BIRTH_WEIGHT_BINS = ["Menos de 500g", "500 a 999g", "1000 a 1499 g", "1500 a 2499 g"]
# 28-36 weeks is reported either as one bin or split into 28-31/32-36
# depending on the year; summing with min_count=0 covers both layouts.
PRETERM_BINS = [
    "Menos de 22 semanas",
    "De 22 a 27 semanas",
    "De 28 a 36 semanas, não especificado",
    "De 28 a 31 semanas",
    "De 32 a 36 semanas",
]


# TabNet bins APGAR as 0-2, 3-5, 6-7 and 8-10, so the clinical cut-off (< 7)
# is not available; < 6 (0-5) and < 8 (0-7) bracket it.
APGAR_BELOW_6_BINS = ["0 a 2", "3 a 5"]
APGAR_BELOW_8_BINS = [*APGAR_BELOW_6_BINS, "6 a 7"]
ANOMALY_DETECTED = "Sim"


def _anomaly_counts(congenital_anomaly):
    counts = congenital_anomaly[["mun_id", "year"]].copy()
    counts["births_anomaly"] = congenital_anomaly.reindex(columns=[ANOMALY_DETECTED]).sum(axis=1, min_count=0)
    counts["births_anomaly_ignored"] = congenital_anomaly.reindex(columns=["Ignorado"]).sum(axis=1, min_count=0)
    return counts


def _apgar_counts(apgar, minute):
    counts = apgar[["mun_id", "year"]].copy()
    counts[f"births_apgar{minute}_below_6"] = apgar.reindex(columns=APGAR_BELOW_6_BINS).sum(axis=1, min_count=0)
    counts[f"births_apgar{minute}_below_8"] = apgar.reindex(columns=APGAR_BELOW_8_BINS).sum(axis=1, min_count=0)
    counts[f"births_apgar{minute}_ignored"] = apgar.reindex(columns=["Ignorado"]).sum(axis=1, min_count=0)
    return counts


def build_birth_summary(birth_weight, gestational_duration, congenital_anomaly=None, apgar_1=None, apgar_5=None):
    """Collapse the cleaned birth-outcome tables into counts per (mun_id, year).

    `*_known` counts exclude the "Ignorado" bin and are the denominators of the
    low-birth-weight and preterm shares. When given, the congenital-anomaly and
    APGAR tables add births with a detected anomaly, with APGAR below 6 and
    below 8 at 1 and 5 minutes, and each table's "Ignorado" count (the
    denominators are `births` minus those).
    """
    births = birth_weight[["mun_id", "year"]].copy()
    births["births"] = birth_weight["Total"]
    births["births_weight_known"] = birth_weight["Total"] - birth_weight["Ignorado"]
    births["births_low_weight"] = birth_weight[LOW_BIRTH_WEIGHT_BINS].sum(axis=1)

    preterm = gestational_duration[["mun_id", "year"]].copy()
    preterm["births_gestation_known"] = gestational_duration["Total"] - gestational_duration["Ignorado"]
    preterm["births_preterm"] = gestational_duration.reindex(columns=PRETERM_BINS).sum(axis=1, min_count=0)
    summary = births.merge(preterm, on=["mun_id", "year"], how="outer", validate="one_to_one")
    for counts in (
        None if congenital_anomaly is None else _anomaly_counts(congenital_anomaly),
        None if apgar_1 is None else _apgar_counts(apgar_1, 1),
        None if apgar_5 is None else _apgar_counts(apgar_5, 5),
    ):
        if counts is not None:
            summary = summary.merge(counts, on=["mun_id", "year"], how="left", validate="one_to_one")
    value_columns = [column for column in summary.columns if column not in {"mun_id", "year"}]
    summary[value_columns] = summary[value_columns].astype("float64")
    return summary


def write_birth_summary_table(root_dir="."):
    """Build `health_births.parquet` from the processed birth-outcome tables.

    Reads the processed (not raw) tables, so it can be rerun without re-fetching SINASC.
    The congenital-anomaly and APGAR tables are used when they have been processed.
    """
    processed_dir = _processed_dir(root_dir)
    optional = {}
    for outcome_name in OPTIONAL_BIRTH_OUTCOMES:
        path = os.path.join(processed_dir, birth_outcome_filename(outcome_name))
        if os.path.exists(path):
            optional[outcome_name] = pd.read_parquet(path)
    summary = build_birth_summary(
        pd.read_parquet(os.path.join(processed_dir, birth_outcome_filename("birth_weight"))),
        pd.read_parquet(os.path.join(processed_dir, birth_outcome_filename("gestational_duration"))),
        **optional,
    )
    output_path = os.path.join(processed_dir, BIRTHS_FILENAME)
    summary.to_parquet(output_path, index=False)
    logger.info("Wrote %s (%d rows)", output_path, len(summary))
    return output_path


def preprocess_health_data(root_dir=".", subtype="all"):
    """Dispatch health-data preprocessors."""
    valid_subtypes = {"all", "mortality", "hospitalization", "birth", "birth_summary"}
    if subtype not in valid_subtypes:
        raise ValueError(
            f"Invalid subtype: {subtype}. Choose from: {', '.join(sorted(valid_subtypes))}"
        )

    outputs = {}
    if subtype in {"all", "mortality"}:
        outputs["mortality"] = preprocess_mortality_age_tables(root_dir=root_dir)
        outputs["infant_deaths"] = preprocess_infant_death_tables(root_dir=root_dir)
    if subtype in {"all", "hospitalization"}:
        outputs["hospitalization"] = preprocess_hospitalization_tables(root_dir=root_dir)
    if subtype in {"all", "birth"}:
        outputs["birth"] = preprocess_birth_outcome_tables(root_dir=root_dir)
    if subtype == "birth_summary":
        outputs["birth_summary"] = write_birth_summary_table(root_dir=root_dir)
    return outputs
