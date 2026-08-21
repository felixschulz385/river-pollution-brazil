import logging
import os
import csv
import unicodedata

import pandas as pd
from pandas.api.types import is_numeric_dtype
from src.data.shared.batches import load_manifest
from src.data.sources.health.fetch.datasus import SIH_CHANNEL_METRICS, SIH_SELECTED_MORBIDITY_CHANNELS
from src.data.sources.health.fetch.datasus import SIH_ALL_METRICS_KEY
from ..constants import HEALTH_DATASET_NAME
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
    path = os.path.join(root_dir, "data", "health")
    os.makedirs(path, exist_ok=True)
    return path


def _raw_dir(root_dir):
    return os.path.join(_health_dir(root_dir), "raw")


def _processed_dir(root_dir):
    path = os.path.join(_health_dir(root_dir), "processed")
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


def _clean_mortality_age_frame(frame):
    frame = frame.copy()
    frame["year"] = frame["year_code"].astype(int).apply(lambda year: year + 1900 if year > 22 else year + 2000)

    age_columns = [
        "Menor 1 ano",
        "1 a 4 anos",
        "5 a 9 anos",
        "10 a 14 anos",
        "15 a 19 anos",
        "20 a 29 anos",
        "30 a 39 anos",
        "40 a 49 anos",
        "50 a 59 anos",
        "60 a 69 anos",
        "70 a 79 anos",
        "80 anos e mais",
        "Idade ignorada",
    ]
    raw_count_columns = age_columns + ["total"]
    frame[raw_count_columns] = (
        frame[raw_count_columns]
        .apply(_coerce_tabnet_numeric, axis=0)
        .fillna(0)
        .astype("float32")
    )
    frame["mun_id"] = frame["Município"].str.extract(r"(\d{6})")[0].str.zfill(6)
    frame["mun_name"] = frame["Município"].str.extract(r"\d{6}(.*)")[0].str.strip()
    frame = frame.drop(columns=["Município", "year_code"])
    frame = frame[["mun_id", "mun_name", "year"] + [col for col in frame.columns if col not in {"mun_id", "mun_name", "year"}]]
    frame.columns = [
        "mun_id",
        "mun_name",
        "year",
        "under_1",
        "1_to_4",
        "5_to_9",
        "10_to_14",
        "15_to_19",
        "20_to_29",
        "30_to_39",
        "40_to_49",
        "50_to_59",
        "60_to_69",
        "70_to_79",
        "80_and_more",
        "age_unknown",
        "total",
    ]
    count_columns = [
        "under_1",
        "1_to_4",
        "5_to_9",
        "10_to_14",
        "15_to_19",
        "20_to_29",
        "30_to_39",
        "40_to_49",
        "50_to_59",
        "60_to_69",
        "70_to_79",
        "80_and_more",
        "age_unknown",
        "total",
    ]
    frame[count_columns] = frame[count_columns].fillna(0)
    frame = frame.dropna(subset=["mun_id", "mun_name", "year"]).copy()

    municipality_frame = (
        frame[["mun_id", "mun_name"]]
        .drop_duplicates()
        .sort_values(["mun_id", "mun_name"], ignore_index=True)
    )
    year_frame = pd.DataFrame({"year": sorted(frame["year"].dropna().unique())})
    complete_index = municipality_frame.merge(year_frame, how="cross")
    frame = complete_index.merge(frame, on=["mun_id", "mun_name", "year"], how="left")
    frame[count_columns] = frame[count_columns].fillna(0)

    return frame.sort_values(["mun_id", "year"], ignore_index=True)


def preprocess_mortality_age_tables(root_dir="."):
    """Clean raw mortality tables and write final CSV outputs."""
    raw_dir = _raw_dir(root_dir)
    processed_dir = _processed_dir(root_dir)
    sources = {
        "pre_1996": (
            os.path.join(raw_dir, "mortality_age_counts_pre_1996_raw.parquet"),
            os.path.join(processed_dir, "scraping_pre_1996.csv"),
        ),
        "post_1995": (
            os.path.join(raw_dir, "mortality_age_counts_post_1995_raw.parquet"),
            os.path.join(processed_dir, "scraping_post_1996.csv"),
        ),
    }

    outputs = {}
    for period, (raw_path, output_path) in sources.items():
        cleaned = _clean_mortality_age_frame(pd.read_parquet(raw_path))
        cleaned.to_csv(output_path, index=False)
        outputs[period] = output_path
    return outputs


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
    municipality_frame = (
        frame[["municipality_code", "municipality_name"]]
        .drop_duplicates()
        .sort_values(["municipality_code", "municipality_name"], ignore_index=True)
    )
    year_source_frame = (
        frame[["year", "source_system"]]
        .drop_duplicates()
        .sort_values(["year", "source_system"], ignore_index=True)
    )
    metric_frame = pd.DataFrame({"metric_name": metric_names})
    complete_index = municipality_frame.merge(year_source_frame, how="cross")
    complete_index = complete_index.merge(category_frame, how="cross")
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
        os.path.join(processed_dir, "hospitalizations.parquet"),
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
        os.path.join(processed_dir, "hospitalizations_icd10_chapter.parquet"),
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
        os.path.join(processed_dir, "hospitalizations_selected_morbidity_list.parquet"),
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


def preprocess_birth_outcome_tables(root_dir=".", outcome_names=None):
    """Clean raw birth tables and write final parquet outputs."""
    raw_dir = _raw_dir(root_dir)
    processed_dir = _processed_dir(root_dir)
    outputs = {}
    selected_outcomes = outcome_names or ["gestational_duration", "birth_weight"]

    for outcome_name in selected_outcomes:
        raw_path = os.path.join(raw_dir, f"{outcome_name}_raw.parquet")
        cleaned = _clean_birth_outcome_frame(pd.read_parquet(raw_path))
        output_path = os.path.join(processed_dir, f"{outcome_name}.parquet")
        cleaned.to_parquet(output_path, index=False)
        outputs[outcome_name] = output_path

    return outputs


def preprocess_health_data(root_dir=".", subtype="all"):
    """Dispatch health-data preprocessors."""
    valid_subtypes = {"all", "mortality", "hospitalization", "birth"}
    if subtype not in valid_subtypes:
        raise ValueError(
            f"Invalid subtype: {subtype}. Choose from: {', '.join(sorted(valid_subtypes))}"
        )

    outputs = {}
    if subtype in {"all", "mortality"}:
        outputs["mortality"] = preprocess_mortality_age_tables(root_dir=root_dir)
    if subtype in {"all", "hospitalization"}:
        outputs["hospitalization"] = preprocess_hospitalization_tables(root_dir=root_dir)
    if subtype in {"all", "birth"}:
        outputs["birth"] = preprocess_birth_outcome_tables(root_dir=root_dir)
    return outputs
