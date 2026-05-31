import os

import pandas as pd


def _health_dir(root_dir):
    path = os.path.join(root_dir, "data", "health")
    os.makedirs(path, exist_ok=True)
    return path


def _raw_dir(root_dir):
    return os.path.join(_health_dir(root_dir), "raw")


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
    frame[age_columns] = frame[age_columns].apply(lambda col: col.str.replace("-", "0"), axis=0).astype("float32")
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
    return frame.dropna()


def preprocess_mortality_age_tables(root_dir="."):
    """Clean raw mortality tables and write final CSV outputs."""
    raw_dir = _raw_dir(root_dir)
    health_dir = _health_dir(root_dir)
    sources = {
        "pre_1996": (
            os.path.join(raw_dir, "mortality_age_counts_pre_1996_raw.parquet"),
            os.path.join(health_dir, "scraping_pre_1996.csv"),
        ),
        "post_1995": (
            os.path.join(raw_dir, "mortality_age_counts_post_1995_raw.parquet"),
            os.path.join(health_dir, "scraping_post_1996.csv"),
        ),
    }

    outputs = {}
    for period, (raw_path, output_path) in sources.items():
        cleaned = _clean_mortality_age_frame(pd.read_parquet(raw_path))
        cleaned.to_csv(output_path, index=False)
        outputs[period] = output_path
    return outputs


def _clean_hospitalization_frame(frame):
    frame = frame.copy()
    frame["year"] = frame["year_code"].astype(int).apply(lambda year: year + 1900 if year > 22 else year + 2000)
    frame["CC_2r"] = frame["Município"].str.extract(r"(\d{6})")[0].str.zfill(6)
    frame = frame.drop(columns=["Município", "year_code"])
    frame = frame[["CC_2r", "year"] + [col for col in frame.columns if col not in {"CC_2r", "year"}]]
    frame.columns = ["CC_2r", "year", "n_approved", "hospitalizations", "total_value"]
    return frame.dropna()


def _coerce_tabnet_numeric(series):
    return pd.to_numeric(
        series.astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.replace("-", "0", regex=False),
        errors="coerce",
    )


def _extract_municipality_fields(frame):
    frame = frame.copy()
    frame["municipality_code"] = frame["Município"].str.extract(r"(\d{6})")[0].str.zfill(6)
    frame["municipality_name"] = frame["Município"].str.extract(r"\d{6}(.*)")[0].str.strip()
    return frame.dropna(subset=["municipality_code"])


def _preprocess_sih_total_request(raw_path, output_path):
    frame = _extract_municipality_fields(pd.read_parquet(raw_path))
    if {"export_year", "metric_key", "Total"}.issubset(frame.columns):
        long_frame = frame[
            [
                "municipality_code",
                "municipality_name",
                "export_year",
                "metric_key",
                "Total",
            ]
        ].rename(
            columns={
                "export_year": "year",
                "metric_key": "metric",
                "Total": "value",
            }
        )
        long_frame["year"] = long_frame["year"].astype(int)
        long_frame["value"] = _coerce_tabnet_numeric(long_frame["value"])
    else:
        year_columns = [column for column in frame.columns if str(column).isdigit()]
        long_frame = frame.melt(
            id_vars=["Município", "municipality_code", "municipality_name", "content_metric"],
            value_vars=year_columns,
            var_name="year",
            value_name="value",
        )
        long_frame["year"] = long_frame["year"].astype(int)
        long_frame["value"] = _coerce_tabnet_numeric(long_frame["value"])
        metric_map = {
            "Internações": "hospitalizations",
            "Valor total": "total_approved_value",
            "Dias permanência": "days_of_stay",
            "Média permanência": "average_length_of_stay",
            "Óbitos": "in_hospital_deaths",
            "Taxa mortalidade": "hospital_mortality_rate",
        }
        long_frame["metric"] = long_frame["content_metric"].map(metric_map)
    wide_frame = (
        long_frame.dropna(subset=["value"])
        .pivot_table(
            index=["municipality_code", "municipality_name", "year"],
            columns="metric",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )
    wide_frame.to_parquet(output_path, index=False)
    return output_path


def _preprocess_sih_channel_request(raw_path, output_path, category_column_name):
    frame = _extract_municipality_fields(pd.read_parquet(raw_path))
    base_id_columns = {
        "request_id",
        "export_year",
        "metric_key",
        "Município",
        "municipality_code",
        "municipality_name",
        "source_key",
    }

    if category_column_name in frame.columns and "Total" in frame.columns:
        long_frame = frame[
            [
                "request_id",
                "export_year",
                "metric_key",
                "Município",
                "municipality_code",
                "municipality_name",
                category_column_name,
                "Total",
            ]
        ].rename(columns={"Total": "value", "metric_key": "metric"})
    else:
        category_columns = [column for column in frame.columns if column not in base_id_columns]
        long_frame = frame.melt(
            id_vars=[
                "request_id",
                "export_year",
                "metric_key",
                "Município",
                "municipality_code",
                "municipality_name",
            ],
            value_vars=category_columns,
            var_name=category_column_name,
            value_name="value",
        )
        long_frame = long_frame[long_frame[category_column_name].ne("Total")].copy()
        long_frame = long_frame.rename(columns={"metric_key": "metric"})

    long_frame["year"] = long_frame["export_year"].astype(int)
    long_frame["value"] = _coerce_tabnet_numeric(long_frame["value"])
    wide_frame = (
        long_frame.dropna(subset=["value"])
        .pivot_table(
            index=["municipality_code", "municipality_name", "year", category_column_name],
            columns="metric",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )
    wide_frame.to_parquet(output_path, index=False)
    return output_path


def preprocess_hospitalization_tables(root_dir="."):
    """Clean SIH residence hospitalization requests and write final parquet outputs."""
    raw_dir = _raw_dir(root_dir)
    health_dir = _health_dir(root_dir)
    outputs = {}
    outputs["SIH_RESIDENCE_TOTAL_MUNICIPALITY_YEAR"] = _preprocess_sih_total_request(
        os.path.join(raw_dir, "sih_residence_total_municipality_year_raw.parquet"),
        os.path.join(health_dir, "hospitalizations.parquet"),
    )
    outputs["SIH_RESIDENCE_ICD10_CHAPTER_MUNICIPALITY_YEAR"] = _preprocess_sih_channel_request(
        os.path.join(raw_dir, "sih_residence_icd10_chapter_municipality_year_raw.parquet"),
        os.path.join(health_dir, "hospitalizations_icd10_chapter.parquet"),
        "icd10_chapter",
    )
    outputs["SIH_RESIDENCE_SELECTED_MORBIDITY_LIST_MUNICIPALITY_YEAR"] = _preprocess_sih_channel_request(
        os.path.join(raw_dir, "sih_residence_selected_morbidity_list_municipality_year_raw.parquet"),
        os.path.join(health_dir, "hospitalizations_selected_morbidity_list.parquet"),
        "morbidity_list_cid10",
    )
    return outputs


def _clean_birth_outcome_frame(frame):
    frame = frame.copy()
    frame["mun_id"] = frame["Município"].str.extract(r"(\d{6})")[0].str.zfill(6)
    frame["mun_name"] = frame["Município"].str.extract(r"\d{6}(.*)")[0].str.strip()
    frame = frame.drop(columns=["Município"])
    value_columns = [col for col in frame.columns if col not in {"mun_id", "mun_name", "year", "Total"}]
    frame[value_columns] = frame[value_columns].apply(lambda col: col.str.replace("-", "0"), axis=0).astype("float32")
    frame = frame[["mun_id", "mun_name", "year"] + value_columns + ["Total"]]
    return frame.dropna(subset=["mun_id"])


def preprocess_birth_outcome_tables(root_dir=".", outcome_names=None):
    """Clean raw birth tables and write final parquet outputs."""
    raw_dir = _raw_dir(root_dir)
    health_dir = _health_dir(root_dir)
    outputs = {}
    selected_outcomes = outcome_names or ["gestational_duration", "birth_weight"]

    for outcome_name in selected_outcomes:
        raw_path = os.path.join(raw_dir, f"{outcome_name}_raw.parquet")
        cleaned = _clean_birth_outcome_frame(pd.read_parquet(raw_path))
        output_path = os.path.join(health_dir, f"{outcome_name}.parquet")
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
