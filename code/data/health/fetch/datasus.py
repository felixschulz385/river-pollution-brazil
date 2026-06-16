import os
import logging
import hashlib

import pandas as pd

from .forms import DatasusTabnetForm
from shared.batches import batch_output_path
from shared.batches import batch_table_dir
from shared.batches import initialize_manifest
from shared.batches import manifest_path
from shared.batches import update_manifest_entry

logger = logging.getLogger(__name__)
HEALTH_DATASET_NAME = "health"
DATASUS_NO_SELECTED_RECORDS = "DATASUS returned no selected records"

MORTALITY_URLS = {
    "pre_1996": "http://tabnet.datasus.gov.br/cgi/deftohtm.exe?sim/cnv/obt09br.def",
    "post_1995": "http://tabnet.datasus.gov.br/cgi/deftohtm.exe?sim/cnv/obt10br.def",
}

BIRTH_COLUMN_OPTIONS = {
    "gestational_duration": "Duração_gestação",
    "birth_weight": "Peso_ao_nascer",
}

SIH_TOTAL_REQUEST_ID = "SIH_RESIDENCE_TOTAL_MUNICIPALITY_YEAR"
SIH_ICD10_CHAPTER_REQUEST_ID = "SIH_RESIDENCE_ICD10_CHAPTER_MUNICIPALITY_YEAR"
SIH_SELECTED_MORBIDITY_LIST_REQUEST_ID = "SIH_RESIDENCE_SELECTED_MORBIDITY_LIST_MUNICIPALITY_YEAR"
SIH_ICD10_YEAR_MIN = 1998
SIH_ALL_METRICS_KEY = "all_metrics"

SIH_MUNICIPALITY_ROW_VALUE = "Município"
SIH_CHAPTER_COLUMN_VALUE = "Capítulo_CID-10"
SIH_NO_ACTIVE_COLUMN_VALUE = "--Não-Ativa--"
SIH_CHAPTER_FILTER_SELECT = "SCapítulo_CID-10"
SIH_MORBIDITY_LIST_FILTER_SELECT = "SLista_Morb__CID-10"
SIH_PERIOD_SELECT = "Arquivos"

SIH_SELECTED_MORBIDITY_LIST_VALUES = [
    "1", "2", "3", "4", "5", "6",
    "25", "26", "27",
    "41", "42", "43", "44", "56", "57", "58", "59", "60", "61", "62", "63", "64",
    "133", "134", "136", "137", "139",
    "169", "170", "171", "172", "174", "175", "177", "178", "179", "180", "188",
    "189", "190", "191", "192", "193", "194", "199", "200", "201", "202", "203",
    "222", "223",
    "235", "236", "237", "238", "239", "240", "241",
    "258", "259", "260", "261", "262", "263", "265", "266", "268",
    "269", "270", "271", "272", "273", "274", "275", "277",
    "278", "279", "280", "283", "285", "289", "290",
    "291", "292", "294",
    "295", "296", "297", "298", "299", "300", "302", "303", "304", "305", "307", "308", "309", "310", "311", "313",
    "323", "324", "325", "326", "327", "328", "329", "330",
]
SIH_SELECTED_MORBIDITY_CHANNELS = {
    "water_sanitation_gastrointestinal": ["1", "2", "3", "4", "5", "6"],
    "leptospirosis_water_exposure": ["25", "26", "27"],
    "vector_borne_ecological": ["41", "42", "43", "44", "56", "57", "58", "59", "60", "61", "62", "63", "64"],
    "respiratory_air_dust": ["189", "190", "191", "192", "193", "194", "199", "200", "201", "202", "203"],
    "cardiovascular_pollution_sensitive": ["169", "170", "171", "172", "174", "175", "177", "178", "179", "180", "188"],
    "renal_urinary_toxic_water": ["235", "236", "237", "238", "239", "240", "241"],
    "pregnancy_maternal": ["258", "259", "260", "261", "262", "263", "265", "266", "268"],
    "perinatal_newborn": ["269", "270", "271", "272", "273", "274", "275", "277"],
    "congenital_anomalies": ["278", "279", "280", "283", "285", "289", "290"],
    "toxic_poisoning": ["308", "309", "327"],
    "injuries_accidents_occupational": ["295", "296", "297", "298", "299", "300", "302", "303", "304", "305", "307", "311", "313", "323", "324", "325", "326", "330"],
    "violence_self_harm": ["310", "328", "329"],
    "mental_health_substance_use": ["133", "134", "136", "137", "139"],
    "skin_contact": ["222", "223"],
    "symptoms_unspecific": ["291", "292", "294"],
}
SIH_RESIDENCE_SOURCES = {
    "legacy": {
        "url": "http://tabnet.datasus.gov.br/cgi/deftohtm.exe?sih/cnv/mrbr.def",
        "year_min": 1995,
        "year_max": 2007,
        "content_labels": {
            "hospitalizations": "Internações",
            "total_approved_value": "Valor Total",
            "days_of_stay": "Dias Permanência",
            "average_length_of_stay": "Média Permanência",
            "in_hospital_deaths": "Óbitos",
            "hospital_mortality_rate": "Taxa Mortalidade",
        },
    },
    "current": {
        "url": "http://tabnet.datasus.gov.br/cgi/deftohtm.exe?sih/cnv/nrbr.def",
        "year_min": 2008,
        "year_max": 2024,
        "content_labels": {
            "hospitalizations": "Internações",
            "total_approved_value": "Valor total",
            "days_of_stay": "Dias permanência",
            "average_length_of_stay": "Média permanência",
            "in_hospital_deaths": "Óbitos",
            "hospital_mortality_rate": "Taxa mortalidade",
        },
    },
}

SIH_TOTAL_METRICS = [
    "hospitalizations",
    "total_approved_value",
    "days_of_stay",
    "average_length_of_stay",
    "in_hospital_deaths",
    "hospital_mortality_rate",
]

SIH_CHANNEL_METRICS = [
    "hospitalizations",
    "total_approved_value",
    "days_of_stay",
    "in_hospital_deaths",
    "hospital_mortality_rate",
]


def _raw_dir(root_dir):
    path = os.path.join(root_dir, "data", "health", "raw")
    os.makedirs(path, exist_ok=True)
    return path


def _save_raw_table(frame, path):
    frame.to_parquet(path, index=False)
    return path


def _slugify_fragment(value):
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


def _batch_id(
    request_id,
    source_key,
    year,
    metric_key,
    *,
    chapter_filter_text=None,
    morbidity_channel=None,
    morbidity_filter_values=None,
    morbidity_filter_value=None,
    morbidity_filter_text=None,
):
    parts = [request_id.lower(), source_key, str(year), metric_key]
    if chapter_filter_text:
        parts.append(_slugify_fragment(chapter_filter_text)[:40] or "chapter")
    if morbidity_channel:
        parts.append(morbidity_channel)
    if morbidity_filter_values:
        parts.append(f"group_{len(morbidity_filter_values)}")
    if morbidity_filter_value is not None:
        parts.append(f"value_{morbidity_filter_value}")
    if morbidity_filter_text:
        text_hash = hashlib.sha1(morbidity_filter_text.encode("utf-8")).hexdigest()[:10]
        parts.append(f"{_slugify_fragment(morbidity_filter_text)[:32] or 'morbidity'}_{text_hash}")
    return "__".join(parts)


def _batch_output_path(root_dir, request_id, batch_id):
    return batch_output_path(root_dir, HEALTH_DATASET_NAME, request_id, batch_id, suffix=".csv")


def _build_sih_manifest_entry(
    root_dir,
    request_id,
    source_key,
    year,
    metric_key,
    *,
    row_value=SIH_MUNICIPALITY_ROW_VALUE,
    column_value=SIH_NO_ACTIVE_COLUMN_VALUE,
    chapter_filter_text=None,
    morbidity_channel=None,
    morbidity_filter_values=None,
    morbidity_filter_value=None,
    morbidity_filter_text=None,
    select_all_content=False,
):
    batch_identifier = _batch_id(
        request_id,
        source_key,
        year,
        metric_key,
        chapter_filter_text=chapter_filter_text,
        morbidity_channel=morbidity_channel,
        morbidity_filter_values=morbidity_filter_values,
        morbidity_filter_value=morbidity_filter_value,
        morbidity_filter_text=morbidity_filter_text,
    )
    return {
        "batch_id": batch_identifier,
        "request_id": request_id,
        "source_key": source_key,
        "export_year": int(year),
        "metric_key": metric_key,
        "row_value": row_value,
        "column_value": column_value,
        "chapter_filter_text": chapter_filter_text,
        "morbidity_channel": morbidity_channel,
        "morbidity_filter_values": morbidity_filter_values,
        "morbidity_filter_value": morbidity_filter_value,
        "morbidity_filter_text": morbidity_filter_text,
        "select_all_content": select_all_content,
        "raw_path": _batch_output_path(root_dir, request_id, batch_identifier),
        "status": "pending",
        "error": None,
    }


def _parse_sih_period_value(period_value):
    yy = int(period_value[4:6])
    year = 1900 + yy if yy >= 95 else 2000 + yy
    month = int(period_value[6:8])
    return year, month


def _list_available_sih_period_values(form):
    return [option["value"] for option in form.get_select_options(SIH_PERIOD_SELECT)]


def _list_complete_sih_years(period_values):
    months_by_year = {}
    for period_value in period_values:
        year, month = _parse_sih_period_value(period_value)
        months_by_year.setdefault(year, set()).add(month)
    return sorted(year for year, months in months_by_year.items() if months == set(range(1, 13)))


def _filter_sih_period_values_for_year(period_values, year):
    return [period_value for period_value in period_values if _parse_sih_period_value(period_value)[0] == year]


def _source_key_for_year(year):
    return "legacy" if year <= 2007 else "current"


def _content_label(source_key, metric_key):
    return SIH_RESIDENCE_SOURCES[source_key]["content_labels"][metric_key]


def _resolve_sih_years(form, *, years=None, year_min=None):
    if years is None:
        years = []
        for source_key in SIH_RESIDENCE_SOURCES:
            source_years, _ = _list_source_years(form, source_key)
            years.extend(source_years)
    resolved_years = sorted(set(int(year) for year in years))
    if year_min is not None:
        resolved_years = [year for year in resolved_years if year >= year_min]
    return resolved_years


def _initialize_sih_manifest(root_dir, request_id, planned_entries):
    manifest_entries = initialize_manifest(
        root_dir,
        HEALTH_DATASET_NAME,
        request_id,
        planned_entries,
    )
    logger.debug(
        "Initialized SIH manifest: request_id=%s, planned_batches=%s, manifest_path=%s",
        request_id,
        len(manifest_entries),
        manifest_path(root_dir, HEALTH_DATASET_NAME, request_id),
    )
    return manifest_entries


def _list_source_years(form, source_key):
    source = SIH_RESIDENCE_SOURCES[source_key]
    form.open(source["url"])
    available_period_values = _list_available_sih_period_values(form)
    complete_years = _list_complete_sih_years(available_period_values)
    year_min = source["year_min"]
    year_max = source["year_max"]
    years = [
        year
        for year in complete_years
        if year >= year_min and (year_max is None or year <= year_max)
    ]
    logger.debug(
        "Discovered SIH source years: source=%s, year_min=%s, year_max=%s, years=%s",
        source_key,
        year_min,
        year_max,
        years,
    )
    return years, available_period_values


def _run_sih_residence_query(
    form,
    *,
    source_url,
    source_key,
    row_value,
    content_label,
    column_value=SIH_NO_ACTIVE_COLUMN_VALUE,
    period_values=None,
    export_year=None,
    chapter_filter_text=None,
    morbidity_channel=None,
    morbidity_filter_values=None,
    morbidity_filter_value=None,
    morbidity_filter_text=None,
    select_all_content=False,
):
    query_bits = [
        f"source={source_key}",
        f"row={row_value}",
        f"column={column_value}",
    ]
    if select_all_content:
        query_bits.append("content=ALL")
    else:
        query_bits.append(f"content={content_label}")
    if export_year is not None:
        query_bits.append(f"year={export_year}")
    if chapter_filter_text is not None:
        query_bits.append(f"chapter={chapter_filter_text}")
    if morbidity_channel is not None:
        query_bits.append(f"morbidity_channel={morbidity_channel}")
    if morbidity_filter_values is not None:
        query_bits.append(f"morbidity_values={','.join(morbidity_filter_values)}")
    if morbidity_filter_value is not None:
        query_bits.append(f"morbidity_value={morbidity_filter_value}")
    if morbidity_filter_text is not None:
        query_bits.append(f"morbidity_list={morbidity_filter_text}")
    logger.info("DATASUS query: %s", ", ".join(query_bits))

    form.open(source_url)
    form.select_line(row_value)
    form.select_column(column_value)
    if select_all_content:
        form.select_all_content_values()
    else:
        form.select_content_text(content_label)

    if period_values is not None:
        form.set_multiselect_values(SIH_PERIOD_SELECT, period_values)

    if chapter_filter_text is not None:
        form.select_filter_option_text(SIH_CHAPTER_FILTER_SELECT, chapter_filter_text, exact=True)

    if morbidity_filter_values is not None:
        available_morbidity_values = {
            option["value"] for option in form.get_select_options(SIH_MORBIDITY_LIST_FILTER_SELECT)
        }
        selected_morbidity_values = [
            value for value in morbidity_filter_values if value in available_morbidity_values
        ]
        missing_morbidity_values = [
            value for value in morbidity_filter_values if value not in available_morbidity_values
        ]
        if missing_morbidity_values:
            logger.warning(
                "DATASUS morbidity values unavailable at execution time: source=%s, year=%s, channel=%s, missing=%s",
                source_key,
                export_year,
                morbidity_channel,
                ",".join(missing_morbidity_values),
            )
        if not selected_morbidity_values:
            raise RuntimeError(
                "DATASUS returned no selected records: requested morbidity values unavailable "
                f"for source={source_key}, year={export_year}, channel={morbidity_channel}"
            )
        form.set_multiselect_values(SIH_MORBIDITY_LIST_FILTER_SELECT, selected_morbidity_values)
    elif morbidity_filter_value is not None:
        form.select_filter_option_value(SIH_MORBIDITY_LIST_FILTER_SELECT, morbidity_filter_value)
    elif morbidity_filter_text is not None:
        form.select_filter_option_text(SIH_MORBIDITY_LIST_FILTER_SELECT, morbidity_filter_text, exact=True)

    form.select_output_format_table()
    form.submit_query()
    return


def _execute_sih_manifest_entries(
    root_dir,
    form,
    request_id,
    manifest_entries,
):
    table_name = request_id
    results = []
    completed_count = 0
    pending_count = 0
    skipped_count = 0
    source_period_values = {}
    for source_key in {entry["source_key"] for entry in manifest_entries}:
        _, source_period_values[source_key] = _list_source_years(form, source_key)

    for entry in manifest_entries:
        if entry["status"] == "completed" and os.path.exists(entry["raw_path"]):
            completed_count += 1
        elif entry["status"] == "skipped":
            skipped_count += 1
        else:
            pending_count += 1

    logger.debug(
        "Executing SIH manifest: request_id=%s, total_batches=%s, completed_batches=%s, skipped_batches=%s, pending_batches=%s",
        request_id,
        len(manifest_entries),
        completed_count,
        skipped_count,
        pending_count,
    )

    for entry in manifest_entries:
        raw_path = entry["raw_path"]
        if entry["status"] == "completed" and os.path.exists(raw_path):
            logger.debug(
                "Skipping completed SIH batch: request_id=%s, batch_id=%s, raw_path=%s",
                request_id,
                entry["batch_id"],
                raw_path,
            )
            continue
        if entry["status"] == "skipped":
            logger.debug(
                "Skipping previously empty SIH batch: request_id=%s, batch_id=%s",
                request_id,
                entry["batch_id"],
            )
            continue

        update_manifest_entry(
            root_dir,
            HEALTH_DATASET_NAME,
            table_name,
            manifest_entries,
            entry["batch_id"],
            status="in_progress",
            error=None,
        )
        logger.debug(
            "Starting SIH batch: request_id=%s, batch_id=%s, source=%s, year=%s, metric=%s",
            request_id,
            entry["batch_id"],
            entry["source_key"],
            entry["export_year"],
            entry["metric_key"],
        )
        try:
            _run_sih_residence_query(
                form,
                source_url=SIH_RESIDENCE_SOURCES[entry["source_key"]]["url"],
                source_key=entry["source_key"],
                row_value=entry["row_value"],
                column_value=entry["column_value"],
                content_label=(
                    None
                    if entry.get("select_all_content", False)
                    else _content_label(entry["source_key"], entry["metric_key"])
                ),
                period_values=_filter_sih_period_values_for_year(
                    source_period_values[entry["source_key"]],
                    entry["export_year"],
                ),
                export_year=entry["export_year"],
                chapter_filter_text=entry["chapter_filter_text"],
                morbidity_channel=entry.get("morbidity_channel"),
                morbidity_filter_values=entry.get("morbidity_filter_values"),
                morbidity_filter_value=entry.get("morbidity_filter_value"),
                morbidity_filter_text=entry["morbidity_filter_text"],
                select_all_content=entry.get("select_all_content", False),
            )
            form.download_result_csv(raw_path)
            update_manifest_entry(
                root_dir,
                HEALTH_DATASET_NAME,
                table_name,
                manifest_entries,
                entry["batch_id"],
                status="completed",
                raw_path=raw_path,
                error=None,
            )
            logger.debug(
                "Completed SIH batch: request_id=%s, batch_id=%s, raw_path=%s",
                request_id,
                entry["batch_id"],
                raw_path,
            )
            results.append(raw_path)
        except Exception as exc:
            if DATASUS_NO_SELECTED_RECORDS in str(exc):
                update_manifest_entry(
                    root_dir,
                    HEALTH_DATASET_NAME,
                    table_name,
                    manifest_entries,
                    entry["batch_id"],
                    status="skipped",
                    error=str(exc),
                )
                logger.info(
                    "Skipping empty DATASUS batch: request_id=%s, batch_id=%s, source=%s, year=%s, metric=%s",
                    request_id,
                    entry["batch_id"],
                    entry["source_key"],
                    entry["export_year"],
                    entry["metric_key"],
                )
                continue
            update_manifest_entry(
                root_dir,
                HEALTH_DATASET_NAME,
                table_name,
                manifest_entries,
                entry["batch_id"],
                status="failed",
                error=str(exc),
            )
            logger.debug(
                "Failed SIH batch: request_id=%s, batch_id=%s, error=%s",
                request_id,
                entry["batch_id"],
                exc,
            )
            raise
        finally:
            if len(form.driver.window_handles) > 1:
                form.reset_query()
    return {
        "table_dir": batch_table_dir(root_dir, HEALTH_DATASET_NAME, table_name),
        "manifest_path": manifest_path(root_dir, HEALTH_DATASET_NAME, table_name),
        "batch_paths": results,
    }


def fetch_mortality_age_tables(root_dir=".", headless=False, download_dir=None):
    """Fetch raw mortality tables grouped by age band."""
    output_dir = _raw_dir(root_dir)
    fetch_plan = {
        "pre_1996": {
            "url": MORTALITY_URLS["pre_1996"],
            "years": [str(year).zfill(2) for year in range(79, 95)],
            "default_year": "22",
            "output": os.path.join(output_dir, "mortality_age_counts_pre_1996_raw.parquet"),
        },
        "post_1995": {
            "url": MORTALITY_URLS["post_1995"],
            "years": [str(year).zfill(2) for year in list(range(96, 100)) + list(range(0, 22))],
            "default_year": "95",
            "output": os.path.join(output_dir, "mortality_age_counts_post_1995_raw.parquet"),
        },
    }

    output_paths = {}
    for period, config in fetch_plan.items():
        form = DatasusTabnetForm(headless=headless, download_dir=download_dir)
        raw_tables = []
        try:
            for year in config["years"]:
                form.open(config["url"])
                form.select_column("Faixa_Etária")
                if year != config["default_year"]:
                    form.select_option_value(f"obtbr{year}.dbf")
                form.select_output_format_prn()
                form.submit_query()

                table = form.read_result_table()
                table.insert(0, "year_code", year)
                raw_tables.append(table)
                form.reset_query()
        finally:
            form.close()

        output_paths[period] = _save_raw_table(pd.concat(raw_tables, ignore_index=True), config["output"])

    return output_paths


def fetch_sih_residence_total_municipality_year(
    root_dir=".",
    headless=False,
    download_dir=None,
):
    """Fetch SIH residence totals by municipality and year."""
    form = DatasusTabnetForm(headless=headless, download_dir=download_dir)
    try:
        logger.debug("Planning SIH request: request_id=%s", SIH_TOTAL_REQUEST_ID)
        planned_entries = []
        for source_key in SIH_RESIDENCE_SOURCES:
            years, _ = _list_source_years(form, source_key)
            for year in years:
                planned_entries.append(
                    _build_sih_manifest_entry(
                        root_dir,
                        SIH_TOTAL_REQUEST_ID,
                        source_key,
                        year,
                        SIH_ALL_METRICS_KEY,
                        select_all_content=True,
                    )
                )
        manifest_entries = _initialize_sih_manifest(root_dir, SIH_TOTAL_REQUEST_ID, planned_entries)
        return _execute_sih_manifest_entries(root_dir, form, SIH_TOTAL_REQUEST_ID, manifest_entries)
    finally:
        form.close()


def fetch_sih_residence_icd10_chapter_municipality_year(
    root_dir=".",
    headless=False,
    download_dir=None,
    years=None,
):
    """Fetch SIH residence municipality by ICD-10 chapter exports, one annual request per file slice."""
    form = DatasusTabnetForm(headless=headless, download_dir=download_dir)
    try:
        logger.debug("Planning SIH request: request_id=%s", SIH_ICD10_CHAPTER_REQUEST_ID)
        years = _resolve_sih_years(form, years=years, year_min=SIH_ICD10_YEAR_MIN)
        planned_entries = []
        for year in years:
            source_key = _source_key_for_year(year)
            for metric_key in SIH_CHANNEL_METRICS:
                planned_entries.append(
                    _build_sih_manifest_entry(
                        root_dir,
                        SIH_ICD10_CHAPTER_REQUEST_ID,
                        source_key,
                        year,
                        metric_key,
                        column_value=SIH_CHAPTER_COLUMN_VALUE,
                    )
                )
        manifest_entries = _initialize_sih_manifest(root_dir, SIH_ICD10_CHAPTER_REQUEST_ID, planned_entries)
        return _execute_sih_manifest_entries(
            root_dir,
            form,
            SIH_ICD10_CHAPTER_REQUEST_ID,
            manifest_entries,
        )
    finally:
        form.close()


def fetch_sih_residence_selected_morbidity_list_municipality_year(
    root_dir=".",
    headless=False,
    download_dir=None,
    years=None,
    morbidity_list_values=None,
):
    """Fetch SIH residence municipality by grouped CID-10 morbidity channels."""
    form = DatasusTabnetForm(headless=headless, download_dir=download_dir)
    try:
        logger.debug("Planning SIH request: request_id=%s", SIH_SELECTED_MORBIDITY_LIST_REQUEST_ID)
        years = _resolve_sih_years(form, years=years, year_min=SIH_ICD10_YEAR_MIN)
        selected_values = [str(value) for value in (morbidity_list_values or SIH_SELECTED_MORBIDITY_LIST_VALUES)]
        planned_entries = []
        matched_options_by_source = {}
        for source_key in sorted({_source_key_for_year(year) for year in years}):
            source_url = SIH_RESIDENCE_SOURCES[source_key]["url"]
            form.open(source_url)
            form.select_line(SIH_MUNICIPALITY_ROW_VALUE)
            form.select_column(SIH_NO_ACTIVE_COLUMN_VALUE)
            form.select_all_content_values()
            selected_morbidity_lists = form.find_options_by_values(
                SIH_MORBIDITY_LIST_FILTER_SELECT,
                selected_values,
            )
            if not selected_morbidity_lists:
                raise ValueError(
                    "No DATASUS morbidity-list options matched the configured values. "
                    f"Values: {selected_values!r}"
                )
            matched_options_by_source[source_key] = {
                option["value"]: option["text"] for option in selected_morbidity_lists
            }

        for year in years:
            source_key = _source_key_for_year(year)
            source_options = matched_options_by_source[source_key]
            for channel_name, channel_values in SIH_SELECTED_MORBIDITY_CHANNELS.items():
                matched_values = [value for value in channel_values if value in source_options]
                if not matched_values:
                    continue
                planned_entries.append(
                    _build_sih_manifest_entry(
                        root_dir,
                        SIH_SELECTED_MORBIDITY_LIST_REQUEST_ID,
                        source_key,
                        year,
                        SIH_ALL_METRICS_KEY,
                        morbidity_channel=channel_name,
                        morbidity_filter_values=matched_values,
                        select_all_content=True,
                    )
                )
        manifest_entries = _initialize_sih_manifest(
            root_dir,
            SIH_SELECTED_MORBIDITY_LIST_REQUEST_ID,
            planned_entries,
        )
        return _execute_sih_manifest_entries(
            root_dir,
            form,
            SIH_SELECTED_MORBIDITY_LIST_REQUEST_ID,
            manifest_entries,
        )
    finally:
        form.close()


def fetch_hospitalization_tables(root_dir=".", headless=False, download_dir=None):
    """Fetch all implemented SIH residence hospitalization requests."""
    logger.debug("Fetching hospitalization tables: total, icd10_chapter, selected_morbidity_list")
    return {
        SIH_TOTAL_REQUEST_ID: fetch_sih_residence_total_municipality_year(
            root_dir=root_dir,
            headless=headless,
            download_dir=download_dir,
        ),
        SIH_ICD10_CHAPTER_REQUEST_ID: fetch_sih_residence_icd10_chapter_municipality_year(
            root_dir=root_dir,
            headless=headless,
            download_dir=download_dir,
        ),
        SIH_SELECTED_MORBIDITY_LIST_REQUEST_ID: fetch_sih_residence_selected_morbidity_list_municipality_year(
            root_dir=root_dir,
            headless=headless,
            download_dir=download_dir,
        ),
    }


def fetch_birth_outcome_tables(root_dir=".", headless=False, download_dir=None, outcome_names=None):
    """Fetch raw birth-outcome tables for gestational duration and birth weight."""
    output_dir = _raw_dir(root_dir)
    years = list(range(1994, 2024))
    latest_year = 2023
    output_paths = {}
    selected_outcomes = outcome_names or list(BIRTH_COLUMN_OPTIONS)

    for outcome_name in selected_outcomes:
        column_option = BIRTH_COLUMN_OPTIONS[outcome_name]
        form = DatasusTabnetForm(headless=headless, download_dir=download_dir)
        raw_tables = []
        try:
            for year in years:
                print(f"Fetching {outcome_name} data for year {year}...")
                form.open("http://tabnet.datasus.gov.br/cgi/tabcgi.exe?sinasc/cnv/nvbr.def")
                form.select_column(column_option)

                year_code = str(year % 100).zfill(2)
                latest_year_code = str(latest_year % 100).zfill(2)
                if year != latest_year:
                    form.select_option_value(f"nvbr{year_code}.dbf")
                    form.select_option_value(f"nvbr{latest_year_code}.dbf")

                form.select_output_format_prn()
                form.submit_query()

                table = form.read_result_table()
                table.insert(0, "year", year)
                raw_tables.append(table)
                form.reset_query()
        finally:
            form.close()

        raw_path = os.path.join(output_dir, f"{outcome_name}_raw.parquet")
        output_paths[outcome_name] = _save_raw_table(pd.concat(raw_tables, ignore_index=True), raw_path)

    return output_paths


def fetch_health_data(root_dir=".", subtype="all", headless=False, download_dir=None):
    """Dispatch raw health-data fetchers."""
    valid_subtypes = {"all", "mortality", "hospitalization", "birth"}
    if subtype not in valid_subtypes:
        raise ValueError(
            f"Invalid subtype: {subtype}. Choose from: {', '.join(sorted(valid_subtypes))}"
        )

    outputs = {}
    if subtype in {"all", "mortality"}:
        outputs["mortality"] = fetch_mortality_age_tables(
            root_dir=root_dir,
            headless=headless,
            download_dir=download_dir,
        )
    if subtype in {"all", "hospitalization"}:
        outputs["hospitalization"] = fetch_hospitalization_tables(
            root_dir=root_dir,
            headless=headless,
            download_dir=download_dir,
        )
    if subtype in {"all", "birth"}:
        outputs["birth"] = fetch_birth_outcome_tables(
            root_dir=root_dir,
            headless=headless,
            download_dir=download_dir,
        )
    return outputs
