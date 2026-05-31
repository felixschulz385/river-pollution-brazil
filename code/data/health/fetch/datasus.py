import os

import pandas as pd

from .forms import DatasusTabnetForm

MORTALITY_URLS = {
    "pre_1996": "http://tabnet.datasus.gov.br/cgi/deftohtm.exe?sim/cnv/obt09br.def",
    "post_1995": "http://tabnet.datasus.gov.br/cgi/deftohtm.exe?sim/cnv/obt10br.def",
}

BIRTH_COLUMN_OPTIONS = {
    "gestational_duration": "Duração_gestação",
    "birth_weight": "Peso_ao_nascer",
}

SIH_RESIDENCE_URL = "http://tabnet.datasus.gov.br/cgi/deftohtm.exe?sih/cnv/nrbr.def"
SIH_TOTAL_REQUEST_ID = "SIH_RESIDENCE_TOTAL_MUNICIPALITY_YEAR"
SIH_ICD10_CHAPTER_REQUEST_ID = "SIH_RESIDENCE_ICD10_CHAPTER_MUNICIPALITY_YEAR"
SIH_SELECTED_MORBIDITY_LIST_REQUEST_ID = "SIH_RESIDENCE_SELECTED_MORBIDITY_LIST_MUNICIPALITY_YEAR"

SIH_TOTAL_CONTENTS = [
    "Internações",
    "Valor total",
    "Dias permanência",
    "Média permanência",
    "Óbitos",
    "Taxa mortalidade",
]

SIH_CHANNEL_CONTENTS = [
    "Internações",
    "Valor total",
    "Dias permanência",
    "Óbitos",
    "Taxa mortalidade",
]

SIH_MUNICIPALITY_ROW_VALUE = "Município"
SIH_YEAR_COLUMN_VALUE = "Ano_atendimento"
SIH_CHAPTER_COLUMN_VALUE = "Capítulo_CID-10"
SIH_NO_ACTIVE_COLUMN_VALUE = "--Não-Ativa--"
SIH_MORBIDITY_LIST_LINE_VALUE = "Lista_Morb__CID-10"
SIH_CHAPTER_FILTER_SELECT = "SCapítulo_CID-10"
SIH_MORBIDITY_LIST_FILTER_SELECT = "SLista_Morb__CID-10"
SIH_PERIOD_SELECT = "Arquivos"

SIH_SELECTED_MORBIDITY_LIST_FRAGMENTS = [
    "Algumas doenças infecciosas e parasitárias",
    "Doenças infecciosas intestinais",
    "Doenças do aparelho respiratório",
    "Doenças do aparelho circulatório",
    "Doenças do aparelho geniturinário",
    "Gravidez parto e puerpério",
    "Algumas afec originadas no período perinatal",
    "Malform congênitas",
    "Lesões enven",
    "Transtornos mentais e comportamentais",
    "Doenças do sistema nervoso",
    "Doenças do aparelho digestivo",
    "Doenças da pele e tecido subcutâneo",
]


def _raw_dir(root_dir):
    path = os.path.join(root_dir, "data", "health", "raw")
    os.makedirs(path, exist_ok=True)
    return path


def _save_raw_table(frame, path):
    frame.to_parquet(path, index=False)
    return path


def _parse_period_value(period_value):
    yy = int(period_value[4:6])
    year = 2000 + yy
    month = int(period_value[6:8])
    return year, month


def _available_sih_period_values(form):
    return [option["value"] for option in form.get_select_options(SIH_PERIOD_SELECT)]


def _complete_sih_years(period_values):
    months_by_year = {}
    for period_value in period_values:
        year, month = _parse_period_value(period_value)
        months_by_year.setdefault(year, set()).add(month)
    return sorted(year for year, months in months_by_year.items() if months == set(range(1, 13)))


def _period_values_for_year(period_values, year):
    return [period_value for period_value in period_values if _parse_period_value(period_value)[0] == year]


def _fetch_sih_residence_export(
    form,
    *,
    row_value,
    column_value,
    content_metric,
    period_values=None,
    chapter_filter_text=None,
    morbidity_filter_text=None,
):
    form.open(SIH_RESIDENCE_URL)
    form.select_line(row_value)
    form.select_column(column_value)
    form.select_content_text(content_metric)

    if period_values is not None:
        form.set_multiselect_values(SIH_PERIOD_SELECT, period_values)

    if chapter_filter_text is not None:
        form.select_filter_option_text(SIH_CHAPTER_FILTER_SELECT, chapter_filter_text, exact=True)

    if morbidity_filter_text is not None:
        form.select_filter_option_text(SIH_MORBIDITY_LIST_FILTER_SELECT, morbidity_filter_text, exact=True)

    form.select_output_format_prn()
    form.submit_query()
    table = form.read_result_table()
    form.reset_query()
    return table

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


def fetch_sih_residence_total_municipality_year(root_dir=".", headless=False, download_dir=None):
    """Fetch SIH residence totals by municipality and year."""
    output_dir = _raw_dir(root_dir)
    form = DatasusTabnetForm(headless=headless, download_dir=download_dir)
    raw_tables = []
    try:
        form.open(SIH_RESIDENCE_URL)
        complete_years = _complete_sih_years(_available_sih_period_values(form))
        annual_period_values = [
            period_value
            for year in complete_years
            for period_value in _period_values_for_year(_available_sih_period_values(form), year)
        ]
        for content_metric in SIH_TOTAL_CONTENTS:
            table = _fetch_sih_residence_export(
                form,
                row_value=SIH_MUNICIPALITY_ROW_VALUE,
                column_value=SIH_YEAR_COLUMN_VALUE,
                content_metric=content_metric,
                period_values=annual_period_values,
            )
            table.insert(0, "request_id", SIH_TOTAL_REQUEST_ID)
            table.insert(1, "content_metric", content_metric)
            raw_tables.append(table)
    finally:
        form.close()

    output_path = os.path.join(output_dir, "sih_residence_total_municipality_year_raw.parquet")
    return _save_raw_table(pd.concat(raw_tables, ignore_index=True), output_path)


def fetch_sih_residence_icd10_chapter_municipality_year(
    root_dir=".",
    headless=False,
    download_dir=None,
    years=None,
):
    """Fetch SIH residence municipality by ICD-10 chapter exports, one annual request per file slice."""
    output_dir = _raw_dir(root_dir)
    form = DatasusTabnetForm(headless=headless, download_dir=download_dir)
    raw_tables = []
    try:
        form.open(SIH_RESIDENCE_URL)
        available_period_values = _available_sih_period_values(form)
        complete_years = _complete_sih_years(available_period_values)
        years = years or complete_years
        for year in years:
            period_values = _period_values_for_year(available_period_values, year)
            for content_metric in SIH_CHANNEL_CONTENTS:
                table = _fetch_sih_residence_export(
                    form,
                    row_value=SIH_MUNICIPALITY_ROW_VALUE,
                    column_value=SIH_CHAPTER_COLUMN_VALUE,
                    content_metric=content_metric,
                    period_values=period_values,
                )
                table.insert(0, "request_id", SIH_ICD10_CHAPTER_REQUEST_ID)
                table.insert(1, "export_year", year)
                table.insert(2, "content_metric", content_metric)
                raw_tables.append(table)
    finally:
        form.close()

    output_path = os.path.join(output_dir, "sih_residence_icd10_chapter_municipality_year_raw.parquet")
    return _save_raw_table(pd.concat(raw_tables, ignore_index=True), output_path)


def fetch_sih_residence_selected_morbidity_list_municipality_year(
    root_dir=".",
    headless=False,
    download_dir=None,
    years=None,
    morbidity_list_fragments=None,
):
    """Fetch SIH residence municipality by selected CID-10 morbidity-list exports."""
    output_dir = _raw_dir(root_dir)
    form = DatasusTabnetForm(headless=headless, download_dir=download_dir)
    raw_tables = []
    try:
        form.open(SIH_RESIDENCE_URL)
        available_period_values = _available_sih_period_values(form)
        complete_years = _complete_sih_years(available_period_values)
        years = years or complete_years
        selected_fragments = morbidity_list_fragments or SIH_SELECTED_MORBIDITY_LIST_FRAGMENTS
        selected_morbidity_lists = form.find_option_texts_by_fragments(
            SIH_MORBIDITY_LIST_FILTER_SELECT,
            selected_fragments,
        )
        for year in years:
            period_values = _period_values_for_year(available_period_values, year)
            for content_metric in SIH_CHANNEL_CONTENTS:
                for morbidity_list_text in selected_morbidity_lists:
                    table = _fetch_sih_residence_export(
                        form,
                        row_value=SIH_MUNICIPALITY_ROW_VALUE,
                        column_value=SIH_NO_ACTIVE_COLUMN_VALUE,
                        content_metric=content_metric,
                        period_values=period_values,
                        morbidity_filter_text=morbidity_list_text,
                    )
                    table.insert(0, "request_id", SIH_SELECTED_MORBIDITY_LIST_REQUEST_ID)
                    table.insert(1, "export_year", year)
                    table.insert(2, "content_metric", content_metric)
                    table.insert(3, "morbidity_list_cid10", morbidity_list_text)
                    raw_tables.append(table)
    finally:
        form.close()

    output_path = os.path.join(output_dir, "sih_residence_selected_morbidity_list_municipality_year_raw.parquet")
    return _save_raw_table(pd.concat(raw_tables, ignore_index=True), output_path)


def fetch_hospitalization_tables(root_dir=".", headless=False, download_dir=None):
    """Fetch all implemented SIH residence hospitalization requests."""
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
