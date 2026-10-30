from __future__ import annotations

from pathlib import Path

from src.data.shared.paths import processed_dir, raw_dir as _shared_raw_dir, source_root


# SIVEP-Malária notifications 2004-2024, record level (Saldanha, CC BY 4.0).
# Coded fields are stored as their labels (e.g. `cod_ocup` "Garimpagem").
ZENODO_RECORD_ID = 20706957
ZENODO_RECORD_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
YEARS = tuple(range(2004, 2025))
# Preliminary years from the LAI requests (Zenodo 20706949, updated monthly;
# the last year is partial). Each fetch that finds the record changed writes a
# new dated snapshot `raw/preliminary_<YYYY-MM-DD>/` with the record's metadata
# (`zenodo_record.json`); the newest complete snapshot is read.
PRELIMINARY_RECORD_ID = 20706949
PRELIMINARY_RECORD_URL = f"https://zenodo.org/api/records/{PRELIMINARY_RECORD_ID}"
PRELIMINARY_YEARS = (2025, 2026)
PRELIMINARY_PREFIX = "preliminary_"
RECORD_FILENAME = "zenodo_record.json"
# Preprocessed years. Outputs stop at the last complete month of notifications
# (yearly tables at the last complete year).
ALL_YEARS = (*YEARS, *PRELIMINARY_YEARS)
YEAR_FILENAME_TEMPLATE = "sivep_malaria_{year}.parquet"
METADATA_FILENAMES = ("sivep_malaria_metadata_csv.zip", "sivep_malaria_full_methodology_description.md")

READ_COLUMNS = (
    "notification_year", "res_exam", "id_lvc", "tipo_lam", "cod_ocup", "pais_inf",
    "mun_infe_code", "mun_infe_nome", "mun_resi_code", "mun_resi_nome",
    "mun_noti_code", "mun_noti_nome", "dt_notif",
)
NEGATIVE_RESULT = "Negativo"
# `id_lvc` is the consistent cure-verification (LVC) flag; `tipo_lam`'s LVC
# code agrees until 2010 and then fades out (none from 2022).
LVC_NOTIFICATION = "Notificacao de LVC"
LVC_DETECTION = "LVC"
GARIMPO_OCCUPATION = "Garimpagem"
BRAZIL = "Brasil"
# 6-digit IBGE codes; 5399xx are SIVEP placeholders, not municipalities.
PLACEHOLDER_MUNICIPALITY_PREFIX = 5399
UNMAPPED_NAME_PREFIX = "Unmapped"

MUN_ID_COLUMN = "mun_id"
YEAR_COLUMN = "year"
MONTH_COLUMN = "month"  # first day of the notification month
# Three perspectives on the same positives (LVC dropped): where the case was
# infected (`inf_`, Brazil only), where the patient lives (`res_`), and where
# it was notified (`noti_`, the only one negatives can be placed by).
INFECTION_COLUMNS = (
    "inf_cases",
    "inf_garimpo",
    "inf_garimpo_local",  # living in the infection municipality
    "inf_garimpo_incoming",  # living in another Brazilian municipality
    "inf_garimpo_incoming_other_state",
)
RESIDENCE_COLUMNS = (
    "res_cases",
    "res_garimpo",
    "res_garimpo_elsewhere",  # infected in another Brazilian municipality
    "res_garimpo_abroad",
)
NOTIFICATION_COLUMNS = ("noti_exams", "noti_positives")
ADM2_COLUMNS = (MUN_ID_COLUMN, YEAR_COLUMN, *INFECTION_COLUMNS, *RESIDENCE_COLUMNS, *NOTIFICATION_COLUMNS)

# Positives by infection municipality and notification month, for timing
# around dated events (raids). Months without a case are absent.
MONTHLY_COLUMNS = (
    MUN_ID_COLUMN, MONTH_COLUMN, "inf_cases", "inf_garimpo", "inf_garimpo_local", "inf_garimpo_incoming",
)

FLOW_RESIDENCE_COLUMN = "res_mun_id"
FLOW_INFECTION_COLUMN = "inf_mun_id"  # missing when infected abroad
FLOW_COUNTRY_COLUMN = "inf_country"
FLOW_CASES_COLUMN = "cases"
FLOW_COLUMNS = (YEAR_COLUMN, FLOW_RESIDENCE_COLUMN, FLOW_INFECTION_COLUMN, FLOW_COUNTRY_COLUMN, FLOW_CASES_COLUMN)

DEFAULT_ADM2_OUTPUT_PATH = str(processed_dir(".", "malaria") / "malaria_adm2.parquet")
DEFAULT_FLOWS_OUTPUT_PATH = str(processed_dir(".", "malaria") / "malaria_garimpo_flows.parquet")
DEFAULT_MONTHLY_OUTPUT_PATH = str(processed_dir(".", "malaria") / "malaria_adm2_monthly.parquet")


def malaria_dir(root_dir: str | Path = ".") -> Path:
    return source_root(root_dir, "malaria")


def raw_dir(root_dir: str | Path = ".") -> Path:
    return _shared_raw_dir(root_dir, "malaria")


def preliminary_snapshot(root_dir: str | Path = ".") -> Path | None:
    """The newest preliminary snapshot folder holding the record and every year file (folders sort by date)."""
    required = [RECORD_FILENAME, *(YEAR_FILENAME_TEMPLATE.format(year=year) for year in PRELIMINARY_YEARS)]
    for snapshot in sorted(raw_dir(root_dir).glob(f"{PRELIMINARY_PREFIX}*"), reverse=True):
        if all((snapshot / name).exists() for name in required):
            return snapshot
    return None


def year_path(root_dir: str | Path, year: int) -> Path:
    """A year's parquet: in the raw folder, or in the newest preliminary snapshot for the preliminary years."""
    name = YEAR_FILENAME_TEMPLATE.format(year=year)
    if year not in PRELIMINARY_YEARS:
        return raw_dir(root_dir) / name
    snapshot = preliminary_snapshot(root_dir)
    return (snapshot or raw_dir(root_dir) / f"{PRELIMINARY_PREFIX}<missing>") / name
