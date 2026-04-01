import logging
import warnings
import zipfile
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pyodbc

from ..database import (
    RAW_ARCHIVES_TABLE,
    SENSOR_ARCHIVE_FILES_TABLE,
    write_dataframe_table,
)
from ..paths import ensure_water_quality_dirs, get_raw_dir

logger = logging.getLogger(__name__)

ARCHIVE_TABLE_RECORD_COLUMNS = [
    "station_code",
    "source_archive_name",
    "source_mdb_name",
    "source_table_name",
    "table_name",
    "row_count",
    "is_empty_table",
]


def _mapping_file_path() -> Path:
    """Return the checked-in table/column translation file."""
    return Path(__file__).resolve().parents[4] / "data" / "sensor_data" / "raw_column_name_mapping.csv"


def _load_name_mapping() -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Load English table and column names derived in the notebook workflow."""
    mapping = pd.read_csv(_mapping_file_path())

    table_map = (
        mapping.loc[:, ["table_original", "table_english"]]
        .dropna(subset=["table_original", "table_english"])
        .drop_duplicates(subset=["table_original"])
        .set_index("table_original")["table_english"]
        .to_dict()
    )

    column_mapping_rows = mapping.dropna(subset=["table_original", "column_original", "column_english"])
    column_map: dict[str, dict[str, str]] = {}
    for row in column_mapping_rows.itertuples(index=False):
        column_map.setdefault(row.table_original, {})[row.column_original] = row.column_english
    return table_map, column_map


def list_valid_raw_archives(root_dir="."):
    """Identify downloaded MDB archives that should be preprocessed."""
    raw_dir = get_raw_dir(root_dir)
    files = pd.DataFrame({"filename": [path.name for path in raw_dir.iterdir() if path.is_file()]})
    if files.empty:
        return files

    files["file_size_bytes"] = files["filename"].apply(lambda name: (raw_dir / name).stat().st_size)
    files["station_code"] = files["filename"].str.extract(r"(^\d+)").iloc[:, 0]
    files = files.loc[
        (files["file_size_bytes"] > 22)
        & files["station_code"].notna()
        & files["filename"].str.contains(r"_mdb.*\.zip$", case=False, regex=True),
        :,
    ].copy()
    if files.empty:
        return files

    return files.reset_index(drop=True)


def _list_archive_mdb_members(archive_path: Path) -> list[str]:
    """Return MDB files contained in one downloaded archive."""
    try:
        with zipfile.ZipFile(archive_path, "r") as archive_file:
            return [
                member_name
                for member_name in archive_file.namelist()
                if member_name.lower().endswith(".mdb")
            ]
    except zipfile.BadZipFile:
        return []


def _extract_mdb_member(archive_path: Path, member_name: str, extract_root: Path) -> Path:
    """Extract one MDB file to a stable local path for the Access ODBC driver."""
    target_path = extract_root / f"{uuid4().hex}_{Path(member_name).name}"
    with zipfile.ZipFile(archive_path, "r") as archive_file:
        with archive_file.open(member_name, "r") as source_handle:
            target_path.write_bytes(source_handle.read())
    return target_path


def _connect_access_database(mdb_path: Path):
    """Open a Microsoft Access MDB file with the Windows Access ODBC driver."""
    return pyodbc.connect(
        rf"Driver={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={mdb_path.resolve()};"
    )


def _list_user_tables(connection) -> list[str]:
    """List non-system tables from one MDB file."""
    cursor = connection.cursor()
    return [
        row.table_name
        for row in cursor.tables(tableType="TABLE")
        if not str(row.table_name).startswith("MSys")
    ]


def _clean_fallback_name(name: str) -> str:
    """Create a stable fallback name if the mapping file has no explicit entry."""
    return (
        str(name)
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .lower()
    )


def _rename_columns(frame: pd.DataFrame, source_table: str, column_map: dict[str, dict[str, str]]) -> pd.DataFrame:
    """Translate raw Portuguese Access column names to readable English names."""
    table_column_map = column_map.get(source_table, {})
    renamed_columns = {
        column_name: table_column_map.get(column_name, _clean_fallback_name(column_name))
        for column_name in frame.columns
    }
    return frame.rename(columns=renamed_columns)


def _normalize_object_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Trim text values and coerce decimal-comma strings when the full column allows it."""
    normalized = frame.copy()
    for column in normalized.columns:
        if normalized[column].dtype != object:
            continue
        text_values = normalized[column].astype(str).str.strip()
        text_values = text_values.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        numeric_candidate = pd.to_numeric(text_values.str.replace(",", ".", regex=False), errors="coerce")
        if text_values.notna().sum() > 0 and numeric_candidate.notna().sum() == text_values.notna().sum():
            normalized[column] = numeric_candidate
        else:
            normalized[column] = text_values
    return normalized


def _load_mdb_tables(
    mdb_path: Path,
    archive_name: str,
    station_code: str,
    table_map: dict[str, str],
    column_map: dict[str, dict[str, str]],
) -> tuple[dict[str, list[pd.DataFrame]], list[dict[str, object]], bool]:
    """Read all MDB tables and split them into non-empty table payloads plus metadata."""
    parsed_tables: dict[str, list[pd.DataFrame]] = {}
    archive_records: list[dict[str, object]] = []
    has_nonempty_table = False

    with _connect_access_database(mdb_path) as connection:
        user_tables = _list_user_tables(connection)
        for source_table in user_tables:
            table_name = table_map.get(source_table, _clean_fallback_name(source_table))
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="pandas only supports SQLAlchemy connectable",
                    category=UserWarning,
                )
                frame = pd.read_sql(f"SELECT * FROM [{source_table}]", connection)
            row_count = len(frame)
            archive_records.append(
                {
                    "station_code": station_code,
                    "source_archive_name": archive_name,
                    "source_mdb_name": mdb_path.name,
                    "source_table_name": source_table,
                    "table_name": table_name,
                    "row_count": row_count,
                    "is_empty_table": int(row_count == 0),
                }
            )
            if row_count == 0:
                continue

            has_nonempty_table = True
            processed = _rename_columns(frame, source_table, column_map)
            processed = _normalize_object_columns(processed)
            processed["source_archive_name"] = archive_name
            processed["source_mdb_name"] = mdb_path.name
            processed["source_table_name"] = source_table
            parsed_tables.setdefault(table_name, []).append(processed)

    return parsed_tables, archive_records, has_nonempty_table


def preprocess_station_data(root_dir=".", single_station=None):
    """Parse downloaded MDB archives and persist translated Access tables into DuckDB."""
    water_quality_dir, raw_dir = ensure_water_quality_dirs(root_dir)
    files = list_valid_raw_archives(root_dir)
    if single_station is not None:
        files = files.loc[files["station_code"].astype(str) == str(single_station)].copy()
    if files.empty:
        raise ValueError(f"No valid raw station MDB archives found in {raw_dir}.")

    extract_root = (water_quality_dir / "_mdb_extract").resolve()
    extract_root.mkdir(parents=True, exist_ok=True)

    table_map, column_map = _load_name_mapping()
    logger.info("Preprocessing %s raw MDB archive(s) from %s.", len(files), raw_dir)

    archive_metadata = files.copy()
    archive_metadata["archive_kind"] = "mdb"
    archive_metadata["is_empty_archive"] = 0

    archive_table_records: list[dict[str, object]] = []
    parsed_tables: dict[str, list[pd.DataFrame]] = {}

    for file_row in files.itertuples(index=False):
        archive_path = raw_dir / file_row.filename
        mdb_members = _list_archive_mdb_members(archive_path)
        if not mdb_members:
            logger.info("Skipping archive without MDB members: %s", archive_path.name)
            archive_metadata.loc[archive_metadata["filename"] == file_row.filename, "is_empty_archive"] = 1
            continue

        archive_has_nonempty_data = False
        extracted_paths: list[Path] = []
        try:
            for member_name in mdb_members:
                extracted_mdb_path = _extract_mdb_member(archive_path, member_name, extract_root)
                extracted_paths.append(extracted_mdb_path)
                member_tables, member_records, has_nonempty_table = _load_mdb_tables(
                    extracted_mdb_path,
                    archive_name=file_row.filename,
                    station_code=str(file_row.station_code),
                    table_map=table_map,
                    column_map=column_map,
                )
                archive_table_records.extend(member_records)
                archive_has_nonempty_data = archive_has_nonempty_data or has_nonempty_table
                for table_name, frames in member_tables.items():
                    parsed_tables.setdefault(table_name, []).extend(frames)
        finally:
            for extracted_path in extracted_paths:
                extracted_path.unlink(missing_ok=True)

        if not archive_has_nonempty_data:
            logger.info("Archive %s contains only empty MDB tables.", file_row.filename)
            archive_metadata.loc[archive_metadata["filename"] == file_row.filename, "is_empty_archive"] = 1

    write_dataframe_table(root_dir, RAW_ARCHIVES_TABLE, archive_metadata)
    write_dataframe_table(
        root_dir,
        SENSOR_ARCHIVE_FILES_TABLE,
        pd.DataFrame(archive_table_records, columns=ARCHIVE_TABLE_RECORD_COLUMNS),
    )

    if not parsed_tables:
        logger.info("All processed MDB archives were empty. No data tables were written.")
        return {}

    for table_name, frames in parsed_tables.items():
        combined = pd.concat(frames, ignore_index=True)
        logger.info("Writing %s rows to DuckDB table %s.", len(combined), table_name)
        write_dataframe_table(root_dir, table_name, combined)

    return parsed_tables
