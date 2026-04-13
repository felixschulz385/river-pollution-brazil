import logging
import shutil
import warnings
import zipfile
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pyodbc

logger = logging.getLogger(__name__)


def clean_fallback_name(name: str) -> str:
    """Create a stable fallback name if the mapping file has no explicit entry."""
    return (
        str(name)
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .lower()
    )


def rename_columns(frame: pd.DataFrame, source_table: str, column_map: dict[str, dict[str, str]]) -> pd.DataFrame:
    """Translate raw Portuguese Access column names to readable English names."""
    table_column_map = column_map.get(source_table, {})
    renamed_columns = {
        column_name: table_column_map.get(column_name, clean_fallback_name(column_name))
        for column_name in frame.columns
    }
    return frame.rename(columns=renamed_columns)


def normalize_object_columns(frame: pd.DataFrame) -> pd.DataFrame:
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


def list_archive_mdb_members(archive_path: str | Path) -> list[str]:
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


def extract_mdb_member(archive_file: zipfile.ZipFile, member_name: str, extract_root: Path) -> Path:
    """Extract one MDB file using ZipFile.extract and return a local Access path."""
    extracted_path = Path(archive_file.extract(member_name, path=extract_root))
    target_path = extract_root / f"{uuid4().hex}_{Path(member_name).name}"
    extracted_path.replace(target_path)
    return target_path


def connect_access_database(mdb_path: str | Path):
    """Open a Microsoft Access MDB file with the Windows Access ODBC driver."""
    return pyodbc.connect(
        rf"Driver={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={Path(mdb_path).resolve()};"
    )


def list_user_tables(connection) -> list[str]:
    """List non-system tables from one MDB file."""
    cursor = connection.cursor()
    return [
        row.table_name
        for row in cursor.tables(tableType="TABLE")
        if not str(row.table_name).startswith("MSys")
    ]


def read_access_table(connection, source_table: str) -> pd.DataFrame:
    """Read one Access table with the direct SELECT path."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.read_sql(f"SELECT * FROM [{source_table}]", connection)


def load_mdb_tables(
    mdb_path: str | Path,
    archive_name: str,
    station_code: str,
    table_map: dict[str, str],
    column_map: dict[str, dict[str, str]],
    source_tables: list[str] | None = None,
) -> tuple[dict[str, list[pd.DataFrame]], list[dict[str, object]], bool]:
    """Read MDB tables and split them into non-empty table payloads plus metadata."""
    parsed_tables: dict[str, list[pd.DataFrame]] = {}
    archive_records: list[dict[str, object]] = []
    has_nonempty_table = False

    with connect_access_database(mdb_path) as connection:
        tables_to_read = source_tables if source_tables is not None else list_user_tables(connection)
        for source_table in tables_to_read:
            try:
                frame = read_access_table(connection, source_table)
            except pyodbc.Error:
                if source_tables is not None:
                    continue
                raise

            table_name = table_map.get(source_table, clean_fallback_name(source_table))
            row_count = len(frame)
            archive_records.append(
                {
                    "station_code": station_code,
                    "source_archive_name": archive_name,
                    "source_mdb_name": Path(mdb_path).name,
                    "source_table_name": source_table,
                    "table_name": table_name,
                    "row_count": row_count,
                    "is_empty_table": int(row_count == 0),
                }
            )
            if row_count == 0:
                continue

            has_nonempty_table = True
            processed = rename_columns(frame, source_table, column_map)
            processed = normalize_object_columns(processed)
            processed["source_archive_name"] = archive_name
            processed["source_mdb_name"] = Path(mdb_path).name
            processed["source_table_name"] = source_table
            parsed_tables.setdefault(table_name, []).append(processed)

    return parsed_tables, archive_records, has_nonempty_table


def read_archive_payload(
    archive_path: str,
    archive_name: str,
    station_code: str,
    table_map: dict[str, str],
    column_map: dict[str, dict[str, str]],
    source_tables: list[str] | None = None,
    extract_base_dir: str | None = None,
) -> dict[str, object]:
    """Extract and read one ZIP archive in an isolated worker process."""
    mdb_members = list_archive_mdb_members(archive_path)
    if not mdb_members:
        return {
            "archive_name": archive_name,
            "station_code": station_code,
            "mdb_members": 0,
            "parsed_tables": {},
            "archive_records": [],
            "has_nonempty_table": False,
            "without_mdb": True,
        }

    parsed_tables: dict[str, list[pd.DataFrame]] = {}
    archive_records: list[dict[str, object]] = []
    archive_has_nonempty_data = False

    extract_parent = Path(extract_base_dir) if extract_base_dir is not None else Path.cwd()
    extract_root = extract_parent / f"sensor_mdb_{uuid4().hex}"
    extract_root.mkdir(parents=True, exist_ok=False)
    try:
        with zipfile.ZipFile(archive_path, "r") as archive_file:
            for member_name in mdb_members:
                extracted_mdb_path = extract_mdb_member(archive_file, member_name, extract_root)
                member_tables, member_records, has_nonempty_table = load_mdb_tables(
                    extracted_mdb_path,
                    archive_name=archive_name,
                    station_code=station_code,
                    table_map=table_map,
                    column_map=column_map,
                    source_tables=source_tables,
                )
                archive_records.extend(member_records)
                archive_has_nonempty_data = archive_has_nonempty_data or has_nonempty_table
                for table_name, frames in member_tables.items():
                    parsed_tables.setdefault(table_name, []).extend(frames)
    finally:
        shutil.rmtree(extract_root, ignore_errors=True)

    return {
        "archive_name": archive_name,
        "station_code": station_code,
        "mdb_members": len(mdb_members),
        "parsed_tables": parsed_tables,
        "archive_records": archive_records,
        "has_nonempty_table": archive_has_nonempty_data,
        "without_mdb": False,
    }
