import logging
import shutil
import warnings
import zipfile
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pyodbc

logger = logging.getLogger(__name__)


def rename_columns(frame: pd.DataFrame, source_table: str, column_map: dict[str, dict[str, str]]) -> pd.DataFrame:
    """Keep raw Access column names for the database conversion step."""
    return frame.copy()


def normalize_object_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Trim text values and coerce decimal-comma strings when the full column allows it."""
    normalized = frame.copy()
    for column in normalized.columns:
        # Text columns from `pd.read_sql` may come back as classic `object`
        # dtype or (pandas >= 3.0's default) `str` dtype depending on the
        # pandas version -- checking only `dtype == object` silently skips
        # every text column under the newer default, defeating the
        # decimal-comma coercion below. `is_string_dtype`/`is_object_dtype`
        # both correctly exclude datetime64 columns, which must stay
        # untouched here.
        column_dtype = normalized[column].dtype
        if not (
            pd.api.types.is_object_dtype(column_dtype)
            or pd.api.types.is_string_dtype(column_dtype)
        ):
            continue
        text_values = normalized[column].astype(str).str.strip()
        text_values = text_values.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        # A value with both "." and "," is unambiguously pt-BR formatted
        # (period = thousands separator, comma = decimal separator, e.g.
        # "1.234,56"); strip the periods before swapping the comma in. A
        # bare comma with no period (e.g. "1,234") is inherently ambiguous
        # between a decimal separator and a thousands separator with no way
        # to tell from the string alone -- this coerces it as a decimal
        # (matching this function's existing pt-BR-decimal assumption), which
        # would silently be a 1000x scale error if that particular column
        # actually used comma as a thousands separator instead.
        has_both_separators = text_values.str.contains(".", regex=False, na=False) & text_values.str.contains(
            ",", regex=False, na=False
        )
        digits_only = text_values.where(~has_both_separators, text_values.str.replace(".", "", regex=False))
        numeric_candidate = pd.to_numeric(digits_only.str.replace(",", ".", regex=False), errors="coerce")
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

    # pyodbc's context manager only commits/rolls back the transaction on
    # exit, it does NOT close the connection -- leaving the ODBC handle (and
    # any file lock the Access driver holds on `mdb_path`) open until GC
    # gets to it. Close explicitly so the caller's `shutil.rmtree` on the
    # extracted temp dir doesn't silently fail on a still-locked file.
    connection = connect_access_database(mdb_path)
    try:
        tables_to_read = source_tables if source_tables is not None else list_user_tables(connection)
        for source_table in tables_to_read:
            try:
                frame = read_access_table(connection, source_table)
            except pyodbc.Error:
                # Skip just this table rather than aborting the whole MDB file,
                # regardless of whether tables were auto-discovered or requested
                # explicitly -- one unreadable/unsupported table (e.g. an Access
                # data type pyodbc can't marshal) shouldn't lose every other
                # table's data for this station.
                logger.warning(
                    "Skipping unreadable table '%s' in %s (station %s).",
                    source_table,
                    mdb_path,
                    station_code,
                    exc_info=True,
                )
                continue

            table_name = source_table
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
    finally:
        connection.close()

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
