"""DuckDB storage helpers for the sensor-data pipeline.

Station inventories, curated tables, and parsed sensor payloads live in the
main sensor-data database. Download history and raw-archive bookkeeping live in
a separate database so scrape retries and resets do not disturb analytical
tables.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd

from .paths import (
    ensure_water_quality_dirs,
    get_download_log_database_path,
    get_sensor_database_path,
)

DATABASE_METADATA_TABLE = "_sensor_data_table_metadata"
RAW_STATIONS_TABLE = "stations_raw"
STATIONS_TABLE = "stations"
STATION_RIVERS_TABLE = "stations_rivers"
SENSOR_DOWNLOADS_TABLE = "sensor_downloads"
RAW_ARCHIVES_TABLE = "raw_archives"
SENSOR_ARCHIVE_FILES_TABLE = "sensor_archive_files"
DOWNLOAD_LOG_TABLES = {
    SENSOR_DOWNLOADS_TABLE,
    RAW_ARCHIVES_TABLE,
}


def _database_path_for_table(root_dir: str, table_name: str | None = None) -> Path:
    if table_name in DOWNLOAD_LOG_TABLES:
        return get_download_log_database_path(root_dir)
    return get_sensor_database_path(root_dir)


def _connect(root_dir: str = ".", table_name: str | None = None) -> duckdb.DuckDBPyConnection:
    """Open the appropriate DuckDB file and ensure the metadata table exists."""
    ensure_water_quality_dirs(root_dir)
    database_path = _database_path_for_table(root_dir, table_name)
    connection = duckdb.connect(str(database_path))
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {DATABASE_METADATA_TABLE} (
            table_name VARCHAR PRIMARY KEY,
            geometry_column VARCHAR,
            geometry_crs VARCHAR,
            json_columns VARCHAR
        )
        """
    )
    return connection


def _quote_identifier(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _normalise_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Persist indexes as columns so tables round-trip cleanly."""
    if isinstance(frame.index, pd.RangeIndex):
        return frame.copy()
    return frame.reset_index()


def clean_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy without changing source database field names."""
    return frame.copy()


def _encode_json_columns(
    frame: pd.DataFrame,
    json_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Serialise list-like columns so DuckDB stores them predictably."""
    encoded = frame.copy()
    for column in json_columns or []:
        encoded[column] = encoded[column].apply(
            lambda value: json.dumps(value) if value is not None else None
        )
    return encoded


def _decode_json_columns(
    frame: pd.DataFrame,
    json_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Restore JSON-backed columns to native Python objects."""
    decoded = frame.copy()
    for column in json_columns or []:
        decoded[column] = decoded[column].apply(
            lambda value: json.loads(value) if value is not None else None
        )
    return decoded


def _write_metadata(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    geometry_column: str | None = None,
    geometry_crs: str | None = None,
    json_columns: list[str] | None = None,
) -> None:
    """Track how a table should be decoded when read back from DuckDB."""
    metadata = pd.DataFrame(
        [
            {
                "table_name": table_name,
                "geometry_column": geometry_column,
                "geometry_crs": geometry_crs,
                "json_columns": json.dumps(json_columns or []),
            }
        ]
    )
    connection.register("_metadata_frame", metadata)
    connection.execute(f"DELETE FROM {DATABASE_METADATA_TABLE} WHERE table_name = ?", [table_name])
    connection.execute(f"INSERT INTO {DATABASE_METADATA_TABLE} SELECT * FROM _metadata_frame")
    connection.unregister("_metadata_frame")


def _read_metadata(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
) -> dict[str, str | list[str] | None]:
    """Return table metadata, or defaults when the table has not been written."""
    metadata = connection.execute(
        f"SELECT * FROM {DATABASE_METADATA_TABLE} WHERE table_name = ?",
        [table_name],
    ).fetchdf()
    if metadata.empty:
        return {"geometry_column": None, "geometry_crs": None, "json_columns": []}
    row = metadata.iloc[0]
    return {
        "geometry_column": row["geometry_column"],
        "geometry_crs": row["geometry_crs"],
        "json_columns": json.loads(row["json_columns"]) if row["json_columns"] else [],
    }


def table_exists(root_dir: str, table_name: str) -> bool:
    """Check whether a given logical table is already present in the database."""
    database_path = _database_path_for_table(root_dir, table_name)
    if not database_path.exists():
        return False
    with _connect(root_dir, table_name=table_name) as connection:
        return (
            connection.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                [table_name],
            ).fetchone()[0]
            > 0
        )


def write_dataframe_table(
    root_dir: str,
    table_name: str,
    frame: pd.DataFrame,
    json_columns: list[str] | None = None,
) -> Path:
    """Persist a pandas DataFrame into DuckDB as a replace-in-place table."""
    prepared_frame = clean_column_names(frame)
    prepared_frame = _encode_json_columns(_normalise_frame(prepared_frame), json_columns=json_columns)
    quoted_table_name = _quote_identifier(table_name)
    with _connect(root_dir, table_name=table_name) as connection:
        connection.register("_table_frame", prepared_frame)
        connection.execute(f"CREATE OR REPLACE TABLE {quoted_table_name} AS SELECT * FROM _table_frame")
        connection.unregister("_table_frame")
        _write_metadata(connection, table_name, json_columns=json_columns)
    return _database_path_for_table(root_dir, table_name)


def append_dataframe_table(
    root_dir: str,
    table_name: str,
    frame: pd.DataFrame,
    json_columns: list[str] | None = None,
) -> Path:
    """Append rows into a DuckDB table, creating it on first write."""
    prepared_frame = clean_column_names(frame)
    prepared_frame = _encode_json_columns(_normalise_frame(prepared_frame), json_columns=json_columns)
    quoted_table_name = _quote_identifier(table_name)
    with _connect(root_dir, table_name=table_name) as connection:
        connection.register("_table_frame", prepared_frame)
        table_already_exists = (
            connection.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                [table_name],
            ).fetchone()[0]
            > 0
        )
        if not table_already_exists:
            connection.execute(f"CREATE TABLE {quoted_table_name} AS SELECT * FROM _table_frame")
            _write_metadata(connection, table_name, json_columns=json_columns)
        else:
            existing_columns = {
                row[1]
                for row in connection.execute(f"PRAGMA table_info({quoted_table_name})").fetchall()
            }
            frame_schema = connection.execute("DESCRIBE SELECT * FROM _table_frame").fetchall()
            for column_name, column_type, *_ in frame_schema:
                if column_name not in existing_columns:
                    quoted_column_name = _quote_identifier(column_name)
                    connection.execute(
                        f"ALTER TABLE {quoted_table_name} ADD COLUMN {quoted_column_name} {column_type}"
                    )
            connection.execute(f"INSERT INTO {quoted_table_name} BY NAME SELECT * FROM _table_frame")
        connection.unregister("_table_frame")
    return _database_path_for_table(root_dir, table_name)


def read_dataframe_table(root_dir: str, table_name: str) -> pd.DataFrame:
    """Load a pandas DataFrame from DuckDB and restore JSON-backed columns."""
    quoted_table_name = _quote_identifier(table_name)
    with _connect(root_dir, table_name=table_name) as connection:
        frame = connection.execute(f"SELECT * FROM {quoted_table_name}").fetchdf()
        metadata = _read_metadata(connection, table_name)
    return _decode_json_columns(frame, json_columns=metadata["json_columns"])


def write_geodataframe_table(
    root_dir: str,
    table_name: str,
    frame: gpd.GeoDataFrame,
    geometry_column: str = "geometry",
    json_columns: list[str] | None = None,
) -> Path:
    """Store geometries as WKT so they remain readable without DuckDB spatial."""
    cleaned_frame = clean_column_names(frame)
    cleaned_geoframe = gpd.GeoDataFrame(
        cleaned_frame,
        geometry=geometry_column,
        crs=frame.crs,
    )
    prepared_frame = _normalise_frame(cleaned_frame)
    prepared_frame = _encode_json_columns(prepared_frame, json_columns=json_columns)
    prepared_frame = prepared_frame.copy()
    # Re-wrap the frame as a GeoDataFrame before serialising to WKT so geopandas
    # still treats the geometry column as active geometry.
    prepared_frame[geometry_column] = cleaned_geoframe.geometry.reset_index(drop=True).to_wkt()
    quoted_table_name = _quote_identifier(table_name)
    with _connect(root_dir, table_name=table_name) as connection:
        connection.register("_table_frame", prepared_frame)
        connection.execute(f"CREATE OR REPLACE TABLE {quoted_table_name} AS SELECT * FROM _table_frame")
        connection.unregister("_table_frame")
        _write_metadata(
            connection,
            table_name,
            geometry_column=geometry_column,
            geometry_crs=frame.crs.to_string() if frame.crs is not None else None,
            json_columns=json_columns,
        )
    return _database_path_for_table(root_dir, table_name)


def read_geodataframe_table(root_dir: str, table_name: str) -> gpd.GeoDataFrame:
    """Load a GeoDataFrame and rebuild geometry objects from WKT text."""
    quoted_table_name = _quote_identifier(table_name)
    with _connect(root_dir, table_name=table_name) as connection:
        frame = connection.execute(f"SELECT * FROM {quoted_table_name}").fetchdf()
        metadata = _read_metadata(connection, table_name)
    frame = _decode_json_columns(frame, json_columns=metadata["json_columns"])
    geometry_column = metadata["geometry_column"] or "geometry"
    geometry = gpd.GeoSeries.from_wkt(frame[geometry_column], crs=metadata["geometry_crs"])
    return gpd.GeoDataFrame(
        frame.drop(columns=[geometry_column]),
        geometry=geometry,
        crs=metadata["geometry_crs"],
    )
