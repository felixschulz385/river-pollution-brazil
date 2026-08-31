"""Export fetch's raw DuckDB tables to parquet -- fetch's public output.

DuckDB remains fetch's internal working store (download log, resumable MDB
parsing), but downstream stages (preprocess, assemble) read only the parquet
files written here, not the DuckDB tables directly.
"""

from __future__ import annotations

from .database import STATIONS_TABLE, read_dataframe_table, read_geodataframe_table, table_exists
from ..constants import ensure_water_quality_dirs, get_raw_dir
from ..schema import (
    RAW_STATIONS_PARQUET,
    RAW_STREAMFLOW_PARQUET,
    RAW_WATER_QUALITY_PARQUET,
    STREAMFLOW_SOURCE_TABLES,
    WATER_QUALITY_SOURCE_TABLES,
)


def _first_available_table(root_dir: str, table_names) -> str:
    for table_name in table_names:
        if table_exists(root_dir, table_name):
            return table_name
    raise ValueError("No source table found. Expected one of: " + ", ".join(table_names))


def export_raw_tables(root_dir=".") -> dict[str, str]:
    """Write the raw station inventory, water-quality, and streamflow tables
    fetch just populated in DuckDB out to parquet under `raw/`."""
    ensure_water_quality_dirs(root_dir)
    raw_dir = get_raw_dir(root_dir)
    outputs: dict[str, str] = {}

    stations = read_geodataframe_table(root_dir, STATIONS_TABLE)
    stations_path = raw_dir / RAW_STATIONS_PARQUET
    stations.to_parquet(stations_path, index=False)
    outputs["stations"] = str(stations_path)

    water_quality_table = _first_available_table(root_dir, WATER_QUALITY_SOURCE_TABLES)
    water_quality = read_dataframe_table(root_dir, water_quality_table)
    water_quality_path = raw_dir / RAW_WATER_QUALITY_PARQUET
    water_quality.to_parquet(water_quality_path, index=False)
    outputs["water_quality"] = str(water_quality_path)

    streamflow_table = _first_available_table(root_dir, STREAMFLOW_SOURCE_TABLES)
    streamflow = read_dataframe_table(root_dir, streamflow_table)
    streamflow_path = raw_dir / RAW_STREAMFLOW_PARQUET
    streamflow.to_parquet(streamflow_path, index=False)
    outputs["streamflow"] = str(streamflow_path)

    return outputs


__all__ = ["export_raw_tables"]
