from __future__ import annotations

from src.data.sources.sensor_data.constants import (
    get_archives_dir,
    get_processed_dir,
    get_raw_dir,
    get_water_quality_cleaning_dir,
)


def test_get_archives_dir_is_a_subfolder_of_raw_dir(tmp_path):
    """Downloaded station .zip archives live in their own subfolder of raw/,
    separate from sensor_data.duckdb/sensor_downloads.duckdb -- otherwise a
    plain directory scan for archives sweeps up the database files too."""
    archives_dir = get_archives_dir(tmp_path)

    assert archives_dir == get_raw_dir(tmp_path) / "archives"
    assert archives_dir != get_raw_dir(tmp_path)


def test_get_water_quality_cleaning_dir_is_a_subfolder_of_extract_dir(tmp_path):
    """Water-quality-cleaning QA byproducts (transformations/flags/summary)
    live in their own extract-stage subfolder, separate from the datasets
    other sources import directly (water_quality/streamflow/stations_rivers)."""
    cleaning_dir = get_water_quality_cleaning_dir(tmp_path)

    assert cleaning_dir == get_processed_dir(tmp_path, stage="extract") / "water_quality_cleaning"
    assert cleaning_dir != get_processed_dir(tmp_path, stage="extract")
