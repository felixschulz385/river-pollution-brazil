from pathlib import Path

from src.data.shared.paths import processed_dir
from .schema import ASSEMBLED_SENSOR_DATA_PARQUET, STATIONS_TRENCHES_PARQUET, WATER_QUALITY_CLEANING_SUBDIR

# Downloaded station archive .zip files live in their own subfolder of raw/,
# separate from the sensor_data.duckdb/sensor_downloads.duckdb files that
# also live directly in raw/ -- otherwise a plain `raw_dir.iterdir()` scan
# for archives (see fetch/data/download.py) sweeps up the database files too.
ARCHIVES_SUBDIR = "archives"


def get_root_path(root_dir="."):
    return Path(root_dir).expanduser()


def get_water_quality_dir(root_dir="."):
    return get_root_path(root_dir) / "data" / "sensor_data"


def get_raw_dir(root_dir="."):
    return get_water_quality_dir(root_dir) / "raw"


def get_archives_dir(root_dir="."):
    """Directory where downloaded station archive .zip files are stored."""
    return get_raw_dir(root_dir) / ARCHIVES_SUBDIR


def get_processed_dir(root_dir=".", stage=None):
    """Processed-output directory for sensor_data, optionally scoped to a stage.

    Mirrors ``src.data.shared.paths.processed_dir``; use ``stage="extract"``
    for the per-station/cleaning-stage outputs and ``stage="aggregate"`` for
    the final joined water-quality/streamflow panel.
    """
    return processed_dir(root_dir, "sensor_data", stage=stage)


def get_water_quality_cleaning_dir(root_dir="."):
    """Subfolder of the extract stage holding water-quality-cleaning QA
    byproducts (transformation recommendations, cleaning flags/summary) --
    kept separate from the datasets other sources import directly
    (water_quality, streamflow, stations_rivers)."""
    return get_processed_dir(root_dir, stage="extract") / WATER_QUALITY_CLEANING_SUBDIR


def get_sensor_database_path(root_dir="."):
    return get_raw_dir(root_dir) / "sensor_data.duckdb"


def get_download_log_database_path(root_dir="."):
    return get_raw_dir(root_dir) / "sensor_downloads.duckdb"


# Canonical output paths, for sources that consume these outside sensor_data
# (e.g. land_cover, climate assembly) to import instead of re-deriving the
# path themselves.
# The assembled panel already contains every cleaned water-quality column
# (plus trench_id and streamflow) -- there's no separate extract-stage
# water-quality file to point at; it's indexed on [station_code, date]
# rather than having those as plain columns, so a reader needs to
# reset_index() them back (climate's prepare_observation_targets already
# does this; land_cover's _assemble_sensor_land_cover does it explicitly).
DEFAULT_WATER_QUALITY_PATH = str(processed_dir(".", "sensor_data", stage="aggregate") / ASSEMBLED_SENSOR_DATA_PARQUET)
# The station-to-trench join requires GADM + river_network, so it's only
# available once sensor_data's aggregate stage has run.
DEFAULT_STATIONS_TRENCHES_PATH = str(processed_dir(".", "sensor_data", stage="aggregate") / STATIONS_TRENCHES_PARQUET)


def ensure_water_quality_dirs(root_dir="."):
    water_quality_dir = get_water_quality_dir(root_dir)
    raw_dir = get_raw_dir(root_dir)
    water_quality_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    return water_quality_dir, raw_dir
