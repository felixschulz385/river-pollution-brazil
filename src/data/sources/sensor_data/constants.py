from pathlib import Path

from src.data.shared.paths import processed_dir


def get_root_path(root_dir="."):
    return Path(root_dir).expanduser()


def get_water_quality_dir(root_dir="."):
    return get_root_path(root_dir) / "data" / "sensor_data"


def get_raw_dir(root_dir="."):
    return get_water_quality_dir(root_dir) / "raw"


def get_processed_dir(root_dir=".", stage=None):
    """Processed-output directory for sensor_data, optionally scoped to a stage.

    Mirrors ``src.data.shared.paths.processed_dir``; use ``stage="extract"``
    for the per-station/cleaning-stage outputs and ``stage="aggregate"`` for
    the final joined water-quality/streamflow panel.
    """
    return processed_dir(root_dir, "sensor_data", stage=stage)


def get_sensor_database_path(root_dir="."):
    return get_raw_dir(root_dir) / "sensor_data.duckdb"


def get_download_log_database_path(root_dir="."):
    return get_raw_dir(root_dir) / "sensor_downloads.duckdb"


def ensure_water_quality_dirs(root_dir="."):
    water_quality_dir = get_water_quality_dir(root_dir)
    raw_dir = get_raw_dir(root_dir)
    water_quality_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    return water_quality_dir, raw_dir
