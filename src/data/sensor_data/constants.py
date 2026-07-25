from pathlib import Path


def get_root_path(root_dir="."):
    return Path(root_dir).expanduser()


def get_water_quality_dir(root_dir="."):
    return get_root_path(root_dir) / "data" / "sensor_data"


def get_raw_dir(root_dir="."):
    return get_water_quality_dir(root_dir) / "raw"


def get_sensor_database_path(root_dir="."):
    return get_water_quality_dir(root_dir) / "sensor_data.duckdb"


def get_download_log_database_path(root_dir="."):
    return get_water_quality_dir(root_dir) / "sensor_downloads.duckdb"


def ensure_water_quality_dirs(root_dir="."):
    water_quality_dir = get_water_quality_dir(root_dir)
    raw_dir = get_raw_dir(root_dir)
    water_quality_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    return water_quality_dir, raw_dir
