from pathlib import Path

import numpy as np


TRENCH_ID_COLUMN = "trench_id"
DATE_COLUMN = "date"
YEAR_COLUMN = "year"
MONTH_COLUMN = "month"
STATION_CODE_COLUMN = "station_code"
DATETIME_COLUMN = "datetime"
UPSTREAM_DISTANCE_COLUMN = "upstream_distance"
DISTANCE_BUCKET_COLUMN = "distance_bucket"
ADM2_ID_COLUMN = "adm2_id"
REACHABLE_TRENCH_COUNT_COLUMN = "reachable_trench_count"
TOTAL_WEIGHT_COLUMN = "total_weight"

ERA5_LAND_PREPROCESS_SUBTYPES = {"era5_land_hourly", "era5_land_daily"}
ERA5_LAND_PREPROCESS_STAGES = {"all", "zarr", "parquet"}
SENSOR_UPSTREAM_DISTANCE_BUCKETS_VARIANT = "sensor_upstream_distance_buckets"
ADM2_UPSTREAM_YEARLY_VARIANT = "adm2_upstream_yearly"
CLIMATE_ASSEMBLE_VARIANTS = {
    SENSOR_UPSTREAM_DISTANCE_BUCKETS_VARIANT,
    ADM2_UPSTREAM_YEARLY_VARIANT,
}

DEFAULT_RIVER_NETWORK_PATH = "data/river_network"
DEFAULT_WATER_QUALITY_PATH = "data/sensor_data/water_quality.parquet"
DEFAULT_STATIONS_RIVERS_PATH = "data/sensor_data/stations_rivers.parquet"
DEFAULT_ERA5_LAND_STORE_PATH = Path("data/climate/processed/era5_land.zarr_nobackup")
DEFAULT_ERA5_LAND_TRENCH_DAY_PATH = Path("data/climate/processed/era5_land.parquet")
DEFAULT_SENSOR_UPSTREAM_OUTPUT_PATH = Path(
    "data/climate/processed/era5_land/climate_sensor_upstream.parquet"
)
DEFAULT_ADM2_UPSTREAM_YEARLY_OUTPUT_PATH = Path(
    "data/climate/processed/era5_land/climate_adm2_upstream_yearly.parquet"
)

SENSOR_WINDOW_LABELS = {
    "7d": 7,
    "30d": 30,
    "90d": 90,
    "180d": 180,
    "365d": 365,
}

ANNUAL_SUM_VARIABLES = {"tp", "sro", "ssro", "pev"}
ANNUAL_MEAN_VARIABLES = {"2t", "2d", "swvl1", "swvl2"}
ANNUAL_MIN_VARIABLES = {"2t_daily_min"}
ANNUAL_MAX_VARIABLES = {"2t_daily_max"}
SENSOR_DISTANCE_BUCKETS = (
    ("0_10km", 0.0, 10.0),
    ("10_50km", 10.0, 50.0),
    ("50_100km", 50.0, 100.0),
    ("100_250km", 100.0, 250.0),
    ("250_500km", 250.0, 500.0),
    ("500km_plus", 500.0, np.inf),
)
