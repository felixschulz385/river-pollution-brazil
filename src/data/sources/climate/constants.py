from pathlib import Path

import numpy as np

from src.data.sources.river_network.constants import PROCESSED_DIR as _RIVER_NETWORK_PROCESSED_DIR


TRENCH_ID_COLUMN = "trench_id"
DATE_COLUMN = "date"
YEAR_COLUMN = "year"
MONTH_COLUMN = "month"
STATION_CODE_COLUMN = "station_code"
DATETIME_COLUMN = "datetime"
UPSTREAM_DISTANCE_COLUMN = "upstream_distance"
ADJUSTED_DISTANCE_COLUMN = "adjusted_distance"
DISTANCE_BUCKET_COLUMN = "distance_bucket"
CLIMATE_VARIABLE_COLUMN = "climate_variable"
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

DEFAULT_RIVER_NETWORK_PATH = _RIVER_NETWORK_PROCESSED_DIR
DEFAULT_WATER_QUALITY_PATH = "data/sensor_data/water_quality.parquet"
DEFAULT_STATIONS_RIVERS_PATH = "data/sensor_data/stations_rivers.parquet"
DEFAULT_ERA5_LAND_STORE_PATH = Path("data/climate/raw/era5_land.zarr_nobackup")
DEFAULT_ERA5_LAND_TRENCH_DAY_PATH = Path("data/climate/processed/extract/era5_land.parquet")
DEFAULT_SENSOR_UPSTREAM_OUTPUT_PATH = Path(
    "data/climate/processed/aggregate/climate_sensor_upstream.parquet"
)
DEFAULT_ADM2_UPSTREAM_YEARLY_OUTPUT_PATH = Path(
    "data/climate/processed/aggregate/climate_adm2_upstream_yearly.parquet"
)

# River-network distances are stored in kilometers. Buckets are 25 km wide,
# labeled by integer lower bound, and computed on the shifted/adjusted
# distance (see ADJUSTED_DISTANCE_COLUMN) so that 0 is the upstream end of
# the seed trench itself — matching the scheme used by land_cover on master.
SENSOR_DISTANCE_BUCKET_WIDTH_KM = 25.0
SENSOR_DISTANCE_BUCKET_STARTS_KM = tuple(range(0, 501, 25))
SENSOR_DISTANCE_BUCKETS = tuple(
    (
        float(lower_bound),
        float(lower_bound + SENSOR_DISTANCE_BUCKET_WIDTH_KM)
        if lower_bound < SENSOR_DISTANCE_BUCKET_STARTS_KM[-1]
        else np.inf,
    )
    for lower_bound in SENSOR_DISTANCE_BUCKET_STARTS_KM
)
