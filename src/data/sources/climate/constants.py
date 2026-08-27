from pathlib import Path

import numpy as np

from src.data.shared.paths import processed_dir, raw_dir
from src.data.sources.river_network.constants import PROCESSED_DIR as _RIVER_NETWORK_PROCESSED_DIR
from src.data.sources.sensor_data.constants import (
    DEFAULT_STATIONS_RIVERS_PATH,
    DEFAULT_WATER_QUALITY_PATH,
)


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

# Single source of truth for which CDS dataset id backs each GRIB-origin
# ERA5-Land preprocess subtype; fetch/era5_land_hourly.py, fetch/era5_land_daily.py,
# fetch/common.py, and preprocess/era5_land.py all key off this instead of
# hardcoding the dataset ids independently.
ERA5_LAND_SUBTYPE_DATASETS = {
    "era5_land_hourly": "reanalysis-era5-land",
    "era5_land_daily": "derived-era5-land-daily-statistics",
}
# Brazil has used a single standard time (UTC-3, "Brasília time") nationwide
# since the 2019 DST repeal; ERA5-Land is archived in UTC. Water-quality and
# sensor observations are dated in Brazil local time, so both daily-climate
# paths below (the CDS-side daily-statistics request and our own hourly->daily
# resample) bucket by this offset rather than by UTC calendar day, to keep
# their "date" values aligned with the outcomes they're joined against.
BRAZIL_UTC_OFFSET_HOURS = -3
# Number of UTC hours from the *next* month's day 1 that a month's own hourly
# GRIB/ARCO input needs appended before resampling by local day, so its own
# last Brazil-local day (built by shifting timestamps back
# `BRAZIL_UTC_OFFSET_HOURS` hours before bucketing by calendar date) isn't
# short by that many hours. See `preprocess/era5_land.py`'s
# `resample_era5l_hourly_to_daily`/`_drop_incomplete_boundary_day`,
# `fetch/era5_land_hourly.py`'s `build_era5_land_hourly_boundary_request`,
# and `preprocess/era5_land_arco.py`'s widened ARCO time slice.
BOUNDARY_HOURS = abs(BRAZIL_UTC_OFFSET_HOURS)

ERA5_LAND_PREPROCESS_SUBTYPES = set(ERA5_LAND_SUBTYPE_DATASETS)
ERA5_LAND_PREPROCESS_STAGES = {"all", "zarr", "parquet"}
SENSOR_UPSTREAM_DISTANCE_BUCKETS_VARIANT = "sensor_upstream_distance_buckets"
ADM2_UPSTREAM_YEARLY_VARIANT = "adm2_upstream_yearly"
CLIMATE_ASSEMBLE_VARIANTS = {
    SENSOR_UPSTREAM_DISTANCE_BUCKETS_VARIANT,
    ADM2_UPSTREAM_YEARLY_VARIANT,
}

DEFAULT_RIVER_NETWORK_PATH = _RIVER_NETWORK_PROCESSED_DIR
DEFAULT_ERA5_LAND_STORE_PATH = raw_dir(".", "climate") / "era5_land.zarr_nobackup"
DEFAULT_ERA5_LAND_TRENCH_DAY_PATH = processed_dir(".", "climate", stage="extract") / "climate_era5_land.parquet"
DEFAULT_SENSOR_UPSTREAM_OUTPUT_PATH = (
    processed_dir(".", "climate", stage="aggregate") / "climate_sensor_upstream.parquet"
)
DEFAULT_ADM2_UPSTREAM_YEARLY_OUTPUT_PATH = (
    processed_dir(".", "climate", stage="aggregate") / "climate_adm2_upstream_yearly.parquet"
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
