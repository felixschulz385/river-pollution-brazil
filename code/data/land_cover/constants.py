import numpy as np


DATADIR = "/scicore/home/meiera/schulz0022/projects/river-pollution-brazil/data/land_cover/raw/lc_mapbiomas8_30/"
DRAINAGE_PATH = "data/river_network/drainage_areas.parquet"
LEGEND_PATH = "data/land_cover/mapbiomas_legend.xlsx"

TRENCH_ID_COLUMN = "trench_id"
YEAR_COLUMN = "year"
ADM2_ID_COLUMN = "adm2_id"
REACHABLE_TRENCH_COUNT_COLUMN = "reachable_trench_count"
TOTAL_WEIGHT_COLUMN = "total_weight"

LAND_COVER_CLASS_PREFIX = "land_cover_class_"
LAND_COVER_TOTAL_COLUMN = "land_cover_total"

SENSOR_UPSTREAM_DISTANCE_BUCKETS_VARIANT = "sensor_upstream_distance_buckets"
DEFAULT_SENSOR_LAND_COVER_PATH = "data/land_cover/land_cover.feather"
DEFAULT_WATER_QUALITY_PATH = "data/sensor_data/water_quality.parquet"
DEFAULT_STATIONS_RIVERS_PATH = "data/sensor_data/stations_rivers.parquet"
DEFAULT_RIVER_NETWORK_PATH = "data/river_network"
DEFAULT_SENSOR_UPSTREAM_OUTPUT_PATH = "data/land_cover/land_cover_sensor_upstream.parquet"

STATION_CODE_COLUMN = "station_code"
DATETIME_COLUMN = "datetime"
DATE_COLUMN = "date"
UPSTREAM_DISTANCE_COLUMN = "upstream_distance"
DISTANCE_BUCKET_COLUMN = "distance_bucket"

# River-network distances are stored in kilometers.
SENSOR_DISTANCE_BUCKETS = (
    ("0_10km", 0.0, 10.0),
    ("10_50km", 10.0, 50.0),
    ("50_100km", 50.0, 100.0),
    ("100_250km", 100.0, 250.0),
    ("250_500km", 250.0, 500.0),
    ("500km_plus", 500.0, np.inf),
)
