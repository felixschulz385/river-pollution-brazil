SENSOR_MODE = "sensor"
ADM2_MODE = "adm2"
ASSEMBLY_MODES = (SENSOR_MODE, ADM2_MODE)

DEFAULT_CONFIG_PATH = "setup/assembly_datasets.yaml"

LAND_COVER_BUCKETED_SOURCE_TYPE = "land_cover_bucketed"
# Same long bucket schema, read from the C11 mining-substances aggregate and
# collapsed into `mining_*` group shares (`land_cover.mining`).
MINING_BUCKETED_SOURCE_TYPE = "mining_bucketed"
CLIMATE_BUCKETED_SOURCE_TYPE = "climate_bucketed"
LONG_PIVOT_SOURCE_TYPE = "long_pivot"
GROUPED_SUM_SOURCE_TYPE = "grouped_sum"
DEFAULT_PIVOT_COLUMN_TEMPLATE = "{pivot_value}_{value_column}"

RING_BUCKETED_SOURCE_TYPES = (LAND_COVER_BUCKETED_SOURCE_TYPE, MINING_BUCKETED_SOURCE_TYPE)
BUCKETED_SOURCE_TYPES = (*RING_BUCKETED_SOURCE_TYPES, CLIMATE_BUCKETED_SOURCE_TYPE)

# Bucketed-source scope: `upstream` kernel-weights the
# 0-500 km rings; `local` averages, with equal weight, the rings of the trenches
# that touch the ADM2 itself (`bucket_intersects_adm2`), prefixing outputs `local_`.
UPSTREAM_SCOPE = "upstream"
LOCAL_SCOPE = "local"
BUCKET_SCOPES = (UPSTREAM_SCOPE, LOCAL_SCOPE)
LOCAL_COLUMN_PREFIX = "local_"
WIDE_SOURCE_TYPE = "wide"
SOURCE_TYPES = (
    WIDE_SOURCE_TYPE,
    LAND_COVER_BUCKETED_SOURCE_TYPE,
    MINING_BUCKETED_SOURCE_TYPE,
    CLIMATE_BUCKETED_SOURCE_TYPE,
    LONG_PIVOT_SOURCE_TYPE,
    GROUPED_SUM_SOURCE_TYPE,
)

DATETIME_COLUMN = "datetime"
DATE_COLUMN = "date"
YEAR_COLUMN = "year"
STATION_CODE_COLUMN = "station_code"
MUN_ID_COLUMN = "mun_id"

CANONICAL_INDEX_COLUMNS = {
    SENSOR_MODE: (STATION_CODE_COLUMN, DATETIME_COLUMN),
    ADM2_MODE: (MUN_ID_COLUMN, YEAR_COLUMN),
}
