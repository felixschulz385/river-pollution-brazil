from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


MAPBIOMAS_COLLECTION = "collection_10"
MAPBIOMAS_FILENAME_TEMPLATE = "brazil_coverage_{year}.tif"


@dataclass(frozen=True)
class LandCoverPaths:
    """Resolved land-cover input paths, rooted at ``root_dir``."""

    datadir: Path
    drainage_path: Path
    legend_path: Path


def build_paths(root_dir: str | Path = ".") -> LandCoverPaths:
    """Derive land-cover input paths relative to ``root_dir``."""
    root = Path(root_dir)
    land_cover_root = root / "data" / "land_cover"
    return LandCoverPaths(
        datadir=land_cover_root / "raw" / "lc_mapbiomas10_30",
        drainage_path=root / "data" / "river_network" / "drainage_areas.parquet",
        legend_path=land_cover_root / "mapbiomas_legend.xlsx",
    )

TRENCH_ID_COLUMN = "trench_id"
YEAR_COLUMN = "year"
ADM2_ID_COLUMN = "adm2_id"
MUN_ID_COLUMN = "mun_id"
REACHABLE_TRENCH_COUNT_COLUMN = "reachable_trench_count"
TOTAL_WEIGHT_COLUMN = "total_weight"

LAND_COVER_CLASS_PREFIX = "land_cover_class_"
LAND_COVER_TOTAL_COLUMN = "land_cover_total"

SENSOR_ASSEMBLY_VARIANT = "sensor"
ADM2_ASSEMBLY_VARIANT = "adm2"
LEGACY_SENSOR_UPSTREAM_DISTANCE_BUCKETS_VARIANT = "sensor_upstream_distance_buckets"
ASSEMBLY_VARIANTS = (
    SENSOR_ASSEMBLY_VARIANT,
    ADM2_ASSEMBLY_VARIANT,
)
DEFAULT_ASSEMBLY_LAND_COVER_PATH = "data/land_cover/land_cover.feather"
DEFAULT_WATER_QUALITY_PATH = "data/sensor_data/water_quality.parquet"
DEFAULT_STATIONS_RIVERS_PATH = "data/sensor_data/stations_rivers.parquet"
DEFAULT_RIVER_NETWORK_PATH = "data/river_network"
DEFAULT_SENSOR_UPSTREAM_OUTPUT_PATH = "data/land_cover/land_cover_sensor_upstream.parquet"
DEFAULT_ADM2_UPSTREAM_OUTPUT_PATH = "data/land_cover/land_cover_adm2_upstream.parquet"

STATION_CODE_COLUMN = "station_code"
DATETIME_COLUMN = "datetime"
DATE_COLUMN = "date"
UPSTREAM_DISTANCE_COLUMN = "upstream_distance"
ADJUSTED_DISTANCE_COLUMN = "adjusted_distance"
DISTANCE_BUCKET_COLUMN = "bucket"
LAND_COVER_CLASS_COLUMN = "land_cover_class"
BUCKET_REACHABLE_COUNT_COLUMN = "n"
BUCKET_COUNT_COLUMN = "cnt"
BUCKET_SHARE_COLUMN = "share"

# River-network distances are stored in kilometers.
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
