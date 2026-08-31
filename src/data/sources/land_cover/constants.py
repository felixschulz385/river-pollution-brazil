from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.sources.river_network.constants import DRAINAGE_AREAS_FILENAME as _RIVER_NETWORK_DRAINAGE_AREAS_FILENAME
from src.data.sources.river_network.constants import PROCESSED_DIR as _RIVER_NETWORK_PROCESSED_DIR
from src.data.sources.sensor_data.constants import (
    DEFAULT_STATIONS_TRENCHES_PATH,
    DEFAULT_WATER_QUALITY_PATH,
)


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
        drainage_path=root / _RIVER_NETWORK_PROCESSED_DIR / _RIVER_NETWORK_DRAINAGE_AREAS_FILENAME,
        legend_path=land_cover_root / "auxiliary" / "mapbiomas_legend.xlsx",
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
DEFAULT_ASSEMBLY_LAND_COVER_PATH = "data/land_cover/processed/extract/land_cover.parquet"
DEFAULT_RIVER_NETWORK_PATH = _RIVER_NETWORK_PROCESSED_DIR
DEFAULT_SENSOR_UPSTREAM_OUTPUT_PATH = (
    "data/land_cover/processed/aggregate/land_cover_sensor_upstream.parquet"
)
DEFAULT_ADM2_UPSTREAM_OUTPUT_PATH = (
    "data/land_cover/processed/aggregate/land_cover_adm2_upstream.parquet"
)
DEFAULT_RIVER_AGGREGATED_OUTPUT_PATH = (
    "data/land_cover/processed/aggregate/land_cover_river_aggregated.parquet"
)

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

# Leaf-level land-cover classes used by `composition.compute_kernel_weighted_composition`,
# after resolving the c3 (-> pasture/agriculture) and c4 (-> urban/mining/other)
# parent-child mismatches present in the raw MapBiomas class codes.
LAND_COVER_LEAF_CLASSES = (
    "forest",
    "nonforest_nat",
    "pasture",
    "agriculture",
    "farming_unclassified",
    "urban",
    "mining",
    "other",
    "water",
)
LAND_COVER_ALR_CLASSES = (
    "pasture",
    "agriculture",
    "farming_unclassified",
    "urban",
    "mining",
    "other",
)
# Raw MapBiomas class codes feeding the bucket pivot, keyed by leaf/parent name.
LAND_COVER_CLASS_CODE_FOREST = 1
LAND_COVER_CLASS_CODE_NONFOREST_NAT = 2
LAND_COVER_CLASS_CODE_FARMING_PARENT = 3  # "c3": pasture + agriculture
LAND_COVER_CLASS_CODE_PASTURE = 30
LAND_COVER_CLASS_CODE_AGRICULTURE = 31
LAND_COVER_CLASS_CODE_URBAN_PARENT = 4  # "c4": urban + mining + other
LAND_COVER_CLASS_CODE_URBAN = 40
LAND_COVER_CLASS_CODE_MINING = 41
LAND_COVER_CLASS_CODE_OTHER_RAW = 42
LAND_COVER_CLASS_CODE_WATER = 5

# 25 km-wide upstream rings out to 500 km, plus an open-ended tail beyond that,
# used to derive inverse-sqrt-distance kernel weights per (entity, bucket).
LAND_COVER_COMPOSITION_RING_WIDTH_KM = 25
LAND_COVER_COMPOSITION_BUCKET_MAP = {
    bucket: (f"{bucket}_{bucket + LAND_COVER_COMPOSITION_RING_WIDTH_KM}km", bucket + LAND_COVER_COMPOSITION_RING_WIDTH_KM / 2)
    for bucket in range(0, 500, LAND_COVER_COMPOSITION_RING_WIDTH_KM)
}
LAND_COVER_COMPOSITION_BUCKET_MAP[500] = ("500km_plus", 750.0)
LAND_COVER_COMPOSITION_PSEUDOCOUNT = 1e-4

ADM2_ID_TO_MUN_ID_TRUNCATION = 1  # trailing digits dropped from `adm2_id` to derive `mun_id`


def derive_mun_id_from_adm2_id(adm2_id):
    """Derive the 6-digit IBGE `mun_id` by dropping `adm2_id`'s trailing check digit.

    Only rejects null input -- callers use varying-length placeholder values in
    tests, so this can't enforce the real 7-digit IBGE length. A null/NaN
    `adm2_id` (`str(nan)` -> `"nan"`) would otherwise be silently truncated into
    a bogus `"na"` join key instead of raising here.
    """
    if pd.isna(adm2_id):
        raise ValueError(f"Cannot derive mun_id from a null adm2_id: {adm2_id!r}")
    return str(adm2_id)[:-ADM2_ID_TO_MUN_ID_TRUNCATION]
