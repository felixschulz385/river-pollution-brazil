"""MapBiomas Collection 11 mining substances: legend groups and composition.

The asset (exported by `scripts/gee_export_mapbiomas_mining.js`, merged to one
GeoTIFF per year by `scripts/prepare_mapbiomas_mining.py`) codes each mined
pixel ABCC: A = 1 garimpo (artisanal), 2 industrial; B = 1 exposed soil,
2 revegetation; CC = substance (15 gold, 58 atypical, 59 no substance, ...).
Pixels without mining are 0.

The land-cover pipeline extracts and aggregates it as its `mining` product
(`data preprocess --source land_cover --phase extract|aggregate --product
mining`, paths below) through the legend built here. It maps every code to a mining group (`Class`) and phase (`Subclass`), and
0 to class 0, so `land_cover_total` is the polygon's full pixel count and
shares are shares of land area.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.shared.sensor_upstream import INV_SQRT_DISTANCE_KERNEL, bucket_kernel_weights

from .constants import LAND_COVER_COMPOSITION_BUCKET_MAP


MINING_DATADIR = "data/land_cover/raw/mapbiomas11_mining"
MINING_TILES_DIR = "data/land_cover/raw/mapbiomas11_mining_tiles"
MINING_SOURCE_LEGEND_PATH = (
    "data/land_cover/auxiliary/"
    "legend_code_mapbiomas_mining_brazil_collection_11_mined_substances.csv"
)
MINING_LEGEND_PATH = "data/land_cover/auxiliary/mapbiomas_mining_legend.xlsx"
MINING_EXTRACT_PATH = "data/land_cover/processed/extract/land_cover_mining.parquet"
MINING_ADM2_UPSTREAM_PATH = "data/land_cover/processed/aggregate/land_cover_mining_adm2_upstream.parquet"
MINING_NATIONAL_AREA_PATH = "data/land_cover/processed/extract/mining_substances_area_by_class_year.csv"
MINING_FILENAME_TEMPLATE = "mapbiomas_mining_substances_c11_{year}.tif"
MINING_BAND_TEMPLATE = "classification_{year}"

NO_MINING_CLASS = 0
GARIMPO = 1
GOLD = 15
UNKNOWN_SUBSTANCES = (58, 59)  # atypical, no substance

# Class id -> name. Garimpo with an unknown substance is kept apart from both
# gold and other ores: around Itaituba it is ~10% of garimpo pixels.
MINING_GROUPS = {
    1: "garimpo_gold",
    2: "garimpo_unknown",
    3: "garimpo_other",
    4: "industrial_gold",
    5: "industrial_other",
}
# Subclass id -> name; subclass columns are `land_cover_class_<group><phase>`.
MINING_PHASES = {1: "active", 2: "revegetation"}

TOTAL_CLASS_STEM = -1  # `schema.land_cover_feature_stem(LAND_COVER_TOTAL_COLUMN)`


def decode(code):
    """Split an ABCC code into (kind, phase, substance)."""
    code = int(code)
    return code // 1000, code // 100 % 10, code % 100


def mining_group(code):
    """Return the `MINING_GROUPS` id of an ABCC code."""
    kind, _, substance = decode(code)
    if kind == GARIMPO:
        if substance == GOLD:
            return 1
        return 2 if substance in UNKNOWN_SUBSTANCES else 3
    return 4 if substance == GOLD else 5


def build_mining_legend(codes):
    """Return the `ID`/`Class`/`Subclass` legend read by `preprocess.create_mappers`."""
    rows = [(NO_MINING_CLASS, NO_MINING_CLASS, None)]
    rows += [(int(code), mining_group(code), decode(code)[1]) for code in sorted(codes)]
    return pd.DataFrame(rows, columns=["ID", "Class", "Subclass"])


def mining_column_names():
    """Map bucketed `land_cover_class` ids to `mining_*` column names."""
    names = {NO_MINING_CLASS: "mining_none"}
    for group, group_name in MINING_GROUPS.items():
        names[group] = f"mining_{group_name}"
        for phase, phase_name in MINING_PHASES.items():
            names[group * 10 + phase] = f"mining_{group_name}_{phase_name}"
    return names


def compute_mining_composition(
    bucket_df,
    *,
    entity_columns,
    bucket_map=None,
    kernel=INV_SQRT_DISTANCE_KERNEL,
    bandwidth=None,
):
    """Collapse a bucketed mining table into kernel-weighted area shares per entity.

    Same ring weights as `composition.compute_kernel_weighted_composition`: the
    kernel at each ring's midpoint, renormalized over the rings an entity has
    with any area. Returns one `mining_*` share column per group and per
    group-phase (`mining_column_names`).
    """
    entity_columns = list(entity_columns)
    bucket_map = LAND_COVER_COMPOSITION_BUCKET_MAP if bucket_map is None else bucket_map
    raw_weights = dict(
        zip(
            bucket_map,
            bucket_kernel_weights(
                [midpoint for _label, midpoint in bucket_map.values()],
                kernel=kernel,
                bandwidth=bandwidth,
            ),
        )
    )
    keys = [*entity_columns, "bucket"]
    names = mining_column_names()
    frame = bucket_df[bucket_df["bucket"].isin(list(bucket_map)) & bucket_df["share"].notna()]
    frame = frame[frame["land_cover_class"] != TOTAL_CLASS_STEM]
    shares = frame.pivot(index=keys, columns="land_cover_class", values="share")
    shares = shares.reindex(columns=list(names)).fillna(0.0).rename(columns=names)
    weight = pd.Series(
        shares.index.get_level_values("bucket").map(raw_weights).to_numpy(dtype=float),
        index=shares.index,
    )
    weighted = shares.mul(weight, axis=0).groupby(level=entity_columns).sum()
    composition = weighted.div(weight.groupby(level=entity_columns).sum(), axis=0)
    return composition.reset_index()


def pixel_area_km2(top_latitude, resolution, n_rows):
    """Area (km^2) of one pixel in each of `n_rows` rows of a lat/lon grid (sphere).

    Uses the authalic radius; within 0.1% of the ellipsoidal area.
    """
    radius_km = 6371.0072
    edges = np.radians(top_latitude - resolution * np.arange(n_rows + 1))
    return radius_km**2 * np.radians(resolution) * (np.sin(edges[:-1]) - np.sin(edges[1:]))
