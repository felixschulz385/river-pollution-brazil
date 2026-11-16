"""Municipality geography shared by the designs: the Legal Amazon, boundaries, neighbours and upstream
municipalities."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np

from src.analysis.core.panels import PROJECT_ROOT
from src.data.sources.biomes.constants import BRAZIL_PROJECTED_CRS
from src.data.sources.biomes.preprocess import _load_adm2_boundaries
from src.data.sources.geology.upstream import municipality_otto_basins, pfafstetter_upstream
from src.data.sources.land_cover.constants import derive_mun_id_from_adm2_id

GADM_PATH = PROJECT_ROOT / "data" / "gadm" / "raw" / "gadm41_BRA.gpkg"
GADM_LAYER, GADM_ID_COLUMN = "ADM_ADM_2", "CC_2"
RIVER_NETWORK_PATH = PROJECT_ROOT / "data" / "river_network" / "processed"

# The Legal Amazon: eight states in full and Maranhão west of 44°W (IBGE counts
# a municipality any part of which lies west of the meridian: 181 of 217).
LEGAL_AMAZON_STATES = ("11", "12", "13", "14", "15", "16", "17", "51")
MARANHAO = "21"
LEGAL_AMAZON_MERIDIAN = -44.0
RORAIMA = "14"  # the Yanomami garimpo rush from 2019

BORDER_BUFFER_M = 1_000  # simplified boundaries: municipalities within 1 km of each other border


def legal_amazon(path=GADM_PATH):
    """`mun_id`s of the Legal Amazon: the eight full states and Maranhão west of 44°W."""
    gadm = gpd.read_file(path, layer=GADM_LAYER, columns=[GADM_ID_COLUMN])
    mun_id = gadm[GADM_ID_COLUMN].astype(str).str[:6]
    state = mun_id.str[:2]
    west = gadm.geometry.bounds["minx"] < LEGAL_AMAZON_MERIDIAN
    return set(mun_id[state.isin(LEGAL_AMAZON_STATES) | (state.eq(MARANHAO) & west)])


def municipality_boundaries(root_dir=PROJECT_ROOT):
    """The simplified GADM municipality polygons with `adm2_id` and `mun_id`."""
    boundaries = _load_adm2_boundaries(Path(root_dir))
    boundaries["mun_id"] = boundaries["adm2_id"].map(derive_mun_id_from_adm2_id)
    return boundaries


def borders(boundaries, buffer_m=BORDER_BUFFER_M):
    """Municipality -> the set of municipalities bordering it (within `buffer_m`)."""
    projected = boundaries[["mun_id", "geometry"]].to_crs(BRAZIL_PROJECTED_CRS)
    buffered = projected.assign(geometry=projected.buffer(buffer_m))
    pairs = gpd.sjoin(buffered, projected, predicate="intersects")
    pairs = pairs[pairs["mun_id_left"] != pairs["mun_id_right"]]
    return pairs.groupby("mun_id_left")["mun_id_right"].agg(set).to_dict()


def rings(boundaries, adjacency, radius_km):
    """Municipality -> the municipalities whose centroid is within `radius_km`, other than itself and its borders."""
    centroids = boundaries.to_crs(BRAZIL_PROJECTED_CRS).set_index("mun_id").centroid
    xy = np.column_stack([centroids.x, centroids.y])
    distance = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1]) / 1000
    ids = centroids.index.to_numpy()
    return {mun: set(ids[distance[i] <= radius_km]) - {mun} - adjacency.get(mun, set()) for i, mun in enumerate(ids)}


def upstream_municipalities(targets, river_network_path=RIVER_NETWORK_PATH):
    """For each target `mun_id`, the other municipalities intersecting an Otto-4 sub-basin upstream of its own."""
    basins = municipality_otto_basins(river_network_path)
    by_basin = {}
    for mun_id, codes in basins.items():
        for code in codes:
            by_basin.setdefault(code, set()).add(mun_id)
    by_parent = {}
    for code in by_basin:
        by_parent.setdefault(code[:-1], []).append(code)
    upstream = {}
    for target in targets:
        members = set()
        for own in basins.get(target, ()):
            for code in by_parent[own[:-1]]:
                if pfafstetter_upstream(code, own):
                    members |= by_basin[code]
        upstream[target] = members - {target}
    return upstream
