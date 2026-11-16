"""Tests for the shared geography: the Legal Amazon, upstream municipalities, borders and rings."""

from __future__ import annotations

from pathlib import Path
import sys

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.core import geography


def test_legal_amazon_takes_full_states_and_maranhao_west_of_44(tmp_path):
    path = tmp_path / "gadm.gpkg"
    gpd.GeoDataFrame(
        {"CC_2": ["1500107", "2100055", "2100105", "2200053"],
         "geometry": [box(-50, -5, -49, -4), box(-44.5, -5, -43.5, -4), box(-43.9, -5, -43, -4),
                      box(-45, -5, -44.5, -4)]},
        crs=4326,
    ).to_file(path, layer=geography.GADM_LAYER, driver="GPKG")

    assert geography.legal_amazon(path) == {"150010", "210005"}


def test_upstream_municipalities_follow_pfafstetter_within_the_parent(tmp_path):
    pd.DataFrame({"trench_id": [1, 2, 3, 4, 5], "otto_basin_l4": ["4583", "4587", "4582", "4589", "4687"]}).to_parquet(
        tmp_path / "river_network_otto_basins.parquet")
    pd.DataFrame({"trench_id": [1, 2, 3, 4, 5],
                  "adm2": ["1500011", "1500021", "1500031", "1500011", "1500041"]}).to_parquet(
        tmp_path / "river_network_trench_adm2_matches.parquet")
    a, b, c = "150001", "150002", "150003"

    upstream = geography.upstream_municipalities([a, b, c], river_network_path=tmp_path)

    assert upstream[a] == {b}  # 4587 drains into 4583; 4582 enters below it; 4687 is another parent
    assert upstream[b] == {a}  # a also holds 4589, upstream of 4587
    assert upstream[c] == set()  # nothing drains into a tributary


def test_borders_find_touching_municipalities():
    boundaries = gpd.GeoDataFrame({"mun_id": ["a", "b", "c"],
                                   "geometry": [box(-56, -6, -55, -5), box(-55, -6, -54, -5), box(-53, -6, -52, -5)]},
                                  crs=4326)
    adjacency = geography.borders(boundaries)
    assert adjacency["a"] == {"b"} and adjacency["b"] == {"a"} and "c" not in adjacency


def test_rings_leave_out_the_municipality_and_its_borders():
    boundaries = gpd.GeoDataFrame({"mun_id": list("abcd")},
                                  geometry=[box(-55 + i, -6, -54 + i, -5) for i in (0, 1, 2, 6)], crs=4326)
    adjacency = {"a": {"b"}, "b": {"a", "c"}, "c": {"b"}}
    ring = geography.rings(boundaries, adjacency, radius_km=250)  # neighbouring centroids are about 110 km apart
    assert ring["a"] == {"c"} and ring["b"] == set() and ring["d"] == set()
