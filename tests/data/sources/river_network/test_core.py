from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point

from src.data.sources.river_network import RiverNetwork


def _fixture_network() -> RiverNetwork:
    """Build a tiny 3-trench linear chain: node 1 -> node 2 -> node 3."""
    network = RiverNetwork()
    network.trenches = gpd.GeoDataFrame(
        {
            "trench_id": [101, 102],
            "upstream_node": [1, 2],
            "downstream_node": [2, 3],
            "distance": [5.0, 3.0],
            "estuary_distance": [3.0, 0.0],
            "geometry": [
                LineString([(0, 1), (0, 0)]),
                LineString([(0, 2), (0, 1)]),
            ],
        },
        crs=4326,
    )
    network.drainage_areas = gpd.GeoDataFrame(
        {
            "trench_id": [101, 102],
            "geometry": [Point(0, 0.5), Point(0, 1.5)],
        },
        crs=4326,
    )
    return network


def test_compute_distance_matrices_and_get_upstream_trenches():
    network = _fixture_network()

    network.compute_subsystems()
    network.compute_distance_matrices()

    # Trench 102 (node 2 -> node 3, downstream-most, closest to the estuary)
    # should see trench 101 (node 1 -> node 2) as upstream. The reported
    # distance is the difference in each trench's own `estuary_distance`
    # (3.0 vs 0.0), not the summed `distance` (length) column.
    upstream_of_102 = network.get_upstream_trenches(102)
    assert set(upstream_of_102["trench_id"]) == {101, 102}
    assert upstream_of_102.set_index("trench_id").loc[101, "upstream_distance"] == 3.0
    assert upstream_of_102.set_index("trench_id").loc[102, "upstream_distance"] == 0.0

    # Trench 101 (node 1 -> node 2, upstream-most) has no upstream trenches
    # besides itself -- nothing flows into node 1.
    upstream_of_101 = network.get_upstream_trenches(101)
    assert set(upstream_of_101["trench_id"]) == {101}


def test_save_load_round_trip_preserves_trenches_and_matrices(tmp_path: Path):
    network = _fixture_network()
    network.compute_subsystems()
    network.compute_distance_matrices()

    output_dir = tmp_path / "river_network"
    network.save(str(output_dir))

    reloaded = RiverNetwork()
    reloaded.load(str(output_dir))

    pd.testing.assert_frame_equal(
        pd.DataFrame(network.trenches.drop(columns="geometry")).sort_values("trench_id").reset_index(drop=True),
        pd.DataFrame(reloaded.trenches.drop(columns="geometry")).sort_values("trench_id").reset_index(drop=True),
    )
    assert reloaded.trench_reachability_matrices.keys() == network.trench_reachability_matrices.keys()
    for system_id, matrix in network.trench_reachability_matrices.items():
        assert (reloaded.trench_reachability_matrices[system_id] != matrix).nnz == 0
    for system_id, matrix in network.trench_distance_matrices.items():
        assert (reloaded.trench_distance_matrices[system_id] != matrix).nnz == 0

    reloaded_upstream = reloaded.get_upstream_trenches(102)
    assert set(reloaded_upstream["trench_id"]) == {101, 102}


def test_build_trench_adm2_table_and_dominant_system_table(tmp_path: Path):
    """`generate()` now chains build_trench_adm2_table() into
    build_adm2_dominant_system_table() -- the dominant system per ADM2 must
    be the one with the greatest *summed* intersecting trench distance, not
    just whichever trench/system happens to join first."""
    network = RiverNetwork()
    base = Point(-47.9, -15.8)  # Brasília
    network.trenches = gpd.GeoDataFrame(
        {
            "trench_id": [201, 202, 203],
            "distance": [10.0, 3.0, 3.0],
            "system_id": ["A", "B", "B"],
            "geometry": [
                LineString([(base.x, base.y), (base.x + 0.01, base.y)]),
                LineString([(base.x, base.y), (base.x, base.y + 0.01)]),
                LineString([(base.x, base.y), (base.x - 0.01, base.y)]),
            ],
        },
        crs=4326,
    )

    gadm_path = tmp_path / "gadm.gpkg"
    adm2_boundary = gpd.GeoDataFrame({"CC_2": ["X"], "geometry": [base.buffer(1.0)]}, crs=4326)
    adm2_boundary.to_file(gadm_path, layer="ADM_ADM_2", driver="GPKG")

    trench_adm2 = network.build_trench_adm2_table(gadm_path=str(gadm_path), layer="ADM_ADM_2")
    assert set(trench_adm2["trench_id"]) == {201, 202, 203}

    dominant = network.build_adm2_dominant_system_table()

    # System A's single trench (distance 10.0) outweighs system B's two
    # trenches summed (3.0 + 3.0 = 6.0), so A must win despite B having more
    # individual trenches in the ADM2.
    assert dominant.set_index("adm2").loc["X", "system_id"] == "A"


def test_save_load_round_trip_preserves_adm2_dominant_system_table(tmp_path: Path):
    network = _fixture_network()
    network.compute_subsystems()
    network.compute_distance_matrices()
    network.trench_adm2_table = pd.DataFrame({"trench_id": [101, 102], "adm2": ["X", "X"]})
    network.build_adm2_dominant_system_table()

    output_dir = tmp_path / "river_network"
    network.save(str(output_dir))
    assert (output_dir / "river_network_adm2_dominant_systems.parquet").exists()

    reloaded = RiverNetwork()
    reloaded.load(str(output_dir))

    pd.testing.assert_frame_equal(
        network.get_adm2_dominant_system_table(), reloaded.get_adm2_dominant_system_table()
    )


def test_annotate_drainage_areas_with_country_membership_reprojects_mismatched_crs(tmp_path: Path):
    # `drainage_areas` lives in a projected CRS (as it would after
    # `build_trench_adm2_table`'s BRAZIL_PROJECTED_CRS reprojection), while
    # the GADM boundary file is in WGS84 -- the two must be reconciled before
    # `.intersects()`, or the raw-coordinate comparison silently returns
    # (mostly) False for everything.
    network = RiverNetwork()
    point_in_brazil_wgs84 = Point(-47.9, -15.8)  # Brasília
    projected = gpd.GeoDataFrame(
        {"trench_id": [101], "geometry": [point_in_brazil_wgs84]}, crs=4326
    ).to_crs(31983)
    network.drainage_areas = projected

    gadm_path = tmp_path / "gadm.gpkg"
    brazil_boundary = gpd.GeoDataFrame(
        {"geometry": [Point(-47.9, -15.8).buffer(5.0)]}, crs=4326
    )
    brazil_boundary.to_file(gadm_path, layer="ADM_ADM_0", driver="GPKG")

    network.annotate_drainage_areas_with_country_membership(str(gadm_path), layer="ADM_ADM_0")

    assert bool(network.drainage_areas["within_brazil"].iloc[0]) is True


def test_annotate_drainage_areas_with_country_membership_handles_missing_crs(tmp_path: Path):
    # `drainage_areas.crs` can be `None` (e.g. dropped by an earlier geometry
    # operation). `.to_crs()` raises on CRS-less geometries, so the mismatch
    # guard must not route a `None` CRS into a reprojection attempt.
    network = RiverNetwork()
    point_in_brazil_wgs84 = Point(-47.9, -15.8)  # Brasília
    no_crs = gpd.GeoDataFrame({"trench_id": [101], "geometry": [point_in_brazil_wgs84]}, crs=None)
    network.drainage_areas = no_crs

    gadm_path = tmp_path / "gadm.gpkg"
    brazil_boundary = gpd.GeoDataFrame(
        {"geometry": [Point(-47.9, -15.8).buffer(5.0)]}, crs=4326
    )
    brazil_boundary.to_file(gadm_path, layer="ADM_ADM_0", driver="GPKG")

    network.annotate_drainage_areas_with_country_membership(str(gadm_path), layer="ADM_ADM_0")

    assert bool(network.drainage_areas["within_brazil"].iloc[0]) is True
