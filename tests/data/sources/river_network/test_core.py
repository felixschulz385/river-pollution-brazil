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
