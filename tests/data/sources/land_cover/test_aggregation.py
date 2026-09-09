from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.data.sources.land_cover import aggregation


class _FakeRiverNetwork:
    """Two trenches in one system: 101 (ADM2-touching) -> 102 (20 km upstream)."""

    def __init__(self):
        self.trenches = pd.DataFrame(
            {
                "trench_id": [101, 102],
                "system_id": [1, 1],
                "trench_index": [0, 1],
                "distance": [1.0, 1.0],
            }
        )
        self.trench_adm2_table = pd.DataFrame(
            {"trench_id": [101, 102], "adm2": ["100011", "200022"]}
        )
        self.drainage_areas = pd.DataFrame({"trench_id": [101, 102]})
        self.trench_reachability_matrices = {
            1: csr_matrix(np.asarray([[1, 1], [0, 1]], dtype=np.int8))
        }
        self.trench_distance_matrices = {
            1: csr_matrix(np.asarray([[0.0, 20.0], [0.0, 0.0]], dtype=float))
        }

    def load(self, path):  # pragma: no cover - trivial
        self.loaded_path = path


def test_aggregate_along_rivers_sums_class_counts_and_shares_per_bucket(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(aggregation.rn_module, "RiverNetwork", _FakeRiverNetwork)

    land_cover = pd.DataFrame(
        {
            "trench_id": [101, 102],
            "year": [2020, 2020],
            "land_cover_class_3": [10.0, 4.0],
            "land_cover_class_15": [30.0, 6.0],
            "land_cover_total": [40.0, 10.0],
        }
    )
    land_cover_path = tmp_path / "land_cover.parquet"
    land_cover.to_parquet(land_cover_path, index=False)
    output_path = tmp_path / "adm2_upstream.parquet"

    result = aggregation.aggregate_along_rivers(
        object(),
        land_cover_path=land_cover_path,
        river_network_path=str(tmp_path / "river_network"),
        n_jobs=1,
        output_path=output_path,
    )

    saved = pd.read_parquet(output_path)
    pd.testing.assert_frame_equal(result, saved)
    assert list(saved.columns) == [
        "mun_id",
        "year",
        "bucket",
        "land_cover_class",
        "n",
        "cnt",
        "share",
        "bucket_intersects_adm2",
    ]

    # mun "10001" (adm2 "100011", trailing check digit dropped) seeded at trench
    # 101: the seed trench sits in bucket -25 and intersects the ADM2 polygon;
    # trench 102, 20 km upstream, lands in bucket 0.
    seed = saved[(saved["mun_id"] == "10001") & (saved["bucket"] == -25)].set_index(
        "land_cover_class"
    )
    assert seed.loc[3, "cnt"] == 10.0
    assert seed.loc[3, "share"] == 10.0 / 40.0
    assert seed.loc[15, "cnt"] == 30.0
    assert seed.loc[-1, "cnt"] == 40.0  # land_cover_total -> class stem -1
    assert (seed["n"] == 1).all()
    assert bool(seed.loc[3, "bucket_intersects_adm2"]) is True

    upstream = saved[(saved["mun_id"] == "10001") & (saved["bucket"] == 0)].set_index(
        "land_cover_class"
    )
    assert upstream.loc[3, "cnt"] == 4.0
    assert upstream.loc[3, "share"] == 4.0 / 10.0
    assert bool(upstream.loc[15, "bucket_intersects_adm2"]) is False


def test_aggregate_along_rivers_returns_empty_frame_when_no_trenches_match(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(aggregation.rn_module, "RiverNetwork", _FakeRiverNetwork)

    # Land cover only for a trench id that isn't in the network -> no overlap.
    land_cover = pd.DataFrame(
        {
            "trench_id": [999],
            "year": [2020],
            "land_cover_class_3": [1.0],
            "land_cover_total": [1.0],
        }
    )
    land_cover_path = tmp_path / "land_cover.parquet"
    land_cover.to_parquet(land_cover_path, index=False)

    result = aggregation.aggregate_along_rivers(
        object(),
        land_cover_path=land_cover_path,
        river_network_path=str(tmp_path / "river_network"),
        n_jobs=1,
        output_path=tmp_path / "adm2_upstream.parquet",
    )
    assert result.empty
