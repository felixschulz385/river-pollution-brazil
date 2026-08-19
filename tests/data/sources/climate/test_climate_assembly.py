from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.data.sources.climate import assembly


class _FakeRiverNetwork:
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
            {
                "trench_id": [101, 102],
                "adm2": ["1001A", "1002B"],
            }
        )
        self.drainage_areas = pd.DataFrame({"trench_id": [101, 102]})
        self.trench_reachability_matrices = {
            1: csr_matrix(np.asarray([[1, 1], [0, 1]], dtype=np.int8))
        }
        self.trench_distance_matrices = {
            1: csr_matrix(np.asarray([[0.0, 20.0], [0.0, 0.0]], dtype=float))
        }

    def load(self, path: str) -> None:
        self.loaded_path = path


def test_assemble_adm2_upstream_bins_by_distance_like_land_cover(
    tmp_path: Path, monkeypatch
):
    climate = pd.DataFrame(
        {
            "trench_id": [101, 101, 102, 102],
            "date": pd.to_datetime(["2020-01-01", "2020-06-01", "2020-01-01", "2020-06-01"]),
            "2t": [10.0, 20.0, 30.0, 40.0],
        }
    )
    climate_path = tmp_path / "climate.parquet"
    climate.to_parquet(climate_path, index=False)
    output_path = tmp_path / "climate_adm2.parquet"

    monkeypatch.setattr(assembly, "RiverNetwork", _FakeRiverNetwork)

    result = assembly._assemble_adm2_upstream_duckdb(
        climate_path=climate_path,
        climate_columns=["2t"],
        river_network_path=str(tmp_path / "river_network"),
        output_path=output_path,
        n_jobs=1,
    )
    result_df = pd.read_parquet(output_path)

    assert list(result_df.columns) == [
        "adm2_id",
        "year",
        "distance_bucket",
        "climate_variable",
        "mean_value",
        "reachable_trench_count",
        "bucket_intersects_adm2",
    ]

    # mun "1001A" seeded from trench 101: itself (bucket -25, intersects) and
    # upstream trench 102 (bucket 0, does not intersect) -- same reachability
    # fixture as land_cover's aggregation test.
    seed = result_df[
        (result_df["adm2_id"] == "1001A")
        & (result_df["distance_bucket"] == -25)
        & (result_df["year"] == 2020)
    ].iloc[0]
    assert seed["mean_value"] == 15.0  # avg(10, 20) at trench 101
    assert bool(seed["bucket_intersects_adm2"]) is True

    upstream = result_df[
        (result_df["adm2_id"] == "1001A")
        & (result_df["distance_bucket"] == 0)
        & (result_df["year"] == 2020)
    ].iloc[0]
    assert upstream["mean_value"] == 35.0  # avg(30, 40) at trench 102
    assert bool(upstream["bucket_intersects_adm2"]) is False
