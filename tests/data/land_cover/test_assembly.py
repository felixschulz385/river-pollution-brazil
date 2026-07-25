from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.data.land_cover import assembly


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


def test_assemble_land_cover_adm2_uses_bucketed_upstream_output(
    tmp_path: Path,
    monkeypatch,
):
    land_cover = pd.DataFrame(
        {
            "trench_id": [101, 102],
            "year": [2020, 2020],
            "land_cover_total": [10.0, 4.0],
            "land_cover_class_41": [6.0, 2.0],
        }
    )
    land_cover_path = tmp_path / "land_cover.feather"
    land_cover.to_feather(land_cover_path)
    output_path = tmp_path / "adm2.parquet"

    monkeypatch.setattr(assembly.rn_module, "RiverNetwork", _FakeRiverNetwork)

    result = assembly.assemble_land_cover(
        object(),
        variant="adm2",
        land_cover_path=str(land_cover_path),
        river_network_path=str(tmp_path / "river_network"),
        output_path=str(output_path),
        n_jobs=1,
    )

    assert output_path.exists()
    assert list(result.columns) == [
        "mun_id",
        "year",
        "bucket",
        "land_cover_class",
        "n",
        "cnt",
        "share",
    ]

    # adm2 ids are truncated by one trailing character (a check-digit-style
    # suffix) into the "mun_id" grouping key -- "1001A"/"1002B" -> "1001"/"1002".
    def _row(mun_id, bucket, land_cover_class):
        match = result[
            (result["mun_id"] == mun_id)
            & (result["bucket"] == bucket)
            & (result["land_cover_class"] == land_cover_class)
        ]
        assert len(match) == 1, f"expected exactly one row for {mun_id}/{bucket}/{land_cover_class}"
        return match.iloc[0]

    # mun_id "1001" is seeded from trench 101, which reaches both 101 (itself,
    # bucket -25) and 102 (upstream, bucket 0) per the fake reachability matrix.
    seed = _row("1001", -25, -1)  # land_cover_class -1 == the "land_cover_total" column
    assert seed["cnt"] == 10.0
    assert seed["share"] == 1.0
    seed_c41 = _row("1001", -25, 41)
    assert seed_c41["cnt"] == 6.0
    assert seed_c41["share"] == 0.6

    upstream = _row("1001", 0, -1)
    assert upstream["cnt"] == 4.0
    assert upstream["share"] == 1.0
    upstream_c41 = _row("1001", 0, 41)
    assert upstream_c41["cnt"] == 2.0
    assert upstream_c41["share"] == 0.5

    # mun_id "1002" is seeded from trench 102, which only reaches itself.
    other_seed = _row("1002", -25, -1)
    assert other_seed["cnt"] == 4.0
    assert other_seed["share"] == 1.0
    other_seed_c41 = _row("1002", -25, 41)
    assert other_seed_c41["cnt"] == 2.0
    assert other_seed_c41["share"] == 0.5
