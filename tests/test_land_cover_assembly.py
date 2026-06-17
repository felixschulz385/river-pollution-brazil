from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from code.data.land_cover import assembly


class _FakeRiverNetwork:
    def __init__(self):
        self.trenches = pd.DataFrame(
            {
                "trench_id": [101, 102],
                "system_id": [1, 1],
                "trench_index": [0, 1],
            }
        )
        self.trench_adm2_table = pd.DataFrame(
            {
                "trench_id": [101, 102],
                "adm2": ["A", "B"],
            }
        )
        self.trench_reachability_matrices = {
            1: csr_matrix(np.asarray([[1, 1], [0, 1]], dtype=np.int8))
        }
        self.trench_distance_matrices = {
            1: csr_matrix(np.asarray([[0.0, 20.0], [0.0, 0.0]], dtype=float))
        }

    def load(self, path: str) -> None:
        self.loaded_path = path


def test_load_trench_adm2_matches_prefers_persisted_relation_table():
    network = _FakeRiverNetwork()
    network.trenches = network.trenches.copy()
    network.trenches["adm2"] = [None, None]

    matches = assembly._load_trench_adm2_matches(network)

    assert matches.to_dict("records") == [
        {"trench_id": 101, "adm2": "A"},
        {"trench_id": 102, "adm2": "B"},
    ]


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
    assert list(result.index.names) == ["adm2_id", "year"]

    adm2_a = result.loc[("A", 2020)]
    assert adm2_a["lc_0_10km_tot"] == 10.0
    assert adm2_a["lc_10_50km_tot"] == 4.0
    assert adm2_a["lc_0_10km_c41_cnt"] == 6.0
    assert adm2_a["lc_10_50km_c41_cnt"] == 2.0
    assert adm2_a["lc_0_10km_c41_shr"] == 0.6
    assert adm2_a["lc_10_50km_c41_shr"] == 0.5
    assert adm2_a["lc_0_10km_n"] == 1
    assert adm2_a["lc_10_50km_n"] == 1

    adm2_b = result.loc[("B", 2020)]
    assert adm2_b["lc_0_10km_tot"] == 4.0
    assert adm2_b["lc_10_50km_tot"] == 0.0
    assert adm2_b["lc_0_10km_c41_cnt"] == 2.0
    assert adm2_b["lc_0_10km_c41_shr"] == 0.5
