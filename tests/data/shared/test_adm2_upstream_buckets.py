from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from src.data.sources import river_network as rn_module
from src.data.shared.adm2_upstream_buckets import build_adm2_upstream_bucket_table


class _FakeRiverNetwork:
    """Three trenches in one linear system: 101 -> 102 -> 103 going upstream."""

    def __init__(self):
        self.trenches = pd.DataFrame(
            {
                "trench_id": [101, 102, 103],
                "system_id": [1, 1, 1],
                "trench_index": [0, 1, 2],
                "distance": [1.0, 1.0, 1.0],
            }
        )
        self.trench_adm2_table = pd.DataFrame(
            {"trench_id": [101, 102, 103], "adm2": ["10001", "20002", "30003"]}
        )
        self.drainage_areas = pd.DataFrame({"trench_id": [101, 102, 103]})
        self.trench_reachability_matrices = {
            1: csr_matrix(np.asarray([[1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=np.int8))
        }
        self.trench_distance_matrices = {
            1: csr_matrix(
                np.asarray(
                    [[0.0, 20.0, 60.0], [0.0, 0.0, 40.0], [0.0, 0.0, 0.0]], dtype=float
                )
            )
        }

    def load(self, path):  # pragma: no cover - trivial
        self.loaded_path = path


def test_membership_table_bins_every_reachable_trench():
    frame = build_adm2_upstream_bucket_table(
        network=_FakeRiverNetwork(),
        rn_module=rn_module,
        n_jobs=1,
    )
    assert not frame.empty
    frame = frame.sort_values(["adm2_id", "trench_id"]).reset_index(drop=True)
    assert list(frame.columns) == [
        "adm2_id",
        "trench_id",
        "distance_bucket",
        "bucket_intersects_adm2",
        "upstream_distance",
        "adjusted_distance",
    ]

    # adm2 "10001" seeded at trench 101: reaches 101 (0 km), 102 (20 km),
    # 103 (60 km); the shifted origin subtracts the 1 km seed-trench length
    # before bucketing into 25 km bins.
    a1 = frame[frame["adm2_id"] == "10001"].set_index("trench_id")
    assert sorted(a1.index) == [101, 102, 103]
    assert a1.loc[101, "distance_bucket"] == -25
    assert a1.loc[102, "distance_bucket"] == 0
    assert a1.loc[103, "distance_bucket"] == 50
    assert bool(a1.loc[101, "bucket_intersects_adm2"]) is True
    assert bool(a1.loc[102, "bucket_intersects_adm2"]) is False
    assert bool(a1.loc[103, "bucket_intersects_adm2"]) is False

    a2 = frame[frame["adm2_id"] == "20002"].set_index("trench_id")
    assert sorted(a2.index) == [102, 103]
    assert a2.loc[102, "distance_bucket"] == -25
    assert a2.loc[103, "distance_bucket"] == 25


def test_resolves_every_unit_across_work_chunks(caplog):
    with caplog.at_level("INFO"):
        frame = build_adm2_upstream_bucket_table(
            network=_FakeRiverNetwork(),
            rn_module=rn_module,
            n_jobs=2,
            chunk_count=3,
        )
    # 3 ADM2 units, 3 work chunks -> one unit per chunk, all still present.
    assert sorted(frame["adm2_id"].unique()) == ["10001", "20002", "30003"]

    # One INFO line per work chunk that produced rows.
    chunk_log_lines = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Resolved ADM2 upstream bucket chunk ")
    ]
    assert len(chunk_log_lines) == 3


def test_reduce_adm2_receives_membership_and_its_output_is_used():
    seen_columns = {}

    def reduce_adm2(adm2_id, membership):
        seen_columns[adm2_id] = list(membership.columns)
        return pd.DataFrame({"adm2_id": [adm2_id], "n_trenches": [len(membership)]})

    frame = build_adm2_upstream_bucket_table(
        network=_FakeRiverNetwork(),
        rn_module=rn_module,
        n_jobs=1,
        reduce_adm2=reduce_adm2,
    ).set_index("adm2_id")

    assert set(seen_columns) == {"10001", "20002", "30003"}
    assert "bucket_intersects_adm2" in seen_columns["10001"]
    assert frame.loc["10001", "n_trenches"] == 3
    assert frame.loc["20002", "n_trenches"] == 2
    assert frame.loc["30003", "n_trenches"] == 1


def test_reduce_adm2_returning_none_skips_the_unit():
    def reduce_adm2(adm2_id, membership):
        if adm2_id == "10001":
            return None
        return pd.DataFrame({"adm2_id": [adm2_id]})

    frame = build_adm2_upstream_bucket_table(
        network=_FakeRiverNetwork(),
        rn_module=rn_module,
        n_jobs=1,
        reduce_adm2=reduce_adm2,
    )
    assert sorted(frame["adm2_id"].unique()) == ["20002", "30003"]


def test_failing_unit_is_skipped_not_fatal(caplog):
    def reduce_adm2(adm2_id, membership):
        if adm2_id == "20002":
            raise RuntimeError("boom")
        return pd.DataFrame({"adm2_id": [adm2_id]})

    with caplog.at_level("WARNING"):
        frame = build_adm2_upstream_bucket_table(
            network=_FakeRiverNetwork(),
            rn_module=rn_module,
            n_jobs=1,
            chunk_count=1,  # all three units share one chunk
            reduce_adm2=reduce_adm2,
        )

    # The raising unit is logged and dropped; its chunk-mates still make it out.
    assert sorted(frame["adm2_id"].unique()) == ["10001", "30003"]
    assert any("20002" in record.getMessage() for record in caplog.records)


def test_missing_drainage_polygons_raises():
    network = _FakeRiverNetwork()
    network.drainage_areas = None
    with pytest.raises(ValueError, match="drainage polygon"):
        build_adm2_upstream_bucket_table(
            network=network,
            rn_module=rn_module,
            n_jobs=1,
        )
