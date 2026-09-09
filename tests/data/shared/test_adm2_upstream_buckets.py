from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from src.data.sources import river_network as rn_module
from src.data.shared.adm2_upstream_buckets import build_adm2_upstream_bucket_parts


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


def _read_parts(paths):
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def test_membership_parts_bin_every_reachable_trench(tmp_path):
    parts = build_adm2_upstream_bucket_parts(
        network=_FakeRiverNetwork(),
        rn_module=rn_module,
        parts_dir=tmp_path / "parts",
        n_jobs=1,
    )
    assert parts, "expected at least one part file"

    frame = (
        _read_parts(parts).sort_values(["adm2_id", "trench_id"]).reset_index(drop=True)
    )
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


def test_streams_one_part_per_chunk_without_losing_units(tmp_path):
    parts = build_adm2_upstream_bucket_parts(
        network=_FakeRiverNetwork(),
        rn_module=rn_module,
        parts_dir=tmp_path / "parts",
        n_jobs=2,
        target_part_count=3,
    )
    # 3 ADM2 units, target 3 parts -> one unit per streamed part file.
    assert len(parts) == 3
    assert all(Path(path).name.startswith("part-") for path in parts)

    frame = _read_parts(parts)
    assert sorted(frame["adm2_id"].unique()) == ["10001", "20002", "30003"]


def test_reduce_adm2_receives_membership_and_its_output_is_written(tmp_path):
    seen_columns = {}

    def reduce_adm2(adm2_id, membership):
        seen_columns[adm2_id] = list(membership.columns)
        return pd.DataFrame({"adm2_id": [adm2_id], "n_trenches": [len(membership)]})

    parts = build_adm2_upstream_bucket_parts(
        network=_FakeRiverNetwork(),
        rn_module=rn_module,
        parts_dir=tmp_path / "parts",
        n_jobs=1,
        reduce_adm2=reduce_adm2,
    )

    frame = _read_parts(parts).set_index("adm2_id")
    assert set(seen_columns) == {"10001", "20002", "30003"}
    assert "bucket_intersects_adm2" in seen_columns["10001"]
    assert frame.loc["10001", "n_trenches"] == 3
    assert frame.loc["20002", "n_trenches"] == 2
    assert frame.loc["30003", "n_trenches"] == 1


def test_reduce_adm2_returning_none_skips_the_unit(tmp_path):
    def reduce_adm2(adm2_id, membership):
        if adm2_id == "10001":
            return None
        return pd.DataFrame({"adm2_id": [adm2_id]})

    parts = build_adm2_upstream_bucket_parts(
        network=_FakeRiverNetwork(),
        rn_module=rn_module,
        parts_dir=tmp_path / "parts",
        n_jobs=1,
        reduce_adm2=reduce_adm2,
    )
    frame = _read_parts(parts)
    assert sorted(frame["adm2_id"].unique()) == ["20002", "30003"]


def test_failing_unit_is_skipped_not_fatal(tmp_path, caplog):
    def reduce_adm2(adm2_id, membership):
        if adm2_id == "20002":
            raise RuntimeError("boom")
        return pd.DataFrame({"adm2_id": [adm2_id]})

    with caplog.at_level("WARNING"):
        parts = build_adm2_upstream_bucket_parts(
            network=_FakeRiverNetwork(),
            rn_module=rn_module,
            parts_dir=tmp_path / "parts",
            n_jobs=1,
            target_part_count=1,  # all three units share one chunk
            reduce_adm2=reduce_adm2,
        )

    # The raising unit is logged and dropped; its chunk-mates still make it out.
    assert len(parts) == 1
    frame = _read_parts(parts)
    assert sorted(frame["adm2_id"].unique()) == ["10001", "30003"]
    assert any("20002" in record.getMessage() for record in caplog.records)


def test_missing_drainage_polygons_raises(tmp_path):
    network = _FakeRiverNetwork()
    network.drainage_areas = None
    with pytest.raises(ValueError, match="drainage polygon"):
        build_adm2_upstream_bucket_parts(
            network=network,
            rn_module=rn_module,
            parts_dir=tmp_path / "parts",
            n_jobs=1,
        )
