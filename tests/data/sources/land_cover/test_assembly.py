from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.data.sources.land_cover import assembly


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


def test_reset_water_quality_index_brings_back_indexed_columns():
    """sensor_data's assembled panel indexes on [station_code, date] (via
    `.set_index(...)` in `assemble_sensor_data`) rather than keeping them as
    plain columns; land_cover's target-building requires them as columns."""
    indexed = pd.DataFrame(
        {"ph": [7.0, 7.2]},
        index=pd.MultiIndex.from_tuples(
            [("11111111", pd.Timestamp("2020-01-01")), ("22222222", pd.Timestamp("2020-01-02"))],
            names=[assembly.STATION_CODE_COLUMN, assembly.DATE_COLUMN],
        ),
    )

    result = assembly._reset_water_quality_index(indexed)

    assert assembly.STATION_CODE_COLUMN in result.columns
    assert assembly.DATE_COLUMN in result.columns
    assert sorted(result[assembly.STATION_CODE_COLUMN].tolist()) == ["11111111", "22222222"]


def test_reset_water_quality_index_is_a_noop_for_plain_columns():
    plain = pd.DataFrame(
        {
            assembly.STATION_CODE_COLUMN: ["11111111"],
            assembly.DATE_COLUMN: [pd.Timestamp("2020-01-01")],
            "ph": [7.0],
        }
    )

    result = assembly._reset_water_quality_index(plain)

    assert list(result.columns) == list(plain.columns)
    assert isinstance(result.index, pd.RangeIndex)


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
    land_cover_path = tmp_path / "land_cover.parquet"
    land_cover.to_parquet(land_cover_path, index=False)
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
        "bucket_intersects_adm2",
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
    assert bool(seed["bucket_intersects_adm2"]) is True
    seed_c41 = _row("1001", -25, 41)
    assert seed_c41["cnt"] == 6.0
    assert seed_c41["share"] == 0.6

    # bucket 0 for mun_id "1001" only contains trench 102, reached purely
    # upstream -- 102 is not itself a trench matched to adm2 "1001A".
    upstream = _row("1001", 0, -1)
    assert upstream["cnt"] == 4.0
    assert upstream["share"] == 1.0
    assert bool(upstream["bucket_intersects_adm2"]) is False
    upstream_c41 = _row("1001", 0, 41)
    assert upstream_c41["cnt"] == 2.0
    assert upstream_c41["share"] == 0.5

    # mun_id "1002" is seeded from trench 102, which only reaches itself.
    other_seed = _row("1002", -25, -1)
    assert other_seed["cnt"] == 4.0
    assert other_seed["share"] == 1.0
    assert bool(other_seed["bucket_intersects_adm2"]) is True
    other_seed_c41 = _row("1002", -25, 41)
    assert other_seed_c41["cnt"] == 2.0
    assert other_seed_c41["share"] == 0.5
