from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.data.sources.climate import assembly


def test_annual_aggregate_sql_nulls_min_max_for_partial_year_coverage():
    # 2021 is not a leap year (365 days).
    full_year = pd.date_range("2021-01-01", "2021-12-31", freq="D")

    # trench 1: every day of the year present with a non-null value -> a
    # real annual MIN.
    # trench 2: full day coverage, but one day's value is NULL (e.g. that day
    # was sourced from era5_land_daily, which never writes the extras) -> the
    # annual MIN must come back NULL rather than silently reflecting only the
    # ARCO-backed days.
    # trench 3: every *present* row is non-null, but 5 calendar days are
    # missing from the group entirely (partial preprocessing run, trench
    # added mid-pipeline, store gap) -> must also come back NULL. Before this
    # fix, `COUNT(identifier) = COUNT(*)` was trivially true here (no nulls
    # among the rows that exist) and would have silently returned a MIN
    # computed over only 360 of the year's 365 days.
    frame = pd.concat(
        [
            pd.DataFrame({"trench_id": 1, "date": full_year, "2t_daily_min": 10.0}),
            pd.DataFrame(
                {
                    "trench_id": 2,
                    "date": full_year,
                    "2t_daily_min": [None] + [9.0] * (len(full_year) - 1),
                }
            ),
            pd.DataFrame(
                {"trench_id": 3, "date": full_year[5:], "2t_daily_min": 8.0}
            ),
        ],
        ignore_index=True,
    )
    connection = duckdb.connect(database=":memory:")
    connection.register("c", frame)
    sql = assembly._annual_aggregate_sql(["2t_daily_min"], source_alias="c")
    result = connection.execute(
        f"SELECT c.trench_id, EXTRACT(YEAR FROM c.date)::BIGINT AS year, {sql} "
        "FROM c GROUP BY 1, 2 ORDER BY 1"
    ).fetchdf()

    assert result.loc[result["trench_id"] == 1, "2t_daily_min"].iloc[0] == 10.0
    assert pd.isna(result.loc[result["trench_id"] == 2, "2t_daily_min"].iloc[0])
    assert pd.isna(result.loc[result["trench_id"] == 3, "2t_daily_min"].iloc[0])


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


def test_partitioned_trench_day_paths_warns_on_full_directory_fallback(tmp_path, caplog):
    climate_path = tmp_path / "climate"
    climate_path.mkdir()

    with caplog.at_level("WARNING", logger="src.data.sources.climate.assembly"):
        paths = assembly._partitioned_trench_day_paths(
            climate_path,
            start_date=pd.Timestamp("2021-01-01"),
            end_date=pd.Timestamp("2021-01-31"),
        )

    assert paths == [climate_path]
    assert any("falling back to a full directory scan" in message for message in caplog.messages)


def test_partitioned_trench_day_paths_uses_existing_partitions_without_warning(tmp_path, caplog):
    climate_path = tmp_path / "climate"
    partition_dir = climate_path / "year=2021" / "month=01"
    partition_dir.mkdir(parents=True)

    with caplog.at_level("WARNING", logger="src.data.sources.climate.assembly"):
        paths = assembly._partitioned_trench_day_paths(
            climate_path,
            start_date=pd.Timestamp("2021-01-01"),
            end_date=pd.Timestamp("2021-01-31"),
        )

    assert paths == [partition_dir]
    assert caplog.messages == []
