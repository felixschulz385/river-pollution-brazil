from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.assembly.build import (
    _aggregate_grouped_sum_source,
    _compute_kernel_weighted_bucket_values,
    _pivot_long_source,
    assemble_dataset,
    write_dataset,
)
from src.data.assembly.constants import (
    ADM2_MODE,
    CLIMATE_BUCKETED_SOURCE_TYPE,
    GROUPED_SUM_SOURCE_TYPE,
    LAND_COVER_BUCKETED_SOURCE_TYPE,
    LOCAL_SCOPE,
    LONG_PIVOT_SOURCE_TYPE,
    MINING_BUCKETED_SOURCE_TYPE,
    SENSOR_MODE,
)
from src.data.assembly.schema import AssemblyDataset, AssemblySource


def _bucket_rows(entity_key, entity_value, year, bucket):
    return [
        {entity_key: entity_value, "year": year, "bucket": bucket, "land_cover_class": 1, "share": 0.6},
        {entity_key: entity_value, "year": year, "bucket": bucket, "land_cover_class": 5, "share": 0.4},
    ]


def test_assemble_dataset_sensor_mode_joins_annual_land_cover_onto_daily_rows(tmp_path):
    water_quality = pd.DataFrame(
        {
            "station_code": ["S1", "S1"],
            "datetime": pd.to_datetime(["2020-03-01", "2020-06-01"]),
            "ph": [7.1, 7.3],
        }
    )
    water_quality_path = tmp_path / "water_quality.parquet"
    water_quality.to_parquet(water_quality_path, index=False)

    land_cover = pd.DataFrame(_bucket_rows("station_code", "S1", 2020, 0))
    land_cover_path = tmp_path / "land_cover.parquet"
    land_cover.to_parquet(land_cover_path, index=False)

    dataset_config = AssemblyDataset(
        id="sensor_panel",
        mode=SENSOR_MODE,
        index=("station_code", "datetime"),
        output_path=str(tmp_path / "out.parquet"),
        sources=(
            AssemblySource(
                name="water_quality",
                path="water_quality.parquet",
                join_keys=("station_code", "datetime"),
                variables=("ph",),
            ),
            AssemblySource(
                name="land_cover",
                path="land_cover.parquet",
                join_keys=("station_code", "year"),
                variables=("lc_forest", "lc_water"),
                type=LAND_COVER_BUCKETED_SOURCE_TYPE,
            ),
        ),
    )

    result = assemble_dataset(dataset_config, root_dir=tmp_path)

    assert list(result["station_code"]) == ["S1", "S1"]
    assert result["ph"].tolist() == [7.1, 7.3]
    # Single bucket with only forest/water present -> composition equals the raw shares.
    assert result["lc_forest"].tolist() == pytest.approx([0.6, 0.6])
    assert result["lc_water"].tolist() == pytest.approx([0.4, 0.4])


def test_assemble_dataset_sensor_mode_joins_pivoted_climate_at_nearest_bucket(tmp_path):
    water_quality = pd.DataFrame(
        {
            "station_code": ["S1", "S1"],
            "datetime": pd.to_datetime(["2020-03-01", "2020-06-01"]),
            "ph": [7.1, 7.3],
        }
    )
    water_quality_path = tmp_path / "water_quality.parquet"
    water_quality.to_parquet(water_quality_path, index=False)

    climate = pd.DataFrame(
        {
            "station_code": ["S1", "S1", "S1", "S1"],
            "date": pd.to_datetime(["2020-03-01", "2020-03-01", "2020-06-01", "2020-06-01"]),
            "distance_bucket": [0, 25, 0, 25],
            "climate_variable": ["2t", "2t", "2t", "2t"],
            "mean_day": [20.0, 99.0, 22.0, 99.0],
        }
    )
    climate_path = tmp_path / "climate.parquet"
    climate.to_parquet(climate_path, index=False)

    dataset_config = AssemblyDataset(
        id="sensor_panel",
        mode=SENSOR_MODE,
        index=("station_code", "datetime"),
        output_path=str(tmp_path / "out.parquet"),
        sources=(
            AssemblySource(
                name="water_quality",
                path="water_quality.parquet",
                join_keys=("station_code", "datetime"),
                variables=("ph",),
            ),
            AssemblySource(
                name="climate",
                path="climate.parquet",
                join_keys=("station_code", "date"),
                variables=("2t_mean_day",),
                type=LONG_PIVOT_SOURCE_TYPE,
                filter={"distance_bucket": 0},
                pivot_column="climate_variable",
                value_columns=("mean_day",),
            ),
        ),
    )

    result = assemble_dataset(dataset_config, root_dir=tmp_path)

    assert result["2t_mean_day"].tolist() == [20.0, 22.0]


def test_assemble_dataset_adm2_mode_derives_mun_id_from_adm2_id(tmp_path):
    land_cover = pd.DataFrame(_bucket_rows("mun_id", "123456", 2019, 0))
    land_cover_path = tmp_path / "land_cover_adm2.parquet"
    land_cover.to_parquet(land_cover_path, index=False)

    climate = pd.DataFrame(
        {
            "adm2_id": [1234567],
            "year": [2019],
            "total_weight": [42.0],
        }
    )
    climate_path = tmp_path / "climate_adm2.parquet"
    climate.to_parquet(climate_path, index=False)

    dataset_config = AssemblyDataset(
        id="adm2_panel",
        mode=ADM2_MODE,
        index=("mun_id", "year"),
        output_path=str(tmp_path / "out.parquet"),
        sources=(
            AssemblySource(
                name="land_cover_adm2",
                path="land_cover_adm2.parquet",
                join_keys=("mun_id", "year"),
                variables=("lc_forest", "lc_water"),
                type=LAND_COVER_BUCKETED_SOURCE_TYPE,
            ),
            AssemblySource(
                name="climate_adm2",
                path="climate_adm2.parquet",
                join_keys=("adm2_id", "year"),
                variables=("total_weight",),
                id_map={"adm2_id": "mun_id"},
            ),
        ),
    )

    result = assemble_dataset(dataset_config, root_dir=tmp_path)

    assert result["mun_id"].tolist() == ["123456"]
    assert result["total_weight"].tolist() == [42.0]


def test_assemble_dataset_adm2_mode_weights_climate_buckets_at_assembly_time(tmp_path):
    land_cover = pd.DataFrame(_bucket_rows("mun_id", "123456", 2019, 0))
    land_cover_path = tmp_path / "land_cover_adm2.parquet"
    land_cover.to_parquet(land_cover_path, index=False)

    # Two buckets for the same ADM2/year/variable -- climate's binned ADM2
    # output, long over (adm2_id, year, distance_bucket, climate_variable).
    climate = pd.DataFrame(
        {
            "adm2_id": [1234567, 1234567],
            "year": [2019, 2019],
            "distance_bucket": [0, 25],
            "climate_variable": ["2t", "2t"],
            "mean_value": [20.0, 30.0],
        }
    )
    climate_path = tmp_path / "climate_adm2.parquet"
    climate.to_parquet(climate_path, index=False)

    dataset_config = AssemblyDataset(
        id="adm2_panel",
        mode=ADM2_MODE,
        index=("mun_id", "year"),
        output_path=str(tmp_path / "out.parquet"),
        sources=(
            AssemblySource(
                name="land_cover_adm2",
                path="land_cover_adm2.parquet",
                join_keys=("mun_id", "year"),
                variables=("lc_forest", "lc_water"),
                type=LAND_COVER_BUCKETED_SOURCE_TYPE,
            ),
            AssemblySource(
                name="climate_adm2",
                path="climate_adm2.parquet",
                join_keys=("adm2_id", "year"),
                variables=("2t",),
                type=CLIMATE_BUCKETED_SOURCE_TYPE,
                id_map={"adm2_id": "mun_id"},
                kernel="uniform",
                bandwidth=1000.0,
            ),
        ),
    )

    result = assemble_dataset(dataset_config, root_dir=tmp_path)

    assert result["mun_id"].tolist() == ["123456"]
    # Both buckets fall within the 1000km uniform-kernel bandwidth -> equal
    # weight -> simple average of the two bucket means.
    assert result["2t"].tolist() == pytest.approx([25.0])


def test_compute_kernel_weighted_bucket_values_ignores_unmapped_bucket_without_zeroing_group():
    # entity A has one row in an unmapped bucket (999, e.g. a bucket label
    # missing from `bucket_map`) alongside a mapped one; entity B has only
    # mapped buckets, as a control. The unmapped row must simply be excluded
    # from A's weighting -- not poison A's weight_sum via NaN and force the
    # whole (entity, category) group's output to 0.
    long_df = pd.DataFrame(
        {
            "entity": ["A", "A", "B"],
            "category": ["x", "x", "x"],
            "bucket": [0, 999, 0],
            "value": [10.0, 999.0, 10.0],
        }
    )
    bucket_map = {0: ("0_25km", 12.5), 25: ("25_50km", 37.5)}

    result = _compute_kernel_weighted_bucket_values(
        long_df,
        entity_columns=["entity"],
        category_column="category",
        value_column="value",
        kernel="uniform",
        bandwidth=1000.0,
        bucket_map=bucket_map,
    )

    row_a = result.loc[result["entity"] == "A", "x"].iloc[0]
    row_b = result.loc[result["entity"] == "B", "x"].iloc[0]
    assert row_a == pytest.approx(10.0)
    assert row_a == pytest.approx(row_b)


def test_compute_kernel_weighted_bucket_values_returns_nan_when_all_buckets_null():
    # Entity A has no climate coverage in any bucket for variable "x" (e.g. no
    # upstream trench had data for this variable) -- weight_sum is 0 for that
    # group. The aggregated value must be NaN ("unknown"), not 0.0, which
    # would be indistinguishable from a genuine zero measurement downstream.
    # Entity B, with real data in both buckets, is a control confirming the
    # normal weighted-average path still works.
    long_df = pd.DataFrame(
        {
            "entity": ["A", "A", "B", "B"],
            "category": ["x", "x", "x", "x"],
            "bucket": [0, 25, 0, 25],
            "value": [np.nan, np.nan, 10.0, 30.0],
        }
    )
    bucket_map = {0: ("0_25km", 12.5), 25: ("25_50km", 37.5)}

    result = _compute_kernel_weighted_bucket_values(
        long_df,
        entity_columns=["entity"],
        category_column="category",
        value_column="value",
        kernel="uniform",
        bandwidth=1000.0,
        bucket_map=bucket_map,
    )

    row_a = result.loc[result["entity"] == "A", "x"].iloc[0]
    row_b = result.loc[result["entity"] == "B", "x"].iloc[0]
    assert np.isnan(row_a)
    # Both buckets fall within the uniform kernel's bandwidth -> equal weight
    # -> simple average of the two bucket means.
    assert row_b == pytest.approx(20.0)


def test_assemble_dataset_adm2_mode_sums_grouped_population_strata(tmp_path):
    # population.parquet is long over sex/age_group -- multiple rows per
    # (mun_id, year) -- which would otherwise fail assembly's
    # `validate="many_to_one"` merge once a second source is joined in.
    land_cover = pd.DataFrame(_bucket_rows("mun_id", "123456", 2019, 0))
    land_cover_path = tmp_path / "land_cover_adm2.parquet"
    land_cover.to_parquet(land_cover_path, index=False)

    population = pd.DataFrame(
        {
            "mun_id": ["123456", "123456", "123456", "123456"],
            "year": [2019, 2019, 2019, 2019],
            "sex": ["female", "female", "male", "male"],
            "age_group": ["00_04", "05_09", "00_04", "05_09"],
            "population": [10.0, 20.0, 15.0, 25.0],
        }
    )
    population_path = tmp_path / "population.parquet"
    population.to_parquet(population_path, index=False)

    dataset_config = AssemblyDataset(
        id="adm2_panel",
        mode=ADM2_MODE,
        index=("mun_id", "year"),
        output_path=str(tmp_path / "out.parquet"),
        sources=(
            AssemblySource(
                name="land_cover_adm2",
                path="land_cover_adm2.parquet",
                join_keys=("mun_id", "year"),
                variables=("lc_forest", "lc_water"),
                type=LAND_COVER_BUCKETED_SOURCE_TYPE,
            ),
            AssemblySource(
                name="population",
                path="population.parquet",
                join_keys=("mun_id", "year"),
                variables=("population",),
                type=GROUPED_SUM_SOURCE_TYPE,
            ),
        ),
    )

    result = assemble_dataset(dataset_config, root_dir=tmp_path)

    assert result["mun_id"].tolist() == ["123456"]
    assert result["population"].tolist() == pytest.approx([70.0])


def test_aggregate_grouped_sum_source_keeps_all_null_group_as_null():
    # A mun_id/year whose strata are all missing must sum to NaN ("unknown"),
    # not a fabricated 0.0 that would be indistinguishable from a genuine
    # zero population downstream.
    frame = pd.DataFrame(
        {
            "mun_id": ["123456", "123456", "654321"],
            "year": [2019, 2019, 2019],
            "population": [np.nan, np.nan, 5.0],
        }
    )
    source = AssemblySource(
        name="population",
        path="population.parquet",
        join_keys=("mun_id", "year"),
        variables=("population",),
        type=GROUPED_SUM_SOURCE_TYPE,
    )

    result = _aggregate_grouped_sum_source(frame, source)

    row = result.set_index("mun_id")["population"]
    assert np.isnan(row.loc["123456"])
    assert row.loc["654321"] == pytest.approx(5.0)


def test_write_dataset_writes_parquet(tmp_path):
    df = pd.DataFrame({"a": [1, 2]})
    output_path = tmp_path / "nested" / "out.parquet"

    write_dataset(df, output_path)

    assert output_path.exists()
    assert pd.read_parquet(output_path).equals(df)


def test_pivot_long_source_raises_clear_error_on_duplicate_join_key_pivot_combination():
    frame = pd.DataFrame(
        {
            "station_code": ["S1", "S1"],
            "date": pd.to_datetime(["2020-03-01", "2020-03-01"]),
            "climate_variable": ["2t", "2t"],
            "mean_day": [20.0, 21.0],
        }
    )
    source = AssemblySource(
        name="climate",
        path="climate.parquet",
        join_keys=("station_code", "date"),
        variables=("2t_mean_day",),
        type=LONG_PIVOT_SOURCE_TYPE,
        pivot_column="climate_variable",
        value_columns=("mean_day",),
    )

    with pytest.raises(ValueError, match="climate.*duplicate rows"):
        _pivot_long_source(frame, source)


def test_pivot_long_source_does_not_blame_duplicates_for_unrelated_pivot_error(
    monkeypatch: pytest.MonkeyPatch,
):
    # A `frame.pivot` failure with no actual duplicate (join_keys,
    # pivot_column) rows present must not be misreported as a duplicate-key
    # error -- that would mask whatever the real cause was.
    frame = pd.DataFrame(
        {
            "station_code": ["S1", "S2"],
            "date": pd.to_datetime(["2020-03-01", "2020-03-01"]),
            "climate_variable": ["2t", "2t"],
            "mean_day": [20.0, 21.0],
        }
    )
    source = AssemblySource(
        name="climate",
        path="climate.parquet",
        join_keys=("station_code", "date"),
        variables=("2t_mean_day",),
        type=LONG_PIVOT_SOURCE_TYPE,
        pivot_column="climate_variable",
        value_columns=("mean_day",),
    )

    def _raise_unrelated_value_error(self, *args, **kwargs):
        raise ValueError("some unrelated pivot failure")

    monkeypatch.setattr(pd.DataFrame, "pivot", _raise_unrelated_value_error)

    with pytest.raises(ValueError, match="climate.*failed to pivot.*unrelated pivot failure") as exc_info:
        _pivot_long_source(frame, source)
    assert "duplicate" not in str(exc_info.value)


def test_assemble_dataset_adm2_mode_pivots_filtered_long_health_source_with_template(tmp_path):
    population = pd.DataFrame({"mun_id": ["123456"], "year": [2019], "population": [1000.0]})
    population.to_parquet(tmp_path / "population.parquet", index=False)

    # SIH-like table: long over metric x ICD chapter, keyed by an already
    # 6-digit `municipality_code` that must be renamed, not id-mapped.
    hospitalizations = pd.DataFrame(
        {
            "municipality_code": ["123456"] * 4,
            "year": [2019] * 4,
            "icd10_chapter_code": ["01", "11", "01", "11"],
            "metric_name": ["hospitalizations_count"] * 2 + ["in_hospital_deaths_count"] * 2,
            "metric_value": [10.0, 20.0, 1.0, 2.0],
        }
    )
    hospitalizations.to_parquet(tmp_path / "hosp.parquet", index=False)

    dataset_config = AssemblyDataset(
        id="adm2_panel",
        mode=ADM2_MODE,
        index=("mun_id", "year"),
        output_path=str(tmp_path / "out.parquet"),
        sources=(
            AssemblySource(
                name="population",
                path="population.parquet",
                join_keys=("mun_id", "year"),
                variables=("population",),
            ),
            AssemblySource(
                name="hospitalizations_icd10_chapter",
                path="hosp.parquet",
                join_keys=("municipality_code", "year"),
                variables=("hosp_icd_01", "hosp_icd_11"),
                type=LONG_PIVOT_SOURCE_TYPE,
                rename={"municipality_code": "mun_id"},
                filter={"metric_name": "hospitalizations_count"},
                pivot_column="icd10_chapter_code",
                value_columns=("metric_value",),
                column_template="hosp_icd_{pivot_value}",
            ),
        ),
    )

    result = assemble_dataset(dataset_config, root_dir=tmp_path)

    assert result["mun_id"].tolist() == ["123456"]
    assert result["hosp_icd_01"].tolist() == [10.0]
    assert result["hosp_icd_11"].tolist() == [20.0]


def test_assemble_dataset_adm2_mode_local_scope_averages_adm2_touching_buckets(tmp_path):
    # Municipality 1 has a local ring (-25, flagged) and an upstream ring (0);
    # municipality 2 is a headwater municipality with local rings only.
    rows = []
    for mun_id, bucket, forest, flagged in [
        ("111111", -25, 0.8, True),
        ("111111", 0, 0.2, False),
        ("222222", -50, 0.4, True),
        ("222222", -25, 0.6, True),
    ]:
        rows += [
            {"mun_id": mun_id, "year": 2019, "bucket": bucket, "land_cover_class": 1, "share": forest,
             "bucket_intersects_adm2": flagged},
            {"mun_id": mun_id, "year": 2019, "bucket": bucket, "land_cover_class": 30, "share": 1 - forest,
             "bucket_intersects_adm2": flagged},
        ]
    pd.DataFrame(rows).to_parquet(tmp_path / "land_cover_adm2.parquet", index=False)
    climate = pd.DataFrame(
        {
            "adm2_id": ["1111117", "1111117", "2222227", "2222227"],
            "year": [2019] * 4,
            "distance_bucket": [-25, 0, -50, -25],
            "climate_variable": ["2t"] * 4,
            "mean_value": [20.0, 30.0, 10.0, 14.0],
            "bucket_intersects_adm2": [True, False, True, True],
        }
    )
    climate.to_parquet(tmp_path / "climate_adm2.parquet", index=False)

    land_cover_source = dict(
        path="land_cover_adm2.parquet",
        join_keys=("mun_id", "year"),
        type=LAND_COVER_BUCKETED_SOURCE_TYPE,
    )
    dataset_config = AssemblyDataset(
        id="adm2_panel",
        mode=ADM2_MODE,
        index=("mun_id", "year"),
        output_path=str(tmp_path / "out.parquet"),
        sources=(
            AssemblySource(name="lc_local", variables=("local_lc_forest",), scope=LOCAL_SCOPE, **land_cover_source),
            AssemblySource(name="lc_upstream", variables=("lc_forest",), **land_cover_source),
            AssemblySource(
                name="climate_local",
                path="climate_adm2.parquet",
                join_keys=("adm2_id", "year"),
                variables=("local_2t",),
                type=CLIMATE_BUCKETED_SOURCE_TYPE,
                id_map={"adm2_id": "mun_id"},
                scope=LOCAL_SCOPE,
            ),
        ),
    )

    result = assemble_dataset(dataset_config, root_dir=tmp_path).set_index("mun_id")

    # The headwater municipality stays in the panel, with no upstream exposure.
    assert result.index.tolist() == ["111111", "222222"]
    assert result["local_lc_forest"].tolist() == pytest.approx([0.8, 0.5], abs=1e-3)
    assert result.loc["111111", "lc_forest"] == pytest.approx(0.2, abs=1e-3)
    assert np.isnan(result.loc["222222", "lc_forest"])
    assert result["local_2t"].tolist() == pytest.approx([20.0, 12.0])


def _mining_bucket_rows(mun_id, bucket, gold, flagged):
    """One bucket of the C11 aggregate: 100 pixels, `gold` of them garimpo gold (all active)."""
    counts = {-1: 100.0, 0: 100.0 - gold, 1: gold, 11: gold}
    return [
        {"mun_id": mun_id, "year": 2020, "bucket": bucket, "land_cover_class": class_id,
         "cnt": counts.get(class_id, 0.0), "share": counts.get(class_id, 0.0) / 100.0,
         "bucket_intersects_adm2": flagged}
        for class_id in (-1, 0, 1, 2, 3, 4, 5, 11, 12, 21, 22, 31, 32, 41, 42, 51, 52)
    ]


def test_assemble_dataset_adm2_mode_mining_shares_by_reach_and_local_scope(tmp_path):
    rows = (
        _mining_bucket_rows("111111", -25, 10.0, True)
        + _mining_bucket_rows("111111", 0, 2.0, False)
        + _mining_bucket_rows("111111", 100, 50.0, False)
    )
    pd.DataFrame(rows).to_parquet(tmp_path / "mining.parquet", index=False)
    mining = dict(path="mining.parquet", join_keys=("mun_id", "year"), type=MINING_BUCKETED_SOURCE_TYPE)
    dataset_config = AssemblyDataset(
        id="adm2_panel",
        mode=ADM2_MODE,
        index=("mun_id", "year"),
        output_path=str(tmp_path / "out.parquet"),
        sources=(
            AssemblySource(name="local", variables=("local_mining_garimpo_gold",), scope=LOCAL_SCOPE, **mining),
            AssemblySource(
                name="upstream_50",
                variables=("gold_50",),
                max_km=50,
                rename={"mining_garimpo_gold": "gold_50"},
                **mining,
            ),
            AssemblySource(
                name="upstream_200",
                variables=("gold_200", "gold_active_200"),
                max_km=200,
                rename={"mining_garimpo_gold": "gold_200", "mining_garimpo_gold_active": "gold_active_200"},
                **mining,
            ),
        ),
    )

    row = assemble_dataset(dataset_config, root_dir=tmp_path).iloc[0]

    assert row["local_mining_garimpo_gold"] == pytest.approx(0.10)
    # Only the 0-25 km ring is within 50 km.
    assert row["gold_50"] == pytest.approx(0.02)
    # Rings 0 and 100 at the inv-sqrt kernel's midpoints 12.5 and 112.5 km.
    w0, w100 = 12.5**-0.5, 112.5**-0.5
    assert row["gold_200"] == pytest.approx((0.02 * w0 + 0.50 * w100) / (w0 + w100))
    assert row["gold_active_200"] == pytest.approx(row["gold_200"])


def test_assemble_dataset_fills_missing_counts_within_the_fill_years(tmp_path):
    pd.DataFrame({"mun_id": ["111111"] * 3 + ["222222"] * 3, "year": [2003, 2004, 2005] * 2,
                  "population": [1.0] * 6}).to_parquet(tmp_path / "spine.parquet", index=False)
    pd.DataFrame({"mun_id": ["111111"], "year": [2005], "cases": [7]}).astype({"mun_id": "string"}).to_parquet(
        tmp_path / "cases.parquet", index=False
    )
    dataset_config = AssemblyDataset(
        id="adm2_panel",
        mode=ADM2_MODE,
        index=("mun_id", "year"),
        output_path=str(tmp_path / "out.parquet"),
        sources=(
            AssemblySource(name="spine", path="spine.parquet", join_keys=("mun_id", "year"), variables=("population",)),
            AssemblySource(
                name="cases",
                path="cases.parquet",
                join_keys=("mun_id", "year"),
                variables=("cases",),
                fill_value=0,
                fill_years=(2004, 2005),
            ),
        ),
    )

    result = assemble_dataset(dataset_config, root_dir=tmp_path).set_index(["mun_id", "year"])["cases"]

    assert np.isnan(result[("111111", 2003)]) and np.isnan(result[("222222", 2003)])
    assert result[("111111", 2004)] == 0 and result[("111111", 2005)] == 7
    assert result[("222222", 2004)] == 0 and result[("222222", 2005)] == 0
