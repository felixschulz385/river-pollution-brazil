from __future__ import annotations

import pandas as pd
import pytest

from src.data.assembly.build import (
    _compute_kernel_weighted_bucket_values,
    _pivot_long_source,
    assemble_dataset,
    write_dataset,
)
from src.data.assembly.constants import (
    ADM2_MODE,
    CLIMATE_BUCKETED_SOURCE_TYPE,
    LAND_COVER_BUCKETED_SOURCE_TYPE,
    LONG_PIVOT_SOURCE_TYPE,
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
    # output, long over (adm2_id, year, bucket, climate_variable).
    climate = pd.DataFrame(
        {
            "adm2_id": [1234567, 1234567],
            "year": [2019, 2019],
            "bucket": [0, 25],
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
