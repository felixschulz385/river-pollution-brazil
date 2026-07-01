from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyfixest as pf
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.modules.pop("code", None)

import code

from code.analysis.cli import main as analysis_main
from code.analysis.sensor_data import faceted_distance_coefplot
from code.analysis.sensor_data.catalog import build_pollutant_catalog
from code.analysis.sensor_data.groups import select_pollutants
from code.analysis.sensor_data.prepare import build_analysis_data
from code.analysis.sensor_data.residualize import residualize_with_map
from code.analysis.sensor_data.runner import run_suite
from code.analysis.sensor_data.specs import build_model_specs
from code.analysis.settings import (
    ClimateVariable,
    ControlVariable,
    DEFAULT_SETTINGS,
    ImportanceTier,
    LassoSettings,
    SensorAnalysisSettings,
)


def _synthetic_settings(tmp_path: Path) -> SensorAnalysisSettings:
    dates = pd.to_datetime(
        [
            "2020-01-15",
            "2020-04-15",
            "2020-07-15",
            "2020-10-15",
            "2021-01-15",
            "2021-04-15",
            "2021-07-15",
            "2021-10-15",
        ]
    )
    bucket_order = DEFAULT_SETTINGS.distance_buckets
    subclasses = DEFAULT_SETTINGS.land_cover_subclasses
    sensor_to_trench = {"s1": 1, "s2": 1, "s3": 2, "s4": 2}
    sensor_effect = {"s1": 0.3, "s2": -0.1, "s3": 0.15, "s4": -0.05}

    land_cover_rows = []
    trench_year_payload = {
        (1, 2020): {"c41": [0.6, 0.2], "c42": [0.2, 0.1]},
        (1, 2021): {"c41": [1.1, 0.4], "c42": [0.3, 0.2]},
        (2, 2020): {"c41": [1.6, 0.5], "c42": [0.4, 0.2]},
        (2, 2021): {"c41": [2.2, 0.7], "c42": [0.5, 0.3]},
    }
    for (trench_id, year), values in trench_year_payload.items():
        row = {"trench_id": trench_id, "year": year}
        for bucket in bucket_order:
            row[f"lc_{bucket}_tot"] = 10.0
            row[f"lc_{bucket}_n"] = 1
        for subclass in subclasses:
            counts = values.get(subclass, [0.0, 0.0])
            padded_counts = [*counts, *([0.0] * (len(bucket_order) - len(counts)))]
            for bucket, count in zip(bucket_order, padded_counts, strict=True):
                row[f"lc_{bucket}_{subclass}_cnt"] = count
                row[f"lc_{bucket}_{subclass}_shr"] = count / 10.0
        land_cover_rows.append(row)
    land_cover_path = tmp_path / "land_cover.parquet"
    pd.DataFrame(land_cover_rows).to_parquet(land_cover_path, index=False)

    trenches_path = tmp_path / "trenches.parquet"
    pd.DataFrame({"trench_id": [1, 2], "system_id": [10, 20]}).to_parquet(
        trenches_path,
        index=False,
    )

    climate_rows = []
    sensor_rows = []
    for sensor_index, (sensor_id, trench_id) in enumerate(sensor_to_trench.items(), start=1):
        for date_index, date in enumerate(dates, start=1):
            year = int(date.year)
            quarter = int(date.quarter)
            c41_0_10, c41_10_50 = trench_year_payload[(trench_id, year)]["c41"]
            c42_0_10, _ = trench_year_payload[(trench_id, year)]["c42"]
            temperature = 18.0 + sensor_index + quarter
            precipitation = 90.0 + 3.0 * date_index + 5.0 * trench_id
            flow_day = 150.0 + 10.0 * sensor_index + 4.0 * quarter
            flow_7d = 130.0 + 8.0 * sensor_index + 3.0 * quarter
            interaction_signal = c41_0_10 * temperature * 0.12
            ph = (
                6.5
                + 0.35 * c41_0_10
                + 0.18 * c41_10_50
                + 0.08 * c42_0_10
                + 0.015 * flow_day
                + interaction_signal
                + sensor_effect[sensor_id]
                + 0.1 * year
            )
            turbidity = (
                12.0
                + 1.4 * c41_0_10
                + 0.5 * precipitation
                + 0.10 * c41_0_10 * precipitation
                + 0.01 * flow_7d
            )
            total_nitrogen = (
                1.2
                + 0.2 * c42_0_10
                + 0.03 * flow_day
                + 0.02 * temperature
            )

            climate_rows.append(
                {
                    "sensor_id": sensor_id,
                    "date": pd.Timestamp(date),
                    "upstream_temperature": temperature,
                    "upstream_precipitation": precipitation,
                }
            )
            sensor_rows.append(
                {
                    "sensor_id": sensor_id,
                    "date": pd.Timestamp(date),
                    "trench_id": trench_id,
                    "ph": ph,
                    "turbidity": turbidity,
                    "total_nitrogen": total_nitrogen,
                    "streamflow_discharge_day": flow_day,
                    "streamflow_discharge_mean_7d": flow_7d,
                    "streamflow_discharge_mean_31d": flow_7d + 10.0,
                }
            )

    sensor_data_path = tmp_path / "sensor.parquet"
    pd.DataFrame(sensor_rows).set_index(["sensor_id", "date"]).to_parquet(sensor_data_path)

    climate_data_path = tmp_path / "climate.parquet"
    pd.DataFrame(climate_rows).to_parquet(climate_data_path, index=False)

    transformations = {
        "schema_version": 1,
        "clean_data_file": "sensor.parquet",
        "recommendations": {
            "ph": {
                "column": "ph",
                "recommended_transform": "identity",
                "expression": "x",
                "apply_to": "analysis",
            },
            "turbidity": {
                "column": "turbidity",
                "recommended_transform": "log10_1p",
                "expression": "log10(1 + x)",
                "apply_to": "analysis",
            },
            "total_nitrogen": {
                "column": "total_nitrogen",
                "recommended_transform": "log10_1p",
                "expression": "log10(1 + x)",
                "apply_to": "analysis",
            },
        },
    }
    transformations_path = tmp_path / "transformations.json"
    transformations_path.write_text(json.dumps(transformations), encoding="utf-8")

    return SensorAnalysisSettings(
        project_root=tmp_path,
        sensor_data_path=sensor_data_path,
        land_cover_path=land_cover_path,
        climate_data_path=climate_data_path,
        transformations_path=transformations_path,
        trenches_path=trenches_path,
        output_dir=tmp_path / "output",
        sensor_id_column="sensor_id",
        date_column="date",
        climate_join_keys=("sensor_id", "date"),
        distance_buckets=bucket_order,
        land_cover_subclasses=subclasses,
        land_cover_statistic="cnt",
        fixed_effects=("sensor_id", "year"),
        cluster_variable="sensor_id",
        vcov_type="CRV1",
        minimum_observations=1,
        map_tolerance=1e-10,
        map_max_iterations=5_000,
        importance_tiers=(
            ImportanceTier("high", 10),
            ImportanceTier("medium", 5),
            ImportanceTier("low", 1),
        ),
        controls=(
            ControlVariable(
                "streamflow_discharge_day",
                "streamflow_discharge_day_scaled",
                100.0,
            ),
            ControlVariable(
                "streamflow_discharge_mean_7d",
                "streamflow_discharge_mean_7d_scaled",
                100.0,
            ),
        ),
        climate_variables=(
            ClimateVariable(
                "upstream_temperature",
                "upstream_temperature_scaled",
                10.0,
            ),
            ClimateVariable(
                "upstream_precipitation",
                "upstream_precipitation_scaled",
                100.0,
            ),
        ),
        model_families=("crude_twfe", "post_lasso"),
        lasso_settings=LassoSettings(cv=3, alphas=25, random_state=0, max_iter=20_000),
        excluded_pollutant_columns=("sensor_id", "date", "trench_id"),
        type_group_names=DEFAULT_SETTINGS.type_group_names,
        subclass_labels=DEFAULT_SETTINGS.subclass_labels,
    )


def test_top_level_code_package_exposes_stdlib_console_api() -> None:
    assert code.InteractiveConsole is not None
    assert code.compile_command is not None


@pytest.fixture
def synthetic_settings(tmp_path: Path) -> SensorAnalysisSettings:
    return _synthetic_settings(tmp_path)


def test_select_pollutants_by_type_and_importance(
    synthetic_settings: SensorAnalysisSettings,
) -> None:
    sensor_data = pd.read_parquet(synthetic_settings.sensor_data_path).reset_index()
    transformations = json.loads(
        synthetic_settings.transformations_path.read_text(encoding="utf-8")
    )["recommendations"]
    catalog = build_pollutant_catalog(sensor_data, transformations, synthetic_settings)

    by_type = select_pollutants(
        catalog,
        group_kind="type",
        group_name="core_physicochemical",
        explicit_pollutants=None,
        minimum_observations=1,
    )
    assert set(by_type.pollutants) == {"ph", "turbidity"}

    by_importance = select_pollutants(
        catalog,
        group_kind="importance",
        group_name="high",
        explicit_pollutants=None,
        minimum_observations=1,
    )
    assert set(by_importance.pollutants) == {"ph", "turbidity", "total_nitrogen"}


def test_build_model_specs_creates_family_specific_specs(
    synthetic_settings: SensorAnalysisSettings,
) -> None:
    specs = build_model_specs(
        synthetic_settings,
        pollutant_selection=["ph"],
        subclass_selection=["c41"],
        max_distance_step=2,
    )

    assert len(specs) == 4
    assert {spec.model_family for spec in specs} == {"crude_twfe", "post_lasso"}
    crude = next(spec for spec in specs if spec.model_family == "crude_twfe" and spec.distance_step_index == 1)
    post = next(spec for spec in specs if spec.model_family == "post_lasso" and spec.distance_step_index == 1)
    assert crude.coefficient_columns == ("lc_0_10km_c41_cnt__log_0p01",)
    assert post.forced_regressor_columns[:1] == ("lc_0_10km_c41_cnt__log_0p01",)
    assert "upstream_temperature_scaled" in post.candidate_regressor_columns
    assert (
        "lc_0_10km_c41_cnt__log_0p01__x__upstream_temperature_scaled"
        in post.candidate_regressor_columns
    )


def test_build_analysis_data_joins_climate_and_creates_interactions(
    synthetic_settings: SensorAnalysisSettings,
) -> None:
    prepared = build_analysis_data(synthetic_settings)

    assert "upstream_temperature_scaled" in prepared.data.columns
    assert "upstream_precipitation_scaled" in prepared.data.columns
    assert "lc_0_10km_c41_cnt__log_0p01__x__upstream_temperature_scaled" in prepared.data.columns
    assert "ph__transformed" in prepared.data.columns
    assert prepared.data["upstream_temperature_scaled"].notna().all()


def test_build_analysis_data_rejects_missing_climate_columns(
    synthetic_settings: SensorAnalysisSettings,
) -> None:
    climate = pd.read_parquet(synthetic_settings.climate_data_path)
    climate = climate.drop(columns=["upstream_temperature"])
    climate.to_parquet(synthetic_settings.climate_data_path, index=False)

    with pytest.raises(ValueError, match="climate_data is missing required columns"):
        build_analysis_data(synthetic_settings)


def test_build_analysis_data_discovers_embedded_climate_and_matches_interactions_by_distance(
    tmp_path: Path,
) -> None:
    sensor = pd.DataFrame(
        {
            "station_code": ["a", "a", "b", "b"],
            "datetime": pd.to_datetime(["2020-01-10", "2021-01-10", "2020-01-11", "2021-01-11"]),
            "date": pd.to_datetime(["2020-01-10", "2021-01-10", "2020-01-11", "2021-01-11"]),
            "trench_id": [1, 1, 2, 2],
            "ph": [7.0, 7.2, 7.1, 7.3],
            "streamflow_discharge_day": [100.0, 110.0, 120.0, 130.0],
            "streamflow_discharge_mean_7d": [90.0, 95.0, 100.0, 105.0],
            "cl_0_10km_tp_mean_7d": [1.0, 1.2, 1.4, 1.6],
            "cl_10_50km_tp_mean_7d": [2.0, 2.2, 2.4, 2.6],
        }
    ).set_index(["station_code", "datetime"])
    sensor_path = tmp_path / "sensor.parquet"
    sensor.to_parquet(sensor_path)

    land_cover = pd.DataFrame(
        {
            "trench_id": [1, 1, 2, 2],
            "year": [2020, 2021, 2020, 2021],
            "lc_0_10km_c41_cnt": [0.2, 0.3, 0.4, 0.5],
            "lc_10_50km_c41_cnt": [0.1, 0.2, 0.3, 0.4],
        }
    )
    for subclass in DEFAULT_SETTINGS.land_cover_subclasses:
        if subclass == "c41":
            continue
        land_cover[f"lc_0_10km_{subclass}_cnt"] = 0.0
        land_cover[f"lc_10_50km_{subclass}_cnt"] = 0.0
    land_cover_path = tmp_path / "land_cover.parquet"
    land_cover.to_parquet(land_cover_path, index=False)

    trenches_path = tmp_path / "trenches.parquet"
    pd.DataFrame({"trench_id": [1, 2], "system_id": [1, 2]}).to_parquet(trenches_path, index=False)

    transformations_path = tmp_path / "transformations.json"
    transformations_path.write_text(
        json.dumps(
            {
                "recommendations": {
                    "ph": {
                        "column": "ph",
                        "recommended_transform": "identity",
                        "expression": "x",
                        "apply_to": "analysis",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    settings = SensorAnalysisSettings(
        project_root=tmp_path,
        sensor_data_path=sensor_path,
        land_cover_path=land_cover_path,
        climate_data_path=None,
        transformations_path=transformations_path,
        trenches_path=trenches_path,
        output_dir=tmp_path / "output",
        sensor_id_column="station_code",
        sensor_id_aliases=("station_code",),
        datetime_column="datetime",
        date_column="date",
        climate_join_keys=("station_code", "date"),
        distance_buckets=("0_10km", "10_50km"),
        land_cover_subclasses=DEFAULT_SETTINGS.land_cover_subclasses,
        land_cover_statistic="cnt",
        fixed_effects=("station_code", "year"),
        cluster_variable="station_code",
        controls=(
            ControlVariable("streamflow_discharge_day", "streamflow_discharge_day_scaled", 100.0),
            ControlVariable("streamflow_discharge_mean_7d", "streamflow_discharge_mean_7d_scaled", 100.0),
        ),
        climate_variables=(),
        excluded_pollutant_columns=("station_code", "datetime", "date", "trench_id"),
        minimum_observations=1,
    )

    prepared = build_analysis_data(settings)
    assert {
        "cl_0_10km_tp_mean_7d__scaled",
        "cl_10_50km_tp_mean_7d__scaled",
    }.issubset(prepared.data.columns)
    assert "lc_0_10km_c41_cnt__log_0p01__x__cl_0_10km_tp_mean_7d__scaled" in prepared.data.columns
    assert "lc_10_50km_c41_cnt__log_0p01__x__cl_10_50km_tp_mean_7d__scaled" in prepared.data.columns
    assert "lc_0_10km_c41_cnt__log_0p01__x__cl_10_50km_tp_mean_7d__scaled" not in prepared.data.columns

    specs = build_model_specs(
        settings,
        pollutant_selection=["ph"],
        subclass_selection=["c41"],
        max_distance_step=1,
        model_families=["post_lasso"],
        climate_variables=prepared.climate_variables,
    )
    assert specs[0].candidate_regressor_columns == (
        "cl_0_10km_tp_mean_7d__scaled",
        "lc_0_10km_c41_cnt__log_0p01__x__cl_0_10km_tp_mean_7d__scaled",
    )


def test_build_analysis_data_reshapes_long_sensor_land_cover(
    tmp_path: Path,
) -> None:
    sensor = pd.DataFrame(
        {
            "station_code": ["101", "101", "202", "202"],
            "datetime": pd.to_datetime(["2020-01-10", "2021-01-10", "2020-01-11", "2021-01-11"]),
            "date": pd.to_datetime(["2020-01-10", "2021-01-10", "2020-01-11", "2021-01-11"]),
            "trench_id": [1, 1, 2, 2],
            "ph": [7.0, 7.2, 7.1, 7.3],
            "streamflow_discharge_day": [100.0, 110.0, 120.0, 130.0],
            "streamflow_discharge_mean_7d": [90.0, 95.0, 100.0, 105.0],
            "cl_0_10km_tp_mean_7d": [1.0, 1.2, 1.4, 1.6],
            "cl_10_50km_tp_mean_7d": [2.0, 2.2, 2.4, 2.6],
        }
    ).set_index(["station_code", "datetime"])
    sensor_path = tmp_path / "sensor.parquet"
    sensor.to_parquet(sensor_path)

    land_cover = pd.DataFrame(
        {
            "station_code": ["101", "101", "101", "101", "202", "202", "202", "202"],
            "year": [2020, 2020, 2021, 2021, 2020, 2020, 2021, 2021],
            "bucket": [0, 25, 0, 25, 0, 25, 0, 25],
            "land_cover_class": ["c41"] * 8,
            "n": [10, 20, 10, 20, 10, 20, 10, 20],
            "cnt": [2.0, 1.0, 3.0, 1.5, 4.0, 2.0, 5.0, 2.5],
            "share": [0.2, 0.05, 0.3, 0.075, 0.4, 0.1, 0.5, 0.125],
        }
    )
    land_cover_path = tmp_path / "land_cover_long.parquet"
    land_cover.to_parquet(land_cover_path, index=False)

    trenches_path = tmp_path / "trenches.parquet"
    pd.DataFrame({"trench_id": [1, 2], "system_id": [1, 2]}).to_parquet(trenches_path, index=False)

    transformations_path = tmp_path / "transformations.json"
    transformations_path.write_text(
        json.dumps(
            {
                "recommendations": {
                    "ph": {
                        "column": "ph",
                        "recommended_transform": "identity",
                        "expression": "x",
                        "apply_to": "analysis",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    settings = SensorAnalysisSettings(
        project_root=tmp_path,
        sensor_data_path=sensor_path,
        land_cover_path=land_cover_path,
        climate_data_path=None,
        transformations_path=transformations_path,
        trenches_path=trenches_path,
        output_dir=tmp_path / "output",
        sensor_id_column="station_code",
        sensor_id_aliases=("station_code",),
        datetime_column="datetime",
        date_column="date",
        climate_join_keys=("station_code", "date"),
        distance_buckets=("0_10km", "10_50km"),
        land_cover_subclasses=("c41",),
        land_cover_statistic="cnt",
        fixed_effects=("station_code", "year"),
        cluster_variable="station_code",
        controls=(
            ControlVariable("streamflow_discharge_day", "streamflow_discharge_day_scaled", 100.0),
            ControlVariable("streamflow_discharge_mean_7d", "streamflow_discharge_mean_7d_scaled", 100.0),
        ),
        climate_variables=(),
        excluded_pollutant_columns=("station_code", "datetime", "date", "trench_id"),
        minimum_observations=1,
    )

    prepared = build_analysis_data(settings)
    row = prepared.data.loc[
        (prepared.data["station_code"] == "101") & (prepared.data["year"] == 2020)
    ].iloc[0]
    assert np.isclose(row["lc_0_10km_c41_cnt"], 2.0)
    assert np.isclose(row["lc_10_50km_c41_cnt"], 1.0)
    assert np.isclose(row["lc_0_10km_c41_cnt__log_0p01"], np.log(2.01))
    assert "lc_0_10km_c41_cnt__log_0p01__x__cl_0_10km_tp_mean_7d__scaled" in prepared.data.columns
    assert "lc_10_50km_c41_cnt__log_0p01__x__cl_10_50km_tp_mean_7d__scaled" in prepared.data.columns


def test_map_residualization_removes_group_means() -> None:
    sample = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "x": [0.0, 1.0, 1.0, 2.0],
            "fe_a": ["a", "a", "b", "b"],
            "fe_b": ["u", "v", "u", "v"],
        }
    )

    result = residualize_with_map(
        sample,
        outcome_column="y",
        feature_columns=["x"],
        fixed_effect_columns=["fe_a", "fe_b"],
        tolerance=1e-12,
        max_iterations=1_000,
    )

    for fixed_effect in ("fe_a", "fe_b"):
        grouped = result.frame.groupby(fixed_effect)[["y", "x"]].mean().abs().max().max()
        assert grouped < 1e-8
    assert result.converged


def test_map_residualization_matches_pyfixest_twfe_coefficients() -> None:
    sensor_ids = [f"s{i}" for i in range(1, 9)]
    years = [2019, 2020, 2021, 2022]
    quarter_effect = {1: -0.2, 2: 0.0, 3: 0.15, 4: 0.3}
    sensor_effect = {sensor_id: (index - 4) * 0.25 for index, sensor_id in enumerate(sensor_ids, start=1)}
    year_effect = {2019: -0.4, 2020: 0.1, 2021: 0.35, 2022: 0.6}

    rows: list[dict[str, float | int | str]] = []
    for sensor_index, sensor_id in enumerate(sensor_ids, start=1):
        for year in years:
            for quarter in range(1, 5):
                x1 = 0.4 * sensor_index + 0.7 * (year - 2019) + 0.3 * quarter
                x2 = (
                    1.1
                    + 0.2 * sensor_index
                    - 0.25 * (year - 2019)
                    + 0.08 * (quarter**2)
                    + 0.03 * sensor_index * quarter
                )
                noise = ((sensor_index * 7 + year + quarter) % 5 - 2) * 0.01
                y = (
                    1.75 * x1
                    - 0.65 * x2
                    + sensor_effect[sensor_id]
                    + year_effect[year]
                    + quarter_effect[quarter]
                    + noise
                )
                rows.append(
                    {
                        "sensor_id": sensor_id,
                        "year": year,
                        "quarter": quarter,
                        "y": y,
                        "x1": x1,
                        "x2": x2,
                    }
                )

    sample = pd.DataFrame(rows)
    fe_fit = pf.feols(
        "y ~ x1 + x2 | sensor_id + year",
        data=sample,
    )
    fe_tidy = fe_fit.tidy().reset_index()
    if "term" not in fe_tidy.columns:
        fe_tidy = fe_tidy.rename(columns={fe_tidy.columns[0]: "term"})
    fe_estimates = fe_tidy.set_index("term")["Estimate"]

    residualized = residualize_with_map(
        sample,
        outcome_column="y",
        feature_columns=["x1", "x2"],
        fixed_effect_columns=["sensor_id", "year"],
        tolerance=1e-12,
        max_iterations=10_000,
    )
    demeaned = residualized.frame.loc[:, ["y", "x1", "x2"]].copy()
    demeaned["cluster_id"] = sample["sensor_id"].to_numpy()
    map_fit = pf.feols(
        "y ~ x1 + x2 - 1",
        data=demeaned,
    )
    map_tidy = map_fit.tidy().reset_index()
    if "term" not in map_tidy.columns:
        map_tidy = map_tidy.rename(columns={map_tidy.columns[0]: "term"})
    map_estimates = map_tidy.set_index("term")["Estimate"]

    for term in ("x1", "x2"):
        assert term in fe_estimates.index
        assert term in map_estimates.index
        assert np.isclose(
            fe_estimates[term],
            map_estimates[term],
            atol=1e-8,
            rtol=1e-8,
        ), (term, fe_estimates[term], map_estimates[term])
    assert residualized.converged


def test_run_suite_produces_crude_and_post_lasso_results(
    synthetic_settings: SensorAnalysisSettings,
) -> None:
    run = run_suite(
        synthetic_settings,
        pollutants=["ph"],
        land_cover_subclasses=["c41"],
        max_distance_step=1,
        min_observations=1,
        save_outputs=False,
    )

    assert run.manifest["status"].eq("ok").all()
    assert set(run.manifest["model_family"]) == {"crude_twfe", "post_lasso"}
    assert set(run.results["model_family"]) == {"crude_twfe", "post_lasso"}
    post_rows = run.manifest.loc[run.manifest["model_family"] == "post_lasso"]
    assert post_rows["lasso_selected_count"].fillna(0).ge(1).all()
    assert run.results["map_converged"].all()
    assert run.output_dir.name == "pollutant_ph"


def test_faceted_distance_coefplot_uses_readable_labels(
    synthetic_settings: SensorAnalysisSettings,
) -> None:
    run = run_suite(
        synthetic_settings,
        pollutants=["ph"],
        land_cover_subclasses=["c41"],
        max_distance_step=1,
        min_observations=1,
        model_families=["crude_twfe"],
        save_outputs=False,
    )

    fig, plot_data = faceted_distance_coefplot(
        run.results,
        pollutants=["ph"],
        land_cover_subclasses=["c41"],
        settings=synthetic_settings,
    )
    assert not plot_data.empty
    assert "distance_bucket_label" in plot_data.columns
    assert "pollutant_label" in plot_data.columns
    assert "land_cover_label" in plot_data.columns
    assert plot_data["distance_bucket_label"].iloc[0] == "0-10 km"
    assert plot_data["land_cover_label"].iloc[0] == "Mining"
    assert plot_data["pollutant_label"].iloc[0] == "Ph"
    fig.clf()


def test_cli_list_groups_outputs_json(
    synthetic_settings: SensorAnalysisSettings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = analysis_main(
        [
            "--sensor-data-path",
            str(synthetic_settings.sensor_data_path),
            "--land-cover-path",
            str(synthetic_settings.land_cover_path),
            "--climate-data-path",
            str(synthetic_settings.climate_data_path),
            "--transformations-path",
            str(synthetic_settings.transformations_path),
            "--trenches-path",
            str(synthetic_settings.trenches_path),
            "--output-dir",
            str(synthetic_settings.output_dir),
            "--min-observations",
            "1",
            "list-groups",
            "--as-json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert "type" in payload
    assert "importance" in payload


def test_cli_run_rejects_invalid_land_cover_subclass(
    synthetic_settings: SensorAnalysisSettings,
) -> None:
    with pytest.raises(ValueError, match="Unknown land-cover subclasses"):
        analysis_main(
            [
                "--sensor-data-path",
                str(synthetic_settings.sensor_data_path),
                "--land-cover-path",
                str(synthetic_settings.land_cover_path),
                "--climate-data-path",
                str(synthetic_settings.climate_data_path),
                "--transformations-path",
                str(synthetic_settings.transformations_path),
                "--trenches-path",
                str(synthetic_settings.trenches_path),
                "--output-dir",
                str(synthetic_settings.output_dir),
                "--min-observations",
                "1",
                "run",
                "--pollutants",
                "ph",
                "--land-cover-subclasses",
                "c99",
            ]
        )


def test_cli_run_writes_to_model_subdirectory(
    synthetic_settings: SensorAnalysisSettings,
) -> None:
    exit_code = analysis_main(
        [
            "--sensor-data-path",
            str(synthetic_settings.sensor_data_path),
            "--land-cover-path",
            str(synthetic_settings.land_cover_path),
            "--climate-data-path",
            str(synthetic_settings.climate_data_path),
            "--transformations-path",
            str(synthetic_settings.transformations_path),
            "--trenches-path",
            str(synthetic_settings.trenches_path),
            "--output-dir",
            str(synthetic_settings.output_dir),
            "--min-observations",
            "1",
            "run",
            "--pollutants",
            "ph",
            "--land-cover-subclasses",
            "c41",
            "--max-distance-step",
            "1",
            "--model-families",
            "crude_twfe,post_lasso",
        ]
    )

    assert exit_code == 0
    assert (synthetic_settings.output_dir / "pollutant_ph" / "manifest.parquet").exists()
    assert (synthetic_settings.output_dir / "pollutant_ph" / "results_readable.csv").exists()
    assert (synthetic_settings.output_dir / "pollutant_ph" / "manifest_readable.csv").exists()
    assert (synthetic_settings.output_dir / "pollutant_ph" / "results_readable.md").exists()
    readable = pd.read_csv(synthetic_settings.output_dir / "pollutant_ph" / "results_readable.csv")
    assert {"model", "pollutant_label", "land_cover_label", "distance_label", "term_label"}.issubset(readable.columns)
