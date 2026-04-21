from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from code.analysis.cli import main as analysis_main
from code.analysis.sensor_data.catalog import build_pollutant_catalog
from code.analysis.sensor_data.groups import select_pollutants
from code.analysis.sensor_data.prepare import build_analysis_data
from code.analysis.sensor_data.runner import run_suite
from code.analysis.sensor_data.specs import build_model_specs
from code.analysis.settings import (
    ControlVariable,
    DEFAULT_SETTINGS,
    ImportanceTier,
    SensorAnalysisSettings,
)


def _synthetic_settings(tmp_path: Path) -> SensorAnalysisSettings:
    dates = pd.to_datetime(
        [
            "2020-01-15",
            "2020-07-15",
            "2021-01-15",
            "2021-07-15",
        ]
    )
    station_effect = {"1001": 0.2, "1002": -0.1, "1003": 0.05}
    quarter_effect = {
        (2020, 1): 0.1,
        (2020, 3): 0.0,
        (2021, 1): 0.15,
        (2021, 3): -0.05,
    }
    land_cover_rows = []
    trench_year_payload = {
        (1, 2020): {"c41": [1.0, 0.5, 0.0], "c42": [0.1, 0.0, 0.0]},
        (1, 2021): {"c41": [1.7, 0.9, 0.0], "c42": [0.4, 0.1, 0.0]},
        (2, 2020): {"c41": [1.8, 0.4, 0.0], "c42": [0.0, 0.3, 0.0]},
        (2, 2021): {"c41": [2.1, 0.2, 0.0], "c42": [0.2, 0.8, 0.0]},
        (3, 2020): {"c41": [2.5, 0.7, 0.0], "c42": [0.2, 0.2, 0.0]},
        (3, 2021): {"c41": [3.4, 1.4, 0.0], "c42": [0.5, 0.1, 0.0]},
    }

    bucket_order = DEFAULT_SETTINGS.distance_buckets
    subclasses = DEFAULT_SETTINGS.land_cover_subclasses

    for (trench_id, year), values in trench_year_payload.items():
        row = {"trench_id": trench_id, "year": year}
        for bucket in bucket_order:
            row[f"lc_{bucket}_tot"] = 10.0
            row[f"lc_{bucket}_n"] = 1
        for subclass in subclasses:
            counts = values.get(subclass, [0.0, 0.0, 0.0])
            padded_counts = [*counts, *([0.0] * (len(bucket_order) - len(counts)))]
            for bucket, count in zip(bucket_order, padded_counts, strict=True):
                row[f"lc_{bucket}_{subclass}_cnt"] = count
        land_cover_rows.append(row)

    land_cover = pd.DataFrame(land_cover_rows)
    land_cover_path = tmp_path / "land_cover.parquet"
    land_cover.to_parquet(land_cover_path, index=False)

    trenches = pd.DataFrame(
        {"trench_id": [1, 2, 3], "system_id": [10, 10, 10]}
    )
    trenches_path = tmp_path / "trenches.parquet"
    trenches.to_parquet(trenches_path, index=False)

    sensor_rows = []
    for station_code, trench_id in [("1001", 1), ("1002", 2), ("1003", 3)]:
        for date in dates:
            year = int(date.year)
            quarter = int(date.quarter)
            land_row = trench_year_payload[(trench_id, year)]
            c41_0_10, c41_10_50, _ = land_row["c41"]
            c42_0_10, _, _ = land_row["c42"]
            flow_day = 100.0 + trench_id * 10 + quarter * 5
            flow_7d = 120.0 + trench_id * 8 + quarter * 4
            ph = (
                6.0
                + 0.45 * c41_0_10
                + 0.20 * c41_10_50
                + 0.10 * c42_0_10
                + 0.01 * flow_day
                + station_effect[station_code]
                + quarter_effect[(year, quarter)]
            )
            turbidity = (
                15.0
                + 2.0 * c41_0_10
                + 1.0 * c41_10_50
                + 0.30 * flow_7d
                + quarter_effect[(year, quarter)] * 5
            )
            total_nitrogen = (
                1.0
                + 0.1 * c42_0_10
                + 0.02 * flow_day
                + station_effect[station_code]
            )
            sensor_rows.append(
                {
                    "station_code": station_code,
                    "datetime": pd.Timestamp(date) + pd.Timedelta(hours=8),
                    "date": pd.Timestamp(date),
                    "trench_id": trench_id,
                    "ph": ph,
                    "turbidity": turbidity,
                    "total_nitrogen": total_nitrogen,
                    "streamflow_discharge_day": flow_day,
                    "streamflow_discharge_mean_7d": flow_7d,
                    "streamflow_discharge_mean_31d": flow_7d + 5,
                }
            )

    sensor_data = pd.DataFrame(sensor_rows).set_index(["station_code", "datetime"])
    sensor_data_path = tmp_path / "sensor.parquet"
    sensor_data.to_parquet(sensor_data_path)

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
        transformations_path=transformations_path,
        trenches_path=trenches_path,
        output_dir=tmp_path / "output",
        distance_buckets=bucket_order,
        land_cover_subclasses=subclasses,
        land_cover_statistic="cnt",
        fixed_effects=("station_code", "quarter_year_system"),
        cluster_variable="station_code",
        vcov_type="CRV1",
        minimum_observations=1,
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
        excluded_pollutant_columns=("date", "trench_id", "station_code", "datetime"),
        type_group_names=(
            "core_physicochemical",
            "nutrients",
            "oxygen_demand_organic_load",
            "microbiological",
            "metals",
            "organics_pesticides",
            "composite_indices",
            "other",
        ),
        subclass_labels=DEFAULT_SETTINGS.subclass_labels,
    )


@pytest.fixture
def synthetic_settings(tmp_path: Path) -> SensorAnalysisSettings:
    return _synthetic_settings(tmp_path)


def test_select_pollutants_by_type_and_importance(synthetic_settings: SensorAnalysisSettings) -> None:
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


def test_build_model_specs_creates_cumulative_distance_steps() -> None:
    settings = SensorAnalysisSettings(
        distance_buckets=("0_10km", "10_50km", "50_100km"),
        land_cover_subclasses=("c41",),
        minimum_observations=1,
    )
    specs = build_model_specs(
        settings,
        pollutant_selection=["ph"],
        subclass_selection=["c41"],
        max_distance_step=3,
    )

    assert len(specs) == 3
    assert specs[0].coefficient_columns == ("lc_0_10km_c41_cnt",)
    assert specs[1].coefficient_columns == (
        "lc_0_10km_c41_cnt",
        "lc_10_50km_c41_cnt",
    )
    assert specs[2].distance_step_name == "50_100km"


def test_build_analysis_data_applies_transforms_and_derived_columns(
    synthetic_settings: SensorAnalysisSettings,
) -> None:
    prepared = build_analysis_data(synthetic_settings)

    assert "quarter_year_system" in prepared.data.columns
    assert "streamflow_discharge_day_scaled" in prepared.data.columns
    assert "ph__transformed" in prepared.data.columns
    assert "turbidity__transformed" in prepared.data.columns
    assert prepared.data["turbidity__transformed"].notna().all()


def test_run_suite_produces_manifest_and_results(
    synthetic_settings: SensorAnalysisSettings,
) -> None:
    run = run_suite(
        synthetic_settings,
        pollutants=["ph"],
        land_cover_subclasses=["c41"],
        max_distance_step=2,
        min_observations=1,
        save_outputs=False,
    )

    assert run.manifest["status"].eq("ok").all()
    assert set(run.manifest["distance_step_name"]) == {"0_10km", "10_50km"}
    assert set(run.results["pollutant"]) == {"ph"}
    assert (run.results["land_cover_subclass"] == "c41").all()
    assert run.output_dir.name == "pollutant_ph"


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
            "--max-distance-step",
            "1",
        ]
    )

    assert exit_code == 0
    assert (synthetic_settings.output_dir / "pollutant_ph" / "manifest.parquet").exists()
