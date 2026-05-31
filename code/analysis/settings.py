"""Shared settings for analysis workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ControlVariable:
    """Definition of one regression control variable."""

    source_column: str
    scaled_column: str
    scale: float = 1.0


@dataclass(frozen=True)
class ImportanceTier:
    """Coverage threshold for an importance tier."""

    name: str
    minimum_observations: int


@dataclass(frozen=True)
class SensorAnalysisSettings:
    """Default configuration for the sensor analysis suite."""

    project_root: Path = PROJECT_ROOT
    sensor_data_path: Path = PROJECT_ROOT / "data/sensor_data/water_quality_assembled.parquet"
    land_cover_path: Path = PROJECT_ROOT / "data/land_cover/land_cover_sensor_upstream.parquet"
    transformations_path: Path = PROJECT_ROOT / "data/sensor_data/water_quality_transformations.json"
    trenches_path: Path = PROJECT_ROOT / "data/river_network/trenches.parquet"
    output_dir: Path = PROJECT_ROOT / "output/analysis/sensor_data"
    distance_buckets: tuple[str, ...] = (
        "0_10km",
        "10_50km",
        "50_100km",
        "100_250km",
        "250_500km",
        "500km_plus",
    )
    land_cover_subclasses: tuple[str, ...] = (
        "c0",
        "c1",
        "c2",
        "c3",
        "c4",
        "c5",
        "c30",
        "c31",
        "c40",
        "c41",
        "c42",
    )
    land_cover_statistic: str = "cnt"
    fixed_effects: tuple[str, ...] = ("station_code", "quarter_year_system")
    cluster_variable: str = "station_code"
    vcov_type: str = "CRV1"
    minimum_observations: int = 5_000
    importance_tiers: tuple[ImportanceTier, ...] = (
        ImportanceTier("high", 100_000),
        ImportanceTier("medium", 25_000),
        ImportanceTier("low", 5_000),
        ImportanceTier("rare", 0),
    )
    controls: tuple[ControlVariable, ...] = (
        ControlVariable(
            source_column="streamflow_discharge_day",
            scaled_column="streamflow_discharge_day_scaled",
            scale=100.0,
        ),
        ControlVariable(
            source_column="streamflow_discharge_mean_7d",
            scaled_column="streamflow_discharge_mean_7d_scaled",
            scale=100.0,
        ),
    )
    excluded_pollutant_columns: tuple[str, ...] = (
        "station_code",
        "datetime",
        "date",
        "trench_id",
        "year",
        "quarter",
        "system_id",
        "quarter_year_system",
        "updated_at",
        "created_at",
        "updated_by",
        "streamflow_match_count",
        "streamflow_nonnull_day_count",
        "streamflow_total_weight",
        "streamflow_nearest_distance_m",
        "streamflow_discharge_day",
        "streamflow_discharge_mean_7d",
        "streamflow_discharge_mean_31d",
        "sample_temperature",
        "air_temperature",
        "depth",
        "depth_parameter",
        "liquid_discharge",
        "rained",
    )
    type_group_names: tuple[str, ...] = (
        "core_physicochemical",
        "nutrients",
        "oxygen_demand_organic_load",
        "microbiological",
        "metals",
        "organics_pesticides",
        "composite_indices",
        "other",
    )
    subclass_labels: dict[str, str] = field(
        default_factory=lambda: {
            "c0": "Forest formation",
            "c1": "Savanna formation",
            "c2": "Mangrove",
            "c3": "Floodable forest",
            "c4": "Wetland",
            "c5": "Grassland",
            "c30": "Pasture",
            "c31": "Agriculture",
            "c40": "Non-vegetated area",
            "c41": "Urban area",
            "c42": "Mining",
        }
    )


DEFAULT_SETTINGS = SensorAnalysisSettings()


__all__ = [
    "ControlVariable",
    "DEFAULT_SETTINGS",
    "ImportanceTier",
    "PROJECT_ROOT",
    "SensorAnalysisSettings",
]
