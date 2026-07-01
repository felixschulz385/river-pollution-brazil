"""Shared settings for analysis workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ControlVariable:
    """Definition of one regression control variable."""

    source_column: str
    scaled_column: str
    scale: float = 1.0


@dataclass(frozen=True)
class ClimateVariable:
    """Definition of one upstream climate variable."""

    source_column: str
    scaled_column: str
    scale: float = 1.0


@dataclass(frozen=True)
class ImportanceTier:
    """Coverage threshold for an importance tier."""

    name: str
    minimum_observations: int


@dataclass(frozen=True)
class LandCoverTransform:
    """Transformation applied to land-cover regressors before estimation."""

    kind: str = "identity"
    offset: float = 0.0

    def column_suffix(self) -> str:
        """Return a deterministic suffix for transformed land-cover columns."""
        if self.kind == "identity":
            return ""
        offset = str(self.offset).replace("-", "m").replace(".", "p")
        return f"__{self.kind}_{offset}"


@dataclass(frozen=True)
class FixedEffectVariable:
    """Atomic fixed-effect component built from a data column."""

    source_column: str
    datetime_accessor: str | None = None


FixedEffectSpec: TypeAlias = str | tuple[str, ...] | list[str]


@dataclass(frozen=True)
class LassoSettings:
    """Hyperparameters for Post-LASSO selection."""

    cv: int = 5
    alphas: int = 100
    eps: float = 1e-3
    random_state: int = 0
    max_iter: int = 10_000
    standardize: bool = True


@dataclass(frozen=True)
class SensorAnalysisSettings:
    """Default configuration for the sensor analysis suite."""

    # Paths
    project_root: Path = PROJECT_ROOT
    sensor_data_path: Path = PROJECT_ROOT / "data/sensor_data/water_quality_assembled.parquet"
    land_cover_path: Path = PROJECT_ROOT / "data/land_cover/land_cover_assembled_sensor.parquet"
    climate_data_path: Path = PROJECT_ROOT / "data/climate/climate_sensor_upstream.parquet"
    transformations_path: Path = PROJECT_ROOT / "data/sensor_data/water_quality_transformations.json"
    trenches_path: Path = PROJECT_ROOT / "data/river_network/trenches.parquet"
    output_dir: Path = PROJECT_ROOT / "output/analysis/sensor_data"
    sensor_id_column: str = "sensor_id"
    date_column: str = "date"
    climate_join_keys: tuple[str, str] = ("sensor_id", "date")

    # Land-cover regressors
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
    land_cover_statistic: str = "shr"
    land_cover_transform: LandCoverTransform = field(
        default_factory=lambda: LandCoverTransform(kind="log", offset=0.01)
    )

    # Fixed effects and inference
    fixed_effect_variables: dict[str, FixedEffectVariable] = field(
        default_factory=lambda: {
            "quarter": FixedEffectVariable("date", "quarter"),
            "year": FixedEffectVariable("date", "year"),
            "system": FixedEffectVariable("system_id"),
        }
    )
    fixed_effects: tuple[FixedEffectSpec, ...] = (
        "sensor_id",
        ("quarter", "system"),
        ("year", "system"),
    )
    cluster_variable: str = "sensor_id"
    vcov_type: str = "CRV1"
    minimum_observations: int = 5_000
    map_tolerance: float = 1e-8
    map_max_iterations: int = 1_000

    # Pollutant grouping
    importance_tiers: tuple[ImportanceTier, ...] = (
        ImportanceTier("high", 100_000),
        ImportanceTier("medium", 25_000),
        ImportanceTier("low", 5_000),
        ImportanceTier("rare", 0),
    )

    # Regression controls
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
    climate_variables: tuple[ClimateVariable, ...] = (
        ClimateVariable(
            source_column="upstream_temperature",
            scaled_column="upstream_temperature_scaled",
            scale=10.0,
        ),
        ClimateVariable(
            source_column="upstream_precipitation",
            scaled_column="upstream_precipitation_scaled",
            scale=100.0,
        ),
    )
    model_families: tuple[str, ...] = ("crude_twfe", "post_lasso")
    lasso_settings: LassoSettings = field(default_factory=LassoSettings)

    # Pollutant catalog exclusions
    excluded_pollutant_columns: tuple[str, ...] = (
        "sensor_id",
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

    # Pollutant catalog grouping
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

    # Land-cover labels
    subclass_labels: dict[str, str] = field(
        default_factory=lambda: {
            "c0": "NA",
            "c1": "Forest",
            "c2": "Non-forest natural vegetation",
            "c3": "Farming",
            "c30": "Pasture",
            "c31": "Agriculture",
            "c4": "Non-vegetated area",
            "c40": "Urban area",
            "c41": "Mining",
            "c5": "Water",
        }
    )
    distance_bucket_labels: dict[str, str] = field(
        default_factory=lambda: {
            "0_10km": "0-10 km",
            "10_50km": "10-50 km",
            "50_100km": "50-100 km",
            "100_250km": "100-250 km",
            "250_500km": "250-500 km",
            "500km_plus": "500+ km",
        }
    )
    model_family_labels: dict[str, str] = field(
        default_factory=lambda: {
            "crude_twfe": "Crude TWFE",
            "post_lasso": "Post-LASSO",
        }
    )

    def land_cover_source_column(self, bucket: str, subclass: str) -> str:
        """Return the raw input column name for a land-cover regressor."""
        return f"lc_{bucket}_{subclass}_{self.land_cover_statistic}"

    def land_cover_column(self, bucket: str, subclass: str) -> str:
        """Return the analysis column name for a land-cover regressor."""
        return (
            f"{self.land_cover_source_column(bucket, subclass)}"
            f"{self.land_cover_transform.column_suffix()}"
        )

    def interaction_column(self, land_cover_column: str, climate_column: str) -> str:
        """Return the analysis column name for a land-cover and climate interaction."""
        return f"{land_cover_column}__x__{climate_column}"

    def distance_bucket_label(self, bucket: str) -> str:
        """Return a human-readable distance-bucket label."""
        return self.distance_bucket_labels.get(bucket, bucket.replace("_", " "))

    def model_family_label(self, family: str) -> str:
        """Return a human-readable model-family label."""
        return self.model_family_labels.get(family, family.replace("_", " ").title())

    def resolve_fixed_effect_name(self, effect: FixedEffectSpec) -> str:
        """Return the materialized column name for a fixed effect."""
        if isinstance(effect, str):
            return effect
        return "_".join(tuple(effect))

    def resolved_fixed_effects(self) -> tuple[str, ...]:
        """Return concrete fixed-effect column names used in formulas."""
        return tuple(self.resolve_fixed_effect_name(effect) for effect in self.fixed_effects)


DEFAULT_SETTINGS = SensorAnalysisSettings()


__all__ = [
    "ClimateVariable",
    "ControlVariable",
    "DEFAULT_SETTINGS",
    "FixedEffectSpec",
    "FixedEffectVariable",
    "ImportanceTier",
    "LandCoverTransform",
    "LassoSettings",
    "PROJECT_ROOT",
    "SensorAnalysisSettings",
]
