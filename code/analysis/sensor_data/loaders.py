"""Loaders and schema validation for analysis inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from ..settings import SensorAnalysisSettings


@dataclass(frozen=True)
class AnalysisInputs:
    """Loaded raw inputs for analysis preparation."""

    sensor_data: pd.DataFrame
    land_cover: pd.DataFrame
    trenches: pd.DataFrame
    transformations: dict[str, dict[str, object]]


def validate_required_columns(
    frame: pd.DataFrame,
    required_columns: set[str],
    frame_name: str,
) -> None:
    """Raise a clear error for missing columns."""
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}.")


def load_sensor_data(settings: SensorAnalysisSettings) -> pd.DataFrame:
    """Load assembled sensor data and materialize index columns."""
    sensor_data = pd.read_parquet(settings.sensor_data_path).reset_index()
    validate_required_columns(
        sensor_data,
        {"station_code", "datetime", "date", "trench_id"},
        "sensor_data",
    )
    return sensor_data


def load_land_cover(settings: SensorAnalysisSettings) -> pd.DataFrame:
    """Load upstream land-cover features."""
    land_cover = pd.read_parquet(settings.land_cover_path)
    required_columns = {"trench_id", "year"}
    for bucket in settings.distance_buckets:
        for subclass in settings.land_cover_subclasses:
            required_columns.add(settings.land_cover_source_column(bucket, subclass))
    validate_required_columns(land_cover, required_columns, "land_cover")
    return land_cover


def load_trenches(settings: SensorAnalysisSettings) -> pd.DataFrame:
    """Load trench to river-system mappings."""
    trenches = pd.read_parquet(settings.trenches_path)
    validate_required_columns(trenches, {"trench_id", "system_id"}, "trenches")
    return trenches.loc[:, ["trench_id", "system_id"]].drop_duplicates()


def load_transformations(settings: SensorAnalysisSettings) -> dict[str, dict[str, object]]:
    """Load pollutant transform recommendations."""
    payload = json.loads(settings.transformations_path.read_text(encoding="utf-8"))
    recommendations = payload.get("recommendations", {})
    if not isinstance(recommendations, dict):
        raise ValueError("Transformations file does not contain `recommendations`.")
    return {
        name: spec
        for name, spec in recommendations.items()
        if spec.get("apply_to") == "analysis"
    }


def load_analysis_inputs(settings: SensorAnalysisSettings) -> AnalysisInputs:
    """Load all raw inputs required for building analysis data."""
    return AnalysisInputs(
        sensor_data=load_sensor_data(settings),
        land_cover=load_land_cover(settings),
        trenches=load_trenches(settings),
        transformations=load_transformations(settings),
    )


__all__ = [
    "AnalysisInputs",
    "load_analysis_inputs",
    "load_land_cover",
    "load_sensor_data",
    "load_transformations",
    "load_trenches",
    "validate_required_columns",
]
