"""Preparation of regression-ready sensor analysis data."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .catalog import PollutantDefinition, build_land_cover_catalog, build_pollutant_catalog
from .loaders import load_analysis_inputs
from ..settings import DEFAULT_SETTINGS, FixedEffectVariable, SensorAnalysisSettings


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedAnalysisData:
    """Prepared merged data plus catalogs used for the run."""

    data: pd.DataFrame
    pollutant_catalog: list[PollutantDefinition]
    land_cover_catalog: list
    transformations: dict[str, dict[str, object]]


def _apply_transform(series: pd.Series, transform_name: str) -> pd.Series:
    if transform_name == "identity":
        return series.astype(float)
    if transform_name == "log10_1p":
        values = series.astype(float)
        values = values.where(values >= 0)
        return np.log10(1.0 + values)
    raise ValueError(f"Unsupported transform `{transform_name}`.")


def _build_transformed_columns(
    frame: pd.DataFrame,
    pollutant_catalog: list[PollutantDefinition],
) -> pd.DataFrame:
    """Build transformed outcome columns in one batch to avoid fragmentation."""
    transformed_series: dict[str, pd.Series] = {}
    for pollutant in pollutant_catalog:
        transformed_column = f"{pollutant.name}__transformed"
        transformed_series[transformed_column] = _apply_transform(
            frame[pollutant.name],
            pollutant.transform,
        )
    return pd.DataFrame(transformed_series, index=frame.index)


def _apply_land_cover_transform(
    series: pd.Series,
    settings: SensorAnalysisSettings,
) -> pd.Series:
    transform = settings.land_cover_transform
    if transform.kind == "identity":
        return series.astype(float)
    values = series.astype(float) + float(transform.offset)
    if transform.kind == "log":
        return np.log(values.where(values > 0))
    if transform.kind == "log10":
        return np.log10(values.where(values > 0))
    raise ValueError(f"Unsupported land-cover transform `{transform.kind}`.")


def _build_land_cover_columns(
    frame: pd.DataFrame,
    settings: SensorAnalysisSettings,
) -> pd.DataFrame:
    """Build transformed land-cover columns in one batch."""
    if settings.land_cover_transform.kind == "identity":
        return pd.DataFrame(index=frame.index)

    transformed_series: dict[str, pd.Series] = {}
    for bucket in settings.distance_buckets:
        for subclass in settings.land_cover_subclasses:
            transformed_series[settings.land_cover_column(bucket, subclass)] = _apply_land_cover_transform(
                frame[settings.land_cover_source_column(bucket, subclass)],
                settings,
            )
    return pd.DataFrame(transformed_series, index=frame.index)


def _extract_fixed_effect_variable(
    frame: pd.DataFrame,
    definition: FixedEffectVariable,
) -> pd.Series:
    """Extract one atomic fixed-effect component from the merged analysis frame."""
    if definition.source_column not in frame.columns:
        raise ValueError(
            f"Fixed-effect source column `{definition.source_column}` is missing."
        )

    source = frame[definition.source_column]
    if definition.datetime_accessor is None:
        return source

    datetimes = pd.to_datetime(source, errors="coerce")
    accessor = getattr(datetimes.dt, definition.datetime_accessor, None)
    if accessor is None:
        raise ValueError(
            f"Unsupported pandas datetime accessor `{definition.datetime_accessor}` "
            f"for fixed-effect source `{definition.source_column}`."
        )
    return accessor() if callable(accessor) else accessor


def _fixed_effect_component(
    frame: pd.DataFrame,
    atomic_columns: dict[str, pd.Series],
    component: str,
) -> pd.Series:
    if component in atomic_columns:
        return atomic_columns[component]
    if component in frame.columns:
        return frame[component]
    raise ValueError(f"Unknown fixed-effect component `{component}`.")


def _stringify_fixed_effect_component(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any() and numeric.dropna().mod(1).eq(0).all():
        return numeric.astype("Int64").astype("string")
    return series.astype("string")


def _build_fixed_effect_columns(
    frame: pd.DataFrame,
    settings: SensorAnalysisSettings,
) -> pd.DataFrame:
    """Build atomic and composite fixed-effect columns in one batch."""
    derived_columns = {
        name: _extract_fixed_effect_variable(frame, definition)
        for name, definition in settings.fixed_effect_variables.items()
    }

    composite_columns: dict[str, pd.Series] = {}
    for effect in settings.fixed_effects:
        if isinstance(effect, str):
            continue
        components = tuple(effect)
        component_series = [
            _stringify_fixed_effect_component(
                _fixed_effect_component(frame, derived_columns, component)
            )
            for component in components
        ]
        composite = component_series[0]
        for series in component_series[1:]:
            composite = composite.str.cat(series, sep="_")
        composite_columns[settings.resolve_fixed_effect_name(effect)] = composite

    return pd.DataFrame({**derived_columns, **composite_columns}, index=frame.index)


def build_analysis_data(
    settings: SensorAnalysisSettings = DEFAULT_SETTINGS,
) -> PreparedAnalysisData:
    """Build a merged regression-ready sensor analysis panel."""
    inputs = load_analysis_inputs(settings)
    sensor_data = inputs.sensor_data.copy()
    sensor_data["station_code"] = sensor_data["station_code"].astype(str)
    sensor_data["datetime"] = pd.to_datetime(sensor_data["datetime"], errors="coerce")
    sensor_data["date"] = pd.to_datetime(sensor_data["date"], errors="coerce")
    sensor_data["date"] = sensor_data["date"].fillna(sensor_data["datetime"])
    missing_date_rows = sensor_data["date"].isna()
    if missing_date_rows.any():
        logger.warning(
            "Dropping %d sensor rows with missing date/datetime values.",
            int(missing_date_rows.sum()),
        )
        sensor_data = sensor_data.loc[~missing_date_rows].copy()
    sensor_data["year"] = sensor_data["date"].dt.year.astype(int)
    sensor_data["quarter"] = sensor_data["date"].dt.quarter.astype(int)

    merged = sensor_data.merge(
        inputs.land_cover,
        on=["trench_id", "year"],
        how="left",
        validate="many_to_one",
    ).merge(
        inputs.trenches,
        on="trench_id",
        how="left",
        validate="many_to_one",
    )

    for control in settings.controls:
        merged[control.scaled_column] = (
            merged[control.source_column] / control.scale
        )

    fixed_effect_columns = _build_fixed_effect_columns(merged, settings)
    land_cover_columns = _build_land_cover_columns(merged, settings)
    replacement_columns = list(fixed_effect_columns.columns) + list(land_cover_columns.columns)
    merged = pd.concat(
        [merged.drop(columns=replacement_columns, errors="ignore"), fixed_effect_columns, land_cover_columns],
        axis=1,
    )

    pollutant_catalog = build_pollutant_catalog(
        merged,
        inputs.transformations,
        settings,
    )
    transformed_columns = _build_transformed_columns(merged, pollutant_catalog)
    merged = pd.concat([merged, transformed_columns], axis=1)

    return PreparedAnalysisData(
        data=merged,
        pollutant_catalog=pollutant_catalog,
        land_cover_catalog=build_land_cover_catalog(settings),
        transformations=inputs.transformations,
    )


__all__ = ["PreparedAnalysisData", "build_analysis_data"]
