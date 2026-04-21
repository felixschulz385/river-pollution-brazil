"""Preparation of regression-ready sensor analysis data."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .catalog import PollutantDefinition, build_land_cover_catalog, build_pollutant_catalog
from .loaders import load_analysis_inputs
from ..settings import DEFAULT_SETTINGS, SensorAnalysisSettings


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

    system_id = merged["system_id"].astype("Int64")
    derived_columns: dict[str, pd.Series] = {
        "system_id": system_id,
        "quarter_year_system": (
            merged["year"].astype(str)
            + "_Q"
            + merged["quarter"].astype(str)
            + "_"
            + system_id.astype(str)
        ),
    }

    for control in settings.controls:
        derived_columns[control.scaled_column] = (
            merged[control.source_column] / control.scale
        )

    merged = pd.concat(
        [merged.drop(columns=["system_id"], errors="ignore"), pd.DataFrame(derived_columns, index=merged.index)],
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
