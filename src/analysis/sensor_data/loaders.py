"""Loaders and schema validation for analysis inputs."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import pandas as pd

from ..settings import SensorAnalysisSettings


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisInputs:
    """Loaded raw inputs for analysis preparation."""

    sensor_data: pd.DataFrame
    land_cover: pd.DataFrame
    climate: pd.DataFrame
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
    if settings.sensor_id_column not in sensor_data.columns:
        for alias in settings.sensor_id_aliases:
            if alias in sensor_data.columns:
                sensor_data = sensor_data.rename(columns={alias: settings.sensor_id_column})
                break
    if settings.date_column not in sensor_data.columns and settings.datetime_column in sensor_data.columns:
        sensor_data[settings.date_column] = sensor_data[settings.datetime_column]
    validate_required_columns(
        sensor_data,
        {settings.sensor_id_column, settings.date_column, "trench_id"},
        "sensor_data",
    )
    return sensor_data


def load_land_cover(settings: SensorAnalysisSettings) -> pd.DataFrame:
    """Load upstream land-cover features."""
    land_cover = pd.read_parquet(settings.land_cover_path).reset_index(drop=True)
    if settings.sensor_id_column not in land_cover.columns:
        for alias in settings.sensor_id_aliases:
            if alias in land_cover.columns:
                land_cover = land_cover.rename(columns={alias: settings.sensor_id_column})
                break
    long_required = {
        settings.sensor_id_column,
        "year",
        "bucket",
        "land_cover_class",
        "n",
        "cnt",
        "share",
    }
    if long_required.issubset(land_cover.columns):
        return _reshape_long_land_cover(land_cover, settings)
    required_columns = {"trench_id", "year"}
    for bucket in settings.distance_buckets:
        for subclass in settings.land_cover_subclasses:
            required_columns.add(settings.land_cover_source_column(bucket, subclass))
    validate_required_columns(land_cover, required_columns, "land_cover")
    return land_cover


def _bucket_name_from_distance(distance: float, settings: SensorAnalysisSettings) -> str:
    if distance < 10:
        return "0_10km"
    if distance < 50:
        return "10_50km"
    if distance < 100:
        return "50_100km"
    if distance < 250:
        return "100_250km"
    if distance < 500:
        return "250_500km"
    return "500km_plus"


def _reshape_long_land_cover(
    land_cover: pd.DataFrame,
    settings: SensorAnalysisSettings,
) -> pd.DataFrame:
    """Convert long station-year-bucket-class land cover into wide analysis columns."""
    land_cover = land_cover.copy(deep=False)
    land_cover.index = pd.RangeIndex(len(land_cover))
    land_cover = land_cover.rename_axis(index=None)
    long_frame = land_cover.loc[
        :,
        [
            settings.sensor_id_column,
            "year",
            "bucket",
            "land_cover_class",
            "n",
            "cnt",
            "share",
        ],
    ].copy().reset_index(drop=True)
    long_frame[settings.sensor_id_column] = long_frame[settings.sensor_id_column].astype(str)
    long_frame["distance_bucket_name"] = long_frame["bucket"].astype(float).map(
        lambda value: _bucket_name_from_distance(value, settings)
    )
    long_frame = long_frame.loc[
        long_frame["land_cover_class"].isin(settings.land_cover_subclasses)
    ].copy()
    long_frame.index = pd.RangeIndex(len(long_frame))
    long_frame = long_frame.rename_axis(index=None)

    keyed_frame = long_frame.assign(
        _sensor_key=long_frame[settings.sensor_id_column].to_numpy(),
        _year_key=long_frame["year"].to_numpy(),
        _bucket_key=long_frame["distance_bucket_name"].to_numpy(),
    )

    totals = (
        keyed_frame.groupby(["_sensor_key", "_year_key", "_bucket_key"], as_index=False)
        .agg(
            lc_tot=("cnt", "sum"),
            lc_n=("n", "max"),
        )
        .rename(
            columns={
                "_sensor_key": settings.sensor_id_column,
                "_year_key": "year",
                "_bucket_key": "distance_bucket_name",
            }
        )
    )
    totals_wide = (
        totals.pivot(
            index=[settings.sensor_id_column, "year"],
            columns="distance_bucket_name",
            values=["lc_tot", "lc_n"],
        )
        .sort_index(axis=1)
    )
    totals_wide.columns = [
        f"lc_{bucket}_{'tot' if metric == 'lc_tot' else 'n'}"
        for metric, bucket in totals_wide.columns
    ]

    values_wide = (
        keyed_frame.pivot_table(
            index=["_sensor_key", "_year_key"],
            columns=["_bucket_key", "land_cover_class"],
            values=["cnt", "share"],
            aggfunc="sum",
            fill_value=0.0,
        )
        .sort_index(axis=1)
    )
    values_wide.columns = [
        f"lc_{bucket}_{subclass}_{'cnt' if metric == 'cnt' else 'shr'}"
        for metric, bucket, subclass in values_wide.columns
    ]

    wide = pd.concat([totals_wide, values_wide], axis=1)
    wide.index = wide.index.set_names([settings.sensor_id_column, "year"])
    wide = wide.reset_index()
    for bucket in settings.distance_buckets:
        if f"lc_{bucket}_tot" not in wide.columns:
            wide[f"lc_{bucket}_tot"] = 0.0
        if f"lc_{bucket}_n" not in wide.columns:
            wide[f"lc_{bucket}_n"] = 0
        for subclass in settings.land_cover_subclasses:
            for suffix, default in (("cnt", 0.0), ("shr", 0.0)):
                column = f"lc_{bucket}_{subclass}_{suffix}"
                if column not in wide.columns:
                    wide[column] = default
    return wide


def load_climate_data(settings: SensorAnalysisSettings) -> pd.DataFrame:
    """Load upstream climate features joined at the sensor-date grain."""
    if settings.climate_data_path is None or not settings.climate_data_path.exists():
        return pd.DataFrame()
    climate = pd.read_parquet(settings.climate_data_path).reset_index(drop=True)
    if settings.sensor_id_column not in climate.columns:
        for alias in settings.sensor_id_aliases:
            if alias in climate.columns:
                climate = climate.rename(columns={alias: settings.sensor_id_column})
                break
    if settings.date_column not in climate.columns and settings.datetime_column in climate.columns:
        climate[settings.date_column] = climate[settings.datetime_column]
    required_columns = set(settings.climate_join_keys)
    if settings.climate_variables:
        required_columns.update(
            variable.source_column for variable in settings.climate_variables
        )
    validate_required_columns(climate, required_columns, "climate_data")
    # Normalize the date join key to day granularity before duplicate
    # detection and merging, so two readings on the same day that differ
    # only by time-of-day are treated as the same key on both sides of the
    # eventual join (see the matching normalization on the sensor side in
    # `build_analysis_data`).
    climate_date_key = settings.climate_join_keys[1]
    if climate_date_key in climate.columns:
        climate[climate_date_key] = pd.to_datetime(
            climate[climate_date_key], errors="coerce"
        ).dt.normalize()
    duplicate_mask = climate.duplicated(subset=list(settings.climate_join_keys), keep=False)
    if duplicate_mask.any():
        duplicate_rows = int(duplicate_mask.sum())
        duplicate_keys = int(
            climate.loc[duplicate_mask, list(settings.climate_join_keys)]
            .drop_duplicates()
            .shape[0]
        )
        # Average numeric measurements across duplicate join keys (e.g. two
        # sub-daily readings on the same station-day) rather than keeping an
        # arbitrary single row; non-numeric columns keep the earliest value.
        numeric_columns = [
            column
            for column in climate.columns
            if column not in settings.climate_join_keys and pd.api.types.is_numeric_dtype(climate[column])
        ]
        other_columns = [
            column
            for column in climate.columns
            if column not in settings.climate_join_keys and column not in numeric_columns
        ]
        logger.warning(
            "Collapsing %d climate rows across %d duplicate join keys onto %s "
            "(mean for numeric columns, earliest for others).",
            duplicate_rows,
            duplicate_keys,
            ",".join(settings.climate_join_keys),
        )
        sort_columns = [
            column
            for column in (settings.datetime_column, *settings.climate_join_keys)
            if column in climate.columns
        ]
        if sort_columns:
            climate = climate.sort_values(sort_columns, kind="stable")
        aggregation = {column: "mean" for column in numeric_columns}
        aggregation.update({column: "first" for column in other_columns})
        climate = climate.groupby(list(settings.climate_join_keys), as_index=False, sort=False).agg(aggregation)

    overlapping_metadata = [
        column
        for column in ("trench_id", settings.datetime_column)
        if column in climate.columns
        and column not in settings.climate_join_keys
        and column not in required_columns
    ]
    if overlapping_metadata:
        climate = climate.drop(columns=overlapping_metadata)
    return climate


def load_trenches(settings: SensorAnalysisSettings) -> pd.DataFrame:
    """Load trench to river-system mappings."""
    trenches = pd.read_parquet(settings.trenches_path).reset_index(drop=True)
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
        climate=load_climate_data(settings),
        trenches=load_trenches(settings),
        transformations=load_transformations(settings),
    )


__all__ = [
    "AnalysisInputs",
    "load_analysis_inputs",
    "load_climate_data",
    "load_land_cover",
    "load_sensor_data",
    "load_transformations",
    "load_trenches",
    "validate_required_columns",
]
