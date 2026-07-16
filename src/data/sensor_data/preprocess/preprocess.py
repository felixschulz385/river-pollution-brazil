import json
import logging
from pathlib import Path

from ..fetch.paths import get_sensor_database_path, get_water_quality_dir
from .clean import clean_measurement_values, resolve_measurement_columns
from .rename import rename_portuguese_fields
from .settings import (
    AUXILIARY_WATER_QUALITY_COLUMNS,
    CLEAN_STREAMFLOW_PARQUET,
    CLEAN_WATER_QUALITY_PARQUET,
    CLEANING_FLAGS_PARQUET,
    CLEANING_SUMMARY_PARQUET,
    DATETIME_COLUMN,
    EXACT_SENTINELS,
    LOG_TRANSFORM_SKEW_THRESHOLD,
    LOG_TRANSFORM_TAIL_RATIO,
    MIN_TRANSFORM_N,
    PREPROCESS_OUTPUT_DIRNAME,
    STATIONS_RIVERS_COLUMNS,
    STATIONS_RIVERS_PARQUET,
    STATIONS_RIVERS_TABLE,
    STREAMFLOW_BAD_STATION_SENTINEL_SHARE,
    STREAMFLOW_DAILY_SCHEMA,
    STREAMFLOW_JUMP_RATIO_THRESHOLD,
    STREAMFLOW_LONG_CONSTANT_RUN_DAYS,
    STREAMFLOW_MAX_VALID_DISCHARGE,
    STREAMFLOW_MISSING_SENTINEL,
    STREAMFLOW_SMALL_LAG_THRESHOLD,
    STREAMFLOW_SOURCE_TABLES,
    STREAMFLOW_SUSPICIOUS_CONSTANT_RUN_DAYS,
    TRANSFORMATION_RECOMMENDATIONS_JSON,
    WATER_QUALITY_SOURCE_TABLES,
)

logger = logging.getLogger(__name__)


def _time_value_to_timedelta(value):
    import datetime as dt

    import pandas as pd

    if pd.isna(value):
        return pd.NaT
    if isinstance(value, pd.Timedelta):
        return value
    if isinstance(value, dt.timedelta):
        return pd.Timedelta(value)
    if isinstance(value, (pd.Timestamp, dt.datetime)):
        return pd.Timedelta(
            hours=value.hour,
            minutes=value.minute,
            seconds=value.second,
            microseconds=value.microsecond,
        )
    if isinstance(value, dt.time):
        return pd.Timedelta(
            hours=value.hour,
            minutes=value.minute,
            seconds=value.second,
            microseconds=value.microsecond,
        )
    if isinstance(value, (int, float)) and 0 <= value < 1:
        return pd.Timedelta(days=float(value))

    text_value = str(value).strip()
    if not text_value:
        return pd.NaT
    parsed_datetime = pd.to_datetime(text_value, errors="coerce")
    if pd.notna(parsed_datetime):
        return pd.Timedelta(
            hours=parsed_datetime.hour,
            minutes=parsed_datetime.minute,
            seconds=parsed_datetime.second,
            microseconds=parsed_datetime.microsecond,
        )
    return pd.to_timedelta(text_value, errors="coerce")


def _merge_date_time_columns(frame):
    """Build a single timestamp from date's date part and time's time part."""
    if "date" not in frame.columns or "time" not in frame.columns:
        return frame

    import pandas as pd

    merged = frame.copy()
    date_position = merged.columns.get_loc("date")
    date_part = pd.to_datetime(merged["date"], errors="coerce").dt.normalize()
    time_part = merged["time"].map(_time_value_to_timedelta)
    merged[DATETIME_COLUMN] = date_part + time_part
    columns = list(merged.columns)
    columns.insert(date_position, columns.pop(columns.index(DATETIME_COLUMN)))
    return merged.loc[:, columns]


def _drop_auxiliary_columns(frame):
    return frame.drop(
        columns=[column for column in AUXILIARY_WATER_QUALITY_COLUMNS if column in frame],
    )


def _quality_flag_rank(values):
    import pandas as pd

    text_values = values.astype("string").str.strip().str.upper()
    ranks = pd.Series(0.0, index=values.index)
    ranks = ranks.mask(text_values.isin({"OK", "GOOD", "VALID", "VALIDO"}), 1000)
    ranks = ranks.mask(
        text_values.isin({"CONSISTED", "CONSISTIDO", "CONSISTIDA", "APPROVED", "APROVADO"}),
        1000,
    )
    ranks = ranks.mask(text_values.isin({"SUSPECT", "SUSPEITO", "REVIEW"}), 100)
    numeric_values = pd.to_numeric(values, errors="coerce")
    numeric_ranks = 999 - numeric_values
    return ranks.mask(numeric_values.notna(), numeric_ranks).fillna(0)


def _streamflow_daily_frames(frame):
    import pandas as pd

    daily_frames = []
    for day in range(1, 32):
        discharge_column = f"discharge_{day}"
        if discharge_column not in frame.columns:
            continue
        status_column = f"discharge_{day}_status"
        daily_frame = frame.loc[
            :,
            [
                "station_code",
                "date",
                "consistency_level",
                "streamflow_acquisition_method",
                "updated_at",
                discharge_column,
            ],
        ].copy()
        daily_frame["day"] = day
        daily_frame["discharge"] = pd.to_numeric(
            daily_frame[discharge_column],
            errors="coerce",
        )
        if status_column in frame.columns:
            daily_frame["quality_flag"] = frame[status_column]
        else:
            daily_frame["quality_flag"] = pd.NA
        daily_frames.append(
            daily_frame.drop(columns=[discharge_column])
        )

    if not daily_frames:
        raise ValueError("No streamflow discharge_1 ... discharge_31 columns found.")
    return pd.concat(daily_frames, ignore_index=True)


def _apply_streamflow_cleaning_rules(daily):
    import pandas as pd

    cleaned = daily.copy()
    sentinel_mask = cleaned["discharge"].eq(STREAMFLOW_MISSING_SENTINEL)
    observed_count = (
        cleaned["discharge"]
        .notna()
        .groupby(cleaned["station_code"])
        .transform("sum")
        .astype(float)
    )
    sentinel_count = sentinel_mask.groupby(cleaned["station_code"]).transform("sum")
    sentinel_share = sentinel_count / observed_count.mask(observed_count == 0)
    cleaned["flag_bad_station"] = sentinel_share > STREAMFLOW_BAD_STATION_SENTINEL_SHARE
    cleaned.loc[sentinel_mask, "discharge"] = pd.NA

    cleaned = cleaned.sort_values(
        by=["station_code", "date"],
        kind="mergesort",
    ).copy()
    run_break = (
        cleaned["station_code"].ne(cleaned["station_code"].shift())
        | cleaned["discharge"].ne(cleaned["discharge"].shift())
        | cleaned["discharge"].isna()
    )
    cleaned["_constant_run_id"] = run_break.cumsum()
    cleaned["_constant_run_length"] = cleaned.groupby(
        ["station_code", "_constant_run_id"],
        observed=True,
    )["discharge"].transform("size")

    nonzero_constant_run = cleaned["discharge"].notna() & cleaned["discharge"].ne(0)
    zero_constant_run = cleaned["discharge"].eq(0)
    long_constant_run = (
        (nonzero_constant_run | zero_constant_run)
        & (cleaned["_constant_run_length"] > STREAMFLOW_LONG_CONSTANT_RUN_DAYS)
    )
    cleaned["flag_constant_run_suspicious"] = (
        nonzero_constant_run
        & (cleaned["_constant_run_length"] >= STREAMFLOW_SUSPICIOUS_CONSTANT_RUN_DAYS)
        & (cleaned["_constant_run_length"] <= STREAMFLOW_LONG_CONSTANT_RUN_DAYS)
    )
    cleaned.loc[long_constant_run, "discharge"] = pd.NA

    lag_discharge = cleaned.groupby("station_code", observed=True)["discharge"].shift()
    positive_small_lag = (
        (lag_discharge > 0)
        & (lag_discharge < STREAMFLOW_SMALL_LAG_THRESHOLD)
    )
    cleaned["flag_jump_suspicious"] = (
        lag_discharge.notna()
        & (
            (
                positive_small_lag
                & ((cleaned["discharge"] / lag_discharge) > STREAMFLOW_JUMP_RATIO_THRESHOLD)
            )
            | (lag_discharge.eq(0) & (cleaned["discharge"] > 0))
        )
    )
    return cleaned.drop(columns=["_constant_run_id", "_constant_run_length"])


def build_clean_streamflow_daily(frame):
    """Return one regression-ready streamflow row per station-day."""
    import pandas as pd

    streamflow = rename_portuguese_fields(frame)
    missing_required = [
        column for column in ["station_code", "date"] if column not in streamflow.columns
    ]
    if missing_required:
        raise KeyError(
            "Missing required streamflow column(s): " + ", ".join(missing_required)
        )
    for column in [
        "consistency_level",
        "streamflow_acquisition_method",
        "updated_at",
    ]:
        if column not in streamflow.columns:
            streamflow[column] = pd.NA

    daily = _streamflow_daily_frames(streamflow)
    month_start = pd.to_datetime(daily["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    daily["date"] = month_start + pd.to_timedelta(daily["day"] - 1, unit="D")
    valid_calendar_day = daily["date"].dt.month.eq(month_start.dt.month)
    daily = daily.loc[valid_calendar_day].copy()
    daily = _apply_streamflow_cleaning_rules(daily)
    invalid_discharge = daily["discharge"].notna() & (
        (daily["discharge"] < 0)
        | (daily["discharge"] > STREAMFLOW_MAX_VALID_DISCHARGE)
        | daily["discharge"].isin(EXACT_SENTINELS)
    )
    daily.loc[invalid_discharge, "discharge"] = pd.NA

    daily["_discharge_present_rank"] = daily["discharge"].notna().astype(int)
    daily["_consistency_rank"] = pd.to_numeric(
        daily["consistency_level"],
        errors="coerce",
    ).fillna(-1)
    daily["_quality_rank"] = _quality_flag_rank(daily["quality_flag"])
    daily["_streamflow_cleaning_rank"] = -(
        daily["flag_bad_station"].astype(int)
        + daily["flag_constant_run_suspicious"].astype(int)
        + daily["flag_jump_suspicious"].astype(int)
    )
    daily["_updated_at_rank"] = pd.to_datetime(
        daily["updated_at"],
        errors="coerce",
    ).fillna(pd.Timestamp.min)
    daily = daily.sort_values(
        by=[
            "station_code",
            "date",
            "_consistency_rank",
            "_discharge_present_rank",
            "_quality_rank",
            "_streamflow_cleaning_rank",
            "_updated_at_rank",
        ],
        ascending=[True, True, False, False, False, False, False],
        kind="mergesort",
    )
    daily = daily.drop_duplicates(subset=["station_code", "date"], keep="first")
    return daily.loc[:, STREAMFLOW_DAILY_SCHEMA].reset_index(drop=True)


def _quote_identifier(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _available_tables(connection) -> set[str]:
    tables = connection.execute(
        "SELECT table_name FROM information_schema.tables"
    ).fetchdf()
    return set(tables["table_name"])


def _resolve_source_table(connection) -> str:
    return _resolve_first_available_table(connection, WATER_QUALITY_SOURCE_TABLES)


def _resolve_first_available_table(connection, table_names) -> str:
    tables = _available_tables(connection)
    for table_name in table_names:
        if table_name in tables:
            return table_name
    raise ValueError(
        "No source table found. Expected one of: " + ", ".join(table_names)
    )


def _read_source_table(root_dir="."):
    import duckdb

    database_path = get_sensor_database_path(root_dir)
    if not Path(database_path).exists():
        raise FileNotFoundError(f"Sensor database not found: {database_path}")
    with duckdb.connect(str(database_path)) as connection:
        source_table = _resolve_source_table(connection)
        frame = connection.execute(
            f"SELECT * FROM {_quote_identifier(source_table)}"
        ).fetchdf()
    return source_table, frame


def _read_table(root_dir=".", table_name=None):
    import duckdb

    database_path = get_sensor_database_path(root_dir)
    if not Path(database_path).exists():
        raise FileNotFoundError(f"Sensor database not found: {database_path}")
    with duckdb.connect(str(database_path)) as connection:
        if table_name not in _available_tables(connection):
            raise ValueError(f"Table {table_name!r} not found in {database_path}.")
        return connection.execute(f"SELECT * FROM {_quote_identifier(table_name)}").fetchdf()


def _read_first_available_table(root_dir=".", table_names=None):
    import duckdb

    database_path = get_sensor_database_path(root_dir)
    if not Path(database_path).exists():
        raise FileNotFoundError(f"Sensor database not found: {database_path}")
    with duckdb.connect(str(database_path)) as connection:
        source_table = _resolve_first_available_table(connection, table_names)
        frame = connection.execute(
            f"SELECT * FROM {_quote_identifier(source_table)}"
        ).fetchdf()
    return source_table, frame


def _preprocess_output_dir(root_dir=".") -> Path:
    output_dir = get_water_quality_dir(root_dir) / PREPROCESS_OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def preprocess_stations_rivers(root_dir=".") -> str:
    """Export a compact GeoParquet version of the station-to-river table."""
    import geopandas as gpd

    stations = _read_table(root_dir, STATIONS_RIVERS_TABLE)
    missing_columns = [
        column for column in STATIONS_RIVERS_COLUMNS if column not in stations.columns
    ]
    if missing_columns:
        raise KeyError(
            "Missing required stations_rivers column(s): "
            + ", ".join(missing_columns)
        )

    stations = stations.loc[:, STATIONS_RIVERS_COLUMNS].copy()
    geometry = gpd.GeoSeries.from_wkt(stations["geometry_wkt"], crs=4326)
    stations_geo = gpd.GeoDataFrame(stations, geometry=geometry, crs=4326)
    output_path = get_water_quality_dir(root_dir) / STATIONS_RIVERS_PARQUET
    stations_geo.to_parquet(output_path, index=False)
    logger.info("Wrote cleaned stations-rivers GeoParquet to %s.", output_path)
    return str(output_path)


def _json_scalar(value):
    if value is None:
        return None
    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except (ImportError, TypeError, ValueError):
        pass
    try:
        if value != value:
            return None
    except TypeError:
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def _recommend_transform(variable_summary: dict) -> tuple[str, str, str]:
    n = _json_scalar(variable_summary.get("n")) or 0
    min_value = _json_scalar(variable_summary.get("min"))
    q1 = _json_scalar(variable_summary.get("q1"))
    median = _json_scalar(variable_summary.get("median"))
    p99 = _json_scalar(variable_summary.get("p99"))
    skew = _json_scalar(variable_summary.get("skew"))
    share_negative = _json_scalar(variable_summary.get("share_negative")) or 0

    if n < MIN_TRANSFORM_N:
        return "identity", "x", "Too few observations for an automatic nonlinear transform."
    if min_value is not None and min_value > 0:
        tail_ratio = (p99 / median) if p99 is not None and median not in (None, 0) else None
        if (
            (skew is not None and skew >= LOG_TRANSFORM_SKEW_THRESHOLD)
            or (tail_ratio is not None and tail_ratio >= LOG_TRANSFORM_TAIL_RATIO)
        ):
            return "log10", "log10(x)", "Positive right-tailed distribution."
    if min_value is not None and min_value >= 0:
        tail_ratio = (p99 / q1) if p99 is not None and q1 not in (None, 0) else None
        if (
            (skew is not None and skew >= LOG_TRANSFORM_SKEW_THRESHOLD)
            or (tail_ratio is not None and tail_ratio >= LOG_TRANSFORM_TAIL_RATIO)
        ):
            return "log10_1p", "log10(1 + x)", "Nonnegative right-tailed distribution."
    if share_negative > 0:
        return "identity", "x", "Contains negative values; no default log transform recommended."
    return "identity", "x", "No automatic nonlinear transform recommended."


def _build_transformation_recommendations(summary, measurement_columns: dict[str, str]) -> dict:
    summary_lookup = summary.set_index("variable").to_dict(orient="index")
    recommendations = {}
    for variable, source_column in sorted(measurement_columns.items()):
        variable_summary = summary_lookup.get(variable, {})
        transform, expression, reason = _recommend_transform(variable_summary)
        recommendations[variable] = {
            "column": source_column,
            "recommended_transform": transform,
            "expression": expression,
            "apply_to": "analysis",
            "reason": reason,
            "statistics": {
                "n": _json_scalar(variable_summary.get("n")),
                "n_sensors": _json_scalar(variable_summary.get("n_sensors")),
                "min": _json_scalar(variable_summary.get("min")),
                "q1": _json_scalar(variable_summary.get("q1")),
                "median": _json_scalar(variable_summary.get("median")),
                "q3": _json_scalar(variable_summary.get("q3")),
                "p99": _json_scalar(variable_summary.get("p99")),
                "max": _json_scalar(variable_summary.get("max")),
                "skew": _json_scalar(variable_summary.get("skew")),
                "share_zero": _json_scalar(variable_summary.get("share_zero")),
                "share_negative": _json_scalar(variable_summary.get("share_negative")),
                "iqr": _json_scalar(variable_summary.get("iqr")),
            },
        }

    return {
        "schema_version": 1,
        "clean_data_file": CLEAN_WATER_QUALITY_PARQUET,
        "recommendations": recommendations,
    }


def preprocess_sensor_data(root_dir=".") -> dict[str, str]:
    """Rename, clean, and export sensor data as parquet files."""
    source_table, raw_frame = _read_source_table(root_dir)
    renamed_frame = rename_portuguese_fields(raw_frame)
    measurement_columns = resolve_measurement_columns(renamed_frame)
    if not measurement_columns:
        raise ValueError(f"No configured pollution variables found in {source_table}.")

    clean_frame, flags, summary = clean_measurement_values(
        renamed_frame,
        measurement_columns,
    )
    clean_frame = _drop_auxiliary_columns(_merge_date_time_columns(clean_frame))
    water_quality_dir = get_water_quality_dir(root_dir)
    output_dir = _preprocess_output_dir(root_dir)
    clean_path = water_quality_dir / CLEAN_WATER_QUALITY_PARQUET
    recommendations_path = water_quality_dir / TRANSFORMATION_RECOMMENDATIONS_JSON
    flags_path = output_dir / CLEANING_FLAGS_PARQUET
    summary_path = output_dir / CLEANING_SUMMARY_PARQUET

    clean_frame.to_parquet(clean_path, index=False)
    flags.to_parquet(flags_path, index=False)
    summary.to_parquet(summary_path, index=False)
    recommendations = _build_transformation_recommendations(summary, measurement_columns)
    recommendations_path.write_text(
        json.dumps(recommendations, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    removed_count = int((flags["cleaning_label"] != "OK").sum())
    logger.info(
        "Wrote cleaned water-quality parquet to %s with %s invalid value(s) set to NA.",
        clean_path,
        removed_count,
    )
    return {
        "clean": str(clean_path),
        "transformations": str(recommendations_path),
        "flags": str(flags_path),
        "summary": str(summary_path),
    }


def preprocess_streamflow(root_dir=".") -> str:
    """Clean monthly wide streamflow records into a daily regression-ready panel."""
    source_table, raw_frame = _read_first_available_table(
        root_dir,
        STREAMFLOW_SOURCE_TABLES,
    )
    clean_streamflow = build_clean_streamflow_daily(raw_frame)
    output_path = get_water_quality_dir(root_dir) / CLEAN_STREAMFLOW_PARQUET
    clean_streamflow.to_parquet(output_path, index=False)
    logger.info(
        "Wrote cleaned streamflow parquet to %s with %s station-day row(s) from %s.",
        output_path,
        len(clean_streamflow),
        source_table,
    )
    return str(output_path)


def preprocess_all(root_dir=".") -> dict[str, str]:
    """Run all sensor-data preprocessing exports."""
    outputs = preprocess_sensor_data(root_dir=root_dir)
    outputs["streamflow"] = preprocess_streamflow(root_dir=root_dir)
    outputs["stations_rivers"] = preprocess_stations_rivers(root_dir=root_dir)
    return outputs
