import pandas as pd

from .settings import (
    EXACT_SENTINELS,
    EXTREME_REVIEW_ONLY_VARIABLES,
    GIANT_VS_P99_REMOVE_MULTIPLIER,
    GLOBAL_RANGE_SENTINELS,
    IQR_MULTIPLIER,
    MICROBIOLOGY_HIGH_THRESHOLD,
    MICROBIOLOGY_VARIABLES,
    NONNEGATIVE_VARIABLES,
    NON_MEASUREMENT_COLUMNS,
    PHYSICAL_LIMITS,
    P99_MULTIPLIER,
    RANGE_SENTINELS,
    SUMMARY_QUANTILES,
    VERY_LARGE_SUSPECT_THRESHOLDS,
)


SUMMARY_COLUMNS = [
    "variable",
    "n",
    "n_sensors",
    "min",
    "q1",
    "median",
    "q3",
    "p99",
    "max",
    "mean",
    "std",
    "skew",
    "share_zero",
    "share_negative",
    "iqr",
    "upper_iqr",
]


def first_present(candidates, available_columns: set[str]) -> str | None:
    for candidate in candidates:
        if candidate in available_columns:
            return candidate
    return None


def resolve_measurement_columns(frame: pd.DataFrame) -> dict[str, str]:
    measurement_columns = {}
    for column in frame.columns:
        if column in NON_MEASUREMENT_COLUMNS:
            continue
        numeric_values = pd.to_numeric(frame[column], errors="coerce")
        if numeric_values.notna().any():
            measurement_columns[column] = column
    return measurement_columns


def classify_value(variable: str, value) -> str:
    if pd.isna(value):
        return "OK"
    if value in EXACT_SENTINELS:
        return "REMOVE_CODED"
    for lower, upper in GLOBAL_RANGE_SENTINELS:
        if lower <= value <= upper:
            return "REMOVE_CODED"
    if variable in NONNEGATIVE_VARIABLES and value < 0:
        return "REMOVE_INVALID"
    if variable in PHYSICAL_LIMITS:
        lower, upper = PHYSICAL_LIMITS[variable]
        if value < lower or value > upper:
            return "REMOVE_INVALID"
    for lower, upper in RANGE_SENTINELS.get(variable, ()):
        if lower <= value <= upper:
            return "REMOVE_CODED"
    if (
        variable in VERY_LARGE_SUSPECT_THRESHOLDS
        and value >= VERY_LARGE_SUSPECT_THRESHOLDS[variable]
    ):
        return "REMOVE_CODED"
    return "OK"


def summarize_measurements(long_frame: pd.DataFrame) -> pd.DataFrame:
    if long_frame.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    summary = (
        long_frame.groupby("variable", observed=True)
        .agg(
            n=("value", "count"),
            n_sensors=("station_code", lambda series: series.dropna().astype(str).nunique()),
            min=("value", "min"),
            q1=("value", lambda series: series.quantile(SUMMARY_QUANTILES[0])),
            median=("value", "median"),
            q3=("value", lambda series: series.quantile(SUMMARY_QUANTILES[1])),
            p99=("value", lambda series: series.quantile(SUMMARY_QUANTILES[2])),
            max=("value", "max"),
            mean=("value", "mean"),
            std=("value", "std"),
            skew=("value", "skew"),
            share_zero=("value", lambda series: (series == 0).mean()),
            share_negative=("value", lambda series: (series < 0).mean()),
        )
        .reset_index()
    )
    summary["iqr"] = summary["q3"] - summary["q1"]
    summary["upper_iqr"] = summary["q3"] + IQR_MULTIPLIER * summary["iqr"]
    return summary


def _mark_giant_values_as_coded(flags: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    if flags.empty or summary.empty:
        return flags

    updated = flags.drop(columns=[column for column in SUMMARY_COLUMNS if column in flags.columns and column != "variable"])
    updated = updated.merge(summary, on="variable", how="left")
    giant_mask = (
        (updated["cleaning_label"] == "OK")
        & (~updated["variable"].isin(EXTREME_REVIEW_ONLY_VARIABLES))
        & (updated["p99"] > 0)
        & (updated["value"] > GIANT_VS_P99_REMOVE_MULTIPLIER * updated["p99"])
    )
    updated.loc[giant_mask, "cleaning_label"] = "REMOVE_CODED"
    return updated


def build_cleaning_flags(frame: pd.DataFrame, measurement_columns: dict[str, str]) -> pd.DataFrame:
    value_columns = list(measurement_columns.values())
    long_frame = frame[value_columns].rename(
        columns={column: variable for variable, column in measurement_columns.items()}
    )
    long_frame = long_frame.melt(var_name="variable", value_name="value", ignore_index=False)
    long_frame = long_frame.reset_index(names="row_id")
    if "station_code" in frame.columns:
        long_frame["station_code"] = long_frame["row_id"].map(frame["station_code"])
    else:
        long_frame["station_code"] = pd.NA
    long_frame["value"] = pd.to_numeric(long_frame["value"], errors="coerce")
    finite_values = long_frame["value"].notna() & ~long_frame["value"].isin(
        [float("inf"), float("-inf")]
    )
    flags = long_frame.loc[finite_values].copy()
    flags["cleaning_label"] = flags.apply(
        lambda row: classify_value(row["variable"], row["value"]),
        axis=1,
    )
    summary = summarize_measurements(flags.loc[flags["cleaning_label"] == "OK"].copy())
    flags = _mark_giant_values_as_coded(flags, summary)
    summary = summarize_measurements(flags.loc[flags["cleaning_label"] == "OK"].copy())
    flags = flags.drop(
        columns=[column for column in SUMMARY_COLUMNS if column in flags.columns and column != "variable"]
    ).merge(summary, on="variable", how="left")
    flags["flag_extreme_review"] = (
        (flags["cleaning_label"] == "OK")
        & (
            (flags["value"] > flags["upper_iqr"])
            | ((flags["p99"] > 0) & (flags["value"] > P99_MULTIPLIER * flags["p99"]))
        )
    )
    flags["flag_microbio_high"] = (
        (flags["cleaning_label"] == "OK")
        & flags["variable"].isin(MICROBIOLOGY_VARIABLES)
        & (flags["value"] > MICROBIOLOGY_HIGH_THRESHOLD)
    )
    flags["source_column"] = flags["variable"].map(
        {variable: column for variable, column in measurement_columns.items()}
    )
    return flags


def clean_measurement_values(
    frame: pd.DataFrame,
    measurement_columns: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    flags = build_cleaning_flags(frame, measurement_columns)
    clean_frame = frame.copy()
    remove_flags = flags.loc[flags["cleaning_label"] != "OK", ["row_id", "source_column"]]
    for source_column, group in remove_flags.groupby("source_column", observed=True):
        clean_frame.loc[group["row_id"], source_column] = pd.NA

    label_counts = (
        flags.groupby(["variable", "cleaning_label"], observed=True)
        .size()
        .rename("row_count")
        .reset_index()
    )
    review_counts = (
        flags.groupby("variable", observed=True)
        .agg(
            extreme_review=("flag_extreme_review", "sum"),
            microbio_high=("flag_microbio_high", "sum"),
        )
        .reset_index()
    )
    label_wide = label_counts.pivot_table(
        index="variable",
        columns="cleaning_label",
        values="row_count",
        fill_value=0,
        aggfunc="sum",
    ).reset_index()
    ok_flags = flags.loc[flags["cleaning_label"] == "OK"].copy()
    if ok_flags.empty:
        summary = pd.DataFrame({"variable": sorted(flags["variable"].unique())})
    else:
        summary = summarize_measurements(ok_flags)
    summary = (
        pd.DataFrame({"variable": sorted(flags["variable"].unique())})
        .merge(summary, on="variable", how="left")
        .merge(review_counts, on="variable", how="left")
        .merge(label_wide, on="variable", how="left")
    )
    summary.columns = [str(column) for column in summary.columns]
    return clean_frame, flags.reset_index(drop=True), summary
