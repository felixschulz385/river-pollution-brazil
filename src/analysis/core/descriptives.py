"""Descriptive-statistics helpers shared by the `output/notebooks/*/01_descriptives` notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pyfixest.estimation import demean as _pyfixest_demean


def demean(frame, columns, fixed_effects):
    """Residualize `columns` on the (possibly multi-way) `fixed_effects` columns.

    Rows with a missing value in any of `columns` or `fixed_effects` are dropped,
    so callers wanting per-column samples should pass one column at a time.
    """
    columns = list(columns)
    fixed_effects = list(fixed_effects)
    data = frame[columns + fixed_effects].dropna()
    codes = np.column_stack(
        [pd.factorize(data[fe])[0] for fe in fixed_effects]
    ).astype(np.uint64)
    residuals, converged = _pyfixest_demean(
        data[columns].to_numpy(dtype=np.float64),
        codes,
        np.ones(len(data), dtype=np.float64),
    )
    if not converged:
        raise RuntimeError(f"Demeaning on {fixed_effects} did not converge.")
    return pd.DataFrame(residuals, index=data.index, columns=columns)


def variation_table(frame, columns, fixed_effect_sets):
    """SD of each column overall and after absorbing each named fixed-effect set.

    `fixed_effect_sets` maps a label (e.g. "station + year") to a list of FE
    columns. The `share:` columns give the fraction of total variance left
    after absorbing that set -- the identifying variation under that design.
    """
    rows = {}
    for column in columns:
        values = frame[column].dropna()
        total_var = values.var()
        row = {"n": len(values), "sd_total": np.sqrt(total_var)}
        for label, fixed_effects in fixed_effect_sets.items():
            residual = demean(frame, [column], fixed_effects)[column]
            row[f"sd: {label}"] = residual.std()
            row[f"share: {label}"] = residual.var() / total_var if total_var > 0 else np.nan
        rows[column] = row
    return pd.DataFrame(rows).T


def distribution_table(frame, columns):
    """Shape statistics that inform transformation, trimming and censoring choices.

    `modal_share` is the share of observations at the single most common
    value -- heaping there usually marks a detection or reporting limit.
    """
    rows = {}
    for column in columns:
        values = frame[column].dropna()
        if values.empty:
            continue
        positive = values[values > 0]
        counts = values.value_counts(normalize=True)
        rows[column] = {
            "n": len(values),
            "share_zero": (values == 0).mean(),
            "share_negative": (values < 0).mean(),
            "n_distinct": values.nunique(),
            "modal_value": counts.index[0],
            "modal_share": counts.iloc[0],
            "min": values.min(),
            "p01": values.quantile(0.01),
            "median": values.median(),
            "p99": values.quantile(0.99),
            "p999": values.quantile(0.999),
            "max": values.max(),
            "skew": values.skew(),
            "skew_log_positive": np.log(positive).skew() if len(positive) > 2 else np.nan,
        }
    return pd.DataFrame(rows).T


def binscatter(x, y, n_bins=20):
    """Mean of `y` within quantile bins of `x` (both aligned Series)."""
    data = pd.DataFrame({"x": x, "y": y}).dropna()
    data["bin"] = pd.qcut(data["x"], n_bins, duplicates="drop")
    return data.groupby("bin", observed=True).agg(x=("x", "mean"), y=("y", "mean"), n=("y", "size"))


def effective_cluster_count(cluster_sizes):
    """Inverse Herfindahl of cluster sizes: the number of equal-sized clusters with the same concentration."""
    shares = np.asarray(cluster_sizes, dtype=float)
    shares = shares / shares.sum()
    return 1.0 / np.square(shares).sum()
