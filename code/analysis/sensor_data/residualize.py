"""MAP residualization utilities for fixed-effects absorption."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MapResidualizationResult:
    """Residualized outcome/regressors plus convergence metadata."""

    frame: pd.DataFrame
    outcome_column: str
    feature_columns: tuple[str, ...]
    fixed_effect_columns: tuple[str, ...]
    nobs: int
    iterations: int
    converged: bool
    max_change: float


def residualize_with_map(
    sample: pd.DataFrame,
    *,
    outcome_column: str,
    feature_columns: list[str] | tuple[str, ...],
    fixed_effect_columns: list[str] | tuple[str, ...],
    tolerance: float,
    max_iterations: int,
) -> MapResidualizationResult:
    """Residualize outcome and regressors with alternating projections."""
    numeric_columns = [outcome_column, *feature_columns]
    work = sample.loc[:, numeric_columns].astype(float).copy()
    group_keys = {
        column: sample[column]
        for column in fixed_effect_columns
    }

    converged = False
    max_change = float("inf")
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        previous = work.to_numpy(copy=True)
        for fixed_effect in fixed_effect_columns:
            demeaned = work.groupby(group_keys[fixed_effect], sort=False).transform("mean")
            work = work - demeaned
        max_change = float(np.nanmax(np.abs(work.to_numpy() - previous)))
        if not np.isfinite(max_change):
            max_change = 0.0
        if max_change <= tolerance:
            converged = True
            break

    residualized = pd.concat(
        [
            work,
            sample.loc[:, list(dict.fromkeys(fixed_effect_columns))].reset_index(drop=True),
        ],
        axis=1,
    )
    return MapResidualizationResult(
        frame=residualized,
        outcome_column=outcome_column,
        feature_columns=tuple(feature_columns),
        fixed_effect_columns=tuple(fixed_effect_columns),
        nobs=int(sample.shape[0]),
        iterations=iterations,
        converged=converged,
        max_change=max_change,
    )


__all__ = ["MapResidualizationResult", "residualize_with_map"]
