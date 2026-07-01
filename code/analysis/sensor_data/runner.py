"""Execution engine for the sensor analysis suite."""

from __future__ import annotations

import logging
import os
import tempfile
import warnings
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV

from .groups import select_pollutants
from .prepare import build_analysis_data
from .residualize import residualize_with_map
from .results import SensorAnalysisRun, manifest_record, pollutant_lookup, save_run, tidy_to_records
from .specs import build_model_specs
from ..settings import DEFAULT_SETTINGS, SensorAnalysisSettings

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import pyfixest as pf  # noqa: E402


logger = logging.getLogger(__name__)


def _configure_runtime_warnings() -> None:
    """Silence known noisy pyfixest warnings during large batch runs."""
    warnings.filterwarnings(
        "ignore",
        message=r"[\s\S]*singleton fixed effect\(s\) dropped from the model[\s\S]*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"[\s\S]*variables dropped due to multicollinearity[\s\S]*",
        category=UserWarning,
    )


def _coerce_settings(
    settings: SensorAnalysisSettings,
    *,
    output_dir: str | Path | None = None,
    minimum_observations: int | None = None,
) -> SensorAnalysisSettings:
    updated = settings
    if output_dir is not None:
        updated = replace(updated, output_dir=Path(output_dir))
    if minimum_observations is not None:
        updated = replace(updated, minimum_observations=minimum_observations)
    return updated


def _slugify(value: str) -> str:
    slug = []
    for character in value.lower():
        if character.isalnum():
            slug.append(character)
        else:
            slug.append("_")
    result = "".join(slug).strip("_")
    while "__" in result:
        result = result.replace("__", "_")
    return result or "run"


def _resolve_model_name(
    *,
    pollutant_group_kind: str,
    pollutant_group: str,
    pollutants: list[str] | None,
) -> str:
    if pollutants:
        if len(pollutants) == 1:
            return f"pollutant_{_slugify(pollutants[0])}"
        return "pollutant_custom"
    return f"{_slugify(pollutant_group_kind)}_{_slugify(pollutant_group)}"


def _analysis_columns(settings: SensorAnalysisSettings, spec) -> list[str]:
    columns = [
        spec.outcome_column,
        *spec.forced_regressor_columns,
        *spec.candidate_regressor_columns,
        *settings.resolved_fixed_effects(),
        settings.cluster_variable,
    ]
    return list(dict.fromkeys(columns))


def _ols_formula(outcome_column: str, regressors: tuple[str, ...]) -> str:
    return f"{outcome_column} ~ {' + '.join(regressors)} - 1"


def _prepare_sample(settings: SensorAnalysisSettings, frame: pd.DataFrame, spec) -> pd.DataFrame:
    sample = frame.loc[:, _analysis_columns(settings, spec)].dropna().copy()
    if sample.empty:
        raise ValueError("No complete observations remain after dropping missing values.")
    if sample[spec.outcome_column].nunique(dropna=True) < 2:
        raise ValueError("Outcome has no variation after filtering.")
    if all(sample[column].nunique(dropna=True) < 2 for column in spec.coefficient_columns):
        raise ValueError("All land-cover regressors are constant after filtering.")
    return sample


def _run_ols(
    settings: SensorAnalysisSettings,
    sample: pd.DataFrame,
    spec,
    regressors: tuple[str, ...],
):
    formula = _ols_formula(spec.outcome_column, regressors)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"[\s\S]*singleton fixed effect\(s\) dropped from the model[\s\S]*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"[\s\S]*variables dropped due to multicollinearity[\s\S]*",
            category=UserWarning,
        )
        fit = pf.feols(
            formula,
            vcov={settings.vcov_type: settings.cluster_variable},
            data=sample.loc[:, [spec.outcome_column, *regressors, settings.cluster_variable]],
        )
    return fit.tidy(), formula


def _standardize_candidates(candidate_frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    valid_columns: list[str] = []
    standardized_arrays: list[np.ndarray] = []
    for column in candidate_frame.columns:
        values = candidate_frame[column].to_numpy(dtype=float)
        std = float(np.std(values))
        if not np.isfinite(std) or std <= 0:
            continue
        centered = values - float(np.mean(values))
        standardized_arrays.append(centered / std)
        valid_columns.append(column)
    if not standardized_arrays:
        return np.empty((candidate_frame.shape[0], 0)), []
    return np.column_stack(standardized_arrays), valid_columns


def _run_model(settings: SensorAnalysisSettings, frame: pd.DataFrame, spec) -> tuple[pd.DataFrame, int, dict[str, object]]:
    sample = _prepare_sample(settings, frame, spec)
    regressors = tuple(spec.forced_regressor_columns)
    residualized = residualize_with_map(
        sample,
        outcome_column=spec.outcome_column,
        feature_columns=[*spec.forced_regressor_columns, *spec.candidate_regressor_columns],
        fixed_effect_columns=settings.resolved_fixed_effects(),
        tolerance=settings.map_tolerance,
        max_iterations=settings.map_max_iterations,
    )
    demeaned = residualized.frame.copy()
    demeaned[settings.cluster_variable] = sample[settings.cluster_variable].to_numpy()

    selected_terms: tuple[str, ...] = ()
    lasso_alpha: float | None = None
    lasso_selected_count: int | None = None

    if spec.model_family == "post_lasso" and spec.candidate_regressor_columns:
        candidate_frame = demeaned.loc[:, list(spec.candidate_regressor_columns)]
        candidate_matrix, valid_columns = _standardize_candidates(candidate_frame)
        if valid_columns:
            lasso = LassoCV(
                cv=settings.lasso_settings.cv,
                alphas=settings.lasso_settings.alphas,
                eps=settings.lasso_settings.eps,
                random_state=settings.lasso_settings.random_state,
                max_iter=settings.lasso_settings.max_iter,
                fit_intercept=False,
            )
            lasso.fit(candidate_matrix, demeaned[spec.outcome_column].to_numpy(dtype=float))
            selected_terms = tuple(
                column
                for column, coefficient in zip(valid_columns, lasso.coef_, strict=True)
                if not np.isclose(coefficient, 0.0)
            )
            lasso_alpha = float(lasso.alpha_)
            lasso_selected_count = int(len(selected_terms))
            regressors = tuple([*spec.forced_regressor_columns, *selected_terms])
        else:
            lasso_selected_count = 0

    tidy, formula = _run_ols(settings, demeaned, spec, regressors)
    metadata = {
        "formula": formula,
        "selected_terms": selected_terms,
        "lasso_alpha": lasso_alpha,
        "lasso_selected_count": lasso_selected_count,
        "map_iterations": residualized.iterations,
        "map_converged": residualized.converged,
        "map_max_change": residualized.max_change,
        "regressors": regressors,
    }
    return tidy, int(sample.shape[0]), metadata


def run_suite(
    settings: SensorAnalysisSettings = DEFAULT_SETTINGS,
    *,
    pollutant_group_kind: str = "all",
    pollutant_group: str = "all",
    pollutants: list[str] | None = None,
    land_cover_subclasses: list[str] | None = None,
    max_distance_step: int | None = None,
    model_families: list[str] | None = None,
    output_dir: str | Path | None = None,
    min_observations: int | None = None,
    save_outputs: bool = True,
) -> SensorAnalysisRun:
    """Run the configured sensor analysis suite."""
    _configure_runtime_warnings()
    model_name = _resolve_model_name(
        pollutant_group_kind=pollutant_group_kind,
        pollutant_group=pollutant_group,
        pollutants=pollutants,
    )
    effective_settings = _coerce_settings(
        settings,
        output_dir=Path(output_dir) / model_name if output_dir is not None else settings.output_dir / model_name,
        minimum_observations=min_observations,
    )
    prepared = build_analysis_data(effective_settings)
    pollutant_selection = select_pollutants(
        prepared.pollutant_catalog,
        group_kind=pollutant_group_kind,
        group_name=pollutant_group,
        explicit_pollutants=pollutants,
        minimum_observations=effective_settings.minimum_observations,
    )
    specs = build_model_specs(
        effective_settings,
        pollutant_selection,
        subclass_selection=land_cover_subclasses,
        max_distance_step=max_distance_step,
        model_families=model_families,
        climate_variables=prepared.climate_variables,
    )
    pollutant_meta = pollutant_lookup(prepared.pollutant_catalog)

    result_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []

    logger.info("Running %d model specification(s).", len(specs))
    for index, spec in enumerate(specs, start=1):
        logger.info(
            "Model %d/%d: family=%s pollutant=%s subclass=%s step=%s",
            index,
            len(specs),
            spec.model_family,
            spec.pollutant,
            spec.land_cover_subclass,
            spec.distance_step_name,
        )
        meta = pollutant_meta[spec.pollutant]
        try:
            tidy_frame, nobs, model_meta = _run_model(effective_settings, prepared.data, spec)
            result_frames.append(
                tidy_to_records(
                    tidy_frame,
                    spec,
                    meta,
                    nobs,
                    formula=model_meta["formula"],
                    selected_terms=model_meta["selected_terms"],
                    lasso_alpha=model_meta["lasso_alpha"],
                    lasso_selected_count=model_meta["lasso_selected_count"],
                    map_iterations=model_meta["map_iterations"],
                    map_converged=model_meta["map_converged"],
                )
            )
            manifest_rows.append(
                manifest_record(
                    spec,
                    meta,
                    status="ok",
                    nobs=nobs,
                    formula=model_meta["formula"],
                    selected_terms=model_meta["selected_terms"],
                    lasso_alpha=model_meta["lasso_alpha"],
                    lasso_selected_count=model_meta["lasso_selected_count"],
                    map_iterations=model_meta["map_iterations"],
                    map_converged=model_meta["map_converged"],
                )
            )
        except Exception as exc:  # pragma: no cover - exercised in CLI/integration flows
            logger.warning(
                "Model failed for family=%s pollutant=%s subclass=%s step=%s: %s",
                spec.model_family,
                spec.pollutant,
                spec.land_cover_subclass,
                spec.distance_step_name,
                exc,
            )
            manifest_rows.append(
                manifest_record(
                    spec,
                    meta,
                    status="failed",
                    nobs=0,
                    error=str(exc),
                )
            )

    results = (
        pd.concat(result_frames, ignore_index=True)
        if result_frames
        else pd.DataFrame()
    )
    manifest = pd.DataFrame.from_records(manifest_rows)
    summary = {
        "models_total": int(len(manifest_rows)),
        "models_succeeded": int((manifest["status"] == "ok").sum()) if not manifest.empty else 0,
        "models_failed": int((manifest["status"] == "failed").sum()) if not manifest.empty else 0,
        "model_name": model_name,
        "pollutants": list(pollutant_selection.pollutants),
        "land_cover_subclasses": land_cover_subclasses or list(effective_settings.land_cover_subclasses),
        "model_families": model_families or list(effective_settings.model_families),
        "max_distance_step": max_distance_step or len(effective_settings.distance_buckets),
        "minimum_observations": effective_settings.minimum_observations,
    }
    run = SensorAnalysisRun(
        results=results,
        manifest=manifest,
        summary=summary,
        output_dir=effective_settings.output_dir,
    )
    if save_outputs:
        save_run(run, settings=effective_settings, settings_payload=asdict(effective_settings))
    return run


__all__ = ["run_suite"]
