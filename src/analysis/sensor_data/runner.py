"""Execution engine for the sensor analysis suite."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import LinAlgWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoCV

from .checkpoints import (
    completed_spec_ids,
    input_fingerprint,
    mark_shard_complete,
    merge_shards,
    shard_dir,
    write_chunk,
    write_shard_metadata,
)
from .groups import select_pollutants
from .prepare import build_analysis_data
from .residualize import residualize_with_map
from .results import SensorAnalysisRun, manifest_record, pollutant_lookup, tidy_to_records
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
    sample = frame.loc[:, _analysis_columns(settings, spec)].dropna().reset_index(drop=True).copy()
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
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", LinAlgWarning)
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
    numerical_warnings = [str(item.message) for item in caught if issubclass(item.category, LinAlgWarning)]
    if numerical_warnings:
        raise ValueError(f"OLS ill-conditioned after rank validation: {numerical_warnings[0]}")
    return fit.tidy(), formula, numerical_warnings


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


def _prune_near_duplicate_candidates(
    candidate_matrix: np.ndarray,
    columns: list[str],
    *,
    correlation_threshold: float,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Keep one deterministic representative of each near-duplicate candidate."""
    if not 0.0 < correlation_threshold <= 1.0:
        raise ValueError("LASSO near-duplicate correlation threshold must be in (0, 1].")
    if candidate_matrix.shape[1] < 2:
        return candidate_matrix, columns, []

    correlations = (candidate_matrix.T @ candidate_matrix) / candidate_matrix.shape[0]
    kept_indices: list[int] = []
    dropped_columns: list[str] = []
    for index, column in enumerate(columns):
        if any(abs(correlations[index, kept_index]) >= correlation_threshold for kept_index in kept_indices):
            dropped_columns.append(column)
        else:
            kept_indices.append(index)
    return candidate_matrix[:, kept_indices], [columns[index] for index in kept_indices], dropped_columns


def _resolve_lasso_jobs(settings: SensorAnalysisSettings, override: int | None) -> int:
    """Resolve workers without assuming a local machine's CPU count is safe."""
    if override is not None:
        if override < 1:
            raise ValueError("--lasso-jobs must be positive.")
        return override
    if settings.lasso_settings.n_jobs is not None:
        return settings.lasso_settings.n_jobs
    try:
        return max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    except ValueError:
        return 1


def _rank_revealing_selected_terms(
    sample: pd.DataFrame,
    forced_terms: tuple[str, ...],
    selected_terms: tuple[str, ...],
    *,
    tolerance: float = 1e-10,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Incrementally construct a QR basis, never dropping forced regressors."""
    basis: list[np.ndarray] = []

    def add(column: str) -> bool:
        vector = sample[column].to_numpy(dtype=float).copy()
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm == 0.0:
            return False
        for _ in range(2):
            for basis_vector in basis:
                vector -= np.dot(basis_vector, vector) * basis_vector
        residual_norm = float(np.linalg.norm(vector))
        if residual_norm <= tolerance * norm:
            return False
        basis.append(vector / residual_norm)
        return True

    for term in forced_terms:
        if not add(term):
            raise ValueError(f"Forced post-LASSO regressor is rank-deficient: {term}")
    retained: list[str] = []
    dropped: list[str] = []
    for term in selected_terms:
        if add(term):
            retained.append(term)
        else:
            dropped.append(term)
    return tuple(retained), tuple(dropped)


def _fit_lasso(
    settings: SensorAnalysisSettings,
    candidate_matrix: np.ndarray,
    outcome: np.ndarray,
    *,
    n_jobs: int,
) -> tuple[LassoCV, bool, int, tuple[str, ...]]:
    """Fit CV LASSO, retrying once only when the CV path fails to converge."""
    messages: list[str] = []
    for attempt, max_iter in enumerate((settings.lasso_settings.max_iter, 50_000), start=1):
        lasso = LassoCV(
            cv=settings.lasso_settings.cv,
            alphas=settings.lasso_settings.alphas,
            eps=settings.lasso_settings.eps,
            random_state=settings.lasso_settings.random_state,
            selection=settings.lasso_settings.selection,
            tol=settings.lasso_settings.tol,
            max_iter=max_iter,
            n_jobs=n_jobs,
            precompute=settings.lasso_settings.precompute,
            fit_intercept=False,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            lasso.fit(candidate_matrix, outcome)
        convergence_messages = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
        messages.extend(convergence_messages)
        if not convergence_messages:
            return lasso, True, attempt, tuple(messages)
    return lasso, False, 2, tuple(messages)


def _fit_residualized_model(
    settings: SensorAnalysisSettings,
    demeaned: pd.DataFrame,
    spec,
    *,
    nobs: int,
    map_iterations: int,
    map_converged: bool,
    map_max_change: float,
    lasso_jobs: int,
    timings: dict[str, float],
) -> tuple[pd.DataFrame, int, dict[str, object]]:
    """Estimate one specification from a shared, residualized work group."""
    regressors = tuple(spec.forced_regressor_columns)

    selected_terms: tuple[str, ...] = ()
    lasso_alpha: float | None = None
    lasso_selected_count: int | None = None
    lasso_candidate_count: int | None = None
    lasso_valid_candidate_count: int | None = None
    lasso_pruned_candidate_count: int | None = None
    lasso_pruned_candidates: tuple[str, ...] = ()
    lasso_selected_share: float | None = None
    lasso_min_cv_mse: float | None = None
    lasso_converged: bool | None = None
    lasso_attempts: int | None = None
    lasso_warnings: tuple[str, ...] = ()
    design_dropped_terms: tuple[str, ...] = ()

    if spec.model_family == "post_lasso" and spec.candidate_regressor_columns:
        lasso_candidate_count = int(len(spec.candidate_regressor_columns))
        candidate_frame = demeaned.loc[:, list(spec.candidate_regressor_columns)]
        candidate_matrix, valid_columns = _standardize_candidates(candidate_frame)
        lasso_valid_candidate_count = int(len(valid_columns))
        if valid_columns:
            candidate_matrix, valid_columns, pruned_columns = _prune_near_duplicate_candidates(
                candidate_matrix,
                valid_columns,
                correlation_threshold=settings.lasso_settings.near_duplicate_correlation,
            )
            lasso_pruned_candidate_count = int(len(pruned_columns))
            lasso_pruned_candidates = tuple(pruned_columns)
            started = time.perf_counter()
            lasso, lasso_converged, lasso_attempts, lasso_warnings = _fit_lasso(
                settings,
                candidate_matrix,
                demeaned[spec.outcome_column].to_numpy(dtype=float),
                n_jobs=lasso_jobs,
            )
            timings["lasso_seconds"] = time.perf_counter() - started
            if not lasso_converged:
                raise ValueError("LASSO CV did not converge after retry.")
            selected_terms = tuple(
                column
                for column, coefficient in zip(valid_columns, lasso.coef_, strict=True)
                if not np.isclose(coefficient, 0.0)
            )
            lasso_alpha = float(lasso.alpha_)
            lasso_selected_count = int(len(selected_terms))
            lasso_selected_share = float(lasso_selected_count / len(valid_columns))
            mean_mse_by_alpha = np.asarray(lasso.mse_path_, dtype=float).mean(axis=1)
            lasso_min_cv_mse = float(np.min(mean_mse_by_alpha))
            regressors = tuple([*spec.forced_regressor_columns, *selected_terms])
        else:
            lasso_selected_count = 0
            lasso_selected_share = 0.0
            lasso_pruned_candidate_count = 0

    if spec.model_family == "post_lasso":
        started = time.perf_counter()
        selected_terms, design_dropped_terms = _rank_revealing_selected_terms(
            demeaned,
            tuple(spec.forced_regressor_columns),
            selected_terms,
        )
        regressors = tuple([*spec.forced_regressor_columns, *selected_terms])
        timings["design_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    tidy, formula, ols_warnings = _run_ols(settings, demeaned, spec, regressors)
    timings["ols_seconds"] = time.perf_counter() - started
    metadata = {
        "formula": formula,
        "selected_terms": selected_terms,
        "lasso_alpha": lasso_alpha,
        "lasso_selected_count": lasso_selected_count,
        "lasso_candidate_count": lasso_candidate_count,
        "lasso_valid_candidate_count": lasso_valid_candidate_count,
        "lasso_pruned_candidate_count": lasso_pruned_candidate_count,
        "lasso_pruned_candidates": lasso_pruned_candidates,
        "lasso_selected_share": lasso_selected_share,
        "lasso_min_cv_mse": lasso_min_cv_mse,
        "lasso_converged": lasso_converged,
        "lasso_attempts": lasso_attempts,
        "lasso_warning_count": len(lasso_warnings),
        "ols_warning_count": len(ols_warnings),
        "numerical_status": "lasso_retried" if lasso_attempts == 2 else "ok",
        "warning_stage": "lasso" if lasso_warnings else None,
        "warning_code": "ConvergenceWarning" if lasso_warnings else None,
        "design_dropped_terms": design_dropped_terms,
        "map_iterations": map_iterations,
        "map_converged": map_converged,
        "map_max_change": map_max_change,
        "timings": timings,
        "regressors": regressors,
    }
    return tidy, nobs, metadata


@dataclass
class _WorkGroup:
    key: str
    mask: np.ndarray
    specs: list[object]

    @property
    def cost(self) -> int:
        return int(self.mask.sum()) * sum(1 + len(spec.candidate_regressor_columns) for spec in self.specs)


def _build_work_groups(settings: SensorAnalysisSettings, frame: pd.DataFrame, specs: list[object]) -> list[_WorkGroup]:
    groups: dict[str, _WorkGroup] = {}
    fixed_effects = ",".join(settings.resolved_fixed_effects())
    for spec in specs:
        mask = frame.loc[:, _analysis_columns(settings, spec)].notna().all(axis=1).to_numpy(dtype=bool)
        mask_digest = hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()
        key = f"{spec.outcome_column}:{fixed_effects}:{mask_digest}"
        groups.setdefault(key, _WorkGroup(key, mask, [])).specs.append(spec)
    return list(groups.values())


def _assign_work_groups(groups: list[_WorkGroup], shard_count: int) -> dict[str, int]:
    loads = [0] * shard_count
    assignments: dict[str, int] = {}
    for group in sorted(groups, key=lambda item: (-item.cost, item.key)):
        shard = min(range(shard_count), key=lambda index: (loads[index], index))
        assignments[group.key] = shard
        loads[shard] += group.cost
    return assignments


def _records_for_group(
    settings: SensorAnalysisSettings,
    frame: pd.DataFrame,
    group: _WorkGroup,
    metadata_by_pollutant: dict,
    *,
    lasso_jobs: int,
) -> tuple[list[pd.DataFrame], list[dict[str, object]]]:
    """Run a work group, sharing its complete-case sample and MAP projection."""
    result_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []
    feature_columns = list(dict.fromkeys(column for spec in group.specs for column in [*spec.forced_regressor_columns, *spec.candidate_regressor_columns]))
    outcome_column = group.specs[0].outcome_column
    columns = list(dict.fromkeys([outcome_column, *feature_columns, *settings.resolved_fixed_effects(), settings.cluster_variable]))
    sample_started = time.perf_counter()
    sample = frame.loc[group.mask, columns].reset_index(drop=True).copy()
    sample_seconds = time.perf_counter() - sample_started
    if sample.empty:
        residualized = None
        residualization_error = "No complete observations remain after dropping missing values."
    else:
        started = time.perf_counter()
        try:
            residualized = residualize_with_map(
                sample,
                outcome_column=outcome_column,
                feature_columns=feature_columns,
                fixed_effect_columns=settings.resolved_fixed_effects(),
                tolerance=settings.map_tolerance,
                max_iterations=settings.map_max_iterations,
            )
            residualization_error = None
            map_seconds = time.perf_counter() - started
            demeaned = residualized.frame.copy()
            demeaned[settings.cluster_variable] = sample[settings.cluster_variable].to_numpy()
        except Exception as exc:  # pragma: no cover - defensive batch boundary
            residualized = None
            residualization_error = str(exc)
    for offset, spec in enumerate(group.specs):
        meta = metadata_by_pollutant[spec.pollutant]
        started = time.perf_counter()
        try:
            if residualized is None:
                raise ValueError(residualization_error)
            if sample[spec.outcome_column].nunique(dropna=True) < 2:
                raise ValueError("Outcome has no variation after filtering.")
            if all(sample[column].nunique(dropna=True) < 2 for column in spec.coefficient_columns):
                raise ValueError("All land-cover regressors are constant after filtering.")
            timings = {"sample_seconds": sample_seconds if offset == 0 else 0.0, "map_seconds": map_seconds if offset == 0 else 0.0, "lasso_seconds": 0.0, "design_seconds": 0.0, "ols_seconds": 0.0}
            tidy, nobs, model_meta = _fit_residualized_model(
                settings,
                demeaned,
                spec,
                nobs=int(sample.shape[0]),
                map_iterations=residualized.iterations,
                map_converged=residualized.converged,
                map_max_change=residualized.max_change,
                lasso_jobs=lasso_jobs,
                timings=timings,
            )
            timings["total_seconds"] = time.perf_counter() - started
            records = tidy_to_records(
                tidy, spec, meta, nobs,
                formula=model_meta["formula"], selected_terms=model_meta["selected_terms"],
                lasso_alpha=model_meta["lasso_alpha"], lasso_selected_count=model_meta["lasso_selected_count"],
                lasso_candidate_count=model_meta["lasso_candidate_count"], lasso_valid_candidate_count=model_meta["lasso_valid_candidate_count"],
                lasso_pruned_candidate_count=model_meta["lasso_pruned_candidate_count"], lasso_pruned_candidates=model_meta["lasso_pruned_candidates"],
                lasso_selected_share=model_meta["lasso_selected_share"], lasso_min_cv_mse=model_meta["lasso_min_cv_mse"],
                map_iterations=model_meta["map_iterations"], map_converged=model_meta["map_converged"],
            )
            records["spec_id"] = spec.spec_id
            for key, value in {**model_meta["timings"], "lasso_converged": model_meta["lasso_converged"], "lasso_attempts": model_meta["lasso_attempts"], "lasso_warning_count": model_meta["lasso_warning_count"], "ols_warning_count": model_meta["ols_warning_count"], "numerical_status": model_meta["numerical_status"], "warning_stage": model_meta["warning_stage"], "warning_code": model_meta["warning_code"], "design_dropped_terms": ",".join(model_meta["design_dropped_terms"])}.items():
                records[key] = value
            result_frames.append(records)
            manifest = manifest_record(spec, meta, status="ok", nobs=nobs, formula=model_meta["formula"], selected_terms=model_meta["selected_terms"], lasso_alpha=model_meta["lasso_alpha"], lasso_selected_count=model_meta["lasso_selected_count"], lasso_candidate_count=model_meta["lasso_candidate_count"], lasso_valid_candidate_count=model_meta["lasso_valid_candidate_count"], lasso_pruned_candidate_count=model_meta["lasso_pruned_candidate_count"], lasso_pruned_candidates=model_meta["lasso_pruned_candidates"], lasso_selected_share=model_meta["lasso_selected_share"], lasso_min_cv_mse=model_meta["lasso_min_cv_mse"], map_iterations=model_meta["map_iterations"], map_converged=model_meta["map_converged"])
            manifest.update({"spec_id": spec.spec_id, **model_meta["timings"], "lasso_converged": model_meta["lasso_converged"], "lasso_attempts": model_meta["lasso_attempts"], "lasso_warning_count": model_meta["lasso_warning_count"], "ols_warning_count": model_meta["ols_warning_count"], "numerical_status": model_meta["numerical_status"], "warning_stage": model_meta["warning_stage"], "warning_code": model_meta["warning_code"], "design_dropped_terms": ",".join(model_meta["design_dropped_terms"])})
            manifest_rows.append(manifest)
        except Exception as exc:  # pragma: no cover - batch boundary
            logger.warning("Model failed for family=%s pollutant=%s subclass=%s step=%s: %s", spec.model_family, spec.pollutant, spec.land_cover_subclass, spec.distance_step_name, exc)
            failed = manifest_record(spec, meta, status="failed", nobs=0, error=str(exc))
            failed.update({"spec_id": spec.spec_id, "total_seconds": time.perf_counter() - started, "numerical_status": "failed", "warning_stage": "ols" if "ill-conditioned" in str(exc).lower() else None, "warning_code": "LinAlgWarning" if "ill-conditioned" in str(exc).lower() else None})
            manifest_rows.append(failed)
    return result_frames, manifest_rows


def run_suite(
    settings: SensorAnalysisSettings = DEFAULT_SETTINGS,
    *,
    pollutant_group_kind: str = "all", pollutant_group: str = "all", pollutants: list[str] | None = None,
    land_cover_subclasses: list[str] | None = None, max_distance_step: int | None = None,
    model_families: list[str] | None = None, output_dir: str | Path | None = None,
    min_observations: int | None = None, save_outputs: bool = True, lasso_jobs: int | None = None,
    shard_count: int = 1, shard_index: int = 0, resume: bool = False, checkpoint_models: int = 25,
) -> SensorAnalysisRun:
    """Run one deterministic shard of the sensor-analysis suite."""
    if shard_count < 1 or not 0 <= shard_index < shard_count or checkpoint_models < 1:
        raise ValueError("Invalid shard count, shard index, or checkpoint size.")
    if not save_outputs and shard_count != 1:
        raise ValueError("Sharded runs require output checkpoints.")
    _configure_runtime_warnings()
    model_name = _resolve_model_name(pollutant_group_kind=pollutant_group_kind, pollutant_group=pollutant_group, pollutants=pollutants)
    effective_settings = _coerce_settings(settings, output_dir=Path(output_dir) / model_name if output_dir is not None else settings.output_dir / model_name, minimum_observations=min_observations)
    prepared = build_analysis_data(effective_settings)
    selection = select_pollutants(prepared.pollutant_catalog, group_kind=pollutant_group_kind, group_name=pollutant_group, explicit_pollutants=pollutants, minimum_observations=effective_settings.minimum_observations)
    specs = build_model_specs(effective_settings, selection, subclass_selection=land_cover_subclasses, max_distance_step=max_distance_step, model_families=model_families, climate_variables=prepared.climate_variables)
    run_inputs = json.dumps({"group_kind": pollutant_group_kind, "group": pollutant_group, "pollutants": pollutants, "subclasses": land_cover_subclasses, "step": max_distance_step, "families": model_families}, sort_keys=True)
    fingerprint = hashlib.sha256(f"{input_fingerprint(effective_settings)}:{run_inputs}".encode()).hexdigest()[:20]
    groups = _build_work_groups(effective_settings, prepared.data, specs)
    assignments = _assign_work_groups(groups, shard_count)
    assigned_groups = [group for group in groups if assignments[group.key] == shard_index]
    assigned_ids = [spec.spec_id for group in assigned_groups for spec in group.specs]
    if save_outputs:
        write_shard_metadata(effective_settings.output_dir, fingerprint, shard_index, shard_count, assigned_ids, effective_settings)
    completed = completed_spec_ids(effective_settings.output_dir, fingerprint, shard_index) if save_outputs and resume else set()
    result_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []
    pending_results: list[pd.DataFrame] = []
    pending_manifest: list[dict[str, object]] = []
    chunk_index = len(list(shard_dir(effective_settings.output_dir, fingerprint, shard_index).glob("chunk-*"))) if save_outputs else 0
    lasso_worker_count = _resolve_lasso_jobs(effective_settings, lasso_jobs)
    logger.info("Running shard %d/%d with %d model specification(s).", shard_index + 1, shard_count, len(assigned_ids))
    for group in assigned_groups:
        remaining = [spec for spec in group.specs if spec.spec_id not in completed]
        if not remaining:
            continue
        group.specs = remaining
        group_results, group_manifest = _records_for_group(effective_settings, prepared.data, group, pollutant_lookup(prepared.pollutant_catalog), lasso_jobs=lasso_worker_count)
        result_frames.extend(group_results); manifest_rows.extend(group_manifest)
        pending_results.extend(group_results); pending_manifest.extend(group_manifest)
        if save_outputs and len(pending_manifest) >= checkpoint_models:
            write_chunk(effective_settings.output_dir, fingerprint, shard_index, chunk_index, pd.concat(pending_results, ignore_index=True) if pending_results else pd.DataFrame(), pd.DataFrame.from_records(pending_manifest))
            chunk_index += 1; pending_results = []; pending_manifest = []
    if save_outputs and pending_manifest:
        write_chunk(effective_settings.output_dir, fingerprint, shard_index, chunk_index, pd.concat(pending_results, ignore_index=True) if pending_results else pd.DataFrame(), pd.DataFrame.from_records(pending_manifest))
    if save_outputs:
        mark_shard_complete(effective_settings.output_dir, fingerprint, shard_index)
        if shard_count == 1:
            return merge_shards(effective_settings.output_dir, fingerprint, 1, effective_settings)
    results = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    manifest = pd.DataFrame.from_records(manifest_rows)
    return SensorAnalysisRun(results, manifest, {"run_fingerprint": fingerprint, "shard_index": shard_index, "shard_count": shard_count}, effective_settings.output_dir)


def merge_suite(settings: SensorAnalysisSettings, *, run_dir: str | Path, run_fingerprint: str | None, expected_shards: int) -> SensorAnalysisRun:
    """Publish canonical outputs from completed shard checkpoints."""
    return merge_shards(Path(run_dir), run_fingerprint, expected_shards, settings)


__all__ = ["merge_suite", "run_suite"]
