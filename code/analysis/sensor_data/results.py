"""Result normalization and persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .catalog import PollutantDefinition
from .specs import ModelSpec
from ..settings import DEFAULT_SETTINGS, SensorAnalysisSettings


@dataclass(frozen=True)
class SensorAnalysisRun:
    """Collected outputs from a suite run."""

    results: pd.DataFrame
    manifest: pd.DataFrame
    summary: dict[str, object]
    output_dir: Path


def _pollutant_label(name: str) -> str:
    return name.replace("_", " ").title()


def _term_label(term: str, settings: SensorAnalysisSettings) -> str:
    if "__x__" in term:
        left, right = term.split("__x__", 1)
        return f"{_term_label(left, settings)} x {_term_label(right, settings)}"
    for subclass, label in settings.subclass_labels.items():
        for bucket in settings.distance_buckets:
            if term == settings.land_cover_column(bucket, subclass):
                return f"{label}, {settings.distance_bucket_label(bucket)}"
    for variable in settings.controls:
        if term == variable.scaled_column:
            return variable.source_column.replace("_", " ").title()
    for variable in settings.climate_variables:
        if term == variable.scaled_column:
            return variable.source_column.replace("_", " ").title()
    return term.replace("_", " ").replace("__", " ").title()


def build_readable_results_table(
    results: pd.DataFrame,
    settings: SensorAnalysisSettings = DEFAULT_SETTINGS,
) -> pd.DataFrame:
    """Return a human-readable result table for export."""
    if results.empty:
        return results.copy()
    frame = results.copy()
    frame["model"] = frame["model_family"].map(settings.model_family_label)
    frame["pollutant_label"] = frame["pollutant"].map(_pollutant_label)
    frame["land_cover_label"] = frame["land_cover_subclass"].map(
        lambda value: settings.subclass_labels.get(value, value)
    )
    frame["distance_label"] = frame["distance_step_name"].map(settings.distance_bucket_label)
    frame["term_label"] = frame["term"].map(lambda value: _term_label(value, settings))
    if {"Estimate", "2.5%", "97.5%"}.issubset(frame.columns):
        frame["estimate_ci"] = frame.apply(
            lambda row: f"{row['Estimate']:.3f} [{row['2.5%']:.3f}, {row['97.5%']:.3f}]",
            axis=1,
        )
    preferred = [
        "model",
        "pollutant_label",
        "land_cover_label",
        "distance_label",
        "term_label",
        "estimate_ci",
        "Estimate",
        "Std. Error",
        "t value",
        "Pr(>|t|)",
        "nobs",
        "selected_by_lasso",
        "lasso_selected_count",
        "map_converged",
    ]
    ordered = [column for column in preferred if column in frame.columns]
    remainder = [column for column in frame.columns if column not in ordered]
    return frame.loc[:, [*ordered, *remainder]]


def build_readable_manifest_table(
    manifest: pd.DataFrame,
    settings: SensorAnalysisSettings = DEFAULT_SETTINGS,
) -> pd.DataFrame:
    """Return a human-readable manifest table for export."""
    if manifest.empty:
        return manifest.copy()
    frame = manifest.copy()
    frame["model"] = frame["model_family"].map(settings.model_family_label)
    frame["pollutant_label"] = frame["pollutant"].map(_pollutant_label)
    frame["land_cover_label"] = frame["land_cover_subclass"].map(
        lambda value: settings.subclass_labels.get(value, value)
    )
    frame["distance_label"] = frame["distance_step_name"].map(settings.distance_bucket_label)
    preferred = [
        "status",
        "model",
        "pollutant_label",
        "land_cover_label",
        "distance_label",
        "nobs",
        "lasso_selected_count",
        "map_converged",
        "error",
    ]
    ordered = [column for column in preferred if column in frame.columns]
    remainder = [column for column in frame.columns if column not in ordered]
    return frame.loc[:, [*ordered, *remainder]]


def pollutant_lookup(
    pollutant_catalog: list[PollutantDefinition],
) -> dict[str, PollutantDefinition]:
    """Create an index from pollutant name to metadata."""
    return {item.name: item for item in pollutant_catalog}


def tidy_to_records(
    tidy_frame: pd.DataFrame,
    spec: ModelSpec,
    pollutant_meta: PollutantDefinition,
    nobs: int,
    *,
    formula: str | None = None,
    selected_terms: tuple[str, ...] = (),
    lasso_alpha: float | None = None,
    lasso_selected_count: int | None = None,
    map_iterations: int | None = None,
    map_converged: bool | None = None,
) -> pd.DataFrame:
    """Attach manifest metadata to a tidy pyfixest output frame."""
    frame = tidy_frame.reset_index().copy()
    if "term" not in frame.columns:
        first_column = frame.columns[0]
        frame = frame.rename(columns={first_column: "term"})
    frame["pollutant"] = spec.pollutant
    frame["pollutant_group_kind"] = spec.pollutant_group_kind
    frame["pollutant_group_name"] = spec.pollutant_group_name
    frame["model_family"] = spec.model_family
    frame["pollutant_type"] = pollutant_meta.type_group
    frame["pollutant_importance"] = pollutant_meta.importance_group
    frame["transform"] = pollutant_meta.transform
    frame["land_cover_subclass"] = spec.land_cover_subclass
    frame["distance_step_index"] = spec.distance_step_index
    frame["distance_step_name"] = spec.distance_step_name
    frame["included_buckets"] = ",".join(spec.included_buckets)
    frame["forced_regressors"] = ",".join(spec.forced_regressor_columns)
    frame["candidate_regressors"] = ",".join(spec.candidate_regressor_columns)
    frame["formula"] = formula
    frame["nobs"] = nobs
    frame["selected_by_lasso"] = frame["term"].isin(selected_terms)
    frame["lasso_alpha"] = lasso_alpha
    frame["lasso_selected_count"] = lasso_selected_count
    frame["map_iterations"] = map_iterations
    frame["map_converged"] = map_converged
    return frame


def manifest_record(
    spec: ModelSpec,
    pollutant_meta: PollutantDefinition,
    *,
    status: str,
    nobs: int,
    error: str | None = None,
    formula: str | None = None,
    selected_terms: tuple[str, ...] = (),
    lasso_alpha: float | None = None,
    lasso_selected_count: int | None = None,
    map_iterations: int | None = None,
    map_converged: bool | None = None,
) -> dict[str, object]:
    """Create one manifest row."""
    return {
        "pollutant": spec.pollutant,
        "pollutant_group_kind": spec.pollutant_group_kind,
        "pollutant_group_name": spec.pollutant_group_name,
        "model_family": spec.model_family,
        "pollutant_type": pollutant_meta.type_group,
        "pollutant_importance": pollutant_meta.importance_group,
        "transform": pollutant_meta.transform,
        "land_cover_subclass": spec.land_cover_subclass,
        "distance_step_index": spec.distance_step_index,
        "distance_step_name": spec.distance_step_name,
        "included_buckets": ",".join(spec.included_buckets),
        "outcome_column": spec.outcome_column,
        "forced_regressors": ",".join(spec.forced_regressor_columns),
        "candidate_regressors": ",".join(spec.candidate_regressor_columns),
        "formula": formula,
        "selected_terms": ",".join(selected_terms),
        "lasso_alpha": lasso_alpha,
        "lasso_selected_count": lasso_selected_count,
        "map_iterations": map_iterations,
        "map_converged": map_converged,
        "nobs": nobs,
        "status": status,
        "error": error,
    }


def save_run(
    run: SensorAnalysisRun,
    *,
    settings: SensorAnalysisSettings = DEFAULT_SETTINGS,
    settings_payload: dict[str, object] | None = None,
) -> None:
    """Persist run artifacts under the configured output directory."""
    run.output_dir.mkdir(parents=True, exist_ok=True)
    run.results.to_parquet(run.output_dir / "results.parquet", index=False)
    run.manifest.to_parquet(run.output_dir / "manifest.parquet", index=False)
    readable_results = build_readable_results_table(run.results, settings)
    readable_manifest = build_readable_manifest_table(run.manifest, settings)
    readable_results.to_csv(run.output_dir / "results_readable.csv", index=False)
    readable_manifest.to_csv(run.output_dir / "manifest_readable.csv", index=False)
    (run.output_dir / "results_readable.md").write_text(
        readable_results.to_markdown(index=False),
        encoding="utf-8",
    )
    (run.output_dir / "manifest_readable.md").write_text(
        readable_manifest.to_markdown(index=False),
        encoding="utf-8",
    )
    (run.output_dir / "summary.json").write_text(
        json.dumps(run.summary, indent=2, default=str),
        encoding="utf-8",
    )
    if settings_payload is not None:
        (run.output_dir / "settings.json").write_text(
            json.dumps(settings_payload, indent=2, default=str),
            encoding="utf-8",
        )


__all__ = [
    "SensorAnalysisRun",
    "build_readable_manifest_table",
    "build_readable_results_table",
    "manifest_record",
    "pollutant_lookup",
    "save_run",
    "tidy_to_records",
]
