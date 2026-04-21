"""Result normalization and persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .catalog import PollutantDefinition
from .specs import ModelSpec


@dataclass(frozen=True)
class SensorAnalysisRun:
    """Collected outputs from a suite run."""

    results: pd.DataFrame
    manifest: pd.DataFrame
    summary: dict[str, object]
    output_dir: Path


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
) -> pd.DataFrame:
    """Attach manifest metadata to a tidy pyfixest output frame."""
    frame = tidy_frame.reset_index().copy()
    if "term" not in frame.columns:
        first_column = frame.columns[0]
        frame = frame.rename(columns={first_column: "term"})
    frame["pollutant"] = spec.pollutant
    frame["pollutant_group_kind"] = spec.pollutant_group_kind
    frame["pollutant_group_name"] = spec.pollutant_group_name
    frame["pollutant_type"] = pollutant_meta.type_group
    frame["pollutant_importance"] = pollutant_meta.importance_group
    frame["transform"] = pollutant_meta.transform
    frame["land_cover_subclass"] = spec.land_cover_subclass
    frame["distance_step_index"] = spec.distance_step_index
    frame["distance_step_name"] = spec.distance_step_name
    frame["included_buckets"] = ",".join(spec.included_buckets)
    frame["formula"] = spec.formula
    frame["nobs"] = nobs
    return frame


def manifest_record(
    spec: ModelSpec,
    pollutant_meta: PollutantDefinition,
    *,
    status: str,
    nobs: int,
    error: str | None = None,
) -> dict[str, object]:
    """Create one manifest row."""
    return {
        "pollutant": spec.pollutant,
        "pollutant_group_kind": spec.pollutant_group_kind,
        "pollutant_group_name": spec.pollutant_group_name,
        "pollutant_type": pollutant_meta.type_group,
        "pollutant_importance": pollutant_meta.importance_group,
        "transform": pollutant_meta.transform,
        "land_cover_subclass": spec.land_cover_subclass,
        "distance_step_index": spec.distance_step_index,
        "distance_step_name": spec.distance_step_name,
        "included_buckets": ",".join(spec.included_buckets),
        "outcome_column": spec.outcome_column,
        "formula": spec.formula,
        "nobs": nobs,
        "status": status,
        "error": error,
    }


def save_run(
    run: SensorAnalysisRun,
    *,
    settings_payload: dict[str, object] | None = None,
) -> None:
    """Persist run artifacts under the configured output directory."""
    run.output_dir.mkdir(parents=True, exist_ok=True)
    run.results.to_parquet(run.output_dir / "results.parquet", index=False)
    run.manifest.to_parquet(run.output_dir / "manifest.parquet", index=False)
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
    "manifest_record",
    "pollutant_lookup",
    "save_run",
    "tidy_to_records",
]
