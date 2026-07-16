"""Focused tests for deterministic sensor-analysis execution artifacts."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.sensor_data.checkpoints import (
    completed_spec_ids,
    latest_fingerprint,
    load_partial_results,
    mark_shard_complete,
    merge_shards,
    shard_progress,
    write_chunk,
    write_shard_metadata,
)
from src.analysis.sensor_data.specs import ModelSpec
from src.analysis.sensor_data.runner import _rank_revealing_selected_terms, _resolve_lasso_jobs
from src.analysis.settings import SensorAnalysisSettings


def _spec() -> ModelSpec:
    return ModelSpec(
        pollutant="ph",
        pollutant_group_kind="explicit",
        pollutant_group_name="custom",
        model_family="post_lasso",
        land_cover_subclass="c41",
        distance_step_index=1,
        distance_step_name="0_10km",
        included_buckets=("0_10km",),
        outcome_column="ph__transformed",
        coefficient_columns=("lc",),
        forced_regressor_columns=("lc",),
        candidate_regressor_columns=("climate",),
    )


def test_model_spec_id_is_stable() -> None:
    assert _spec().spec_id == _spec().spec_id


def test_rank_validation_keeps_forced_terms_and_drops_redundant_selected_terms() -> None:
    sample = pd.DataFrame({"forced": [1.0, 2.0, 3.0], "same": [1.0, 2.0, 3.0], "new": [1.0, 0.0, -1.0]})
    retained, dropped = _rank_revealing_selected_terms(sample, ("forced",), ("same", "new"))
    assert retained == ("new",)
    assert dropped == ("same",)


def test_lasso_jobs_uses_slurm_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "3")
    assert _resolve_lasso_jobs(SensorAnalysisSettings(), None) == 3


def test_checkpoint_merge_requires_complete_unique_specs(tmp_path) -> None:
    fingerprint = "test-fingerprint"
    settings = SensorAnalysisSettings(output_dir=tmp_path)
    spec_id = _spec().spec_id
    write_shard_metadata(tmp_path, fingerprint, 0, 1, [spec_id], settings)
    write_chunk(
        tmp_path,
        fingerprint,
        0,
        0,
        pd.DataFrame({"spec_id": [spec_id], "Estimate": [1.0]}),
        pd.DataFrame({"spec_id": [spec_id], "status": ["ok"]}),
    )
    assert completed_spec_ids(tmp_path, fingerprint, 0) == {spec_id}
    with pytest.raises(ValueError, match="incomplete"):
        merge_shards(tmp_path, fingerprint, 1, settings)
    mark_shard_complete(tmp_path, fingerprint, 0)
    merged = merge_shards(tmp_path, fingerprint, 1, settings)
    assert merged.manifest["spec_id"].tolist() == [spec_id]


def test_shard_progress_and_partial_results_before_completion(tmp_path) -> None:
    fingerprint = "test-fingerprint"
    settings = SensorAnalysisSettings(output_dir=tmp_path)
    spec_a = _spec()
    spec_b = ModelSpec(
        pollutant="ph",
        pollutant_group_kind="explicit",
        pollutant_group_name="custom",
        model_family="post_lasso",
        land_cover_subclass="c41",
        distance_step_index=2,
        distance_step_name="10_50km",
        included_buckets=("0_10km", "10_50km"),
        outcome_column="ph__transformed",
        coefficient_columns=("lc",),
        forced_regressor_columns=("lc",),
        candidate_regressor_columns=("climate",),
    )

    assert latest_fingerprint(tmp_path) is None

    write_shard_metadata(tmp_path, fingerprint, 0, 2, [spec_a.spec_id], settings)
    write_shard_metadata(tmp_path, fingerprint, 1, 2, [spec_b.spec_id], settings)
    write_chunk(
        tmp_path,
        fingerprint,
        0,
        0,
        pd.DataFrame({"spec_id": [spec_a.spec_id], "Estimate": [1.0]}),
        pd.DataFrame({"spec_id": [spec_a.spec_id], "status": ["ok"]}),
    )
    mark_shard_complete(tmp_path, fingerprint, 0)
    # Shard 1 has a checkpointed chunk but has not finished (no _SUCCESS marker).

    write_chunk(
        tmp_path,
        fingerprint,
        1,
        0,
        pd.DataFrame({"spec_id": [spec_b.spec_id], "Estimate": [2.0]}),
        pd.DataFrame({"spec_id": [spec_b.spec_id], "status": ["ok"]}),
    )

    assert latest_fingerprint(tmp_path) == fingerprint

    progress = shard_progress(tmp_path, fingerprint)
    assert {(entry["shard_index"], entry["complete"]) for entry in progress} == {(0, True), (1, False)}
    assert sum(entry["specs_expected"] for entry in progress) == 2
    assert sum(entry["specs_done"] for entry in progress) == 2

    results, manifest = load_partial_results(tmp_path, fingerprint)
    assert sorted(manifest["spec_id"].tolist()) == sorted([spec_a.spec_id, spec_b.spec_id])
    assert sorted(results["spec_id"].tolist()) == sorted([spec_a.spec_id, spec_b.spec_id])
