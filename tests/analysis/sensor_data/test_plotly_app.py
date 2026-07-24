from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.cli import main as analysis_main
from src.analysis import cli as analysis_cli
from src.analysis.sensor_data.checkpoints import (
    mark_shard_complete,
    write_chunk,
    write_shard_metadata,
)
from src.analysis.sensor_data.plotly_app import (
    build_model_comparison_table,
    build_significance_matrix_table,
    discover_pending_runs,
    discover_result_runs,
    filter_app_frame,
    load_pending_run,
    load_result_run,
    make_diagnostics_table,
    make_lasso_stats_table,
)
from src.analysis.sensor_data.specs import ModelSpec
from src.analysis.settings import DEFAULT_SETTINGS


def _write_synthetic_run(base_dir: Path) -> Path:
    run_dir = base_dir / "type_nutrients"
    run_dir.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(
        [
            {
                "term": DEFAULT_SETTINGS.land_cover_column("0_10km", "c41"),
                "Estimate": 0.25,
                "2.5%": 0.10,
                "97.5%": 0.40,
                "t value": 3.2,
                "Pr(>|t|)": 0.01,
                "pollutant": "ph",
                "pollutant_group_kind": "type",
                "pollutant_group_name": "nutrients",
                "model_family": "crude_twfe",
                "pollutant_type": "core_physicochemical",
                "pollutant_importance": "high",
                "transform": "identity",
                "land_cover_subclass": "c41",
                "distance_step_index": 1,
                "distance_step_name": "0_10km",
                "included_buckets": "0_10km",
                "forced_regressors": DEFAULT_SETTINGS.land_cover_column("0_10km", "c41"),
                "candidate_regressors": "",
                "formula": "y ~ x - 1",
                "nobs": 100,
                "selected_by_lasso": False,
                "lasso_alpha": None,
                "lasso_selected_count": 0,
                "lasso_candidate_count": None,
                "lasso_valid_candidate_count": None,
                "lasso_selected_share": None,
                "lasso_min_cv_mse": None,
                "map_iterations": 3,
                "map_converged": True,
            },
            {
                "term": "cl_0_10km_tp_mean_7d__scaled",
                "Estimate": 0.18,
                "2.5%": 0.05,
                "97.5%": 0.32,
                "t value": 2.7,
                "Pr(>|t|)": 0.02,
                "pollutant": "ph",
                "pollutant_group_kind": "type",
                "pollutant_group_name": "nutrients",
                "model_family": "post_lasso",
                "pollutant_type": "core_physicochemical",
                "pollutant_importance": "high",
                "transform": "identity",
                "land_cover_subclass": "c41",
                "distance_step_index": 1,
                "distance_step_name": "0_10km",
                "included_buckets": "0_10km",
                "forced_regressors": DEFAULT_SETTINGS.land_cover_column("0_10km", "c41"),
                "candidate_regressors": "cl_0_10km_tp_mean_7d__scaled",
                "formula": "y ~ x + z - 1",
                "nobs": 100,
                "selected_by_lasso": True,
                "lasso_alpha": 0.01,
                "lasso_selected_count": 2,
                "lasso_candidate_count": 2,
                "lasso_valid_candidate_count": 2,
                "lasso_selected_share": 1.0,
                "lasso_min_cv_mse": 0.12,
                "map_iterations": 3,
                "map_converged": True,
            },
            {
                "term": DEFAULT_SETTINGS.land_cover_column("0_10km", "c41"),
                "Estimate": 0.20,
                "2.5%": -0.02,
                "97.5%": 0.38,
                "t value": 1.8,
                "Pr(>|t|)": 0.08,
                "pollutant": "ph",
                "pollutant_group_kind": "type",
                "pollutant_group_name": "nutrients",
                "model_family": "post_lasso",
                "pollutant_type": "core_physicochemical",
                "pollutant_importance": "high",
                "transform": "identity",
                "land_cover_subclass": "c41",
                "distance_step_index": 1,
                "distance_step_name": "0_10km",
                "included_buckets": "0_10km",
                "forced_regressors": DEFAULT_SETTINGS.land_cover_column("0_10km", "c41"),
                "candidate_regressors": "cl_0_10km_tp_mean_7d__scaled",
                "formula": "y ~ x + z + xz - 1",
                "nobs": 100,
                "selected_by_lasso": False,
                "lasso_alpha": 0.01,
                "lasso_selected_count": 2,
                "lasso_candidate_count": 2,
                "lasso_valid_candidate_count": 2,
                "lasso_selected_share": 1.0,
                "lasso_min_cv_mse": 0.12,
                "map_iterations": 3,
                "map_converged": True,
            },
            {
                "term": f"{DEFAULT_SETTINGS.land_cover_column('0_10km', 'c41')}__x__cl_0_10km_tp_mean_7d__scaled",
                "Estimate": 0.31,
                "2.5%": 0.08,
                "97.5%": 0.52,
                "t value": 2.1,
                "Pr(>|t|)": 0.04,
                "pollutant": "ph",
                "pollutant_group_kind": "type",
                "pollutant_group_name": "nutrients",
                "model_family": "post_lasso",
                "pollutant_type": "core_physicochemical",
                "pollutant_importance": "high",
                "transform": "identity",
                "land_cover_subclass": "c41",
                "distance_step_index": 1,
                "distance_step_name": "0_10km",
                "included_buckets": "0_10km",
                "forced_regressors": DEFAULT_SETTINGS.land_cover_column("0_10km", "c41"),
                "candidate_regressors": "cl_0_10km_tp_mean_7d__scaled",
                "formula": "y ~ x + z + xz - 1",
                "nobs": 100,
                "selected_by_lasso": True,
                "lasso_alpha": 0.01,
                "lasso_selected_count": 2,
                "lasso_candidate_count": 2,
                "lasso_valid_candidate_count": 2,
                "lasso_selected_share": 1.0,
                "lasso_min_cv_mse": 0.12,
                "map_iterations": 3,
                "map_converged": True,
            },
        ]
    )
    manifest = pd.DataFrame(
        [
            {
                "pollutant": "ph",
                "pollutant_group_kind": "type",
                "pollutant_group_name": "nutrients",
                "model_family": "crude_twfe",
                "pollutant_type": "core_physicochemical",
                "pollutant_importance": "high",
                "transform": "identity",
                "land_cover_subclass": "c41",
                "distance_step_index": 1,
                "distance_step_name": "0_10km",
                "included_buckets": "0_10km",
                "outcome_column": "ph__transformed",
                "forced_regressors": DEFAULT_SETTINGS.land_cover_column("0_10km", "c41"),
                "candidate_regressors": "",
                "formula": "y ~ x - 1",
                "selected_terms": "",
                "lasso_alpha": None,
                "lasso_selected_count": 0,
                "lasso_candidate_count": None,
                "lasso_valid_candidate_count": None,
                "lasso_selected_share": None,
                "lasso_min_cv_mse": None,
                "map_iterations": 3,
                "map_converged": True,
                "nobs": 100,
                "status": "ok",
                "error": None,
            },
            {
                "pollutant": "ph",
                "pollutant_group_kind": "type",
                "pollutant_group_name": "nutrients",
                "model_family": "post_lasso",
                "pollutant_type": "core_physicochemical",
                "pollutant_importance": "high",
                "transform": "identity",
                "land_cover_subclass": "c41",
                "distance_step_index": 1,
                "distance_step_name": "0_10km",
                "included_buckets": "0_10km",
                "outcome_column": "ph__transformed",
                "forced_regressors": DEFAULT_SETTINGS.land_cover_column("0_10km", "c41"),
                "candidate_regressors": "cl_0_10km_tp_mean_7d__scaled",
                "formula": "y ~ x + z + xz - 1",
                "selected_terms": "cl_0_10km_tp_mean_7d__scaled",
                "lasso_alpha": 0.01,
                "lasso_selected_count": 2,
                "lasso_candidate_count": 2,
                "lasso_valid_candidate_count": 2,
                "lasso_selected_share": 1.0,
                "lasso_min_cv_mse": 0.12,
                "map_iterations": 3,
                "map_converged": True,
                "nobs": 100,
                "status": "ok",
                "error": None,
            },
            {
                "pollutant": "ph",
                "pollutant_group_kind": "type",
                "pollutant_group_name": "nutrients",
                "model_family": "post_lasso",
                "pollutant_type": "core_physicochemical",
                "pollutant_importance": "high",
                "transform": "identity",
                "land_cover_subclass": "c41",
                "distance_step_index": 2,
                "distance_step_name": "10_50km",
                "included_buckets": "0_10km,10_50km",
                "outcome_column": "ph__transformed",
                "forced_regressors": DEFAULT_SETTINGS.land_cover_column("10_50km", "c41"),
                "candidate_regressors": "cl_0_10km_tp_mean_7d__scaled",
                "formula": "y ~ x + z + xz - 1",
                "selected_terms": "",
                "lasso_alpha": None,
                "lasso_selected_count": 0,
                "lasso_candidate_count": 2,
                "lasso_valid_candidate_count": 0,
                "lasso_selected_share": 0.0,
                "lasso_min_cv_mse": None,
                "map_iterations": 1000,
                "map_converged": False,
                "nobs": 50,
                "status": "error",
                "error": "synthetic failure",
            },
        ]
    )
    results.to_parquet(run_dir / "results.parquet", index=False)
    manifest.to_parquet(run_dir / "manifest.parquet", index=False)
    return run_dir


def test_plotly_app_helpers_load_and_filter_runs(tmp_path: Path) -> None:
    run_dir = _write_synthetic_run(tmp_path)

    discovered = discover_result_runs(tmp_path)
    assert discovered == [run_dir]

    run = load_result_run(run_dir)
    assert {"land_cover", "climate", "interaction"} == set(run.app_results["term_group"])
    assert "through 0-10 km" in run.app_results["profile_facet_label"].iloc[0]

    filtered = filter_app_frame(
        run.app_results,
        model_families=["post_lasso"],
        term_groups=["interaction"],
        selected_only=True,
        significant_only=True,
    )
    assert filtered["model_family"].unique().tolist() == ["post_lasso"]
    assert filtered["term_group"].unique().tolist() == ["interaction"]
    assert filtered["selected_by_lasso"].all()
    assert filtered["is_significant"].all()


def test_dashboard_analysis_helpers_summarize_and_compare(tmp_path: Path) -> None:
    run_dir = _write_synthetic_run(tmp_path)
    run = load_result_run(run_dir)

    diagnostics = make_diagnostics_table(run.app_manifest, run.app_results)
    diagnostic_values = dict(diagnostics.to_records(index=False))
    assert diagnostic_values["Total models"] == "3"
    assert diagnostic_values["Failed models"] == "1"
    assert diagnostic_values["Success rate"] == "66.7%"
    assert diagnostic_values["LASSO-selected coefficient rows"] == "2"
    assert diagnostic_values["Mean LASSO candidate terms"] == "2.0"

    comparison = build_model_comparison_table(run.app_results)
    assert comparison["crude_twfe"].tolist() == [0.25]
    assert comparison["post_lasso"].tolist() == [0.20]
    assert comparison["estimate_delta"].round(2).tolist() == [-0.05]

    significance = build_significance_matrix_table(run.app_results)
    assert significance["status"].tolist() == ["Significant positive"]

    lasso_stats = make_lasso_stats_table(run.app_manifest)
    assert lasso_stats["lasso_candidate_count"].tolist() == [2, 2]
    assert lasso_stats["lasso_selected_count"].tolist() == [2, 0]

    empty_results = run.app_results.iloc[0:0]
    assert build_model_comparison_table(empty_results).empty
    assert build_significance_matrix_table(empty_results).empty


def test_cli_plotly_app_command_invokes_server(monkeypatch, tmp_path: Path) -> None:
    run_dir = _write_synthetic_run(tmp_path)
    captured: dict[str, object] = {}

    def _fake_run_plotly_app(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(analysis_cli, "run_plotly_app", _fake_run_plotly_app)
    exit_code = analysis_main(
        [
            "--output-dir",
            str(tmp_path),
            "plotly-app",
            "--results-dir",
            str(tmp_path),
            "--run-name",
            run_dir.name,
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--max-facets",
            "8",
            "--top-terms",
            "15",
        ]
    )

    assert exit_code == 0
    assert Path(captured["results_dir"]) == tmp_path
    assert captured["run_name"] == run_dir.name
    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 9000
    assert captured["max_facets"] == 8
    assert captured["top_terms"] == 15


def test_diagnostics_and_lasso_stats_surface_new_columns() -> None:
    manifest = pd.DataFrame(
        [
            {
                "model_family": "post_lasso",
                "pollutant_label": "Ph",
                "land_cover_label": "Forest",
                "distance_step_name": "0_10km",
                "distance_step_label": "0-10 km",
                "status": "ok",
                "nobs": 100,
                "lasso_alpha": 0.01,
                "lasso_candidate_count": 4,
                "lasso_valid_candidate_count": 4,
                "lasso_pruned_candidate_count": 1,
                "lasso_selected_count": 2,
                "lasso_selected_share": 0.5,
                "lasso_min_cv_mse": 0.1,
                "lasso_converged": True,
                "lasso_attempts": 1,
                "numerical_status": "ok",
                "total_seconds": 2.5,
                "ols_warning_count": 0,
                "map_converged": True,
            },
            {
                "model_family": "post_lasso",
                "pollutant_label": "Ph",
                "land_cover_label": "Forest",
                "distance_step_name": "10_50km",
                "distance_step_label": "10-50 km",
                "status": "ok",
                "nobs": 90,
                "lasso_alpha": 0.02,
                "lasso_candidate_count": 4,
                "lasso_valid_candidate_count": 3,
                "lasso_pruned_candidate_count": 3,
                "lasso_selected_count": 1,
                "lasso_selected_share": 0.25,
                "lasso_min_cv_mse": 0.2,
                "lasso_converged": False,
                "lasso_attempts": 2,
                "numerical_status": "lasso_retried",
                "total_seconds": 3.5,
                "ols_warning_count": 1,
                "map_converged": True,
            },
        ]
    )
    results = pd.DataFrame({"selected_by_lasso": [True, False]})

    diagnostics = make_diagnostics_table(manifest, results)
    diagnostic_values = dict(diagnostics.to_records(index=False))
    assert diagnostic_values["Mean LASSO pruned candidates"] == "2.0"
    assert diagnostic_values["LASSO convergence rate"] == "50.0%"
    assert diagnostic_values["Mean LASSO fit attempts"] == "1.50"
    assert diagnostic_values["Mean total runtime (s)"] == "3.00"
    assert diagnostic_values["OLS warnings"] == "1"

    lasso_stats = make_lasso_stats_table(manifest)
    assert lasso_stats["lasso_pruned_candidate_count"].tolist() == [1, 3]
    assert lasso_stats["lasso_converged"].tolist() == [True, False]
    assert lasso_stats["lasso_attempts"].tolist() == [1, 2]
    assert lasso_stats["numerical_status"].tolist() == ["ok", "lasso_retried"]


def _spec(distance_step_index: int, distance_step_name: str, included_buckets: tuple[str, ...]) -> ModelSpec:
    return ModelSpec(
        pollutant="ph",
        pollutant_group_kind="explicit",
        pollutant_group_name="custom",
        model_family="post_lasso",
        land_cover_subclass="c41",
        distance_step_index=distance_step_index,
        distance_step_name=distance_step_name,
        included_buckets=included_buckets,
        outcome_column="ph__transformed",
        coefficient_columns=("lc",),
        forced_regressor_columns=("lc",),
        candidate_regressor_columns=("climate",),
    )


def test_pending_run_discovery_and_partial_loading(tmp_path: Path) -> None:
    complete_dir = _write_synthetic_run(tmp_path)
    pending_dir = tmp_path / "type_metals"
    pending_dir.mkdir(parents=True, exist_ok=True)

    fingerprint = "wip-fingerprint"
    spec_a = _spec(1, "0_10km", ("0_10km",))
    spec_b = _spec(2, "10_50km", ("0_10km", "10_50km"))
    settings = DEFAULT_SETTINGS
    write_shard_metadata(pending_dir, fingerprint, 0, 2, [spec_a.spec_id], settings)
    write_shard_metadata(pending_dir, fingerprint, 1, 2, [spec_b.spec_id], settings)
    write_chunk(
        pending_dir,
        fingerprint,
        0,
        0,
        pd.DataFrame(
            {
                "spec_id": [spec_a.spec_id],
                "term": [DEFAULT_SETTINGS.land_cover_column("0_10km", "c41")],
                "Estimate": [0.3],
                "Std. Error": [0.05],
                "t value": [6.0],
                "Pr(>|t|)": [0.001],
                "2.5%": [0.20],
                "97.5%": [0.40],
                "pollutant": ["ph"],
                "pollutant_group_kind": ["explicit"],
                "pollutant_group_name": ["custom"],
                "model_family": ["post_lasso"],
                "pollutant_type": ["core_physicochemical"],
                "pollutant_importance": ["high"],
                "transform": ["identity"],
                "land_cover_subclass": ["c41"],
                "distance_step_index": [1],
                "distance_step_name": ["0_10km"],
                "included_buckets": ["0_10km"],
                "forced_regressors": [DEFAULT_SETTINGS.land_cover_column("0_10km", "c41")],
                "candidate_regressors": [""],
                "formula": ["y ~ x - 1"],
                "nobs": [80],
                "selected_by_lasso": [False],
                "lasso_alpha": [None],
                "lasso_selected_count": [0],
                "lasso_candidate_count": [None],
                "lasso_valid_candidate_count": [None],
                "lasso_selected_share": [None],
                "lasso_min_cv_mse": [None],
                "map_iterations": [3],
                "map_converged": [True],
            }
        ),
        pd.DataFrame(
            {
                "spec_id": [spec_a.spec_id],
                "pollutant": ["ph"],
                "pollutant_group_kind": ["explicit"],
                "pollutant_group_name": ["custom"],
                "model_family": ["post_lasso"],
                "pollutant_type": ["core_physicochemical"],
                "pollutant_importance": ["high"],
                "transform": ["identity"],
                "land_cover_subclass": ["c41"],
                "distance_step_index": [1],
                "distance_step_name": ["0_10km"],
                "included_buckets": ["0_10km"],
                "outcome_column": ["ph__transformed"],
                "forced_regressors": [DEFAULT_SETTINGS.land_cover_column("0_10km", "c41")],
                "candidate_regressors": [""],
                "formula": ["y ~ x - 1"],
                "selected_terms": [""],
                "lasso_alpha": [None],
                "lasso_selected_count": [0],
                "lasso_candidate_count": [None],
                "lasso_valid_candidate_count": [None],
                "lasso_selected_share": [None],
                "lasso_min_cv_mse": [None],
                "map_iterations": [3],
                "map_converged": [True],
                "nobs": [80],
                "status": ["ok"],
                "error": [None],
            }
        ),
    )
    mark_shard_complete(pending_dir, fingerprint, 0)
    # Shard 1 has no chunks yet, so the run is still in progress overall.

    discovered = discover_result_runs(tmp_path)
    assert discovered == [complete_dir]

    pending = discover_pending_runs(tmp_path)
    assert pending == [pending_dir]

    run, progress = load_pending_run(pending_dir)
    assert progress.shard_count == 2
    assert progress.shards_complete == 1
    assert progress.specs_expected == 2
    assert progress.specs_done == 1
    assert run.app_manifest["status"].tolist() == ["ok"]
    assert run.app_results.shape[0] == 1
