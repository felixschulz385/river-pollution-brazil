from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.modules.pop("code", None)

from code.analysis.cli import main as analysis_main
from code.analysis import cli as analysis_cli
from code.analysis.sensor_data.plotly_app import (
    discover_result_runs,
    filter_app_frame,
    load_result_run,
)
from code.analysis.settings import DEFAULT_SETTINGS


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
                "map_iterations": 3,
                "map_converged": True,
                "nobs": 100,
                "status": "ok",
                "error": None,
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
