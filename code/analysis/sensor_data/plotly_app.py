"""Interactive Plotly/Dash app for sensor-analysis regression outputs."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from .results import build_readable_manifest_table, build_readable_results_table
from ..settings import DEFAULT_SETTINGS, SensorAnalysisSettings


DEFAULT_MAX_FACETS = 12
DEFAULT_TOP_TERMS = 20


@dataclass(frozen=True)
class SensorResultRun:
    """Loaded regression outputs for one saved analysis run."""

    run_name: str
    run_dir: Path
    results: pd.DataFrame
    manifest: pd.DataFrame
    app_results: pd.DataFrame
    app_manifest: pd.DataFrame


def discover_result_runs(base_dir: str | Path) -> list[Path]:
    """Return result-run directories containing persisted manifests."""
    root = Path(base_dir)
    if not root.exists():
        return []
    return sorted(
        [
            path
            for path in root.iterdir()
            if path.is_dir() and (path / "manifest.parquet").exists()
        ],
        key=lambda path: path.name,
    )


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
            pieces = []
            if variable.variable_name is not None:
                pieces.append(variable.variable_name.replace("_", " ").title())
            if variable.window_name is not None:
                pieces.append(variable.window_name.replace("_", " "))
            if variable.distance_bucket is not None:
                pieces.append(settings.distance_bucket_label(variable.distance_bucket))
            return ", ".join(pieces) if pieces else variable.source_column.replace("_", " ").title()
    return term.replace("_", " ").replace("__", " ").title()


def _term_group(term: str, settings: SensorAnalysisSettings) -> str:
    if "__x__" in term:
        return "interaction"
    if term.endswith("__scaled"):
        return "climate"
    control_terms = {variable.scaled_column for variable in settings.controls}
    if term in control_terms:
        return "control"
    land_cover_terms = {
        settings.land_cover_column(bucket, subclass)
        for bucket in settings.distance_buckets
        for subclass in settings.land_cover_subclasses
    }
    if term in land_cover_terms:
        return "land_cover"
    return "other"


def _land_cover_lookup(
    settings: SensorAnalysisSettings,
) -> dict[str, tuple[str, str]]:
    return {
        settings.land_cover_column(bucket, subclass): (bucket, subclass)
        for bucket in settings.distance_buckets
        for subclass in settings.land_cover_subclasses
    }


def _enrich_results_for_app(
    results: pd.DataFrame,
    settings: SensorAnalysisSettings,
) -> pd.DataFrame:
    if results.empty:
        frame = build_readable_results_table(results, settings)
        frame["pollutant_label"] = pd.Series(dtype="object")
        frame["land_cover_label"] = pd.Series(dtype="object")
        frame["distance_label"] = pd.Series(dtype="object")
        frame["distance_step_label"] = pd.Series(dtype="object")
        frame["term_label"] = pd.Series(dtype="object")
        frame["model_family_label"] = pd.Series(dtype="object")
        frame["term_group"] = pd.Series(dtype="object")
        frame["is_significant"] = pd.Series(dtype="bool")
        frame["facet_label"] = pd.Series(dtype="object")
        frame["term_bucket"] = pd.Series(dtype="object")
        frame["term_subclass"] = pd.Series(dtype="object")
        frame["term_bucket_label"] = pd.Series(dtype="object")
        frame["profile_facet_label"] = pd.Series(dtype="object")
        return frame

    frame = build_readable_results_table(results, settings)
    if "pollutant_label" not in frame.columns:
        frame["pollutant_label"] = frame["pollutant"].map(_pollutant_label)
    if "land_cover_label" not in frame.columns:
        frame["land_cover_label"] = frame["land_cover_subclass"].map(
            lambda value: settings.subclass_labels.get(str(value), str(value))
        )
    if "distance_label" not in frame.columns:
        frame["distance_label"] = frame["distance_step_name"].map(settings.distance_bucket_label)
    frame["distance_step_label"] = frame["distance_step_name"].map(settings.distance_bucket_label)
    if "term_label" not in frame.columns:
        frame["term_label"] = frame["term"].map(lambda value: _term_label(str(value), settings))
    frame["model_family_label"] = frame["model_family"].map(settings.model_family_label)
    frame["term_group"] = frame["term"].map(lambda value: _term_group(str(value), settings))
    p_value_column = "Pr(>|t|)"
    if p_value_column in frame.columns:
        frame["is_significant"] = pd.to_numeric(
            frame[p_value_column],
            errors="coerce",
        ).le(0.05)
    else:
        frame["is_significant"] = False
    frame["abs_estimate"] = pd.to_numeric(frame.get("Estimate"), errors="coerce").abs()
    frame["abs_t_value"] = pd.to_numeric(frame.get("t value"), errors="coerce").abs()
    frame["facet_label"] = (
        frame["pollutant_label"].astype(str) + " | " + frame["land_cover_label"].astype(str)
    )

    lookup = _land_cover_lookup(settings)
    mapped = frame["term"].map(lookup)
    frame["term_bucket"] = mapped.map(lambda value: value[0] if isinstance(value, tuple) else None)
    frame["term_subclass"] = mapped.map(lambda value: value[1] if isinstance(value, tuple) else None)
    frame["term_bucket_label"] = frame["term_bucket"].map(
        lambda value: settings.distance_bucket_label(str(value)) if pd.notna(value) else None
    )
    frame["profile_facet_label"] = (
        frame["facet_label"].astype(str) + " | through " + frame["distance_step_label"].astype(str)
    )
    return frame


def _enrich_manifest_for_app(
    manifest: pd.DataFrame,
    settings: SensorAnalysisSettings,
) -> pd.DataFrame:
    if manifest.empty:
        frame = build_readable_manifest_table(manifest, settings)
        frame["status_ok"] = pd.Series(dtype="bool")
        frame["model_family_label"] = pd.Series(dtype="object")
        frame["distance_step_label"] = pd.Series(dtype="object")
        return frame

    frame = build_readable_manifest_table(manifest, settings)
    frame["model_family_label"] = frame["model_family"].map(settings.model_family_label)
    frame["status_ok"] = frame["status"].eq("ok")
    frame["distance_step_label"] = frame["distance_step_name"].map(settings.distance_bucket_label)
    return frame


@lru_cache(maxsize=16)
def _load_result_tables(run_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cache raw parquet loads by run directory."""
    directory = Path(run_dir)
    results_path = directory / "results.parquet"
    manifest_path = directory / "manifest.parquet"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results parquet: {results_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest parquet: {manifest_path}")

    return pd.read_parquet(results_path), pd.read_parquet(manifest_path)


def load_result_run(
    run_dir: str | Path,
    *,
    settings: SensorAnalysisSettings = DEFAULT_SETTINGS,
) -> SensorResultRun:
    """Load one persisted run and enrich it for interactive display."""
    directory = Path(run_dir)
    results, manifest = _load_result_tables(str(directory))
    return SensorResultRun(
        run_name=directory.name,
        run_dir=directory,
        results=results.copy(),
        manifest=manifest.copy(),
        app_results=_enrich_results_for_app(results, settings),
        app_manifest=_enrich_manifest_for_app(manifest, settings),
    )


def _limit_facets(
    frame: pd.DataFrame,
    *,
    facet_column: str,
    max_facets: int,
) -> tuple[pd.DataFrame, int]:
    if frame.empty:
        return frame, 0
    facet_counts = (
        frame.groupby(facet_column, dropna=False)
        .size()
        .sort_values(ascending=False)
    )
    keep = facet_counts.head(max_facets).index
    limited = frame.loc[frame[facet_column].isin(keep)].copy()
    return limited, int(facet_counts.shape[0] - len(keep))


def filter_app_frame(
    frame: pd.DataFrame,
    *,
    model_families: list[str] | None = None,
    pollutants: list[str] | None = None,
    subclasses: list[str] | None = None,
    distance_steps: list[str] | None = None,
    term_groups: list[str] | None = None,
    significant_only: bool = False,
    selected_only: bool = False,
) -> pd.DataFrame:
    """Apply common dashboard filters to a results or manifest frame."""
    filtered = frame.copy()
    if model_families:
        filtered = filtered.loc[filtered["model_family"].isin(model_families)].copy()
    if pollutants and "pollutant" in filtered.columns:
        filtered = filtered.loc[filtered["pollutant"].isin(pollutants)].copy()
    if subclasses and "land_cover_subclass" in filtered.columns:
        filtered = filtered.loc[filtered["land_cover_subclass"].isin(subclasses)].copy()
    if distance_steps and "distance_step_name" in filtered.columns:
        filtered = filtered.loc[filtered["distance_step_name"].isin(distance_steps)].copy()
    if term_groups and "term_group" in filtered.columns:
        filtered = filtered.loc[filtered["term_group"].isin(term_groups)].copy()
    if significant_only and "is_significant" in filtered.columns:
        filtered = filtered.loc[filtered["is_significant"]].copy()
    if selected_only and "selected_by_lasso" in filtered.columns:
        filtered = filtered.loc[filtered["selected_by_lasso"]].copy()
    return filtered


def _import_dash() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        from dash import Dash, Input, Output, dcc, html, dash_table
    except ImportError as exc:  # pragma: no cover - dependency-gated
        raise RuntimeError(
            "The Plotly app requires `dash` and `plotly`. "
            "Install them in the active environment first."
        ) from exc
    return Dash, Input, Output, dcc, html, dash_table


def _import_plotly() -> tuple[Any, Any]:
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - dependency-gated
        raise RuntimeError(
            "The Plotly app requires `plotly`. Install it in the active environment first."
        ) from exc
    return px, go


def _empty_figure(title: str):
    _, go = _import_plotly()
    figure = go.Figure()
    figure.update_layout(
        template="plotly_white",
        title=title,
        annotations=[
            {
                "text": "No data available for the current filters.",
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 15},
            }
        ],
    )
    return figure


def make_status_heatmap(manifest: pd.DataFrame):
    """Build a heatmap of successful model shares."""
    px, _ = _import_plotly()
    if manifest.empty:
        return _empty_figure("Model success by pollutant and land cover")

    summary = (
        manifest.groupby(["pollutant_label", "land_cover_label"], dropna=False)["status_ok"]
        .mean()
        .reset_index(name="ok_share")
    )
    figure = px.density_heatmap(
        summary,
        x="land_cover_label",
        y="pollutant_label",
        z="ok_share",
        color_continuous_scale="Tealgrn",
        range_color=(0.0, 1.0),
        labels={
            "land_cover_label": "Land cover",
            "pollutant_label": "Pollutant",
            "ok_share": "Successful share",
        },
        title="Model success by pollutant and land cover",
    )
    figure.update_layout(template="plotly_white", height=600)
    return figure


def make_distance_profile(results: pd.DataFrame, *, max_facets: int = DEFAULT_MAX_FACETS):
    """Plot land-cover coefficients within cumulative distance-step specifications."""
    px, _ = _import_plotly()
    subset = _land_cover_results(results)
    if subset.empty:
        return _empty_figure("Land-cover coefficient profiles")

    subset, hidden_count = _limit_facets(
        subset,
        facet_column="profile_facet_label",
        max_facets=max_facets,
    )
    subset["term_bucket_label"] = pd.Categorical(
        subset["term_bucket_label"],
        categories=[
            settings_label
            for settings_label in list(dict.fromkeys(subset["term_bucket_label"]))
            if settings_label is not None
        ],
        ordered=True,
    )
    title = "Land-cover coefficients within cumulative distance-step specifications"
    if hidden_count > 0:
        title = f"{title} (showing top {max_facets} panels)"
    figure = px.line(
        subset.sort_values(
            ["profile_facet_label", "distance_step_index", "term_bucket", "model_family_label"]
        ),
        x="term_bucket_label",
        y="Estimate",
        error_y=subset["97.5%"] - subset["Estimate"] if {"97.5%", "Estimate"}.issubset(subset.columns) else None,
        error_y_minus=subset["Estimate"] - subset["2.5%"] if {"2.5%", "Estimate"}.issubset(subset.columns) else None,
        color="model_family_label",
        markers=True,
        facet_col="profile_facet_label",
        facet_col_wrap=3,
        hover_data=[
            "term_label",
            "distance_step_label",
            "nobs",
            "lasso_alpha",
            "lasso_selected_count",
            "lasso_candidate_count",
            "map_converged",
        ],
        labels={
            "term_bucket_label": "Coefficient bucket",
            "Estimate": "Coefficient estimate",
            "model_family_label": "Model family",
            "profile_facet_label": "Panel",
        },
        title=title,
    )
    figure.update_layout(template="plotly_white", height=320 * min(max_facets, 4))
    figure.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split("=")[-1]))
    figure.add_hline(y=0.0, line_dash="dash", line_color="black", opacity=0.5)
    return figure


def make_top_terms(results: pd.DataFrame, *, top_n: int = DEFAULT_TOP_TERMS):
    """Plot the most selected or strongest non-land-cover terms."""
    px, _ = _import_plotly()
    subset = results.loc[results["term_group"].isin(["climate", "interaction", "control"])].copy()
    if subset.empty:
        return _empty_figure("Top climate and interaction terms")

    if "selected_by_lasso" in subset.columns:
        selected_subset = subset.loc[subset["selected_by_lasso"]].copy()
    else:
        selected_subset = subset.iloc[0:0].copy()
    if not selected_subset.empty:
        grouped = (
            selected_subset.groupby(["term_label", "term_group"], dropna=False)
            .agg(
                selected_count=("selected_by_lasso", "sum"),
                mean_abs_t=("abs_t_value", "mean"),
                mean_estimate=("Estimate", "mean"),
            )
            .reset_index()
            .sort_values(["selected_count", "mean_abs_t"], ascending=[False, False])
            .head(top_n)
        )
        title = "Most frequently selected climate and interaction terms"
        x_column = "selected_count"
        x_label = "Selected by lasso (count)"
    else:
        grouped = (
            subset.groupby(["term_label", "term_group"], dropna=False)
            .agg(
                mean_abs_t=("abs_t_value", "mean"),
                mean_estimate=("Estimate", "mean"),
            )
            .reset_index()
            .sort_values("mean_abs_t", ascending=False)
            .head(top_n)
        )
        title = "Strongest climate and interaction terms by |t|"
        x_column = "mean_abs_t"
        x_label = "Mean |t value|"

    grouped = grouped.sort_values(x_column, ascending=True)
    figure = px.bar(
        grouped,
        x=x_column,
        y="term_label",
        color="term_group",
        orientation="h",
        hover_data=["mean_estimate"],
        labels={
            x_column: x_label,
            "term_label": "Term",
            "term_group": "Term group",
        },
        title=title,
    )
    figure.update_layout(template="plotly_white", height=max(500, 24 * len(grouped)))
    return figure


def make_term_forest(results: pd.DataFrame, *, top_n: int = DEFAULT_TOP_TERMS):
    """Plot a compact forest plot for non-land-cover terms."""
    px, _ = _import_plotly()
    subset = results.loc[
        results["term_group"].isin(["climate", "interaction", "control"])
    ].copy()
    if subset.empty:
        return _empty_figure("Non-land-cover coefficient forest")

    score = subset["abs_t_value"].fillna(0.0)
    subset = subset.assign(_score=score)
    subset = subset.sort_values("_score", ascending=False).head(top_n).sort_values("Estimate")
    figure = px.scatter(
        subset,
        x="Estimate",
        y="term_label",
        color="model_family_label",
        symbol="term_group",
        error_x=subset["97.5%"] - subset["Estimate"] if {"97.5%", "Estimate"}.issubset(subset.columns) else None,
        error_x_minus=subset["Estimate"] - subset["2.5%"] if {"2.5%", "Estimate"}.issubset(subset.columns) else None,
        hover_data=["pollutant_label", "land_cover_label", "distance_label", "nobs"],
        labels={
            "Estimate": "Coefficient estimate",
            "term_label": "Term",
            "model_family_label": "Model family",
            "term_group": "Term group",
        },
        title="Top non-land-cover coefficients",
    )
    figure.update_layout(template="plotly_white", height=max(500, 26 * len(subset)))
    figure.add_vline(x=0.0, line_dash="dash", line_color="black", opacity=0.5)
    return figure


def make_diagnostics_table(manifest: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Build compact diagnostics for the active dashboard filters."""
    total_models = int(manifest.shape[0])
    ok_models = int(manifest["status_ok"].sum()) if "status_ok" in manifest.columns else 0
    failed_models = total_models - ok_models
    nobs = (
        pd.to_numeric(manifest["nobs"], errors="coerce")
        if "nobs" in manifest.columns
        else pd.Series(dtype="float64")
    )
    lasso_counts = (
        pd.to_numeric(manifest["lasso_selected_count"], errors="coerce")
        if "lasso_selected_count" in manifest.columns
        else pd.Series(dtype="float64")
    )
    lasso_alpha = (
        pd.to_numeric(manifest["lasso_alpha"], errors="coerce")
        if "lasso_alpha" in manifest.columns
        else pd.Series(dtype="float64")
    )
    lasso_candidates = (
        pd.to_numeric(manifest["lasso_candidate_count"], errors="coerce")
        if "lasso_candidate_count" in manifest.columns
        else pd.Series(dtype="float64")
    )
    lasso_valid = (
        pd.to_numeric(manifest["lasso_valid_candidate_count"], errors="coerce")
        if "lasso_valid_candidate_count" in manifest.columns
        else pd.Series(dtype="float64")
    )
    lasso_share = (
        pd.to_numeric(manifest["lasso_selected_share"], errors="coerce")
        if "lasso_selected_share" in manifest.columns
        else pd.Series(dtype="float64")
    )

    if "map_converged" in manifest.columns and total_models > 0:
        converged_share = float(manifest["map_converged"].fillna(False).mean())
    else:
        converged_share = 0.0
    selected_rows = (
        int(results["selected_by_lasso"].fillna(False).sum())
        if "selected_by_lasso" in results.columns
        else 0
    )

    diagnostics = [
        ("Total models", f"{total_models:,}"),
        ("Successful models", f"{ok_models:,}"),
        ("Failed models", f"{failed_models:,}"),
        ("Success rate", f"{ok_models / total_models:.1%}" if total_models else "0.0%"),
        ("Converged share", f"{converged_share:.1%}" if total_models else "0.0%"),
        ("Mean sample size", f"{nobs.mean():,.0f}" if nobs.notna().any() else "0"),
        ("Median sample size", f"{nobs.median():,.0f}" if nobs.notna().any() else "0"),
        (
            "Mean LASSO selected terms",
            f"{lasso_counts.mean():.1f}" if lasso_counts.notna().any() else "0.0",
        ),
        (
            "Mean LASSO alpha",
            f"{lasso_alpha.mean():.4f}" if lasso_alpha.notna().any() else "NA",
        ),
        (
            "Mean LASSO candidate terms",
            f"{lasso_candidates.mean():.1f}" if lasso_candidates.notna().any() else "0.0",
        ),
        (
            "Mean valid LASSO candidates",
            f"{lasso_valid.mean():.1f}" if lasso_valid.notna().any() else "0.0",
        ),
        (
            "Mean LASSO selection share",
            f"{lasso_share.mean():.1%}" if lasso_share.notna().any() else "0.0%",
        ),
        ("LASSO-selected coefficient rows", f"{selected_rows:,}"),
    ]
    return pd.DataFrame(diagnostics, columns=["metric", "value"])


def make_lasso_stats_table(manifest: pd.DataFrame) -> pd.DataFrame:
    """Build a compact per-specification LASSO diagnostics table."""
    if manifest.empty or "model_family" not in manifest.columns:
        return pd.DataFrame(
            columns=[
                "pollutant_label",
                "land_cover_label",
                "distance_step_label",
                "lasso_alpha",
                "lasso_candidate_count",
                "lasso_valid_candidate_count",
                "lasso_selected_count",
                "lasso_selected_share",
                "lasso_min_cv_mse",
                "status",
            ]
        )
    subset = manifest.loc[manifest["model_family"].eq("post_lasso")].copy()
    if subset.empty:
        return pd.DataFrame(
            columns=[
                "pollutant_label",
                "land_cover_label",
                "distance_step_label",
                "lasso_alpha",
                "lasso_candidate_count",
                "lasso_valid_candidate_count",
                "lasso_selected_count",
                "lasso_selected_share",
                "lasso_min_cv_mse",
                "status",
            ]
        )
    preferred = [
        "pollutant_label",
        "land_cover_label",
        "distance_step_label",
        "lasso_alpha",
        "lasso_candidate_count",
        "lasso_valid_candidate_count",
        "lasso_selected_count",
        "lasso_selected_share",
        "lasso_min_cv_mse",
        "status",
        "selected_terms",
        "nobs",
        "map_converged",
    ]
    subset = subset.sort_values(
        ["pollutant_label", "land_cover_label", "distance_step_name"],
        kind="stable",
    )
    available = [column for column in preferred if column in subset.columns]
    return subset.loc[:, available]


def _land_cover_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty or "term_group" not in results.columns:
        return results.iloc[0:0].copy()
    return results.loc[
        results["term_group"].eq("land_cover")
        & results["term_subclass"].eq(results["land_cover_subclass"])
    ].copy()


def build_model_comparison_table(results: pd.DataFrame) -> pd.DataFrame:
    """Align land-cover estimates for comparison across model families."""
    subset = _land_cover_results(results)
    required = {
        "pollutant",
        "pollutant_label",
        "land_cover_subclass",
        "land_cover_label",
        "distance_step_name",
        "distance_step_index",
        "distance_label",
        "term",
        "model_family",
        "Estimate",
    }
    if subset.empty or not required.issubset(subset.columns):
        return pd.DataFrame()

    index_columns = [
        "pollutant",
        "pollutant_label",
        "land_cover_subclass",
        "land_cover_label",
        "distance_step_name",
        "distance_step_index",
        "distance_label",
        "term",
    ]
    grouped = (
        subset.assign(Estimate=pd.to_numeric(subset["Estimate"], errors="coerce"))
        .groupby(index_columns + ["model_family"], dropna=False)
        .agg(
            Estimate=("Estimate", "mean"),
            nobs=("nobs", "mean") if "nobs" in subset.columns else ("Estimate", "count"),
        )
        .reset_index()
    )
    estimates = grouped.pivot_table(
        index=index_columns,
        columns="model_family",
        values="Estimate",
        aggfunc="mean",
    ).reset_index()
    nobs = grouped.groupby(index_columns, dropna=False)["nobs"].mean().reset_index()
    comparison = estimates.merge(nobs, on=index_columns, how="left")
    if not {"crude_twfe", "post_lasso"}.issubset(comparison.columns):
        return pd.DataFrame()

    comparison = comparison.dropna(subset=["crude_twfe", "post_lasso"]).copy()
    if comparison.empty:
        return pd.DataFrame()

    comparison["estimate_delta"] = comparison["post_lasso"] - comparison["crude_twfe"]
    return comparison


def make_model_comparison(results: pd.DataFrame):
    """Compare land-cover estimates for the same specification across model families."""
    px, go = _import_plotly()
    comparison = build_model_comparison_table(results)
    if comparison.empty:
        return _empty_figure("Crude TWFE vs Post-LASSO land-cover estimates")

    figure = px.scatter(
        comparison,
        x="crude_twfe",
        y="post_lasso",
        color="land_cover_label",
        symbol="distance_label",
        hover_data=[
            "pollutant_label",
            "land_cover_label",
            "distance_label",
            "nobs",
            "estimate_delta",
        ],
        labels={
            "crude_twfe": "Crude TWFE estimate",
            "post_lasso": "Post-LASSO estimate",
            "land_cover_label": "Land cover",
            "distance_label": "Distance",
            "estimate_delta": "Post-LASSO minus Crude TWFE",
        },
        title="Crude TWFE vs Post-LASSO land-cover estimates",
    )
    axis_values = pd.concat([comparison["crude_twfe"], comparison["post_lasso"]]).dropna()
    if not axis_values.empty:
        padding = max((axis_values.max() - axis_values.min()) * 0.05, 0.01)
        lower = float(axis_values.min() - padding)
        upper = float(axis_values.max() + padding)
        figure.add_trace(
            go.Scatter(
                x=[lower, upper],
                y=[lower, upper],
                mode="lines",
                name="Equal estimates",
                line={"color": "black", "dash": "dash"},
            )
        )
        figure.update_xaxes(range=[lower, upper])
        figure.update_yaxes(range=[lower, upper])
    figure.update_layout(template="plotly_white", height=620)
    return figure


def build_significance_matrix_table(results: pd.DataFrame) -> pd.DataFrame:
    """Classify land-cover coefficient significance by pollutant and distance."""
    subset = _land_cover_results(results)
    required = {
        "facet_label",
        "distance_label",
        "distance_step_index",
        "Estimate",
        "is_significant",
    }
    if subset.empty or not required.issubset(subset.columns):
        return pd.DataFrame()

    subset = subset.assign(
        Estimate=pd.to_numeric(subset["Estimate"], errors="coerce"),
        is_significant=subset["is_significant"].fillna(False).astype(bool),
    )
    grouped = (
        subset.groupby(["facet_label", "distance_label", "distance_step_index"], dropna=False)
        .agg(
            has_estimate=("Estimate", lambda values: values.notna().any()),
            significant_positive=(
                "Estimate",
                lambda values: bool(
                    ((values > 0) & subset.loc[values.index, "is_significant"]).any()
                ),
            ),
            significant_negative=(
                "Estimate",
                lambda values: bool(
                    ((values < 0) & subset.loc[values.index, "is_significant"]).any()
                ),
            ),
        )
        .reset_index()
    )

    def _status(row: pd.Series) -> str:
        if not row["has_estimate"]:
            return "Missing"
        if row["significant_positive"] and row["significant_negative"]:
            return "Mixed significant"
        if row["significant_positive"]:
            return "Significant positive"
        if row["significant_negative"]:
            return "Significant negative"
        return "Insignificant"

    grouped["status"] = grouped.apply(_status, axis=1)
    return grouped


def make_significance_matrix(results: pd.DataFrame):
    """Show the sign and significance of land-cover estimates by distance bucket."""
    _, go = _import_plotly()
    grouped = build_significance_matrix_table(results)
    if grouped.empty:
        return _empty_figure("Land-cover significance by distance")

    status_order = [
        "Missing",
        "Insignificant",
        "Significant negative",
        "Mixed significant",
        "Significant positive",
    ]
    status_codes = {status: index for index, status in enumerate(status_order)}
    grouped["status_code"] = grouped["status"].map(status_codes)
    x_order = (
        grouped[["distance_label", "distance_step_index"]]
        .drop_duplicates()
        .sort_values("distance_step_index")["distance_label"]
        .tolist()
    )
    y_order = sorted(grouped["facet_label"].dropna().unique().tolist())
    matrix = (
        grouped.pivot(index="facet_label", columns="distance_label", values="status_code")
        .reindex(index=y_order, columns=x_order)
    )
    hover = (
        grouped.pivot(index="facet_label", columns="distance_label", values="status")
        .reindex(index=y_order, columns=x_order)
    )
    colorscale = [
        [0.00, "#d8d8d8"],
        [0.24, "#d8d8d8"],
        [0.25, "#f1eadf"],
        [0.49, "#f1eadf"],
        [0.50, "#3b7ea1"],
        [0.74, "#3b7ea1"],
        [0.75, "#756bb1"],
        [0.87, "#756bb1"],
        [0.88, "#b24b45"],
        [1.00, "#b24b45"],
    ]
    figure = go.Figure(
        data=[
            go.Heatmap(
                z=matrix.to_numpy(),
                x=x_order,
                y=y_order,
                text=hover.to_numpy(),
                hovertemplate="%{y}<br>%{x}<br>%{text}<extra></extra>",
                zmin=0,
                zmax=len(status_order) - 1,
                colorscale=colorscale,
                colorbar={
                    "tickmode": "array",
                    "tickvals": list(status_codes.values()),
                    "ticktext": status_order,
                    "title": "Status",
                },
            )
        ]
    )
    figure.update_layout(
        template="plotly_white",
        title="Land-cover significance by distance",
        xaxis_title="Distance bucket",
        yaxis_title="Pollutant | land cover",
        height=max(520, 28 * len(y_order)),
    )
    return figure


def _summary_cards(html, manifest: pd.DataFrame, results: pd.DataFrame):
    total_models = int(manifest.shape[0])
    ok_models = int(manifest["status_ok"].sum()) if "status_ok" in manifest.columns else 0
    pollutants = int(results["pollutant"].nunique()) if "pollutant" in results.columns else 0
    mean_nobs = float(pd.to_numeric(manifest.get("nobs"), errors="coerce").mean() or 0.0)

    card_style = {
        "padding": "18px",
        "borderRadius": "14px",
        "background": "linear-gradient(180deg, #f7f3ea 0%, #efe8d9 100%)",
        "border": "1px solid #d8c9a8",
        "minWidth": "180px",
    }
    label_style = {"fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "0.08em", "color": "#665c4b"}
    value_style = {"fontSize": "28px", "fontWeight": 700, "color": "#17241f"}
    return html.Div(
        [
            html.Div([html.Div("Successful models", style=label_style), html.Div(f"{ok_models:,}", style=value_style)], style=card_style),
            html.Div([html.Div("Total models", style=label_style), html.Div(f"{total_models:,}", style=value_style)], style=card_style),
            html.Div([html.Div("Pollutants", style=label_style), html.Div(f"{pollutants:,}", style=value_style)], style=card_style),
            html.Div([html.Div("Mean sample size", style=label_style), html.Div(f"{mean_nobs:,.0f}", style=value_style)], style=card_style),
        ],
        style={"display": "flex", "gap": "14px", "flexWrap": "wrap"},
    )


def _dropdown_options(
    values: list[str],
    labeler: Any | None = None,
) -> list[dict[str, str]]:
    options = []
    for value in values:
        label = str(labeler(value)) if labeler is not None else str(value)
        if label != str(value):
            label = f"{label} ({value})"
        options.append({"label": label, "value": value})
    return options


def run_plotly_app(
    *,
    results_dir: str | Path,
    run_name: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = False,
    settings: SensorAnalysisSettings = DEFAULT_SETTINGS,
    max_facets: int = DEFAULT_MAX_FACETS,
    top_terms: int = DEFAULT_TOP_TERMS,
) -> None:
    """Launch the interactive Dash app for saved sensor-analysis outputs."""
    Dash, Input, Output, dcc, html, dash_table = _import_dash()

    run_directories = discover_result_runs(results_dir)
    if not run_directories:
        raise ValueError(f"No result runs found under {results_dir}.")

    run_lookup = {path.name: path for path in run_directories}
    initial_run_name = run_name or run_directories[0].name
    if initial_run_name not in run_lookup:
        raise ValueError(
            f"Run `{initial_run_name}` not found under {results_dir}. "
            f"Available runs: {sorted(run_lookup)}."
        )

    app = Dash(__name__)
    app.title = "Sensor Analysis Results"

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H1("Sensor Analysis Results", style={"marginBottom": "4px"}),
                    html.Div(
                        "Interactive review of TWFE and Post-LASSO regression outputs.",
                        style={"color": "#5b5448"},
                    ),
                ],
                style={
                    "padding": "28px 32px 12px 32px",
                    "background": "linear-gradient(135deg, #f3eee2 0%, #dfe8de 100%)",
                    "borderBottom": "1px solid #d6d2c5",
                },
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Run", style={"fontWeight": 700}),
                            dcc.Dropdown(
                                id="run-name",
                                options=_dropdown_options(sorted(run_lookup)),
                                value=initial_run_name,
                                clearable=False,
                            ),
                            html.Label("Model families", style={"fontWeight": 700, "marginTop": "14px"}),
                            dcc.Dropdown(id="model-families", multi=True),
                            html.Label("Pollutants", style={"fontWeight": 700, "marginTop": "14px"}),
                            dcc.Dropdown(id="pollutants", multi=True),
                            html.Label("Land-cover subclasses", style={"fontWeight": 700, "marginTop": "14px"}),
                            dcc.Dropdown(id="subclasses", multi=True),
                            html.Label("Distance steps", style={"fontWeight": 700, "marginTop": "14px"}),
                            dcc.Dropdown(id="distance-steps", multi=True),
                            html.Label("Term groups", style={"fontWeight": 700, "marginTop": "14px"}),
                            dcc.Dropdown(id="term-groups", multi=True),
                            dcc.Checklist(
                                id="flags",
                                options=[
                                    {"label": "Only significant coefficients", "value": "significant"},
                                    {"label": "Only lasso-selected terms", "value": "selected"},
                                ],
                                value=[],
                                style={"marginTop": "18px"},
                            ),
                        ],
                        style={
                            "width": "340px",
                            "padding": "24px",
                            "borderRight": "1px solid #ddd7ca",
                            "background": "#fbfaf6",
                        },
                    ),
                    html.Div(
                        [
                            html.Div(id="summary-cards", style={"marginBottom": "18px"}),
                            dcc.Tabs(
                                [
                                    dcc.Tab(label="Overview", children=[dcc.Graph(id="status-heatmap")]),
                                    dcc.Tab(
                                        label="Land Cover",
                                        children=[
                                            html.Div(
                                                "Each panel corresponds to a cumulative specification that includes buckets through the panel label.",
                                                style={"color": "#5b5448", "margin": "10px 0 4px 8px"},
                                            ),
                                            dcc.Graph(id="distance-profile"),
                                        ],
                                    ),
                                    dcc.Tab(
                                        label="Comparison",
                                        children=[
                                            dcc.Graph(id="model-comparison"),
                                            dcc.Graph(id="significance-matrix"),
                                            html.H3("Diagnostics"),
                                            dash_table.DataTable(
                                                id="diagnostics-table",
                                                page_size=20,
                                                style_table={"maxWidth": "620px"},
                                                style_cell={
                                                    "textAlign": "left",
                                                    "fontFamily": "Avenir Next, Helvetica Neue, sans-serif",
                                                    "fontSize": "13px",
                                                    "padding": "8px",
                                                },
                                                style_header={"fontWeight": 700},
                                            ),
                                        ],
                                    ),
                                    dcc.Tab(
                                        label="Climate and Lasso",
                                        children=[
                                            dcc.Graph(id="top-terms"),
                                            dcc.Graph(id="term-forest"),
                                            html.H3("LASSO Stats"),
                                            dash_table.DataTable(
                                                id="lasso-stats-table",
                                                page_size=15,
                                                sort_action="native",
                                                filter_action="native",
                                                style_table={"overflowX": "auto"},
                                                style_cell={
                                                    "textAlign": "left",
                                                    "fontFamily": "Menlo, monospace",
                                                    "fontSize": "12px",
                                                },
                                            ),
                                        ],
                                    ),
                                    dcc.Tab(
                                        label="Tables",
                                        children=[
                                            html.H3("Manifest"),
                                            dash_table.DataTable(
                                                id="manifest-table",
                                                page_size=15,
                                                sort_action="native",
                                                filter_action="native",
                                                style_table={"overflowX": "auto"},
                                                style_cell={"textAlign": "left", "fontFamily": "Menlo, monospace", "fontSize": "12px"},
                                            ),
                                            html.H3("Results", style={"marginTop": "18px"}),
                                            dash_table.DataTable(
                                                id="results-table",
                                                page_size=15,
                                                sort_action="native",
                                                filter_action="native",
                                                style_table={"overflowX": "auto"},
                                                style_cell={"textAlign": "left", "fontFamily": "Menlo, monospace", "fontSize": "12px"},
                                            ),
                                        ],
                                    ),
                                ]
                            ),
                        ],
                        style={"flex": "1", "padding": "20px 28px 28px 28px"},
                    ),
                ],
                style={"display": "flex", "minHeight": "calc(100vh - 110px)"},
            ),
        ],
        style={"fontFamily": "Avenir Next, Helvetica Neue, sans-serif", "background": "#f3f1ea"},
    )

    @app.callback(
        Output("model-families", "options"),
        Output("model-families", "value"),
        Output("pollutants", "options"),
        Output("pollutants", "value"),
        Output("subclasses", "options"),
        Output("subclasses", "value"),
        Output("distance-steps", "options"),
        Output("distance-steps", "value"),
        Output("term-groups", "options"),
        Output("term-groups", "value"),
        Input("run-name", "value"),
    )
    def _refresh_filter_options(selected_run_name: str):
        run = load_result_run(run_lookup[selected_run_name], settings=settings)
        results = run.app_results
        families = sorted(results["model_family"].dropna().unique().tolist())
        pollutants = sorted(results["pollutant"].dropna().unique().tolist())
        subclasses = sorted(results["land_cover_subclass"].dropna().unique().tolist())
        distance_steps = sorted(
            results["distance_step_name"].dropna().unique().tolist(),
            key=lambda value: settings.distance_buckets.index(value)
            if value in settings.distance_buckets
            else len(settings.distance_buckets),
        )
        term_groups = sorted(results["term_group"].dropna().unique().tolist())
        return (
            _dropdown_options(families, settings.model_family_label),
            families,
            _dropdown_options(pollutants, _pollutant_label),
            pollutants[: min(len(pollutants), 6)],
            _dropdown_options(
                subclasses,
                lambda value: settings.subclass_labels.get(str(value), str(value)),
            ),
            subclasses[: min(len(subclasses), 4)],
            _dropdown_options(distance_steps, settings.distance_bucket_label),
            distance_steps,
            _dropdown_options(term_groups),
            term_groups,
        )

    @app.callback(
        Output("summary-cards", "children"),
        Output("status-heatmap", "figure"),
        Output("distance-profile", "figure"),
        Output("model-comparison", "figure"),
        Output("significance-matrix", "figure"),
        Output("diagnostics-table", "data"),
        Output("diagnostics-table", "columns"),
        Output("top-terms", "figure"),
        Output("term-forest", "figure"),
        Output("lasso-stats-table", "data"),
        Output("lasso-stats-table", "columns"),
        Output("manifest-table", "data"),
        Output("manifest-table", "columns"),
        Output("results-table", "data"),
        Output("results-table", "columns"),
        Input("run-name", "value"),
        Input("model-families", "value"),
        Input("pollutants", "value"),
        Input("subclasses", "value"),
        Input("distance-steps", "value"),
        Input("term-groups", "value"),
        Input("flags", "value"),
    )
    def _refresh_dashboard(
        selected_run_name: str,
        model_families: list[str] | None,
        pollutants: list[str] | None,
        subclasses: list[str] | None,
        distance_steps: list[str] | None,
        term_groups: list[str] | None,
        flags: list[str] | None,
    ):
        flag_values = set(flags or [])
        run = load_result_run(run_lookup[selected_run_name], settings=settings)
        filtered_results = filter_app_frame(
            run.app_results,
            model_families=model_families,
            pollutants=pollutants,
            subclasses=subclasses,
            distance_steps=distance_steps,
            term_groups=term_groups,
            significant_only="significant" in flag_values,
            selected_only="selected" in flag_values,
        )
        filtered_manifest = filter_app_frame(
            run.app_manifest,
            model_families=model_families,
            pollutants=pollutants,
            subclasses=subclasses,
            distance_steps=distance_steps,
        )

        summary = _summary_cards(html, filtered_manifest, filtered_results)
        status_heatmap = make_status_heatmap(filtered_manifest)
        distance_profile = make_distance_profile(filtered_results, max_facets=max_facets)
        model_comparison = make_model_comparison(filtered_results)
        significance_matrix = make_significance_matrix(filtered_results)
        diagnostics_table = make_diagnostics_table(filtered_manifest, filtered_results)
        top_terms_figure = make_top_terms(filtered_results, top_n=top_terms)
        term_forest = make_term_forest(filtered_results, top_n=top_terms)
        lasso_stats_table = make_lasso_stats_table(filtered_manifest).head(250)

        manifest_table = filtered_manifest.head(250)
        results_table = filtered_results.head(250)
        return (
            summary,
            status_heatmap,
            distance_profile,
            model_comparison,
            significance_matrix,
            diagnostics_table.to_dict("records"),
            [{"name": column, "id": column} for column in diagnostics_table.columns],
            top_terms_figure,
            term_forest,
            lasso_stats_table.to_dict("records"),
            [{"name": column, "id": column} for column in lasso_stats_table.columns],
            manifest_table.to_dict("records"),
            [{"name": column, "id": column} for column in manifest_table.columns],
            results_table.to_dict("records"),
            [{"name": column, "id": column} for column in results_table.columns],
        )

    app.run(host=host, port=port, debug=debug)


__all__ = [
    "DEFAULT_MAX_FACETS",
    "DEFAULT_TOP_TERMS",
    "SensorResultRun",
    "build_model_comparison_table",
    "build_significance_matrix_table",
    "discover_result_runs",
    "filter_app_frame",
    "load_result_run",
    "make_diagnostics_table",
    "make_lasso_stats_table",
    "make_model_comparison",
    "make_significance_matrix",
    "run_plotly_app",
]
