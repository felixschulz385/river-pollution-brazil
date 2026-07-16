"""Interactive Plotly/Dash app for sensor-analysis regression outputs."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from .checkpoints import latest_fingerprint, load_partial_results, shard_progress
from .results import build_readable_manifest_table, build_readable_results_table
from ..settings import DEFAULT_SETTINGS, SensorAnalysisSettings


DEFAULT_MAX_FACETS = 12
DEFAULT_TOP_TERMS = 20

_TABLE_HEADER_STYLE = {
    "fontWeight": 700,
    "backgroundColor": "#efe8d9",
    "borderBottom": "2px solid #d8c9a8",
}
_TABLE_ZEBRA_CONDITIONAL = [
    {"if": {"row_index": "odd"}, "backgroundColor": "#f7f5ee"},
]


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


def discover_pending_runs(base_dir: str | Path) -> list[Path]:
    """Return run directories with in-progress checkpoints but no published manifest."""
    root = Path(base_dir)
    if not root.exists():
        return []
    return sorted(
        [
            path
            for path in root.iterdir()
            if path.is_dir()
            and not (path / "manifest.parquet").exists()
            and (path / "_work").is_dir()
            and latest_fingerprint(path) is not None
        ],
        key=lambda path: path.name,
    )


@dataclass(frozen=True)
class SensorRunProgress:
    """Shard-completion summary for an in-progress checkpoint run."""

    fingerprint: str
    shard_count: int
    shards_complete: int
    specs_expected: int
    specs_done: int


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

    def _bucket_subclass_key(term: str) -> str:
        return term.split("__x__", 1)[0] if "__x__" in term else term

    mapped = frame["term"].map(lambda value: lookup.get(_bucket_subclass_key(str(value))))
    frame["term_bucket"] = mapped.map(lambda value: value[0] if isinstance(value, tuple) else None)
    frame["term_subclass"] = mapped.map(lambda value: value[1] if isinstance(value, tuple) else None)

    # Direct climate terms (term_group == "climate") aren't land-cover columns, so
    # the lookup above misses them. Resolve their bucket via the same name-pattern
    # parser `_climate_scale_info` uses, since `settings.climate_variables` is
    # typically empty at render time (discovered per-run, not stored on settings).
    def _climate_term_bucket(term: str) -> str | None:
        candidate = term.removesuffix("__scaled") if term.endswith("__scaled") else term
        parsed = settings.parse_climate_source_column(candidate)
        return parsed.distance_bucket if parsed is not None else None

    missing_bucket = frame["term_bucket"].isna()
    if missing_bucket.any():
        frame.loc[missing_bucket, "term_bucket"] = frame.loc[missing_bucket, "term"].map(
            lambda value: _climate_term_bucket(str(value))
        )

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


def load_pending_run(
    run_dir: str | Path,
    *,
    settings: SensorAnalysisSettings = DEFAULT_SETTINGS,
) -> tuple[SensorResultRun, SensorRunProgress]:
    """Load whatever checkpointed chunks exist for an in-progress run."""
    directory = Path(run_dir)
    fingerprint = latest_fingerprint(directory)
    if fingerprint is None:
        raise FileNotFoundError(f"No checkpoint work found under {directory}.")

    results, manifest = load_partial_results(directory, fingerprint)
    shards = shard_progress(directory, fingerprint)
    progress = SensorRunProgress(
        fingerprint=fingerprint,
        shard_count=shards[0]["shard_count"] if shards else 0,
        shards_complete=sum(1 for shard in shards if shard["complete"]),
        specs_expected=sum(shard["specs_expected"] for shard in shards),
        specs_done=sum(shard["specs_done"] for shard in shards),
    )
    run = SensorResultRun(
        run_name=directory.name,
        run_dir=directory,
        results=results.copy(),
        manifest=manifest.copy(),
        app_results=_enrich_results_for_app(results, settings),
        app_manifest=_enrich_manifest_for_app(manifest, settings),
    )
    return run, progress


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
    figure.update_layout(
        template="plotly_white",
        height=600,
        font={"size": 12},
        title={"font": {"size": 16}},
    )
    figure.update_xaxes(tickangle=-35, automargin=True)
    figure.update_yaxes(automargin=True)
    return figure


# Fixed order/color/between-group-offset for the four coefficient groups shown
# together in the merged "Land Cover & Climate" tab. Offsets are in bucket-position
# units (each bucket occupies one integer step on the x-axis); within-group items
# (e.g. two climate variables matching the same bucket) get a smaller nested offset
# around their group's slot, computed in make_coefficient_dodge_chart.
_COEF_GROUP_ORDER: list[str] = [
    "Crude",
    "Post-Lasso land cover",
    "Interactions (Post-Lasso)",
    "Climate (Post-Lasso)",
]
_COEF_GROUP_COLORS: dict[str, str] = {
    "Crude": "#2a78d6",
    "Post-Lasso land cover": "#eb6834",
    "Interactions (Post-Lasso)": "#4a3aa7",
    "Climate (Post-Lasso)": "#1baf7a",
}
_COEF_GROUP_OFFSETS: dict[str, float] = {
    "Crude": -0.3,
    "Post-Lasso land cover": -0.1,
    "Interactions (Post-Lasso)": 0.1,
    "Climate (Post-Lasso)": 0.3,
}
_COEF_GROUP_SPECS: list[tuple[str, str, str]] = [
    ("Crude", "land_cover", "crude_twfe"),
    ("Post-Lasso land cover", "land_cover", "post_lasso"),
    ("Interactions (Post-Lasso)", "interaction", "post_lasso"),
    ("Climate (Post-Lasso)", "climate", "post_lasso"),
]

_TRANSFORM_LABELS: dict[str, str] = {
    "identity": "raw units",
    "log10_1p": "log10(1 + x)",
}


def _transform_label(transform_name: object) -> str:
    """Human-readable label for a pollutant outcome transform name."""
    return _TRANSFORM_LABELS.get(str(transform_name), str(transform_name))


def _land_cover_transform_caption(settings: SensorAnalysisSettings) -> str:
    """Static note describing the (uniform, run-wide) land-cover regressor transform."""
    transform = settings.land_cover_transform
    if transform.kind == "identity":
        return "Land-cover shares are not transformed before estimation."
    return (
        f"Land-cover shares are transformed as {transform.kind}(share + {transform.offset:g}) "
        "before estimation."
    )


def _coefficient_group_frame(results: pd.DataFrame, *, term_group: str, model_family: str) -> pd.DataFrame:
    """Select one coefficient group's rows (land-cover/interaction/climate x model family)."""
    if results.empty or not {"term_group", "model_family"}.issubset(results.columns):
        return results.iloc[0:0].copy()
    if term_group == "land_cover":
        base = _land_cover_results(results)
    elif term_group == "interaction":
        base = _interaction_results(results)
    else:
        base = results.loc[results["term_group"].eq(term_group)].copy()
    return base.loc[base["model_family"].eq(model_family)].copy()


def make_coefficient_dodge_chart(
    results: pd.DataFrame,
    *,
    groups: list[str] | None = None,
    significant_only: bool = False,
    max_facets: int = DEFAULT_MAX_FACETS,
    settings: SensorAnalysisSettings = DEFAULT_SETTINGS,
):
    """Plot Crude/Post-Lasso land-cover, interaction, and climate coefficients by bucket.

    All four groups are dodged apart at each bucket: a larger fixed offset between
    groups, and a smaller nested offset within a group when it has more than one
    term at the same bucket (e.g. two climate variables matching the same bucket).
    """
    px, _go = _import_plotly()
    selected_groups = set(groups) if groups is not None else set(_COEF_GROUP_ORDER)

    frames = []
    for group_label, term_group, model_family in _COEF_GROUP_SPECS:
        if group_label not in selected_groups:
            continue
        frame = _coefficient_group_frame(results, term_group=term_group, model_family=model_family)
        if frame.empty:
            continue
        frames.append(frame.assign(coef_group=group_label))

    if not frames:
        return _empty_figure("Land-cover and climate coefficients")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["term_bucket_label", "pollutant_label"])
    if significant_only and "is_significant" in combined.columns:
        combined = combined.loc[combined["is_significant"]].copy()
    if combined.empty:
        return _empty_figure("Land-cover and climate coefficients")

    combined, hidden_count = _limit_facets(combined, facet_column="pollutant_label", max_facets=max_facets)

    # Shared numeric bucket axis (independent per-facet range via matches=None below)
    # with manual tick labels, so both between- and within-group dodge offsets can
    # be added as small fractional shifts around each bucket's integer position.
    bucket_categories = [
        label for label in dict.fromkeys(combined["term_bucket_label"]) if label is not None
    ]
    bucket_position = {label: index for index, label in enumerate(bucket_categories)}
    combined["_bucket_position"] = combined["term_bucket_label"].map(bucket_position).astype(float)
    combined = combined.dropna(subset=["_bucket_position"])

    combined["_between_offset"] = combined["coef_group"].map(_COEF_GROUP_OFFSETS).astype(float)
    cluster_keys = ["pollutant_label", "_bucket_position", "coef_group"]
    combined["_within_rank"] = combined.groupby(cluster_keys).cumcount()
    combined["_within_count"] = combined.groupby(cluster_keys)["_within_rank"].transform("count")
    combined["_within_offset"] = (
        (combined["_within_rank"] - (combined["_within_count"] - 1) / 2)
        * (0.12 / (combined["_within_count"] - 1).clip(lower=1))
    )
    combined["dodged_position"] = (
        combined["_bucket_position"] + combined["_between_offset"] + combined["_within_offset"]
    )

    has_ci = {"97.5%", "2.5%", "Estimate"}.issubset(combined.columns)
    if has_ci:
        combined["_error_y_plus"] = combined["97.5%"] - combined["Estimate"]
        combined["_error_y_minus"] = combined["Estimate"] - combined["2.5%"]

    scale_caption = _scale_caption(
        combined.loc[
            combined["coef_group"].isin(["Interactions (Post-Lasso)", "Climate (Post-Lasso)"]), "term"
        ],
        settings,
    )
    title = "Land-cover and climate coefficients by distance bucket"
    if hidden_count > 0:
        title = f"{title} (showing top {max_facets} pollutants)"
    if scale_caption:
        title = f"{title}<br><sup>{scale_caption}</sup>"

    figure = px.scatter(
        combined.sort_values(["pollutant_label", "_bucket_position", "coef_group"]),
        x="dodged_position",
        y="Estimate",
        color="coef_group",
        color_discrete_map=_COEF_GROUP_COLORS,
        category_orders={"coef_group": _COEF_GROUP_ORDER},
        facet_col="pollutant_label",
        facet_col_wrap=3,
        error_y="_error_y_plus" if has_ci else None,
        error_y_minus="_error_y_minus" if has_ci else None,
        hover_data=["term_label", "model_family_label", "nobs", "distance_step_label"],
        labels={
            "dodged_position": "Coefficient bucket",
            "Estimate": "Coefficient estimate",
            "coef_group": "Group",
            "pollutant_label": "Pollutant",
        },
        title=title,
    )
    figure.update_traces(marker={"size": 9, "line": {"width": 1, "color": "#fcfcfb"}})

    transform_by_pollutant = (
        combined[["pollutant_label", "transform"]]
        .drop_duplicates("pollutant_label")
        .set_index("pollutant_label")["transform"]
        if "transform" in combined.columns
        else pd.Series(dtype="object")
    )

    def _annotate_facet(annotation) -> None:
        pollutant_label = annotation.text.split("=")[-1]
        transform_name = transform_by_pollutant.get(pollutant_label)
        suffix = f" ({_transform_label(transform_name)})" if transform_name is not None else ""
        annotation.update(text=f"{pollutant_label}{suffix}", font={"size": 11})

    figure.for_each_annotation(_annotate_facet)

    figure.update_layout(
        template="plotly_white",
        height=340 * min(max_facets, 4),
        font={"size": 12},
        title={"font": {"size": 16}},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"t": 100 if scale_caption else 90},
    )
    figure.update_xaxes(
        gridcolor="#e1e0d9",
        tickfont={"size": 10},
        tickmode="array",
        tickvals=list(range(len(bucket_categories))),
        ticktext=bucket_categories,
        matches=None,
        showticklabels=True,
    )
    figure.update_yaxes(gridcolor="#e1e0d9", zeroline=False, matches=None, showticklabels=True)
    figure.add_hline(y=0.0, line_dash="dash", line_color="#898781", opacity=0.8)
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
    lasso_pruned = (
        pd.to_numeric(manifest["lasso_pruned_candidate_count"], errors="coerce")
        if "lasso_pruned_candidate_count" in manifest.columns
        else pd.Series(dtype="float64")
    )
    lasso_attempts = (
        pd.to_numeric(manifest["lasso_attempts"], errors="coerce")
        if "lasso_attempts" in manifest.columns
        else pd.Series(dtype="float64")
    )
    total_seconds = (
        pd.to_numeric(manifest["total_seconds"], errors="coerce")
        if "total_seconds" in manifest.columns
        else pd.Series(dtype="float64")
    )
    ols_warnings = (
        pd.to_numeric(manifest["ols_warning_count"], errors="coerce")
        if "ols_warning_count" in manifest.columns
        else pd.Series(dtype="float64")
    )
    if "lasso_converged" in manifest.columns and manifest["lasso_converged"].notna().any():
        lasso_converged_share = float(manifest["lasso_converged"].fillna(False).mean())
    else:
        lasso_converged_share = None

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
        (
            "Mean LASSO pruned candidates",
            f"{lasso_pruned.mean():.1f}" if lasso_pruned.notna().any() else "0.0",
        ),
        (
            "LASSO convergence rate",
            f"{lasso_converged_share:.1%}" if lasso_converged_share is not None else "NA",
        ),
        (
            "Mean LASSO fit attempts",
            f"{lasso_attempts.mean():.2f}" if lasso_attempts.notna().any() else "NA",
        ),
        (
            "Mean total runtime (s)",
            f"{total_seconds.mean():.2f}" if total_seconds.notna().any() else "NA",
        ),
        (
            "OLS warnings",
            f"{int(ols_warnings.sum()):,}" if ols_warnings.notna().any() else "0",
        ),
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
                "lasso_pruned_candidate_count",
                "lasso_selected_count",
                "lasso_selected_share",
                "lasso_min_cv_mse",
                "lasso_converged",
                "lasso_attempts",
                "numerical_status",
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
                "lasso_pruned_candidate_count",
                "lasso_selected_count",
                "lasso_selected_share",
                "lasso_min_cv_mse",
                "lasso_converged",
                "lasso_attempts",
                "numerical_status",
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
        "lasso_pruned_candidate_count",
        "lasso_selected_count",
        "lasso_selected_share",
        "lasso_min_cv_mse",
        "lasso_converged",
        "lasso_attempts",
        "numerical_status",
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


def _interaction_results(results: pd.DataFrame) -> pd.DataFrame:
    """Return interaction-term rows whose land-cover side matches the row's facet."""
    if results.empty or "term_group" not in results.columns:
        return results.iloc[0:0].copy()
    return results.loc[
        results["term_group"].eq("interaction")
        & results["term_subclass"].eq(results["land_cover_subclass"])
    ].copy()


def _climate_scale_info(term: str, settings: SensorAnalysisSettings) -> tuple[str, float] | None:
    """Return (display label, scale divisor) for a term's climate component, if any."""
    candidate = term.split("__x__", 1)[1] if "__x__" in term else term
    for variable in (*settings.climate_variables, *settings.controls):
        if variable.scaled_column == candidate:
            return variable.source_column.replace("_", " ").title(), variable.scale
    # `settings.climate_variables` is often empty at render time (climate
    # variables are discovered per-run from the assembled columns rather than
    # stored back onto settings), so fall back to the same name-based parser
    # the analysis pipeline itself uses to recover the source column and scale.
    if candidate.endswith("__scaled"):
        parsed = settings.parse_climate_source_column(candidate.removesuffix("__scaled"))
        if parsed is not None:
            return parsed.source_column.replace("_", " ").title(), parsed.scale
    return None


def _scale_caption(terms: pd.Series, settings: SensorAnalysisSettings) -> str:
    """Summarize the scale divisor applied to each distinct climate variable shown."""
    scales: dict[str, float] = {}
    for term in terms.dropna().unique():
        info = _climate_scale_info(str(term), settings)
        if info is None:
            continue
        label, scale = info
        scales[label] = scale
    if not scales:
        return ""
    parts = [
        f"{label} (raw units)" if scale == 1.0 else f"{label} ÷ {scale:g}"
        for label, scale in sorted(scales.items())
    ]
    return "Climate scaling — " + "; ".join(parts)


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
    figure.update_traces(marker={"size": 9, "line": {"width": 1, "color": "#fcfcfb"}})
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
                line={"color": "#898781", "dash": "dash"},
            )
        )
        figure.update_xaxes(range=[lower, upper])
        figure.update_yaxes(range=[lower, upper])
    figure.update_layout(
        template="plotly_white",
        height=620,
        font={"size": 12},
        title={"font": {"size": 16}},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"t": 90},
    )
    figure.update_xaxes(gridcolor="#e1e0d9")
    figure.update_yaxes(gridcolor="#e1e0d9")
    return figure


def build_significance_matrix_table(
    results: pd.DataFrame,
    *,
    settings: SensorAnalysisSettings = DEFAULT_SETTINGS,
) -> pd.DataFrame:
    """Classify land-cover coefficient significance by pollutant and bucket.

    Grouped by individual coefficient bucket (`term_bucket`) rather than the
    cumulative spec (`distance_step`), so every bucket included in the
    currently selected distance step gets its own column — not just the
    step's terminal/largest bucket.
    """
    subset = _land_cover_results(results)
    required = {
        "facet_label",
        "term_bucket_label",
        "term_bucket",
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
        subset.groupby(["facet_label", "term_bucket_label", "term_bucket"], dropna=False)
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

    def _bucket_order(bucket: object) -> int:
        return (
            settings.distance_buckets.index(bucket)
            if bucket in settings.distance_buckets
            else len(settings.distance_buckets)
        )

    grouped["_bucket_order"] = grouped["term_bucket"].map(_bucket_order)
    grouped = grouped.sort_values(["facet_label", "_bucket_order"]).drop(columns="_bucket_order")
    return grouped.reset_index(drop=True)


def make_significance_matrix(
    results: pd.DataFrame,
    *,
    settings: SensorAnalysisSettings = DEFAULT_SETTINGS,
):
    """Show the sign and significance of land-cover estimates by distance bucket."""
    _, go = _import_plotly()
    grouped = build_significance_matrix_table(results, settings=settings)
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

    def _bucket_order(bucket: object) -> int:
        return (
            settings.distance_buckets.index(bucket)
            if bucket in settings.distance_buckets
            else len(settings.distance_buckets)
        )

    x_order = (
        grouped[["term_bucket_label", "term_bucket"]]
        .drop_duplicates()
        .assign(_order=lambda frame: frame["term_bucket"].map(_bucket_order))
        .sort_values("_order")["term_bucket_label"]
        .tolist()
    )
    y_order = sorted(grouped["facet_label"].dropna().unique().tolist())
    matrix = (
        grouped.pivot(index="facet_label", columns="term_bucket_label", values="status_code")
        .reindex(index=y_order, columns=x_order)
    )
    hover = (
        grouped.pivot(index="facet_label", columns="term_bucket_label", values="status")
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
                xgap=2,
                ygap=2,
                colorbar={
                    "tickmode": "array",
                    "tickvals": list(status_codes.values()),
                    "ticktext": status_order,
                    "title": "Status",
                    "tickfont": {"size": 11},
                },
            )
        ]
    )
    figure.update_layout(
        template="plotly_white",
        title={"text": "Land-cover significance by distance", "font": {"size": 16}},
        xaxis_title="Distance bucket",
        yaxis_title="Pollutant | land cover",
        height=max(520, 28 * len(y_order)),
        font={"size": 12},
        margin={"l": 10},
    )
    figure.update_xaxes(tickangle=-30, automargin=True)
    figure.update_yaxes(automargin=True)
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
    pending_directories = discover_pending_runs(results_dir)
    if not run_directories and not pending_directories:
        raise ValueError(f"No result runs found under {results_dir}.")

    run_lookup = {path.name: path for path in run_directories}
    pending_lookup = {path.name: path for path in pending_directories}
    all_run_names = sorted(run_lookup) + sorted(pending_lookup)
    initial_run_name = run_name or all_run_names[0]
    if initial_run_name not in run_lookup and initial_run_name not in pending_lookup:
        raise ValueError(
            f"Run `{initial_run_name}` not found under {results_dir}. "
            f"Available runs: {all_run_names}."
        )

    def _pending_run_label(path: Path) -> str:
        fingerprint = latest_fingerprint(path)
        if fingerprint is None:
            return path.name
        shards = shard_progress(path, fingerprint)
        shard_count = shards[0]["shard_count"] if shards else 0
        shards_complete = sum(1 for shard in shards if shard["complete"])
        specs_expected = sum(shard["specs_expected"] for shard in shards)
        specs_done = sum(shard["specs_done"] for shard in shards)
        return (
            f"{path.name} (in progress: {shards_complete}/{shard_count} shards, "
            f"{specs_done}/{specs_expected} specs)"
        )

    def _load_selected_run(selected_run_name: str) -> SensorResultRun:
        if selected_run_name in run_lookup:
            return load_result_run(run_lookup[selected_run_name], settings=settings)
        run, _ = load_pending_run(pending_lookup[selected_run_name], settings=settings)
        return run

    def _run_progress_banner(html, selected_run_name: str):
        if selected_run_name not in pending_lookup:
            return None
        _, progress = load_pending_run(pending_lookup[selected_run_name], settings=settings)
        return html.Div(
            f"Run in progress — {progress.shards_complete}/{progress.shard_count} shards complete, "
            f"{progress.specs_done}/{progress.specs_expected} specifications done. "
            "Showing partial results from the latest checkpoint.",
            style={
                "padding": "10px 32px",
                "background": "#f6e9c9",
                "borderBottom": "1px solid #d6c48a",
                "color": "#5b4a1a",
                "fontSize": "13px",
            },
        )

    app = Dash(__name__)
    app.title = "Sensor Analysis Results"

    land_cover_transform_note = _land_cover_transform_caption(settings)

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
            html.Div(id="run-progress-banner"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Run", style={"fontWeight": 700}),
                            dcc.Dropdown(
                                id="run-name",
                                options=[
                                    {"label": name, "value": name} for name in sorted(run_lookup)
                                ]
                                + [
                                    {"label": _pending_run_label(path), "value": name}
                                    for name, path in sorted(pending_lookup.items())
                                ],
                                value=initial_run_name,
                                clearable=False,
                                style={"minWidth": "260px"},
                            ),
                        ],
                        style={"minWidth": "260px"},
                    ),
                    html.Div(
                        [
                            html.Label("Pollutants", style={"fontWeight": 700}),
                            dcc.Dropdown(id="pollutants", multi=True),
                        ],
                        style={"flex": "1", "minWidth": "220px"},
                    ),
                    html.Div(
                        [
                            html.Label("Land-cover subclass", style={"fontWeight": 700}),
                            dcc.Dropdown(id="subclasses", multi=False, clearable=False),
                        ],
                        style={"minWidth": "220px"},
                    ),
                    html.Div(
                        [
                            html.Label("Distance step", style={"fontWeight": 700}),
                            dcc.Dropdown(id="distance-steps", multi=False, clearable=False),
                        ],
                        style={"minWidth": "220px"},
                    ),
                ],
                style={
                    "display": "flex",
                    "flexWrap": "wrap",
                    "gap": "18px",
                    "alignItems": "flex-end",
                    "padding": "16px 32px",
                    "background": "#fbfaf6",
                    "borderBottom": "1px solid #ddd7ca",
                },
            ),
            html.Div(
                [
                    html.Div(id="summary-cards", style={"marginBottom": "18px"}),
                    dcc.Tabs(
                        [
                            dcc.Tab(
                                label="Overview",
                                children=[
                                    dcc.Graph(id="status-heatmap"),
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
                                        style_header=_TABLE_HEADER_STYLE,
                                        style_data_conditional=_TABLE_ZEBRA_CONDITIONAL,
                                    ),
                                ],
                            ),
                            dcc.Tab(
                                label="Land Cover & Climate",
                                children=[
                                    html.Div(
                                        [
                                            html.Div(
                                                "Each panel is one pollutant, faceted at the currently selected land-cover "
                                                "subclass and distance step. Each panel scales its own axes independently. "
                                                "Groups are dodged apart at each bucket (small offset within a group, larger "
                                                "offset between groups).",
                                                style={"color": "#5b5448", "margin": "10px 0 4px 0"},
                                            ),
                                            html.Div(
                                                land_cover_transform_note,
                                                style={"color": "#5b5448", "margin": "0 0 10px 0", "fontStyle": "italic"},
                                            ),
                                        ],
                                        style={"margin": "0 0 4px 8px"},
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Label("Show groups", style={"fontWeight": 700}),
                                                    dcc.Checklist(
                                                        id="coef-groups",
                                                        options=[
                                                            {"label": label, "value": label}
                                                            for label in _COEF_GROUP_ORDER
                                                        ],
                                                        value=list(_COEF_GROUP_ORDER),
                                                        inline=True,
                                                        style={"display": "flex", "gap": "14px"},
                                                    ),
                                                ],
                                                style={"marginRight": "32px"},
                                            ),
                                            dcc.Checklist(
                                                id="coef-significant-only",
                                                options=[
                                                    {"label": "Only significant coefficients", "value": "significant"},
                                                ],
                                                value=[],
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "flexWrap": "wrap",
                                            "alignItems": "center",
                                            "gap": "18px",
                                            "margin": "0 0 10px 8px",
                                        },
                                    ),
                                    dcc.Graph(id="coefficient-dodge-chart"),
                                ],
                            ),
                            dcc.Tab(
                                label="Comparison",
                                children=[dcc.Graph(id="model-comparison")],
                            ),
                        ]
                    ),
                ],
                style={"padding": "20px 28px 28px 28px"},
            ),
        ],
        style={
            "fontFamily": "Avenir Next, Helvetica Neue, sans-serif",
            "background": "#f3f1ea",
            "overflowX": "hidden",
        },
    )

    @app.callback(
        Output("run-progress-banner", "children"),
        Output("pollutants", "options"),
        Output("pollutants", "value"),
        Output("subclasses", "options"),
        Output("subclasses", "value"),
        Output("distance-steps", "options"),
        Output("distance-steps", "value"),
        Input("run-name", "value"),
    )
    def _refresh_filter_options(selected_run_name: str):
        run = _load_selected_run(selected_run_name)
        results = run.app_results
        pollutants = sorted(results["pollutant"].dropna().unique().tolist())
        subclasses = sorted(results["land_cover_subclass"].dropna().unique().tolist())
        distance_steps = sorted(
            results["distance_step_name"].dropna().unique().tolist(),
            key=lambda value: settings.distance_buckets.index(value)
            if value in settings.distance_buckets
            else len(settings.distance_buckets),
        )
        return (
            _run_progress_banner(html, selected_run_name),
            _dropdown_options(pollutants, _pollutant_label),
            pollutants[: min(len(pollutants), 6)],
            _dropdown_options(
                subclasses,
                lambda value: settings.subclass_labels.get(str(value), str(value)),
            ),
            subclasses[0] if subclasses else None,
            _dropdown_options(distance_steps, settings.distance_bucket_label),
            distance_steps[-1] if distance_steps else None,
        )

    @app.callback(
        Output("summary-cards", "children"),
        Output("status-heatmap", "figure"),
        Output("significance-matrix", "figure"),
        Output("diagnostics-table", "data"),
        Output("diagnostics-table", "columns"),
        Output("coefficient-dodge-chart", "figure"),
        Output("model-comparison", "figure"),
        Input("run-name", "value"),
        Input("pollutants", "value"),
        Input("subclasses", "value"),
        Input("distance-steps", "value"),
        Input("coef-groups", "value"),
        Input("coef-significant-only", "value"),
    )
    def _refresh_dashboard(
        selected_run_name: str,
        pollutants: list[str] | None,
        subclass: str | None,
        distance_step: str | None,
        coef_groups: list[str] | None,
        coef_significant_only: list[str] | None,
    ):
        subclasses = [subclass] if subclass else None
        distance_steps = [distance_step] if distance_step else None
        run = _load_selected_run(selected_run_name)
        filtered_results = filter_app_frame(
            run.app_results,
            pollutants=pollutants,
            subclasses=subclasses,
            distance_steps=distance_steps,
        )
        filtered_manifest = filter_app_frame(
            run.app_manifest,
            pollutants=pollutants,
            subclasses=subclasses,
            distance_steps=distance_steps,
        )

        summary = _summary_cards(html, filtered_manifest, filtered_results)
        status_heatmap = make_status_heatmap(filtered_manifest)
        significance_matrix = make_significance_matrix(filtered_results, settings=settings)
        diagnostics_table = make_diagnostics_table(filtered_manifest, filtered_results)
        coefficient_chart = make_coefficient_dodge_chart(
            filtered_results,
            groups=coef_groups,
            significant_only="significant" in set(coef_significant_only or []),
            max_facets=max_facets,
            settings=settings,
        )
        model_comparison = make_model_comparison(filtered_results)

        return (
            summary,
            status_heatmap,
            significance_matrix,
            diagnostics_table.to_dict("records"),
            [{"name": column, "id": column} for column in diagnostics_table.columns],
            coefficient_chart,
            model_comparison,
        )

    app.run(host=host, port=port, debug=debug)


__all__ = [
    "DEFAULT_MAX_FACETS",
    "DEFAULT_TOP_TERMS",
    "SensorResultRun",
    "SensorRunProgress",
    "build_model_comparison_table",
    "build_significance_matrix_table",
    "discover_pending_runs",
    "discover_result_runs",
    "filter_app_frame",
    "load_pending_run",
    "load_result_run",
    "make_coefficient_dodge_chart",
    "make_diagnostics_table",
    "make_lasso_stats_table",
    "make_model_comparison",
    "make_significance_matrix",
    "run_plotly_app",
]
