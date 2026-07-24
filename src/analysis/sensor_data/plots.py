"""Plotting helpers for sensor-data regression outputs."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import pandas as pd

from ..settings import DEFAULT_SETTINGS, SensorAnalysisSettings


def _pollutant_label(name: str) -> str:
    """Turn a raw pollutant column name into a readable panel title."""
    return name.replace("_", " ").title()


def _distance_bucket_label(bucket: str, settings: SensorAnalysisSettings) -> str:
    return settings.distance_bucket_label(bucket)


def _distance_term_lookup(
    subclasses: Sequence[str],
    settings: SensorAnalysisSettings,
) -> dict[str, tuple[str, str]]:
    """Map model term names to their distance bucket and land-cover subclass."""
    lookup: dict[str, tuple[str, str]] = {}
    for subclass in subclasses:
        for bucket in settings.distance_buckets:
            lookup[settings.land_cover_column(bucket, subclass)] = (bucket, subclass)
    return lookup


def faceted_distance_coefplot(
    results: pd.DataFrame,
    *,
    pollutants: Sequence[str] | None = None,
    land_cover_subclasses: Sequence[str] | None = None,
    settings: SensorAnalysisSettings = DEFAULT_SETTINGS,
    estimate_column: str = "Estimate",
    lower_column: str = "2.5%",
    upper_column: str = "97.5%",
    figsize: tuple[float, float] | None = None,
    sharex: bool = False,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Plot faceted coefficient estimates for land-cover distance terms.

    Rows correspond to land-cover subclasses and columns to pollutants. Within
    each panel, only the distance terms for that subclass are stacked on the
    y-axis and shown with confidence intervals.
    """
    pollutant_order = list(
        pollutants
        if pollutants is not None
        else pd.Index(results["pollutant"]).dropna().drop_duplicates().tolist()
    )
    subclass_order = list(
        land_cover_subclasses
        if land_cover_subclasses is not None
        else [
            subclass
            for subclass in settings.land_cover_subclasses
            if subclass in set(results["land_cover_subclass"].dropna())
        ]
    )
    if not pollutant_order:
        raise ValueError("No pollutants available for plotting.")
    if not subclass_order:
        raise ValueError("No land-cover subclasses available for plotting.")

    term_lookup = _distance_term_lookup(subclass_order, settings)
    plot_data = results.copy()
    plot_data = plot_data.loc[
        plot_data["pollutant"].isin(pollutant_order)
        & plot_data["land_cover_subclass"].isin(subclass_order)
        & plot_data["term"].isin(term_lookup)
    ].copy()
    if plot_data.empty:
        raise ValueError(
            "No matching land-cover distance coefficients found in `results`."
        )

    plot_data[["distance_bucket", "term_subclass"]] = pd.DataFrame(
        plot_data["term"].map(term_lookup).tolist(),
        index=plot_data.index,
    )
    plot_data = plot_data.loc[
        plot_data["term_subclass"] == plot_data["land_cover_subclass"]
    ].copy()
    plot_data["distance_bucket"] = pd.Categorical(
        plot_data["distance_bucket"],
        categories=list(settings.distance_buckets),
        ordered=True,
    )
    plot_data["distance_bucket_label"] = plot_data["distance_bucket"].map(
        lambda value: _distance_bucket_label(str(value), settings)
    )
    plot_data["pollutant"] = pd.Categorical(
        plot_data["pollutant"],
        categories=pollutant_order,
        ordered=True,
    )
    plot_data["pollutant_label"] = plot_data["pollutant"].map(
        lambda value: _pollutant_label(str(value))
    )
    plot_data["land_cover_subclass"] = pd.Categorical(
        plot_data["land_cover_subclass"],
        categories=subclass_order,
        ordered=True,
    )
    plot_data["land_cover_label"] = plot_data["land_cover_subclass"].map(
        lambda value: settings.subclass_labels.get(str(value), str(value))
    )
    if "model_family" in plot_data.columns:
        plot_data["model_family_label"] = plot_data["model_family"].map(
            settings.model_family_label
        )
    plot_data = plot_data.sort_values(
        ["land_cover_subclass", "pollutant", "distance_bucket"]
    )

    nrows = len(subclass_order)
    ncols = len(pollutant_order)
    if figsize is None:
        figsize = (4.2 * ncols, 2.8 * nrows)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=sharex,
        squeeze=False,
        constrained_layout=True,
    )

    for row_index, subclass in enumerate(subclass_order):
        for col_index, pollutant in enumerate(pollutant_order):
            ax = axes[row_index, col_index]
            subset = plot_data.loc[
                (plot_data["land_cover_subclass"] == subclass)
                & (plot_data["pollutant"] == pollutant)
            ].sort_values("distance_bucket")
            if subset.empty:
                ax.set_visible(False)
                continue

            ax.errorbar(
                subset[estimate_column],
                subset["distance_bucket_label"],
                xerr=[
                    subset[estimate_column] - subset[lower_column],
                    subset[upper_column] - subset[estimate_column],
                ],
                fmt="o",
                color="#0072B2",
                ecolor="#56B4E9",
                elinewidth=2,
                capsize=3,
                markersize=6,
            )
            ax.axvline(0, color="black", linestyle="--", linewidth=1)

            if row_index == 0:
                title = _pollutant_label(pollutant)
                families = (
                    subset["model_family_label"].dropna().drop_duplicates().tolist()
                    if "model_family_label" in subset.columns
                    else []
                )
                if len(families) == 1:
                    title = f"{title}\n{families[0]}"
                ax.set_title(title)
            if col_index == 0:
                ax.set_ylabel(
                    settings.subclass_labels.get(subclass, subclass),
                )
            else:
                ax.set_ylabel("")
            if row_index == nrows - 1:
                ax.set_xlabel("Coefficient estimate")
            else:
                ax.set_xlabel("")

    return fig, plot_data


__all__ = ["faceted_distance_coefplot"]
