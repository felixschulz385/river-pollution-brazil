"""Regression specification builders."""

from __future__ import annotations

from dataclasses import dataclass

from .groups import PollutantSelection
from ..settings import DEFAULT_SETTINGS, SensorAnalysisSettings


@dataclass(frozen=True)
class ModelSpec:
    """One regression model to be estimated."""

    pollutant: str
    pollutant_group_kind: str
    pollutant_group_name: str
    land_cover_subclass: str
    distance_step_index: int
    distance_step_name: str
    included_buckets: tuple[str, ...]
    outcome_column: str
    coefficient_columns: tuple[str, ...]
    formula: str


def _land_cover_column(
    bucket: str,
    subclass: str,
    settings: SensorAnalysisSettings,
) -> str:
    return settings.land_cover_column(bucket, subclass)


def build_model_specs(
    settings: SensorAnalysisSettings = DEFAULT_SETTINGS,
    pollutant_selection: PollutantSelection | list[str] | tuple[str, ...] = (),
    subclass_selection: list[str] | tuple[str, ...] | None = None,
    *,
    max_distance_step: int | None = None,
) -> list[ModelSpec]:
    """Construct cumulative distance-bucket model specifications."""
    if isinstance(pollutant_selection, PollutantSelection):
        pollutants = pollutant_selection.pollutants
        group_kind = pollutant_selection.group_kind
        group_name = pollutant_selection.group_name
    else:
        pollutants = tuple(pollutant_selection)
        group_kind = "explicit"
        group_name = "custom"

    subclasses = (
        tuple(subclass_selection)
        if subclass_selection is not None
        else settings.land_cover_subclasses
    )
    invalid = sorted(set(subclasses).difference(settings.land_cover_subclasses))
    if invalid:
        raise ValueError(
            f"Unknown land-cover subclasses requested: {invalid}. "
            f"Available: {list(settings.land_cover_subclasses)}."
        )

    bucket_limit = len(settings.distance_buckets)
    if max_distance_step is not None:
        bucket_limit = min(bucket_limit, max_distance_step)

    control_terms = [control.scaled_column for control in settings.controls]
    fixed_effects = " + ".join(settings.resolved_fixed_effects())
    specs: list[ModelSpec] = []
    for pollutant in pollutants:
        outcome_column = f"{pollutant}__transformed"
        for subclass in subclasses:
            for index in range(bucket_limit):
                included_buckets = settings.distance_buckets[: index + 1]
                coefficient_columns = tuple(
                    _land_cover_column(bucket, subclass, settings)
                    for bucket in included_buckets
                )
                rhs_terms = [*coefficient_columns, *control_terms]
                formula = (
                    f"{outcome_column} ~ {' + '.join(rhs_terms)} | {fixed_effects}"
                )
                specs.append(
                    ModelSpec(
                        pollutant=pollutant,
                        pollutant_group_kind=group_kind,
                        pollutant_group_name=group_name,
                        land_cover_subclass=subclass,
                        distance_step_index=index + 1,
                        distance_step_name=included_buckets[-1],
                        included_buckets=included_buckets,
                        outcome_column=outcome_column,
                        coefficient_columns=coefficient_columns,
                        formula=formula,
                    )
                )
    return specs


__all__ = ["ModelSpec", "build_model_specs"]
