"""Regression specification builders."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from .groups import PollutantSelection
from ..settings import DEFAULT_SETTINGS, SensorAnalysisSettings


@dataclass(frozen=True)
class ModelSpec:
    """One regression model to be estimated."""

    pollutant: str
    pollutant_group_kind: str
    pollutant_group_name: str
    model_family: str
    land_cover_subclass: str
    distance_step_index: int
    distance_step_name: str
    included_buckets: tuple[str, ...]
    outcome_column: str
    coefficient_columns: tuple[str, ...]
    forced_regressor_columns: tuple[str, ...]
    candidate_regressor_columns: tuple[str, ...]

    @property
    def spec_id(self) -> str:
        """Return a stable identifier suitable for checkpoints and merging."""
        payload = json.dumps(
            {
                "pollutant": self.pollutant,
                "group_kind": self.pollutant_group_kind,
                "group_name": self.pollutant_group_name,
                "family": self.model_family,
                "subclass": self.land_cover_subclass,
                "step": self.distance_step_index,
                "outcome": self.outcome_column,
                "coefficients": self.coefficient_columns,
                "forced": self.forced_regressor_columns,
                "candidates": self.candidate_regressor_columns,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


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
    model_families: list[str] | tuple[str, ...] | None = None,
    climate_variables=None,
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
    resolved_climate_variables = (
        tuple(climate_variables)
        if climate_variables is not None
        else settings.climate_variables
    )
    families = tuple(model_families) if model_families is not None else settings.model_families
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
                candidate_climate_variables = tuple(
                    variable
                    for variable in resolved_climate_variables
                    if variable.distance_bucket is None
                    or any(
                        settings.climate_matches_bucket(variable.distance_bucket, bucket)
                        for bucket in included_buckets
                    )
                )
                climate_terms = tuple(
                    variable.scaled_column for variable in candidate_climate_variables
                )
                interaction_terms = []
                for bucket, land_cover_column in zip(included_buckets, coefficient_columns, strict=True):
                    matching_climate_variables = [
                        variable
                        for variable in candidate_climate_variables
                        if settings.climate_matches_bucket(variable.distance_bucket, bucket)
                    ]
                    interaction_terms.extend(
                        settings.interaction_column(land_cover_column, variable.scaled_column)
                        for variable in matching_climate_variables
                    )
                for model_family in families:
                    if model_family not in {"crude_twfe", "post_lasso"}:
                        raise ValueError(f"Unsupported model family `{model_family}`.")
                    forced_regressor_columns = tuple([*coefficient_columns, *control_terms])
                    candidate_regressor_columns = (
                        tuple()
                        if model_family == "crude_twfe"
                        else tuple([*climate_terms, *interaction_terms])
                    )
                    specs.append(
                        ModelSpec(
                            pollutant=pollutant,
                            pollutant_group_kind=group_kind,
                            pollutant_group_name=group_name,
                            model_family=model_family,
                            land_cover_subclass=subclass,
                            distance_step_index=index + 1,
                            distance_step_name=included_buckets[-1],
                            included_buckets=included_buckets,
                            outcome_column=outcome_column,
                            coefficient_columns=coefficient_columns,
                            forced_regressor_columns=forced_regressor_columns,
                            candidate_regressor_columns=candidate_regressor_columns,
                        )
                    )
    return specs


__all__ = ["ModelSpec", "build_model_specs"]
