"""Grouping and selection logic for pollutants."""

from __future__ import annotations

from dataclasses import dataclass

from .catalog import PollutantDefinition, build_pollutant_catalog
from .loaders import load_sensor_data, load_transformations
from ..settings import DEFAULT_SETTINGS, SensorAnalysisSettings


@dataclass(frozen=True)
class PollutantSelection:
    """Concrete pollutant selection for one analysis run."""

    pollutants: tuple[str, ...]
    group_kind: str
    group_name: str


def _catalog_by_name(
    catalog: list[PollutantDefinition],
) -> dict[str, PollutantDefinition]:
    return {item.name: item for item in catalog}


def _filter_by_minimum_observations(
    catalog: list[PollutantDefinition],
    minimum_observations: int,
) -> list[PollutantDefinition]:
    return [
        pollutant
        for pollutant in catalog
        if pollutant.observation_count >= minimum_observations
    ]


def select_pollutants(
    catalog: list[PollutantDefinition],
    *,
    group_kind: str = "all",
    group_name: str = "all",
    explicit_pollutants: list[str] | None = None,
    minimum_observations: int,
) -> PollutantSelection:
    """Resolve a pollutant selection from explicit names or a configured group."""
    filtered_catalog = _filter_by_minimum_observations(catalog, minimum_observations)
    catalog_lookup = _catalog_by_name(filtered_catalog)
    if explicit_pollutants:
        missing = sorted(set(explicit_pollutants).difference(catalog_lookup))
        if missing:
            raise ValueError(
                "Unknown or under-threshold pollutants requested: "
                f"{missing}. Increase `--min-observations` if needed."
            )
        return PollutantSelection(
            pollutants=tuple(explicit_pollutants),
            group_kind="explicit",
            group_name="custom",
        )

    if group_kind == "all":
        return PollutantSelection(
            pollutants=tuple(item.name for item in filtered_catalog),
            group_kind="all",
            group_name="all",
        )

    if group_kind == "type":
        selected = [item.name for item in filtered_catalog if item.type_group == group_name]
    elif group_kind == "importance":
        selected = [
            item.name for item in filtered_catalog if item.importance_group == group_name
        ]
    else:
        raise ValueError(
            f"Unsupported pollutant group kind `{group_kind}`. Use all, type, or importance."
        )

    if not selected:
        raise ValueError(
            f"No pollutants matched group kind `{group_kind}` and group `{group_name}` "
            f"with minimum_observations={minimum_observations}."
        )

    return PollutantSelection(
        pollutants=tuple(selected),
        group_kind=group_kind,
        group_name=group_name,
    )


def summarize_groups(
    catalog: list[PollutantDefinition],
    *,
    minimum_observations: int,
) -> dict[str, dict[str, list[str]]]:
    """Return type and importance groups for discovery and CLI output."""
    filtered_catalog = _filter_by_minimum_observations(catalog, minimum_observations)
    summary: dict[str, dict[str, list[str]]] = {"type": {}, "importance": {}}
    for pollutant in filtered_catalog:
        summary["type"].setdefault(pollutant.type_group, []).append(pollutant.name)
        summary["importance"].setdefault(pollutant.importance_group, []).append(
            pollutant.name
        )
    for group_kind in summary.values():
        for pollutants in group_kind.values():
            pollutants.sort()
    return summary


def list_groups(
    settings: SensorAnalysisSettings = DEFAULT_SETTINGS,
    *,
    minimum_observations: int | None = None,
) -> dict[str, dict[str, list[str]]]:
    """Build a current group listing from the configured sensor data."""
    sensor_data = load_sensor_data(settings)
    transformations = load_transformations(settings)
    catalog = build_pollutant_catalog(sensor_data, transformations, settings)
    threshold = (
        settings.minimum_observations
        if minimum_observations is None
        else minimum_observations
    )
    return summarize_groups(catalog, minimum_observations=threshold)


__all__ = [
    "PollutantSelection",
    "list_groups",
    "select_pollutants",
    "summarize_groups",
]
