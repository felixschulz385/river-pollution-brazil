from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .constants import (
    ASSEMBLY_MODES,
    BUCKET_SCOPES,
    BUCKETED_SOURCE_TYPES,
    DEFAULT_PIVOT_COLUMN_TEMPLATE,
    RING_BUCKETED_SOURCE_TYPES,
    SOURCE_TYPES,
    UPSTREAM_SCOPE,
    WIDE_SOURCE_TYPE,
)


def validate_required_columns(frame, required_columns, frame_name):
    """Raise a clear error if a table is missing required columns."""
    missing_columns = set(required_columns).difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"{frame_name} is missing required columns: {sorted(missing_columns)}."
        )


@dataclass(frozen=True)
class AssemblySource:
    """One input table contributing variables to an assembled dataset."""

    name: str
    path: str
    join_keys: tuple[str, ...]
    variables: tuple[str, ...]
    type: str = WIDE_SOURCE_TYPE
    id_map: dict[str, str] = field(default_factory=dict)
    # Plain column renames applied like `id_map` but without deriving a new id
    # (e.g. SIH's 6-digit `municipality_code` is already a `mun_id`).
    rename: dict[str, str] = field(default_factory=dict)
    # `long_pivot` only: keep rows where `frame[column] == value` for each pair
    # in `filter`, then pivot `pivot_column`'s values across `value_columns`
    # into `{pivot_value}_{value_column}` columns (e.g. long climate rows keyed
    # by `climate_variable` -> `2t_mean_day`, `2t_mean_7d`, ...).
    filter: dict[str, object] = field(default_factory=dict)
    pivot_column: str | None = None
    value_columns: tuple[str, ...] = ()
    # `long_pivot` only: output column name, formatted with `pivot_value` and
    # `value_column` (e.g. `hosp_icd_{pivot_value}` for a single value column).
    column_template: str = DEFAULT_PIVOT_COLUMN_TEMPLATE
    # `land_cover_bucketed`/`climate_bucketed` only: kernel used to weight
    # discrete distance buckets when collapsing them into one value per
    # entity. Defaults are set by the transform (`inv_sqrt_distance` for land
    # cover, the shared Gaussian ADM2 default for climate) when omitted.
    kernel: str | None = None
    bandwidth: float | None = None
    # `land_cover_bucketed`/`climate_bucketed` only: `upstream` (default) or
    # `local` -- see `constants.BUCKET_SCOPES`.
    scope: str = UPSTREAM_SCOPE
    # Variables that are categorical/identifiers rather than measurements
    # (e.g. biome_adm2's dominant-biome label) and must be excluded from the
    # numeric-dtype coercion `build._load_source_frame` applies to variables.
    categorical_variables: tuple[str, ...] = ()
    # `land_cover_bucketed`/`mining_bucketed` upstream scope only: keep the
    # rings that start below `max_km` (e.g. 50 -> the 0-25 and 25-50 km rings).
    max_km: float | None = None
    # Value given to this source's variables where the merge left them missing,
    # in `fill_years` (inclusive) only, or every year if `fill_years` is unset:
    # for count sources that list only non-zero rows (e.g. malaria cases).
    fill_value: float | None = None
    fill_years: tuple[int, int] | None = None


@dataclass(frozen=True)
class AssemblyDataset:
    """One output analysis-ready table declared in the assembly config."""

    id: str
    mode: str
    index: tuple[str, ...]
    output_path: str
    sources: tuple[AssemblySource, ...]


def _parse_source(raw_source, *, dataset_id):
    required_keys = {"path", "join_keys", "variables"}
    missing_keys = required_keys.difference(raw_source)
    if missing_keys:
        raise ValueError(
            f"Dataset '{dataset_id}' source {raw_source.get('name', raw_source)} "
            f"is missing required keys: {sorted(missing_keys)}."
        )
    source_type = raw_source.get("type", WIDE_SOURCE_TYPE)
    if source_type not in SOURCE_TYPES:
        raise ValueError(
            f"Dataset '{dataset_id}' source '{raw_source.get('name', raw_source.get('path'))}' "
            f"has unknown type '{source_type}'; must be one of {sorted(SOURCE_TYPES)}. "
            "An unrecognized type would otherwise silently fall through "
            "`build.py`'s type dispatch and be treated as a plain wide source."
        )
    if source_type == "long_pivot" and not raw_source.get("pivot_column"):
        raise ValueError(
            f"Dataset '{dataset_id}' source '{raw_source.get('name', raw_source['path'])}' "
            "has type 'long_pivot' but no 'pivot_column'."
        )
    scope = raw_source.get("scope", UPSTREAM_SCOPE)
    if scope not in BUCKET_SCOPES:
        raise ValueError(
            f"Dataset '{dataset_id}' source '{raw_source.get('name', raw_source['path'])}' "
            f"has unknown scope '{scope}'; must be one of {list(BUCKET_SCOPES)}."
        )
    name = raw_source.get("name", raw_source["path"])
    if scope != UPSTREAM_SCOPE and source_type not in BUCKETED_SOURCE_TYPES:
        raise ValueError(
            f"Dataset '{dataset_id}' source '{name}' "
            f"sets scope '{scope}', which only applies to bucketed sources."
        )
    max_km = raw_source.get("max_km")
    if max_km is not None and (source_type not in RING_BUCKETED_SOURCE_TYPES or scope != UPSTREAM_SCOPE):
        raise ValueError(
            f"Dataset '{dataset_id}' source '{name}' sets max_km, which only applies "
            f"to the upstream scope of {list(RING_BUCKETED_SOURCE_TYPES)} sources."
        )
    fill = raw_source.get("fill") or {}
    unknown_fill_keys = set(fill) - {"value", "years"}
    if unknown_fill_keys or (fill and "value" not in fill):
        raise ValueError(
            f"Dataset '{dataset_id}' source '{name}' has an invalid fill {fill!r}; "
            "expected {value: <number>, years: [<first>, <last>]} (years optional)."
        )
    return AssemblySource(
        name=raw_source.get("name", raw_source["path"]),
        path=raw_source["path"],
        join_keys=tuple(raw_source["join_keys"]),
        variables=tuple(raw_source["variables"]),
        type=source_type,
        id_map=dict(raw_source.get("id_map", {})),
        rename=dict(raw_source.get("rename", {})),
        filter=dict(raw_source.get("filter", {})),
        pivot_column=raw_source.get("pivot_column"),
        value_columns=tuple(raw_source.get("value_columns", ())),
        column_template=raw_source.get("column_template", DEFAULT_PIVOT_COLUMN_TEMPLATE),
        kernel=raw_source.get("kernel"),
        bandwidth=raw_source.get("bandwidth"),
        scope=scope,
        categorical_variables=tuple(raw_source.get("categorical_variables", ())),
        max_km=max_km,
        fill_value=fill.get("value"),
        fill_years=tuple(fill["years"]) if fill.get("years") else None,
    )


def _parse_dataset(raw_dataset):
    required_keys = {"id", "mode", "index", "output_path", "sources"}
    missing_keys = required_keys.difference(raw_dataset)
    if missing_keys:
        raise ValueError(
            f"Assembly dataset {raw_dataset.get('id', raw_dataset)} is missing "
            f"required keys: {sorted(missing_keys)}."
        )
    if raw_dataset["mode"] not in ASSEMBLY_MODES:
        raise ValueError(
            f"Dataset '{raw_dataset['id']}' has unsupported mode "
            f"{raw_dataset['mode']!r}; expected one of {ASSEMBLY_MODES}."
        )
    return AssemblyDataset(
        id=raw_dataset["id"],
        mode=raw_dataset["mode"],
        index=tuple(raw_dataset["index"]),
        output_path=raw_dataset["output_path"],
        sources=tuple(
            _parse_source(raw_source, dataset_id=raw_dataset["id"])
            for raw_source in raw_dataset["sources"]
        ),
    )


def load_assembly_config(config_path):
    """Parse the assembly datasets config file into `AssemblyDataset` records."""
    with open(Path(config_path)) as config_file:
        raw_config = yaml.safe_load(config_file) or {}
    raw_datasets = raw_config.get("datasets", [])
    datasets = [_parse_dataset(raw_dataset) for raw_dataset in raw_datasets]
    id_counts = Counter(dataset.id for dataset in datasets)
    duplicate_ids = {dataset_id for dataset_id, count in id_counts.items() if count > 1}
    if duplicate_ids:
        raise ValueError(f"Duplicate dataset ids in {config_path}: {sorted(duplicate_ids)}.")
    return {dataset.id: dataset for dataset in datasets}


def get_dataset_config(config_path, dataset_id):
    """Load the config and return the single dataset matching `dataset_id`."""
    datasets = load_assembly_config(config_path)
    if dataset_id not in datasets:
        raise ValueError(
            f"Dataset '{dataset_id}' not found in {config_path}. "
            f"Available datasets: {sorted(datasets)}."
        )
    return datasets[dataset_id]
