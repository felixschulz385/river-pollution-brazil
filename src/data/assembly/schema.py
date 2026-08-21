from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .constants import ASSEMBLY_MODES, WIDE_SOURCE_TYPE


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
    # `long_pivot` only: keep rows where `frame[column] == value` for each pair
    # in `filter`, then pivot `pivot_column`'s values across `value_columns`
    # into `{pivot_value}_{value_column}` columns (e.g. long climate rows keyed
    # by `climate_variable` -> `2t_mean_day`, `2t_mean_7d`, ...).
    filter: dict[str, object] = field(default_factory=dict)
    pivot_column: str | None = None
    value_columns: tuple[str, ...] = ()
    # `land_cover_bucketed`/`climate_bucketed` only: kernel used to weight
    # discrete distance buckets when collapsing them into one value per
    # entity. Defaults are set by the transform (`inv_sqrt_distance` for land
    # cover, the shared Gaussian ADM2 default for climate) when omitted.
    kernel: str | None = None
    bandwidth: float | None = None
    # Variables that are categorical/identifiers rather than measurements
    # (e.g. population's `sex`/`age_group`) and must be excluded from the
    # numeric-dtype coercion `build._load_source_frame` applies to variables.
    categorical_variables: tuple[str, ...] = ()


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
    if source_type == "long_pivot" and not raw_source.get("pivot_column"):
        raise ValueError(
            f"Dataset '{dataset_id}' source '{raw_source.get('name', raw_source['path'])}' "
            "has type 'long_pivot' but no 'pivot_column'."
        )
    return AssemblySource(
        name=raw_source.get("name", raw_source["path"]),
        path=raw_source["path"],
        join_keys=tuple(raw_source["join_keys"]),
        variables=tuple(raw_source["variables"]),
        type=source_type,
        id_map=dict(raw_source.get("id_map", {})),
        filter=dict(raw_source.get("filter", {})),
        pivot_column=raw_source.get("pivot_column"),
        value_columns=tuple(raw_source.get("value_columns", ())),
        kernel=raw_source.get("kernel"),
        bandwidth=raw_source.get("bandwidth"),
        categorical_variables=tuple(raw_source.get("categorical_variables", ())),
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
