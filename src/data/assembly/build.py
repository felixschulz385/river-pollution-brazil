from __future__ import annotations

import logging
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.sources.land_cover.composition import compute_kernel_weighted_composition
from src.data.sources.land_cover.constants import (
    LAND_COVER_COMPOSITION_BUCKET_MAP,
    derive_mun_id_from_adm2_id,
)
from src.data.shared.sensor_upstream import (
    DEFAULT_ADM2_DISTANCE_KERNEL,
    DEFAULT_ADM2_KERNEL_BANDWIDTH_KM,
    bucket_kernel_weights,
)

from .constants import (
    CLIMATE_BUCKETED_SOURCE_TYPE,
    DATE_COLUMN,
    DATETIME_COLUMN,
    LAND_COVER_BUCKETED_SOURCE_TYPE,
    LONG_PIVOT_SOURCE_TYPE,
    SENSOR_MODE,
    YEAR_COLUMN,
)
from .schema import validate_required_columns


logger = logging.getLogger(__name__)


def _compute_kernel_weighted_bucket_values(
    long_df,
    *,
    entity_columns,
    category_column,
    value_column,
    kernel,
    bandwidth,
    bucket_column="bucket",
    bucket_map=None,
):
    """Collapse a table long over a distance `bucket_column` into one weighted value
    per (entity, category), e.g. climate's ADM2 output long over `distance_bucket`
    x `climate_variable`. Weights use the same bucket-kernel machinery as
    land-cover composition (`src.data.shared.sensor_upstream.bucket_kernel_weights`),
    renormalized per (entity, category) across the buckets that have a value.
    """
    bucket_map = LAND_COVER_COMPOSITION_BUCKET_MAP if bucket_map is None else bucket_map
    raw_weights = dict(
        zip(
            bucket_map,
            bucket_kernel_weights(
                [midpoint for _label, midpoint in bucket_map.values()],
                kernel=kernel,
                bandwidth=bandwidth,
            ),
        )
    )

    df = long_df.copy()
    # A bucket label absent from `bucket_map` (e.g. a negative self-trench bucket,
    # which `land_cover.composition`'s equivalent inner join also excludes) must
    # drop out of the weighting, not poison it: `.map` alone would leave NaN for
    # those rows, and NaN propagating into the per-group `weight_sum` below would
    # silently zero the *entire* (entity, category) group's output via
    # `np.where(weight_sum > 0, ...)`, not just exclude the unmapped row.
    df["_raw_weight"] = df[bucket_column].map(raw_weights).astype(float).fillna(0.0)
    df.loc[df[value_column].isna(), "_raw_weight"] = 0.0
    group_columns = [*entity_columns, category_column]
    weight_sum = df.groupby(group_columns)["_raw_weight"].transform("sum")
    df["_weight"] = np.where(weight_sum > 0, df["_raw_weight"] / weight_sum, 0.0)
    df["_weighted_value"] = df[value_column].fillna(0.0) * df["_weight"]

    weighted = df.groupby(group_columns, as_index=False)["_weighted_value"].sum()
    wide = weighted.pivot(
        index=list(entity_columns), columns=category_column, values="_weighted_value"
    ).reset_index()
    wide.columns.name = None
    return wide


def _pivot_long_source(frame, source):
    """Filter a long table and pivot `pivot_column`'s values into wide columns.

    Used for sources like climate's sensor-bucketed output, which is long over
    both `distance_bucket` and `climate_variable`; `source.filter` narrows to a
    single row per (join_keys, pivot value) (e.g. one distance bucket) before
    the pivot, and `source.value_columns` are the measurement columns pivoted
    out per value of `pivot_column`, named `{pivot_value}_{value_column}`.
    """
    for column, value in source.filter.items():
        frame = frame[frame[column] == value]

    try:
        wide = frame.pivot(
            index=list(source.join_keys),
            columns=source.pivot_column,
            values=list(source.value_columns),
        )
    except ValueError as exc:
        duplicate_keys = frame.loc[
            frame.duplicated(subset=[*source.join_keys, source.pivot_column], keep=False),
            [*source.join_keys, source.pivot_column],
        ].drop_duplicates()
        raise ValueError(
            f"Source {source.name!r} has duplicate rows for the same "
            f"({', '.join(source.join_keys)}, {source.pivot_column}) combination, so it can't "
            f"be pivoted; first few duplicated keys:\n{duplicate_keys.head(5)}"
        ) from exc
    wide.columns = [
        f"{pivot_value}_{value_column}" for value_column, pivot_value in wide.columns
    ]
    return wide.reset_index()


def _load_source_frame(source, *, root_dir):
    """Load one configured source and return it alongside its canonical join keys.

    `source.join_keys` names the join columns as they exist in the raw source
    table; `source.id_map` (if set) renames a raw id column (e.g. `adm2_id`) to
    the canonical column used across sources (e.g. `mun_id`). The returned join
    keys reflect that renaming so callers can merge on a consistent key set.
    """
    source_path = Path(root_dir) / source.path
    logger.info("Loading assembly source '%s' from %s", source.name, source_path)
    frame = pd.read_parquet(source_path)
    index_names = [name for name in frame.index.names if name is not None]
    if index_names:
        # Upstream writers persist their join keys as the parquet row index,
        # either instead of (sensor-data assembly) or alongside (land-cover
        # assembly, which keeps the index columns via `drop=False`) plain
        # columns; join_keys below must be selectable as columns either way.
        duplicate_names = [name for name in index_names if name in frame.columns]
        if duplicate_names:
            frame = frame.reset_index(level=duplicate_names, drop=True)
        if len(duplicate_names) < len(index_names):
            frame = frame.reset_index()

    if source.type == LAND_COVER_BUCKETED_SOURCE_TYPE:
        kwargs = {"entity_columns": source.join_keys}
        if source.kernel is not None:
            kwargs["kernel"] = source.kernel
        if source.bandwidth is not None:
            kwargs["bandwidth"] = source.bandwidth
        frame = compute_kernel_weighted_composition(frame, **kwargs)
    elif source.type == CLIMATE_BUCKETED_SOURCE_TYPE:
        frame = _compute_kernel_weighted_bucket_values(
            frame,
            entity_columns=source.join_keys,
            category_column="climate_variable",
            value_column="mean_value",
            kernel=source.kernel or DEFAULT_ADM2_DISTANCE_KERNEL,
            bandwidth=(
                source.bandwidth if source.bandwidth is not None else DEFAULT_ADM2_KERNEL_BANDWIDTH_KM
            ),
        )
    elif source.type == LONG_PIVOT_SOURCE_TYPE:
        frame = _pivot_long_source(frame, source)

    for datetime_like_column in (DATE_COLUMN, DATETIME_COLUMN):
        if datetime_like_column in frame.columns:
            # DuckDB-written sources (e.g. climate's DATE columns) round-trip
            # through parquet as plain `object`/python-date values rather than
            # pandas datetime64, which breaks merges against datetime64 join
            # keys from other sources with a dtype mismatch.
            frame[datetime_like_column] = pd.to_datetime(frame[datetime_like_column])

    canonical_join_keys = list(source.join_keys)
    if source.id_map:
        for from_column, to_column in source.id_map.items():
            frame[to_column] = frame[from_column].map(derive_mun_id_from_adm2_id)
            canonical_join_keys = [
                to_column if key == from_column else key for key in canonical_join_keys
            ]

    canonical_join_keys = list(dict.fromkeys(canonical_join_keys))
    selected_columns = list(dict.fromkeys([*canonical_join_keys, *source.variables]))
    validate_required_columns(frame, selected_columns, source.name)
    selected_frame = frame[selected_columns].copy()

    numeric_columns = [
        column
        for column in source.variables
        if column not in source.categorical_variables
        and column not in (DATE_COLUMN, DATETIME_COLUMN)
    ]
    for column in numeric_columns:
        if pd.api.types.is_numeric_dtype(selected_frame[column]):
            continue
        coerced = pd.to_numeric(selected_frame[column], errors="coerce")
        newly_unparsable = coerced.isna() & selected_frame[column].notna()
        if newly_unparsable.any():
            logger.warning(
                "Source '%s' column '%s' has %d non-numeric value(s) that were "
                "coerced to NaN while enforcing float dtype.",
                source.name,
                column,
                int(newly_unparsable.sum()),
            )
        selected_frame[column] = coerced.astype(float)

    return selected_frame, canonical_join_keys


def _order_assembled_columns(df, dataset_config):
    """Reorder assembled columns so they read as index, then source-by-source.

    Merge order (and thus raw column order) depends on incidental details like
    dict/list ordering inside `_load_source_frame`; this makes the output
    columns deterministic and grouped by the config's declared source order
    instead, which is far more useful for a human skimming the table: the
    dataset index first, any derived date/year helper columns next, then each
    source's variables in the order it lists them.
    """
    ordered_columns = list(dataset_config.index)
    for helper_column in (YEAR_COLUMN, DATE_COLUMN, DATETIME_COLUMN):
        if helper_column in df.columns and helper_column not in ordered_columns:
            ordered_columns.append(helper_column)
    for source in dataset_config.sources:
        for variable in source.variables:
            if variable in df.columns and variable not in ordered_columns:
                ordered_columns.append(variable)
    remaining_columns = [column for column in df.columns if column not in ordered_columns]
    return df[[*ordered_columns, *remaining_columns]]


def assemble_dataset(dataset_config, *, root_dir="."):
    """Join the configured sources for one assembly dataset into a single wide table."""
    if not dataset_config.sources:
        raise ValueError(f"Dataset '{dataset_config.id}' has no sources configured.")

    frames = [
        _load_source_frame(source, root_dir=root_dir) for source in dataset_config.sources
    ]

    if dataset_config.mode == SENSOR_MODE:
        primary_frame, primary_keys = frames[0]
        if DATETIME_COLUMN in primary_frame.columns:
            primary_datetime = pd.to_datetime(primary_frame[DATETIME_COLUMN])
            primary_frame = primary_frame.copy()
            if YEAR_COLUMN not in primary_frame.columns:
                primary_frame[YEAR_COLUMN] = primary_datetime.dt.year
            if DATE_COLUMN not in primary_frame.columns:
                primary_frame[DATE_COLUMN] = primary_datetime.dt.floor("D")
            frames[0] = (primary_frame, primary_keys)

    merged = reduce(
        lambda left, right: pd.merge(
            left,
            right[0],
            on=right[1],
            how="left",
            validate="many_to_one",
        ),
        frames[1:],
        frames[0][0],
    )
    merged = merged.sort_values(list(dataset_config.index)).reset_index(drop=True)
    merged = _order_assembled_columns(merged, dataset_config)
    return merged


def write_dataset(df, output_path):
    """Persist an assembled dataset, honoring the output path's file suffix."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".feather":
        df.to_feather(output_path)
    else:
        df.to_parquet(output_path, index=False)
    logger.info("Assembled dataset written to %s (shape=%s)", output_path, df.shape)
    return output_path
