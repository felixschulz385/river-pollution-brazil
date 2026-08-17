import numpy as np
import pandas as pd


def prepare_entity_links(
    assignments_df,
    *,
    entity_column,
    location_column,
):
    required_columns = {entity_column, location_column}
    missing_columns = required_columns.difference(assignments_df.columns)
    if missing_columns:
        raise ValueError(
            "Assignment data is missing required columns: "
            f"{sorted(missing_columns)}."
        )

    links = assignments_df[[entity_column, location_column]].dropna().copy()
    links[entity_column] = links[entity_column].astype(str)
    links[location_column] = links[location_column].astype(np.int64)
    return links.drop_duplicates(
        subset=[entity_column, location_column],
        keep="first",
    )


def collapse_same_period_observations(
    observations,
    *,
    entity_column,
    period_column,
    ordering_column,
):
    if observations.empty:
        return observations

    collapsed = observations.sort_values(
        [entity_column, period_column, ordering_column],
        kind="mergesort",
    )
    group_columns = [entity_column, period_column]
    duplicate_mask = collapsed.duplicated(subset=group_columns, keep=False)
    if not duplicate_mask.any():
        return collapsed.reset_index(drop=True)

    duplicate_rows = collapsed.loc[duplicate_mask].copy()
    duplicate_rows = (
        duplicate_rows.groupby(
            group_columns,
            sort=False,
            observed=True,
            as_index=False,
            dropna=False,
        )
        .first()
    )

    collapsed = pd.concat(
        [collapsed.loc[~duplicate_mask], duplicate_rows],
        ignore_index=True,
    ).sort_values(
        [entity_column, period_column, ordering_column],
        kind="mergesort",
    )
    return collapsed.loc[:, observations.columns].reset_index(drop=True)


def prepare_observation_targets(
    observations_df,
    entity_links_df,
    *,
    entity_column,
    date_column,
    timestamp_column,
    location_column,
):
    index_names = [name for name in observations_df.index.names if name is not None]
    missing_index_columns = [
        column
        for column in (entity_column, date_column, timestamp_column)
        if column in index_names and column not in observations_df.columns
    ]
    if missing_index_columns:
        observations_df = observations_df.reset_index(level=missing_index_columns)
    else:
        observations_df = observations_df.copy()

    if entity_column not in observations_df.columns:
        raise ValueError(
            "Observation data must include "
            f"`{entity_column}` for upstream assembly."
        )

    if date_column in observations_df.columns:
        columns = [entity_column, date_column]
        if timestamp_column in observations_df.columns:
            columns.append(timestamp_column)
        targets = observations_df[columns].copy()
        targets[date_column] = pd.to_datetime(
            targets[date_column],
            errors="coerce",
        ).dt.normalize()
        if timestamp_column in targets.columns:
            targets[timestamp_column] = pd.to_datetime(
                targets[timestamp_column],
                errors="coerce",
            )
        else:
            targets[timestamp_column] = targets[date_column]
    elif timestamp_column in observations_df.columns:
        targets = observations_df[[entity_column, timestamp_column]].copy()
        targets[timestamp_column] = pd.to_datetime(
            targets[timestamp_column],
            errors="coerce",
        )
        targets[date_column] = targets[timestamp_column].dt.normalize()
    else:
        raise ValueError(
            "Observation data must include either "
            f"`{date_column}` or `{timestamp_column}`."
        )

    targets[entity_column] = targets[entity_column].astype(str)
    targets = targets.dropna(subset=[entity_column, timestamp_column, date_column])
    targets = collapse_same_period_observations(
        targets,
        entity_column=entity_column,
        period_column=date_column,
        ordering_column=timestamp_column,
    )

    location_lookup = entity_links_df.drop_duplicates(
        subset=[entity_column],
        keep="first",
    )
    targets = targets.merge(
        location_lookup,
        on=entity_column,
        how="inner",
        validate="many_to_one",
    )
    targets[location_column] = targets[location_column].astype(np.int64)
    return targets.drop_duplicates(
        subset=[entity_column, date_column],
        keep="first",
    ).reset_index(drop=True)


def build_location_period_targets(
    observations_df,
    assignments_df,
    *,
    entity_column,
    date_column,
    timestamp_column,
    location_column,
    period_value_column,
):
    entity_links = prepare_entity_links(
        assignments_df,
        entity_column=entity_column,
        location_column=location_column,
    )
    targets = prepare_observation_targets(
        observations_df,
        entity_links,
        entity_column=entity_column,
        date_column=date_column,
        timestamp_column=timestamp_column,
        location_column=location_column,
    )
    targets[period_value_column] = pd.to_datetime(
        targets[date_column],
        errors="coerce",
    ).dt.year
    targets = targets.dropna(subset=[period_value_column])
    targets[period_value_column] = targets[period_value_column].astype(int)
    return (
        targets[[location_column, period_value_column]]
        .drop_duplicates()
        .sort_values([location_column, period_value_column])
        .reset_index(drop=True)
    )


def normalize_network_frame(frame):
    """Return a copy with a simple RangeIndex to avoid index/column ambiguity."""
    if frame is None:
        return None
    return frame.reset_index(drop=True).copy()


def validate_network_index_tables(
    network,
    *,
    location_column,
    system_column,
    position_column,
):
    if network.trenches is None:
        raise ValueError("River network must include trench data.")
    if not network.trench_reachability_matrices:
        raise ValueError("River network must have trench reachability data computed.")
    if not network.trench_distance_matrices:
        raise ValueError("River network must have trench distance data computed.")

    required_columns = {location_column, system_column, position_column}
    missing_columns = required_columns.difference(network.trenches.columns)
    if missing_columns:
        raise ValueError(
            "River trench data is missing matrix index columns: "
            f"{sorted(missing_columns)}."
        )


def build_group_index_lookup(
    frame,
    *,
    location_column,
    system_column,
    position_column,
):
    frame = frame.reset_index(drop=True)
    system_tables = {
        int(system_id): system_rows[[location_column, position_column]]
        .sort_values(position_column)
        .reset_index(drop=True)
        for system_id, system_rows in frame.groupby(system_column)
    }
    system_location_arrays = {
        system_id: system_rows[location_column].to_numpy(dtype=np.int64)
        for system_id, system_rows in system_tables.items()
    }
    system_positions = {
        system_id: dict(
            zip(
                system_rows[location_column].to_numpy(dtype=np.int64),
                system_rows[position_column].to_numpy(dtype=np.int64),
            )
        )
        for system_id, system_rows in system_tables.items()
    }
    return system_location_arrays, system_positions


def sparse_row(matrix, row_idx):
    if hasattr(matrix, "getrow"):
        return matrix.getrow(row_idx)
    return matrix[row_idx : row_idx + 1, :]


def resolve_reachable_distances(
    location_id,
    network,
    system_location_arrays,
    system_positions,
    *,
    location_column,
    distance_column,
    system_column,
    position_column,
):
    location_row = (
        network.trenches.reset_index(drop=True)
        .loc[
            lambda frame: frame[location_column] == location_id,
            [system_column, position_column],
        ]
        .drop_duplicates()
    )
    if len(location_row) == 0:
        raise KeyError(f"Unknown location id in river network: {location_id}")
    if len(location_row) > 1:
        raise ValueError(f"Expected one trench row for location id {location_id}.")

    system_id = int(location_row.iloc[0][system_column])
    target_position = int(location_row.iloc[0][position_column])
    system_ids = system_location_arrays.get(system_id, np.asarray([], dtype=np.int64))
    if len(system_ids) == 0:
        return pd.DataFrame(columns=[location_column, distance_column])

    if target_position not in set(system_positions[system_id].values()):
        raise ValueError(
            f"Trench index {target_position} for location id {location_id} is invalid."
        )

    reach_row = sparse_row(
        network.trench_reachability_matrices[system_id],
        target_position,
    )
    dist_row = sparse_row(
        network.trench_distance_matrices[system_id],
        target_position,
    )
    distance_lookup = dict(zip(dist_row.indices.tolist(), dist_row.data.tolist()))

    reachable_records = [
        {
            location_column: int(system_ids[col_idx]),
            distance_column: float(distance_lookup.get(col_idx, 0.0)),
        }
        for col_idx in reach_row.indices.tolist()
    ]
    if location_id not in [record[location_column] for record in reachable_records]:
        reachable_records.append(
            {
                location_column: int(location_id),
                distance_column: 0.0,
            }
        )

    return pd.DataFrame(reachable_records).sort_values(
        [distance_column, location_column]
    ).reset_index(drop=True)


DISTANCE_KERNELS = ("uniform", "triangular", "epanechnikov", "gaussian", "exponential")

# Shared default continuous-kernel settings for ADM2 upstream aggregation, so
# climate and land-cover ADM2 outputs weight upstream distance the same way
# unless a caller deliberately overrides one of them.
DEFAULT_ADM2_DISTANCE_KERNEL = "gaussian"
DEFAULT_ADM2_KERNEL_BANDWIDTH_KM = 1_000_000

# The notebook-derived kernel (weight = 1/sqrt(distance)), kept as an
# alternative to the continuous DISTANCE_KERNELS for bucketed aggregation.
INV_SQRT_DISTANCE_KERNEL = "inv_sqrt_distance"
BUCKET_DISTANCE_KERNELS = (INV_SQRT_DISTANCE_KERNEL, *DISTANCE_KERNELS)

# Both land-cover and climate ADM2 aggregation flag, per discrete upstream
# distance bucket, whether any trench in that bucket directly intersects the
# ADM2 polygon (as opposed to being purely upstream of it).
BUCKET_INTERSECTS_ADM2_COLUMN = "bucket_intersects_adm2"


def bucket_kernel_weights(bucket_midpoints_km, *, kernel, bandwidth=None):
    """Return raw (unnormalized) kernel weights for a set of bucket midpoint distances.

    Used to weight discrete upstream distance buckets (rather than individual
    trench distances) when collapsing a bucketed table into one value per
    entity -- shared between land-cover composition and climate ADM2 assembly
    so both weight distance the same way for a given `kernel`/`bandwidth`.
    """
    midpoints = np.asarray(bucket_midpoints_km, dtype=float)
    if kernel == INV_SQRT_DISTANCE_KERNEL:
        return 1.0 / np.sqrt(midpoints)
    if bandwidth is None:
        raise ValueError(f"kernel={kernel!r} requires a bandwidth.")
    return distance_kernel_weights(midpoints, kernel=kernel, bandwidth=bandwidth)


def distance_kernel_weights(distances, *, kernel, bandwidth):
    """Convert upstream distances into continuous kernel-decayed weights.

    Where `label_values_by_intervals` assigns each distance to one discrete
    bucket, this assigns a continuous weight that decays with `bandwidth`
    under the chosen `kernel` -- used for ADM2 upstream aggregation that
    blends contributions across trenches rather than binning them.
    """
    if kernel not in DISTANCE_KERNELS:
        raise ValueError(f"Unknown kernel: {kernel}. Available: {DISTANCE_KERNELS}")

    distances = np.asarray(distances, dtype=float)
    scaled = distances / bandwidth
    if kernel == "uniform":
        return (distances <= bandwidth).astype(float)
    if kernel == "triangular":
        return np.clip(1 - scaled, 0, None)
    if kernel == "epanechnikov":
        return np.clip(1 - scaled**2, 0, None)
    if kernel == "gaussian":
        return np.exp(-(scaled**2))
    return np.exp(-scaled)  # exponential


def label_values_by_intervals(values, intervals):
    """Assign values to configured closed/open interval labels."""
    values = pd.Series(values, copy=False)
    labels = pd.Series(pd.NA, index=values.index, dtype="object")
    for interval_name, lower_bound, upper_bound in intervals:
        if lower_bound == 0:
            mask = values.ge(lower_bound) & values.le(upper_bound)
        elif np.isinf(upper_bound):
            mask = values.gt(lower_bound)
        else:
            mask = values.gt(lower_bound) & values.le(upper_bound)
        labels.loc[mask] = interval_name
    return labels


def build_target_reachability_lookup(
    network,
    target_location_ids,
    *,
    location_column,
    distance_column,
    category_column,
    system_column,
    position_column,
    categorize_distances,
):
    validate_network_index_tables(
        network,
        location_column=location_column,
        system_column=system_column,
        position_column=position_column,
    )
    system_location_arrays, system_positions = build_group_index_lookup(
        network.trenches,
        location_column=location_column,
        system_column=system_column,
        position_column=position_column,
    )

    upstream_frames = []
    for location_id in target_location_ids:
        upstream = resolve_reachable_distances(
            int(location_id),
            network,
            system_location_arrays,
            system_positions,
            location_column=location_column,
            distance_column=distance_column,
            system_column=system_column,
            position_column=position_column,
        )
        upstream[category_column] = categorize_distances(upstream[distance_column])
        upstream = upstream.dropna(subset=[category_column])
        if upstream.empty:
            continue
        upstream["target_location_id"] = int(location_id)
        upstream = upstream.rename(columns={location_column: "source_location_id"})
        upstream_frames.append(
            upstream[["target_location_id", "source_location_id", category_column]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

    if not upstream_frames:
        return pd.DataFrame(
            columns=["target_location_id", "source_location_id", category_column]
        )
    return pd.concat(upstream_frames, ignore_index=True)


def explode_list_matches(
    frame,
    *,
    id_columns,
    values_column,
    value_name,
    weights_column=None,
    weight_name=None,
):
    required_columns = set(id_columns) | {values_column}
    if weights_column is not None:
        required_columns.add(weights_column)
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            "Input data is missing required match columns: "
            f"{sorted(missing_columns)}."
        )

    matches = frame[list(id_columns) + [values_column] + ([weights_column] if weights_column else [])].copy()
    matches = matches.rename(columns={values_column: value_name})
    matches[value_name] = matches[value_name].apply(
        lambda values: values if isinstance(values, list) else []
    )
    explode_columns = [value_name]
    if weights_column is not None:
        renamed_weight = weight_name or weights_column
        matches = matches.rename(columns={weights_column: renamed_weight})
        matches[renamed_weight] = matches[renamed_weight].apply(
            lambda values: values if isinstance(values, list) else []
        )
        explode_columns.append(renamed_weight)
    matches = matches.explode(explode_columns, ignore_index=True)
    return matches.dropna(subset=[value_name])


def prepare_trench_adm2_matches(
    network,
    *,
    rn_module,
    trench_id_column,
):
    if network.trenches is None:
        raise ValueError("River network must include trench data.")

    trenches = network.trenches.reset_index(drop=True)
    system_column = rn_module.SYSTEM_ID_KEY
    required_columns = {trench_id_column, system_column}
    missing_columns = required_columns.difference(trenches.columns)
    if missing_columns:
        raise ValueError(
            "River trench data is missing required column(s): "
            f"{sorted(missing_columns)}."
        )

    system_lookup = trenches[[trench_id_column, system_column]].drop_duplicates()
    trench_adm2_table = getattr(network, "trench_adm2_table", None)
    adm2_column = getattr(rn_module, "ADM2_COLUMN", "adm2")
    if trench_adm2_table is not None and not trench_adm2_table.empty:
        trench_adm2_table = trench_adm2_table.reset_index(drop=True)
        missing_adm2_columns = {trench_id_column, adm2_column}.difference(
            trench_adm2_table.columns
        )
        if missing_adm2_columns:
            raise ValueError(
                "River network trench-to-ADM2 table is missing required column(s): "
                f"{sorted(missing_adm2_columns)}."
            )

        trench_adm2 = trench_adm2_table[[trench_id_column, adm2_column]].dropna().drop_duplicates()
        trench_adm2 = trench_adm2.rename(columns={adm2_column: "adm2"})
        return trench_adm2.merge(
            system_lookup,
            on=trench_id_column,
            how="inner",
            validate="many_to_one",
        )

    adm2_list_column = getattr(rn_module, "ADM2_LIST_COLUMN", None)
    intersection_lengths_column = getattr(
        rn_module,
        "ADM2_INTERSECTION_LENGTHS_COLUMN",
        None,
    )
    if adm2_list_column is None or adm2_list_column not in trenches.columns:
        if "adm2" not in trenches.columns:
            raise ValueError(
                "River trench data must include either `adm2` or saved ADM2 list columns "
                "for upstream aggregation."
            )
        trench_adm2 = trenches[[trench_id_column, system_column, "adm2"]].copy()
        trench_adm2 = trench_adm2.dropna(subset=["adm2"])
        trench_adm2["intersection_length"] = np.nan
        return trench_adm2

    return explode_list_matches(
        trenches,
        id_columns=[trench_id_column, system_column],
        values_column=adm2_list_column,
        value_name="adm2",
        weights_column=intersection_lengths_column,
        weight_name="intersection_length",
    )


def resolve_multi_seed_reachable_distances(
    network,
    seed_assignments,
    *,
    location_column,
    distance_column,
    system_column,
    position_column,
    system_location_arrays=None,
    system_positions=None,
):
    """Resolve min-distance-reachable locations from a set of seed locations.

    `system_location_arrays`/`system_positions` may be precomputed once via
    `build_group_index_lookup` and reused across many calls (e.g. one call per
    ADM2 unit) -- rebuilding them from `network.trenches` on every call means
    redoing a full-network groupby per unit instead of once overall.
    """
    if system_location_arrays is None or system_positions is None:
        validate_network_index_tables(
            network,
            location_column=location_column,
            system_column=system_column,
            position_column=position_column,
        )
        system_location_arrays, system_positions = build_group_index_lookup(
            network.trenches,
            location_column=location_column,
            system_column=system_column,
            position_column=position_column,
        )

    distance_frames = []
    for system_id, system_seed_rows in seed_assignments.groupby(system_column):
        system_id = int(system_id)
        system_ids = system_location_arrays.get(system_id, np.asarray([], dtype=np.int64))
        if len(system_ids) == 0:
            continue

        position_lookup = system_positions[system_id]
        seed_positions = np.asarray(
            [
                position_lookup[location_id]
                for location_id in system_seed_rows[location_column]
                if location_id in position_lookup
            ],
            dtype=np.int64,
        )
        if len(seed_positions) == 0:
            continue

        system_reachability = network.trench_reachability_matrices[system_id][seed_positions, :].tocsr()
        system_distance = network.trench_distance_matrices[system_id][seed_positions, :].tocsr()

        min_distances = np.full(len(system_ids), np.inf)
        for row_idx in range(system_reachability.shape[0]):
            reach_start = system_reachability.indptr[row_idx]
            reach_end = system_reachability.indptr[row_idx + 1]
            reachable_cols = system_reachability.indices[reach_start:reach_end]
            if len(reachable_cols) == 0:
                continue

            dist_start = system_distance.indptr[row_idx]
            dist_end = system_distance.indptr[row_idx + 1]
            distance_lookup = dict(
                zip(
                    system_distance.indices[dist_start:dist_end],
                    system_distance.data[dist_start:dist_end],
                )
            )
            reachable_distances = np.asarray(
                [distance_lookup.get(col, 0.0) for col in reachable_cols],
                dtype=float,
            )
            np.minimum.at(min_distances, reachable_cols, reachable_distances)

        reachable_mask = np.isfinite(min_distances)
        if not np.any(reachable_mask):
            continue

        distance_frames.append(
            pd.DataFrame(
                {
                    location_column: system_ids[reachable_mask],
                    distance_column: min_distances[reachable_mask],
                }
            )
        )

    if not distance_frames:
        return pd.DataFrame(columns=[location_column, distance_column])

    return (
        pd.concat(distance_frames, ignore_index=True)
        .groupby(location_column, as_index=False)[distance_column]
        .min()
    )

