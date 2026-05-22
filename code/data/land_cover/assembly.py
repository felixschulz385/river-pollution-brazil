import logging
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from .constants import (
    ADM2_ASSEMBLY_VARIANT,
    ADM2_ID_COLUMN,
    ASSEMBLY_VARIANTS,
    DATE_COLUMN,
    DATETIME_COLUMN,
    DEFAULT_ADM2_UPSTREAM_OUTPUT_PATH,
    DEFAULT_ASSEMBLY_LAND_COVER_PATH,
    DEFAULT_RIVER_NETWORK_PATH,
    DEFAULT_SENSOR_UPSTREAM_OUTPUT_PATH,
    DEFAULT_STATIONS_RIVERS_PATH,
    DEFAULT_WATER_QUALITY_PATH,
    DISTANCE_BUCKET_COLUMN,
    LAND_COVER_CLASS_PREFIX,
    LAND_COVER_TOTAL_COLUMN,
    LEGACY_SENSOR_UPSTREAM_DISTANCE_BUCKETS_VARIANT,
    SENSOR_ASSEMBLY_VARIANT,
    SENSOR_DISTANCE_BUCKETS,
    STATION_CODE_COLUMN,
    TRENCH_ID_COLUMN,
    UPSTREAM_DISTANCE_COLUMN,
    YEAR_COLUMN,
)
from .. import river_network as rn_module
from .schema import land_cover_assembly_columns, validate_required_columns


logger = logging.getLogger(__name__)

ASSEMBLY_VARIANT_ALIASES = {
    SENSOR_ASSEMBLY_VARIANT: SENSOR_ASSEMBLY_VARIANT,
    LEGACY_SENSOR_UPSTREAM_DISTANCE_BUCKETS_VARIANT: SENSOR_ASSEMBLY_VARIANT,
    ADM2_ASSEMBLY_VARIANT: ADM2_ASSEMBLY_VARIANT,
}


def _normalize_assembly_variant(variant):
    """Map legacy variant names to the canonical public interface."""
    normalized = ASSEMBLY_VARIANT_ALIASES.get(variant)
    if normalized is None:
        raise ValueError(
            f"Unknown land-cover assembly variant: {variant}. "
            f"Available variants: {list(ASSEMBLY_VARIANTS)}"
        )
    return normalized


def _default_output_path_for_variant(variant):
    """Return the standard output file for each assembly variant."""
    if variant == SENSOR_ASSEMBLY_VARIANT:
        return DEFAULT_SENSOR_UPSTREAM_OUTPUT_PATH
    if variant == ADM2_ASSEMBLY_VARIANT:
        return DEFAULT_ADM2_UPSTREAM_OUTPUT_PATH
    raise ValueError(f"Unknown normalized variant: {variant}")


def _derive_water_quality_years(water_quality_df):
    """Add a year column from cleaned water-quality timestamps."""
    if DATETIME_COLUMN in water_quality_df.columns:
        date_source_column = DATETIME_COLUMN
    elif DATE_COLUMN in water_quality_df.columns:
        date_source_column = DATE_COLUMN
    else:
        raise ValueError(
            "Cleaned water-quality data must include either "
            f"`{DATETIME_COLUMN}` or `{DATE_COLUMN}` to derive `{YEAR_COLUMN}`."
        )

    water_quality = water_quality_df.copy()
    water_quality[YEAR_COLUMN] = pd.to_datetime(
        water_quality[date_source_column],
        errors="coerce",
    ).dt.year
    water_quality = water_quality.dropna(subset=[YEAR_COLUMN])
    water_quality[YEAR_COLUMN] = water_quality[YEAR_COLUMN].astype(int)
    return water_quality


def _build_sensor_trench_year_targets(water_quality_df, stations_rivers_df):
    """Return unique station-trench-year rows observed in cleaned water quality."""
    validate_required_columns(
        water_quality_df,
        {STATION_CODE_COLUMN},
        "Cleaned water-quality data",
    )
    validate_required_columns(
        stations_rivers_df,
        {STATION_CODE_COLUMN, TRENCH_ID_COLUMN},
        "Stations-rivers data",
    )

    water_quality = _derive_water_quality_years(water_quality_df)
    water_quality[STATION_CODE_COLUMN] = water_quality[STATION_CODE_COLUMN].astype(str)

    stations_rivers = stations_rivers_df[
        [STATION_CODE_COLUMN, TRENCH_ID_COLUMN]
    ].dropna().copy()
    stations_rivers[STATION_CODE_COLUMN] = stations_rivers[STATION_CODE_COLUMN].astype(str)
    stations_rivers = stations_rivers.drop_duplicates(
        subset=[STATION_CODE_COLUMN, TRENCH_ID_COLUMN],
        keep="first",
    )

    targets = water_quality[[STATION_CODE_COLUMN, YEAR_COLUMN]].merge(
        stations_rivers,
        on=STATION_CODE_COLUMN,
        how="inner",
        validate="many_to_many",
    )
    targets = targets.dropna(subset=[TRENCH_ID_COLUMN])
    targets[TRENCH_ID_COLUMN] = targets[TRENCH_ID_COLUMN].astype(np.int64)
    return (
        targets[[STATION_CODE_COLUMN, TRENCH_ID_COLUMN, YEAR_COLUMN]]
        .drop_duplicates()
        .sort_values([STATION_CODE_COLUMN, YEAR_COLUMN, TRENCH_ID_COLUMN])
        .reset_index(drop=True)
    )


def _validate_river_network_for_trench_aggregation(network):
    """Validate river-network tables and matrices for upstream trench lookup."""
    if network.trenches is None:
        raise ValueError("River network must include trench data.")
    if not network.trench_reachability_matrices:
        raise ValueError("River network must have trench reachability data computed.")
    if not network.trench_distance_matrices:
        raise ValueError("River network must have trench distance data computed.")

    validate_required_columns(
        network.trenches,
        {TRENCH_ID_COLUMN, rn_module.SYSTEM_ID_KEY, rn_module.TRENCH_INDEX_COLUMN},
        "River trench data",
    )


def _build_system_trench_lookup(rivers):
    """Build per-system trench id arrays and target-position lookups."""
    system_trench_tables = {
        int(system_id): system_trenches[
            [TRENCH_ID_COLUMN, rn_module.TRENCH_INDEX_COLUMN]
        ]
        .sort_values(rn_module.TRENCH_INDEX_COLUMN)
        .reset_index(drop=True)
        for system_id, system_trenches in rivers.groupby(rn_module.SYSTEM_ID_KEY)
    }
    system_trench_id_arrays = {
        system_id: system_trenches[TRENCH_ID_COLUMN].to_numpy(dtype=np.int64)
        for system_id, system_trenches in system_trench_tables.items()
    }
    system_trench_positions = {
        system_id: dict(
            zip(
                system_trenches[TRENCH_ID_COLUMN].to_numpy(dtype=np.int64),
                system_trenches[rn_module.TRENCH_INDEX_COLUMN].to_numpy(dtype=np.int64),
            )
        )
        for system_id, system_trenches in system_trench_tables.items()
    }
    system_valid_positions = {
        system_id: set(trench_positions.values())
        for system_id, trench_positions in system_trench_positions.items()
    }
    return system_trench_id_arrays, system_trench_positions, system_valid_positions


def _build_trench_system_position_lookup(rivers):
    """Build an O(1) lookup from trench id to system id and trench index."""
    trench_rows = rivers[
        [TRENCH_ID_COLUMN, rn_module.SYSTEM_ID_KEY, rn_module.TRENCH_INDEX_COLUMN]
    ].drop_duplicates()

    duplicated = trench_rows[TRENCH_ID_COLUMN].duplicated(keep=False)
    if duplicated.any():
        duplicate_ids = trench_rows.loc[duplicated, TRENCH_ID_COLUMN].unique()[:10]
        raise ValueError(
            "Expected one river-network row per trench id. "
            f"Found duplicate trench ids, e.g. {duplicate_ids}."
        )

    return {
        int(trench_id): (
            int(system_id),
            int(trench_index),
        )
        for trench_id, system_id, trench_index in trench_rows.itertuples(
            index=False,
            name=None,
        )
    }


def _sparse_row(matrix, row_idx):
    """Return one sparse row for both csr_matrix and newer csr_array objects."""
    if hasattr(matrix, "getrow"):
        return matrix.getrow(row_idx)
    return matrix[row_idx : row_idx + 1, :]


def _resolve_upstream_trench_distances(
    trench_id,
    network,
    system_trench_id_arrays,
    system_trench_positions,
    system_valid_positions,
    trench_system_position_lookup=None,
):
    """Return upstream trench ids and distances for one target trench."""
    if trench_system_position_lookup is None:
        trench_row = network.trenches.loc[
            network.trenches[TRENCH_ID_COLUMN] == trench_id,
            [rn_module.SYSTEM_ID_KEY, rn_module.TRENCH_INDEX_COLUMN],
        ].drop_duplicates()
        if len(trench_row) == 0:
            raise KeyError(f"Unknown trench_id in river network: {trench_id}")
        if len(trench_row) > 1:
            raise ValueError(f"Expected one trench row for trench_id {trench_id}.")

        system_id = int(trench_row.iloc[0][rn_module.SYSTEM_ID_KEY])
        target_position = int(trench_row.iloc[0][rn_module.TRENCH_INDEX_COLUMN])
    else:
        try:
            system_id, target_position = trench_system_position_lookup[int(trench_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown trench_id in river network: {trench_id}") from exc

    system_trench_ids = system_trench_id_arrays.get(
        system_id,
        np.asarray([], dtype=np.int64),
    )
    if len(system_trench_ids) == 0:
        return pd.DataFrame(columns=[TRENCH_ID_COLUMN, UPSTREAM_DISTANCE_COLUMN])

    if target_position not in system_valid_positions[system_id]:
        raise ValueError(
            f"Trench index {target_position} for trench_id {trench_id} is invalid."
        )

    reach_row = _sparse_row(
        network.trench_reachability_matrices[system_id],
        target_position,
    )
    dist_row = _sparse_row(
        network.trench_distance_matrices[system_id],
        target_position,
    )

    reach_indices = reach_row.indices.astype(np.int64, copy=False)
    dist_indices = dist_row.indices.astype(np.int64, copy=False)
    dist_values = dist_row.data.astype(float, copy=False)

    if len(reach_indices) == 0:
        upstream_trench_ids = np.asarray([int(trench_id)], dtype=np.int64)
        upstream_distances = np.asarray([0.0], dtype=float)
    else:
        if len(dist_indices) == 0:
            distance_lookup = {}
        else:
            distance_lookup = dict(zip(dist_indices.tolist(), dist_values.tolist()))

        upstream_trench_ids = system_trench_ids[reach_indices].astype(np.int64, copy=False)
        upstream_distances = np.asarray(
            [float(distance_lookup.get(int(col_idx), 0.0)) for col_idx in reach_indices],
            dtype=float,
        )

        if int(trench_id) not in set(upstream_trench_ids.tolist()):
            upstream_trench_ids = np.append(upstream_trench_ids, int(trench_id))
            upstream_distances = np.append(upstream_distances, 0.0)

    upstream = pd.DataFrame(
        {
            TRENCH_ID_COLUMN: upstream_trench_ids,
            UPSTREAM_DISTANCE_COLUMN: upstream_distances,
        }
    )
    return upstream.sort_values(
        [UPSTREAM_DISTANCE_COLUMN, TRENCH_ID_COLUMN]
    ).reset_index(drop=True)


def _assign_distance_buckets(distances):
    """Assign upstream distances to the configured buckets."""
    distances = pd.Series(distances, copy=False)
    bucket_values = pd.Series(pd.NA, index=distances.index, dtype="object")
    for bucket_name, lower_bound, upper_bound in SENSOR_DISTANCE_BUCKETS:
        if lower_bound == 0:
            mask = distances.ge(lower_bound) & distances.le(upper_bound)
        elif np.isinf(upper_bound):
            mask = distances.gt(lower_bound)
        else:
            mask = distances.gt(lower_bound) & distances.le(upper_bound)
        bucket_values.loc[mask] = bucket_name
    return bucket_values


def _bucket_total_column(bucket_name):
    return f"lc_{bucket_name}_tot"


def _bucket_reachable_column(bucket_name):
    return f"lc_{bucket_name}_n"


def _land_cover_feature_stem(lc_column):
    """Return a compact output stem for a preprocessed land-cover class column."""
    if lc_column.startswith(LAND_COVER_CLASS_PREFIX):
        return f"c{lc_column.removeprefix(LAND_COVER_CLASS_PREFIX)}"
    return lc_column


def _bucket_class_column(bucket_name, lc_column, statistic):
    return f"lc_{bucket_name}_{_land_cover_feature_stem(lc_column)}_{statistic}"


def _empty_bucket_result(lc_columns):
    """Return zero/NA-filled output columns for one target row."""
    result = {}
    for bucket_name, _, _ in SENSOR_DISTANCE_BUCKETS:
        result[_bucket_total_column(bucket_name)] = 0.0
        result[_bucket_reachable_column(bucket_name)] = 0
        for lc_column in lc_columns:
            result[_bucket_class_column(bucket_name, lc_column, "cnt")] = 0.0
            result[_bucket_class_column(bucket_name, lc_column, "shr")] = np.nan
    return result


def _aggregate_bucketed_land_cover(
    upstream_distances,
    target_year,
    land_cover_by_trench_year,
    lc_columns,
    empty_template=None,
    land_cover_by_year=None,
):
    """Aggregate one target and year into distance-bucket counts and shares.

    This function is kept for the sensor path. It accepts optional precomputed
    structures to avoid repeated index work and repeated empty-dict construction.
    """
    result = (empty_template or _empty_bucket_result(lc_columns)).copy()
    if upstream_distances.empty:
        return result

    if land_cover_by_year is None:
        available_years = land_cover_by_trench_year.index.get_level_values(YEAR_COLUMN)
        year_land_cover = (
            land_cover_by_trench_year.xs(
                target_year,
                level=YEAR_COLUMN,
                drop_level=False,
            ).reset_index()
            if target_year in available_years
            else None
        )
    else:
        year_land_cover = land_cover_by_year.get(int(target_year))

    if year_land_cover is None or year_land_cover.empty:
        return result

    upstream = upstream_distances.copy()
    upstream[DISTANCE_BUCKET_COLUMN] = _assign_distance_buckets(
        upstream[UPSTREAM_DISTANCE_COLUMN]
    )
    upstream = upstream.dropna(subset=[DISTANCE_BUCKET_COLUMN])
    if upstream.empty:
        return result

    matched = upstream.merge(
        year_land_cover,
        on=TRENCH_ID_COLUMN,
        how="left",
    )
    fill_columns = [LAND_COVER_TOTAL_COLUMN, *lc_columns]
    matched[fill_columns] = matched[fill_columns].fillna(0)

    for bucket_name, _, _ in SENSOR_DISTANCE_BUCKETS:
        bucket = matched.loc[matched[DISTANCE_BUCKET_COLUMN] == bucket_name]
        if bucket.empty:
            continue

        bucket_total = float(bucket[LAND_COVER_TOTAL_COLUMN].sum())
        result[_bucket_total_column(bucket_name)] = bucket_total
        result[_bucket_reachable_column(bucket_name)] = int(
            bucket[TRENCH_ID_COLUMN].nunique()
        )
        bucket_sums = bucket[lc_columns].sum()
        for lc_column in lc_columns:
            count_value = float(bucket_sums[lc_column])
            result[_bucket_class_column(bucket_name, lc_column, "cnt")] = count_value
            if bucket_total > 0:
                result[_bucket_class_column(bucket_name, lc_column, "shr")] = (
                    count_value / bucket_total
                )
    return result


def _aggregate_bucketed_land_cover_all_years(
    upstream_distances,
    years,
    land_cover_reset,
    lc_columns,
    empty_template,
):
    """Aggregate one ADM2's upstream trenches across all years at once.

    The ADM2 upstream distance table is invariant across years, so this avoids
    repeating bucket assignment, year slicing, index resetting, and merging for
    every ADM2-year pair.
    """
    rows = []

    if upstream_distances.empty:
        for year in years:
            result = {YEAR_COLUMN: int(year)}
            result.update(empty_template.copy())
            rows.append(result)
        return rows

    upstream = upstream_distances.copy()
    upstream[DISTANCE_BUCKET_COLUMN] = _assign_distance_buckets(
        upstream[UPSTREAM_DISTANCE_COLUMN]
    )
    upstream = upstream.dropna(subset=[DISTANCE_BUCKET_COLUMN])

    if upstream.empty:
        for year in years:
            result = {YEAR_COLUMN: int(year)}
            result.update(empty_template.copy())
            rows.append(result)
        return rows

    reachable_by_bucket = (
        upstream.groupby(DISTANCE_BUCKET_COLUMN, observed=True)[TRENCH_ID_COLUMN]
        .nunique()
        .to_dict()
    )

    matched = upstream.merge(
        land_cover_reset,
        on=TRENCH_ID_COLUMN,
        how="left",
    )

    fill_columns = [LAND_COVER_TOTAL_COLUMN, *lc_columns]
    matched[fill_columns] = matched[fill_columns].fillna(0)
    matched = matched.dropna(subset=[YEAR_COLUMN])

    if matched.empty:
        grouped = None
    else:
        matched[YEAR_COLUMN] = matched[YEAR_COLUMN].astype(int)
        grouped = matched.groupby(
            [YEAR_COLUMN, DISTANCE_BUCKET_COLUMN],
            observed=True,
        )[fill_columns].sum()

    for year in years:
        result = {YEAR_COLUMN: int(year)}
        result.update(empty_template.copy())

        for bucket_name, _, _ in SENSOR_DISTANCE_BUCKETS:
            result[_bucket_reachable_column(bucket_name)] = int(
                reachable_by_bucket.get(bucket_name, 0)
            )

            if grouped is None:
                continue

            key = (int(year), bucket_name)
            if key not in grouped.index:
                continue

            bucket_sums = grouped.loc[key]
            bucket_total = float(bucket_sums[LAND_COVER_TOTAL_COLUMN])
            result[_bucket_total_column(bucket_name)] = bucket_total

            for lc_column in lc_columns:
                count_value = float(bucket_sums[lc_column])
                result[_bucket_class_column(bucket_name, lc_column, "cnt")] = count_value
                if bucket_total > 0:
                    result[_bucket_class_column(bucket_name, lc_column, "shr")] = (
                        count_value / bucket_total
                    )

        rows.append(result)

    return rows


def _load_trench_adm2_matches(network):
    """Load the canonical trench-to-ADM2 table from saved network outputs."""
    trench_adm2_table = getattr(network, "trench_adm2_table", None)
    if trench_adm2_table is not None and len(trench_adm2_table) > 0:
        validate_required_columns(
            trench_adm2_table,
            {TRENCH_ID_COLUMN, rn_module.ADM2_COLUMN},
            "River trench ADM2 matches",
        )
        matches = trench_adm2_table[[TRENCH_ID_COLUMN, rn_module.ADM2_COLUMN]].copy()
        matches = matches.dropna(subset=[rn_module.ADM2_COLUMN])
        matches[rn_module.ADM2_COLUMN] = matches[rn_module.ADM2_COLUMN].astype(str)
        return matches.drop_duplicates().reset_index(drop=True)

    rivers = network.trenches
    if rivers is None:
        raise ValueError("River network must include trench data.")

    adm2_column = getattr(rn_module, "ADM2_COLUMN", "adm2")
    if adm2_column not in rivers.columns:
        raise ValueError(
            "River network does not include saved trench-to-ADM2 matches. "
            "Run river-network generation with ADM2 matching enabled."
        )

    matches = rivers[[TRENCH_ID_COLUMN, adm2_column]].dropna().copy()
    matches[adm2_column] = matches[adm2_column].astype(str)
    return matches.drop_duplicates().reset_index(drop=True)


def _build_adm2_targets(network):
    """Return one row per ADM2 seed trench using the persisted match table."""
    trench_adm2_matches = _load_trench_adm2_matches(network)
    if trench_adm2_matches.empty:
        return pd.DataFrame(columns=[ADM2_ID_COLUMN, TRENCH_ID_COLUMN])

    targets = trench_adm2_matches.rename(columns={rn_module.ADM2_COLUMN: ADM2_ID_COLUMN})
    targets = targets[[ADM2_ID_COLUMN, TRENCH_ID_COLUMN]].drop_duplicates()
    targets[ADM2_ID_COLUMN] = targets[ADM2_ID_COLUMN].astype(str)
    targets[TRENCH_ID_COLUMN] = targets[TRENCH_ID_COLUMN].astype(np.int64)
    return targets.sort_values([ADM2_ID_COLUMN, TRENCH_ID_COLUMN]).reset_index(drop=True)


def _resolve_adm2_upstream_distances(
    adm2_target_trenches,
    upstream_distance_cache,
):
    """Collapse multiple seed-trench lookups to one min-distance table per ADM2."""
    distance_frames = []
    for trench_id in adm2_target_trenches:
        upstream = upstream_distance_cache.get(int(trench_id))
        if upstream is None or upstream.empty:
            continue
        distance_frames.append(upstream[[TRENCH_ID_COLUMN, UPSTREAM_DISTANCE_COLUMN]])

    if not distance_frames:
        return pd.DataFrame(columns=[TRENCH_ID_COLUMN, UPSTREAM_DISTANCE_COLUMN])

    combined = pd.concat(distance_frames, ignore_index=True)
    return (
        combined.groupby(TRENCH_ID_COLUMN, as_index=False, sort=False)[
            UPSTREAM_DISTANCE_COLUMN
        ]
        .min()
        .sort_values([UPSTREAM_DISTANCE_COLUMN, TRENCH_ID_COLUMN])
        .reset_index(drop=True)
    )


def _build_land_cover_grouped(land_cover_path):
    """Load land cover and return grouped trench-year data plus class columns."""
    logger.info("Loading land-cover data from %s", land_cover_path)
    land_cover_df = pd.read_feather(land_cover_path)
    lc_columns = land_cover_assembly_columns(land_cover_df)
    land_cover_class_columns = [
        column for column in lc_columns if column != LAND_COVER_TOTAL_COLUMN
    ]
    land_cover_by_trench_year = land_cover_df.groupby(
        [TRENCH_ID_COLUMN, YEAR_COLUMN],
    )[lc_columns].sum().sort_index()
    return land_cover_by_trench_year, land_cover_class_columns


def _load_network_with_lookup(river_network_path):
    """Load the saved river network and derive shared lookup structures."""
    logger.info("Loading river network from %s", river_network_path)
    network = rn_module.RiverNetwork()
    network.load(str(river_network_path))
    _validate_river_network_for_trench_aggregation(network)
    lookup = _build_system_trench_lookup(network.trenches)
    trench_system_position_lookup = _build_trench_system_position_lookup(network.trenches)
    return network, lookup, trench_system_position_lookup


def _resolve_upstream_distance_cache(
    target_trench_ids,
    network,
    lookup,
    trench_system_position_lookup,
    n_jobs,
    progress_label,
):
    """Resolve upstream distance tables for a set of target trenches."""
    (
        system_trench_id_arrays,
        system_trench_positions,
        system_valid_positions,
    ) = lookup

    logger.info(
        "Resolving upstream distances for %d target trench(es) with %s thread(s).",
        len(target_trench_ids),
        n_jobs,
    )

    def resolve_target_trench(trench_id):
        return (
            int(trench_id),
            _resolve_upstream_trench_distances(
                int(trench_id),
                network,
                system_trench_id_arrays,
                system_trench_positions,
                system_valid_positions,
                trench_system_position_lookup=trench_system_position_lookup,
            ),
        )

    if n_jobs == 1:
        upstream_distance_items = [
            resolve_target_trench(trench_id)
            for trench_id in tqdm(target_trench_ids, desc=progress_label)
        ]
    else:
        upstream_distance_items = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(resolve_target_trench)(trench_id)
            for trench_id in tqdm(target_trench_ids, desc=progress_label)
        )
    return dict(upstream_distance_items)


def _assemble_sensor_land_cover(
    land_cover_by_trench_year,
    land_cover_class_columns,
    network,
    lookup,
    trench_system_position_lookup,
    water_quality_path,
    stations_rivers_path,
    n_jobs,
):
    """Assemble the sensor-matched land-cover dataset."""
    logger.info("Loading cleaned water-quality data from %s", water_quality_path)
    water_quality_df = pd.read_parquet(water_quality_path)
    logger.info("Loading station-river matches from %s", stations_rivers_path)
    stations_rivers_df = pd.read_parquet(stations_rivers_path)
    targets = _build_sensor_trench_year_targets(water_quality_df, stations_rivers_df)
    logger.info(
        "Found %d observed trench-year target(s) for sensor assembly.",
        len(targets),
    )

    target_trench_ids = targets[TRENCH_ID_COLUMN].drop_duplicates().astype(np.int64).tolist()
    upstream_distance_cache = _resolve_upstream_distance_cache(
        target_trench_ids,
        network,
        lookup,
        trench_system_position_lookup,
        n_jobs,
        progress_label="Sensor upstream trenches",
    )

    logger.info(
        "Aggregating %d sensor trench-year target(s) with %s thread(s).",
        len(targets),
        n_jobs,
    )

    land_cover_by_year = {
        int(year): frame.reset_index()
        for year, frame in land_cover_by_trench_year.groupby(level=YEAR_COLUMN, sort=False)
    }
    empty_template = _empty_bucket_result(land_cover_class_columns)

    def aggregate_target(target):
        station_code = str(getattr(target, STATION_CODE_COLUMN))
        trench_id = int(getattr(target, TRENCH_ID_COLUMN))
        year = int(getattr(target, YEAR_COLUMN))
        result = {
            STATION_CODE_COLUMN: station_code,
            TRENCH_ID_COLUMN: trench_id,
            YEAR_COLUMN: year,
        }
        result.update(
            _aggregate_bucketed_land_cover(
                upstream_distance_cache[trench_id],
                year,
                land_cover_by_trench_year,
                land_cover_class_columns,
                empty_template=empty_template,
                land_cover_by_year=land_cover_by_year,
            )
        )
        return result

    target_records = list(targets.itertuples(index=False))
    if n_jobs == 1:
        records = [
            aggregate_target(target)
            for target in tqdm(target_records, desc="Sensor trench-years")
        ]
    else:
        records = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(aggregate_target)(target)
            for target in tqdm(target_records, desc="Sensor trench-years")
        )

    if records:
        return pd.DataFrame(records).sort_values(
            [STATION_CODE_COLUMN, YEAR_COLUMN, TRENCH_ID_COLUMN]
        )
    return pd.DataFrame(
        columns=[
            STATION_CODE_COLUMN,
            TRENCH_ID_COLUMN,
            YEAR_COLUMN,
            *empty_template.keys(),
        ]
    )


def _assemble_adm2_land_cover(
    land_cover_by_trench_year,
    land_cover_class_columns,
    network,
    lookup,
    trench_system_position_lookup,
    n_jobs,
):
    """Assemble the ADM2 upstream land-cover dataset."""
    adm2_targets = _build_adm2_targets(network)
    logger.info(
        "Found %d trench-to-ADM2 match row(s) for ADM2 assembly.",
        len(adm2_targets),
    )

    empty_template = _empty_bucket_result(land_cover_class_columns)

    if adm2_targets.empty:
        return pd.DataFrame(
            columns=[
                ADM2_ID_COLUMN,
                YEAR_COLUMN,
                *empty_template.keys(),
            ]
        )

    target_trench_ids = (
        adm2_targets[TRENCH_ID_COLUMN].drop_duplicates().astype(np.int64).tolist()
    )
    upstream_distance_cache = _resolve_upstream_distance_cache(
        target_trench_ids,
        network,
        lookup,
        trench_system_position_lookup,
        n_jobs,
        progress_label="ADM2 seed trenches",
    )

    years = (
        land_cover_by_trench_year.index.get_level_values(YEAR_COLUMN)
        .unique()
        .astype(int)
        .tolist()
    )
    adm2_seed_lookup = {
        adm2_id: group[TRENCH_ID_COLUMN].astype(np.int64).tolist()
        for adm2_id, group in adm2_targets.groupby(ADM2_ID_COLUMN, sort=True)
    }
    adm2_ids = list(adm2_seed_lookup)

    land_cover_reset = land_cover_by_trench_year.reset_index()

    logger.info(
        "Aggregating %d ADM2 unit(s) across %d year(s) with %s thread(s).",
        len(adm2_ids),
        len(years),
        n_jobs,
    )

    def aggregate_adm2(adm2_id):
        upstream_distances = _resolve_adm2_upstream_distances(
            adm2_seed_lookup[adm2_id],
            upstream_distance_cache,
        )
        rows = _aggregate_bucketed_land_cover_all_years(
            upstream_distances=upstream_distances,
            years=years,
            land_cover_reset=land_cover_reset,
            lc_columns=land_cover_class_columns,
            empty_template=empty_template,
        )
        for row in rows:
            row[ADM2_ID_COLUMN] = adm2_id
        return rows

    if n_jobs == 1:
        nested_records = [
            aggregate_adm2(adm2_id) for adm2_id in tqdm(adm2_ids, desc="ADM2 units")
        ]
    else:
        nested_records = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(aggregate_adm2)(adm2_id)
            for adm2_id in tqdm(adm2_ids, desc="ADM2 units")
        )

    records = [row for rows in nested_records for row in rows]
    return pd.DataFrame(records).sort_values([ADM2_ID_COLUMN, YEAR_COLUMN])


def assemble_land_cover(
    self,
    variant=SENSOR_ASSEMBLY_VARIANT,
    land_cover_path=DEFAULT_ASSEMBLY_LAND_COVER_PATH,
    water_quality_path=DEFAULT_WATER_QUALITY_PATH,
    stations_rivers_path=DEFAULT_STATIONS_RIVERS_PATH,
    river_network_path=DEFAULT_RIVER_NETWORK_PATH,
    output_path=None,
    n_jobs=None,
):
    """Assemble analysis-ready land-cover datasets."""
    normalized_variant = _normalize_assembly_variant(variant)
    if n_jobs is None:
        n_jobs = cpu_count()
    if output_path is None:
        output_path = _default_output_path_for_variant(normalized_variant)

    land_cover_by_trench_year, land_cover_class_columns = _build_land_cover_grouped(
        land_cover_path
    )
    network, lookup, trench_system_position_lookup = _load_network_with_lookup(
        river_network_path
    )

    if normalized_variant == SENSOR_ASSEMBLY_VARIANT:
        result_df = _assemble_sensor_land_cover(
            land_cover_by_trench_year,
            land_cover_class_columns,
            network,
            lookup,
            trench_system_position_lookup,
            water_quality_path,
            stations_rivers_path,
            n_jobs,
        )
        index_columns = [STATION_CODE_COLUMN, YEAR_COLUMN]
    elif normalized_variant == ADM2_ASSEMBLY_VARIANT:
        result_df = _assemble_adm2_land_cover(
            land_cover_by_trench_year,
            land_cover_class_columns,
            network,
            lookup,
            trench_system_position_lookup,
            n_jobs,
        )
        index_columns = [ADM2_ID_COLUMN, YEAR_COLUMN]
    else:
        raise AssertionError(f"Unhandled normalized variant: {normalized_variant}")

    result_df = result_df.set_index(index_columns)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.reset_index().to_parquet(output_path, index=False)
    logger.info(
        "Saved %s land-cover assembly output to %s",
        normalized_variant,
        output_path,
    )
    logger.info("Output shape: %s", result_df.shape)
    return result_df
