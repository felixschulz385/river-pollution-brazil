import logging
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from .constants import (
    DATE_COLUMN,
    DATETIME_COLUMN,
    DEFAULT_RIVER_NETWORK_PATH,
    DEFAULT_SENSOR_LAND_COVER_PATH,
    DEFAULT_SENSOR_UPSTREAM_OUTPUT_PATH,
    DEFAULT_STATIONS_RIVERS_PATH,
    DEFAULT_WATER_QUALITY_PATH,
    DISTANCE_BUCKET_COLUMN,
    LAND_COVER_CLASS_PREFIX,
    LAND_COVER_TOTAL_COLUMN,
    SENSOR_DISTANCE_BUCKETS,
    SENSOR_UPSTREAM_DISTANCE_BUCKETS_VARIANT,
    STATION_CODE_COLUMN,
    TRENCH_ID_COLUMN,
    UPSTREAM_DISTANCE_COLUMN,
    YEAR_COLUMN,
)
from .river_network_import import rn_module
from .schema import land_cover_assembly_columns, validate_required_columns


logger = logging.getLogger(__name__)


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
    """Return unique trench-year pairs observed in cleaned water quality."""
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
        targets[[TRENCH_ID_COLUMN, YEAR_COLUMN]]
        .drop_duplicates()
        .sort_values([TRENCH_ID_COLUMN, YEAR_COLUMN])
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
    return system_trench_id_arrays, system_trench_positions


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
):
    """Return upstream trench ids and distances for one target trench."""
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
    system_trench_ids = system_trench_id_arrays.get(
        system_id,
        np.asarray([], dtype=np.int64),
    )
    if len(system_trench_ids) == 0:
        return pd.DataFrame(columns=[TRENCH_ID_COLUMN, UPSTREAM_DISTANCE_COLUMN])

    if target_position not in set(system_trench_positions[system_id].values()):
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
    distance_lookup = dict(zip(dist_row.indices.tolist(), dist_row.data.tolist()))

    upstream_records = [
        {
            TRENCH_ID_COLUMN: int(system_trench_ids[col_idx]),
            UPSTREAM_DISTANCE_COLUMN: float(distance_lookup.get(col_idx, 0.0)),
        }
        for col_idx in reach_row.indices.tolist()
    ]
    if trench_id not in [record[TRENCH_ID_COLUMN] for record in upstream_records]:
        upstream_records.append(
            {
                TRENCH_ID_COLUMN: int(trench_id),
                UPSTREAM_DISTANCE_COLUMN: 0.0,
            }
        )

    return pd.DataFrame(upstream_records).sort_values(
        [UPSTREAM_DISTANCE_COLUMN, TRENCH_ID_COLUMN]
    ).reset_index(drop=True)


def _assign_sensor_distance_buckets(distances):
    """Assign upstream distances to the configured sensor buckets."""
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


def _sensor_bucket_total_column(bucket_name):
    return f"lc_{bucket_name}_tot"


def _sensor_bucket_reachable_column(bucket_name):
    return f"lc_{bucket_name}_n"


def _land_cover_feature_stem(lc_column):
    """Return a compact output stem for a preprocessed land-cover class column."""
    if lc_column.startswith(LAND_COVER_CLASS_PREFIX):
        return f"c{lc_column.removeprefix(LAND_COVER_CLASS_PREFIX)}"
    return lc_column


def _sensor_bucket_class_column(bucket_name, lc_column, statistic):
    return f"lc_{bucket_name}_{_land_cover_feature_stem(lc_column)}_{statistic}"


def _empty_sensor_bucket_result(lc_columns):
    """Return zero/NA-filled output columns for one target row."""
    result = {}
    for bucket_name, _, _ in SENSOR_DISTANCE_BUCKETS:
        result[_sensor_bucket_total_column(bucket_name)] = 0.0
        result[_sensor_bucket_reachable_column(bucket_name)] = 0
        for lc_column in lc_columns:
            result[_sensor_bucket_class_column(bucket_name, lc_column, "cnt")] = 0.0
            result[_sensor_bucket_class_column(bucket_name, lc_column, "shr")] = np.nan
    return result


def _aggregate_sensor_trench_year(
    upstream_distances,
    target_year,
    land_cover_by_trench_year,
    lc_columns,
):
    """Aggregate one target trench-year into distance-bucket counts and shares."""
    result = _empty_sensor_bucket_result(lc_columns)
    if upstream_distances.empty:
        return result

    available_years = land_cover_by_trench_year.index.get_level_values(YEAR_COLUMN)
    year_land_cover = (
        land_cover_by_trench_year.xs(
            target_year,
            level=YEAR_COLUMN,
            drop_level=False,
        )
        if target_year in available_years
        else None
    )
    if year_land_cover is None or year_land_cover.empty:
        return result

    upstream = upstream_distances.copy()
    upstream[DISTANCE_BUCKET_COLUMN] = _assign_sensor_distance_buckets(
        upstream[UPSTREAM_DISTANCE_COLUMN]
    )
    upstream = upstream.dropna(subset=[DISTANCE_BUCKET_COLUMN])
    if upstream.empty:
        return result

    matched = upstream.merge(
        year_land_cover.reset_index(),
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
        result[_sensor_bucket_total_column(bucket_name)] = bucket_total
        result[_sensor_bucket_reachable_column(bucket_name)] = int(
            bucket[TRENCH_ID_COLUMN].nunique()
        )
        bucket_sums = bucket[lc_columns].sum()
        for lc_column in lc_columns:
            count_value = float(bucket_sums[lc_column])
            result[_sensor_bucket_class_column(bucket_name, lc_column, "cnt")] = count_value
            if bucket_total > 0:
                result[_sensor_bucket_class_column(bucket_name, lc_column, "shr")] = (
                    count_value / bucket_total
                )
    return result


def assemble_land_cover(
    self,
    variant=SENSOR_UPSTREAM_DISTANCE_BUCKETS_VARIANT,
    land_cover_path=DEFAULT_SENSOR_LAND_COVER_PATH,
    water_quality_path=DEFAULT_WATER_QUALITY_PATH,
    stations_rivers_path=DEFAULT_STATIONS_RIVERS_PATH,
    river_network_path=DEFAULT_RIVER_NETWORK_PATH,
    output_path=DEFAULT_SENSOR_UPSTREAM_OUTPUT_PATH,
    n_jobs=None,
):
    """Assemble analysis-ready land-cover datasets."""
    if variant != SENSOR_UPSTREAM_DISTANCE_BUCKETS_VARIANT:
        raise ValueError(
            f"Unknown land-cover assembly variant: {variant}. "
            f"Available variants: {[SENSOR_UPSTREAM_DISTANCE_BUCKETS_VARIANT]}"
        )
    if n_jobs is None:
        n_jobs = cpu_count()

    logger.info("Loading cleaned water-quality data from %s", water_quality_path)
    water_quality_df = pd.read_parquet(water_quality_path)
    logger.info("Loading station-river matches from %s", stations_rivers_path)
    stations_rivers_df = pd.read_parquet(stations_rivers_path)
    targets = _build_sensor_trench_year_targets(water_quality_df, stations_rivers_df)
    logger.info(
        "Found %d observed trench-year target(s) for sensor-matched assembly.",
        len(targets),
    )

    logger.info("Loading land-cover data from %s", land_cover_path)
    land_cover_df = pd.read_feather(land_cover_path)
    lc_columns = land_cover_assembly_columns(land_cover_df)
    land_cover_class_columns = [
        column for column in lc_columns if column != LAND_COVER_TOTAL_COLUMN
    ]
    land_cover_by_trench_year = land_cover_df.groupby(
        [TRENCH_ID_COLUMN, YEAR_COLUMN],
    )[lc_columns].sum().sort_index()

    logger.info("Loading river network from %s", river_network_path)
    network = rn_module.RiverNetwork()
    network.load(str(river_network_path))
    _validate_river_network_for_trench_aggregation(network)
    system_trench_id_arrays, system_trench_positions = _build_system_trench_lookup(
        network.trenches,
    )

    target_trench_ids = targets[TRENCH_ID_COLUMN].drop_duplicates().astype(np.int64).tolist()
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
            ),
        )

    if n_jobs == 1:
        upstream_distance_items = [
            resolve_target_trench(trench_id)
            for trench_id in tqdm(target_trench_ids, desc="Upstream trenches")
        ]
    else:
        upstream_distance_items = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(resolve_target_trench)(trench_id)
            for trench_id in tqdm(target_trench_ids, desc="Upstream trenches")
        )
    upstream_distance_cache = dict(upstream_distance_items)

    logger.info(
        "Aggregating %d sensor trench-year target(s) with %s thread(s).",
        len(targets),
        n_jobs,
    )

    def aggregate_target(target):
        trench_id = int(getattr(target, TRENCH_ID_COLUMN))
        year = int(getattr(target, YEAR_COLUMN))
        result = {TRENCH_ID_COLUMN: trench_id, YEAR_COLUMN: year}
        result.update(
            _aggregate_sensor_trench_year(
                upstream_distance_cache[trench_id],
                year,
                land_cover_by_trench_year,
                land_cover_class_columns,
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
        result_df = pd.DataFrame(records).sort_values([TRENCH_ID_COLUMN, YEAR_COLUMN])
    else:
        result_df = pd.DataFrame(
            columns=[
                TRENCH_ID_COLUMN,
                YEAR_COLUMN,
                *_empty_sensor_bucket_result(land_cover_class_columns).keys(),
            ]
        )

    result_df = result_df.set_index([TRENCH_ID_COLUMN, YEAR_COLUMN])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.reset_index().to_parquet(output_path, index=False)
    logger.info("Saved sensor-matched upstream land cover to %s", output_path)
    logger.info("Output shape: %s", result_df.shape)
    return result_df
