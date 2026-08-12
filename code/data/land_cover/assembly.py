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
from .schema import land_cover_assembly_columns
from shared.sensor_upstream import (
    build_group_index_lookup,
    build_location_period_targets,
    label_values_by_intervals,
    resolve_reachable_distances,
    validate_network_index_tables,
)


logger = logging.getLogger(__name__)


def _assign_sensor_distance_buckets(distances):
    """Assign upstream distances to the configured sensor buckets."""
    return label_values_by_intervals(distances, SENSOR_DISTANCE_BUCKETS)


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
    targets = build_location_period_targets(
        water_quality_df,
        stations_rivers_df,
        entity_column=STATION_CODE_COLUMN,
        date_column=DATE_COLUMN,
        timestamp_column=DATETIME_COLUMN,
        location_column=TRENCH_ID_COLUMN,
        period_value_column=YEAR_COLUMN,
    )
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
    validate_network_index_tables(
        network,
        location_column=TRENCH_ID_COLUMN,
        system_column=rn_module.SYSTEM_ID_KEY,
        position_column=rn_module.TRENCH_INDEX_COLUMN,
    )
    system_trench_id_arrays, system_trench_positions = build_group_index_lookup(
        network.trenches,
        location_column=TRENCH_ID_COLUMN,
        system_column=rn_module.SYSTEM_ID_KEY,
        position_column=rn_module.TRENCH_INDEX_COLUMN,
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
            resolve_reachable_distances(
                int(trench_id),
                network,
                system_trench_id_arrays,
                system_trench_positions,
                location_column=TRENCH_ID_COLUMN,
                distance_column=UPSTREAM_DISTANCE_COLUMN,
                system_column=rn_module.SYSTEM_ID_KEY,
                position_column=rn_module.TRENCH_INDEX_COLUMN,
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
