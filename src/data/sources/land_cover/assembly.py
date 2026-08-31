import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from src.data.shared.slurm import resolve_n_jobs

from .constants import (
    ADM2_ASSEMBLY_VARIANT,
    ADJUSTED_DISTANCE_COLUMN,
    ASSEMBLY_VARIANTS,
    BUCKET_COUNT_COLUMN,
    BUCKET_REACHABLE_COUNT_COLUMN,
    BUCKET_SHARE_COLUMN,
    DATE_COLUMN,
    DATETIME_COLUMN,
    DEFAULT_ADM2_UPSTREAM_OUTPUT_PATH,
    DEFAULT_ASSEMBLY_LAND_COVER_PATH,
    DEFAULT_RIVER_NETWORK_PATH,
    DEFAULT_SENSOR_UPSTREAM_OUTPUT_PATH,
    DEFAULT_STATIONS_TRENCHES_PATH,
    DEFAULT_WATER_QUALITY_PATH,
    DISTANCE_BUCKET_COLUMN,
    LAND_COVER_CLASS_COLUMN,
    LAND_COVER_TOTAL_COLUMN,
    LEGACY_SENSOR_UPSTREAM_DISTANCE_BUCKETS_VARIANT,
    SENSOR_ASSEMBLY_VARIANT,
    SENSOR_DISTANCE_BUCKETS,
    STATION_CODE_COLUMN,
    TRENCH_ID_COLUMN,
    UPSTREAM_DISTANCE_COLUMN,
    YEAR_COLUMN,
)
from src.data.sources import river_network as rn_module
from .schema import (
    build_trench_length_lookup,
    land_cover_assembly_columns,
    land_cover_feature_stem,
    validate_required_columns,
)
from src.data.shared.sensor_upstream import (
    assign_distance_buckets,
    bucket_label,
    build_system_trench_lookup,
    build_trench_system_position_lookup,
    combine_station_upstream_distances,
    normalize_network_frame,
    resolve_upstream_trench_distances,
    shift_upstream_distances,
    validate_network_index_tables,
)


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
    """Return the standard output path for one assembly variant."""
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


def _build_sensor_targets(water_quality_df, stations_rivers_df):
    """Return unique station-year targets and a deduplicated station-trench map."""
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

    station_trenches = stations_rivers_df[
        [STATION_CODE_COLUMN, TRENCH_ID_COLUMN]
    ].dropna().copy()
    station_trenches[STATION_CODE_COLUMN] = station_trenches[
        STATION_CODE_COLUMN
    ].astype(str)
    station_trenches[TRENCH_ID_COLUMN] = station_trenches[TRENCH_ID_COLUMN].astype(
        np.int64
    )
    station_trenches = station_trenches.drop_duplicates(
        subset=[STATION_CODE_COLUMN, TRENCH_ID_COLUMN],
        keep="first",
    )

    station_year_targets = (
        water_quality[[STATION_CODE_COLUMN, YEAR_COLUMN]]
        .drop_duplicates()
        .sort_values([STATION_CODE_COLUMN, YEAR_COLUMN])
        .reset_index(drop=True)
    )
    return station_year_targets, station_trenches


def _load_network(river_network_path):
    """Load the river network and normalize core tables."""
    logger.info("Loading river network from %s", river_network_path)
    network = rn_module.RiverNetwork()
    network.load(str(river_network_path))
    validate_network_index_tables(
        network,
        location_column=TRENCH_ID_COLUMN,
        system_column=rn_module.SYSTEM_ID_KEY,
        position_column=rn_module.TRENCH_INDEX_COLUMN,
    )
    validate_required_columns(network.trenches, {"distance"}, "River trench data")
    network.trenches = normalize_network_frame(network.trenches)
    if network.drainage_areas is not None:
        network.drainage_areas = normalize_network_frame(network.drainage_areas)
    if getattr(network, "trench_adm2_table", None) is not None:
        network.trench_adm2_table = normalize_network_frame(network.trench_adm2_table)
    return network


def _build_system_trench_lookup(rivers):
    return build_system_trench_lookup(
        rivers, rn_module=rn_module, trench_id_column=TRENCH_ID_COLUMN
    )


def _build_trench_system_position_lookup(rivers):
    return build_trench_system_position_lookup(
        rivers, rn_module=rn_module, trench_id_column=TRENCH_ID_COLUMN
    )


def _resolve_upstream_trench_distances(
    trench_id,
    network,
    system_trench_id_arrays,
    system_valid_positions,
    trench_system_position_lookup,
):
    return resolve_upstream_trench_distances(
        trench_id,
        network,
        system_trench_id_arrays,
        system_valid_positions,
        trench_system_position_lookup,
        trench_id_column=TRENCH_ID_COLUMN,
        distance_column=UPSTREAM_DISTANCE_COLUMN,
    )


def _shift_upstream_distances(upstream_distances, trench_lengths):
    return shift_upstream_distances(
        upstream_distances,
        trench_lengths,
        trench_id_column=TRENCH_ID_COLUMN,
        distance_column=UPSTREAM_DISTANCE_COLUMN,
        adjusted_distance_column=ADJUSTED_DISTANCE_COLUMN,
    )


def _combine_station_upstream_distances(station_trench_ids, upstream_distance_cache):
    return combine_station_upstream_distances(
        station_trench_ids,
        upstream_distance_cache,
        trench_id_column=TRENCH_ID_COLUMN,
        distance_column=UPSTREAM_DISTANCE_COLUMN,
        adjusted_distance_column=ADJUSTED_DISTANCE_COLUMN,
    )


def _bucket_label(lower_bound_km):
    return bucket_label(lower_bound_km)


def _assign_sensor_distance_buckets(distances):
    return assign_distance_buckets(distances, SENSOR_DISTANCE_BUCKETS)


def _empty_sensor_bucket_rows(target_key, lc_columns):
    """Return zero/NA-filled long-format bucket rows for one target row."""
    station_code, year = target_key
    rows = []
    for lower_bound, _ in SENSOR_DISTANCE_BUCKETS:
        bucket_value = _bucket_label(lower_bound)
        for lc_column in lc_columns:
            rows.append(
                {
                    STATION_CODE_COLUMN: station_code,
                    YEAR_COLUMN: year,
                    DISTANCE_BUCKET_COLUMN: bucket_value,
                    LAND_COVER_CLASS_COLUMN: land_cover_feature_stem(lc_column),
                    BUCKET_REACHABLE_COUNT_COLUMN: 0,
                    BUCKET_COUNT_COLUMN: 0.0,
                    BUCKET_SHARE_COLUMN: np.nan,
                }
            )
    return rows


def _aggregate_sensor_station_year(
    target_key,
    upstream_distances,
    year_land_cover,
    lc_columns,
):
    """Aggregate one station-year into long-format distance-bucket rows."""
    empty_rows = _empty_sensor_bucket_rows(target_key, lc_columns)
    if upstream_distances.empty or year_land_cover is None or year_land_cover.empty:
        return empty_rows

    upstream = upstream_distances.copy()
    upstream[DISTANCE_BUCKET_COLUMN] = _assign_sensor_distance_buckets(
        upstream[ADJUSTED_DISTANCE_COLUMN]
    )
    upstream = upstream.dropna(subset=[DISTANCE_BUCKET_COLUMN])
    if upstream.empty:
        return empty_rows

    matched = upstream.merge(
        year_land_cover,
        on=TRENCH_ID_COLUMN,
        how="left",
    )
    matched[lc_columns] = matched[lc_columns].fillna(0.0)

    bucket_summaries = {}
    for lower_bound, _ in SENSOR_DISTANCE_BUCKETS:
        bucket_value = _bucket_label(lower_bound)
        bucket = matched.loc[matched[DISTANCE_BUCKET_COLUMN] == bucket_value]
        if bucket.empty:
            bucket_summaries[bucket_value] = {
                BUCKET_REACHABLE_COUNT_COLUMN: 0,
                "total": 0.0,
                "counts": {},
            }
            continue

        bucket_total = float(bucket[LAND_COVER_TOTAL_COLUMN].sum())
        bucket_counts = {
            land_cover_feature_stem(lc_column): float(bucket[lc_column].sum())
            for lc_column in lc_columns
        }
        bucket_summaries[bucket_value] = {
            BUCKET_REACHABLE_COUNT_COLUMN: int(bucket[TRENCH_ID_COLUMN].nunique()),
            "total": bucket_total,
            "counts": bucket_counts,
        }

    station_code, year = target_key
    rows = []
    for lower_bound, _ in SENSOR_DISTANCE_BUCKETS:
        bucket_value = _bucket_label(lower_bound)
        summary = bucket_summaries[bucket_value]
        bucket_total = summary["total"]
        for lc_column in lc_columns:
            class_name = land_cover_feature_stem(lc_column)
            count_value = float(summary["counts"].get(class_name, 0.0))
            rows.append(
                {
                    STATION_CODE_COLUMN: station_code,
                    YEAR_COLUMN: year,
                    DISTANCE_BUCKET_COLUMN: bucket_value,
                    LAND_COVER_CLASS_COLUMN: class_name,
                    BUCKET_REACHABLE_COUNT_COLUMN: summary[
                        BUCKET_REACHABLE_COUNT_COLUMN
                    ],
                    BUCKET_COUNT_COLUMN: count_value,
                    BUCKET_SHARE_COLUMN: (
                        count_value / bucket_total if bucket_total > 0 else np.nan
                    ),
                }
            )
    return rows


def _reset_water_quality_index(water_quality_df):
    """Bring `station_code`/`date` back as columns if sensor_data's assembled
    output indexed on them (mirrors climate's prepare_observation_targets in
    sensor_upstream.py). A no-op for a plain-column frame."""
    index_names = [name for name in water_quality_df.index.names if name is not None]
    missing_index_columns = [
        column
        for column in (STATION_CODE_COLUMN, DATE_COLUMN)
        if column in index_names and column not in water_quality_df.columns
    ]
    if missing_index_columns:
        return water_quality_df.reset_index(level=missing_index_columns)
    return water_quality_df


def _assemble_sensor_land_cover(
    land_cover_path,
    water_quality_path,
    stations_rivers_path,
    river_network_path,
    output_path,
    n_jobs,
):
    """Assemble long-format sensor upstream land-cover buckets."""
    logger.info("Loading cleaned water-quality data from %s", water_quality_path)
    water_quality_df = _reset_water_quality_index(pd.read_parquet(water_quality_path))
    logger.info("Loading station-river matches from %s", stations_rivers_path)
    stations_rivers_df = pd.read_parquet(stations_rivers_path)
    targets, station_trenches = _build_sensor_targets(
        water_quality_df,
        stations_rivers_df,
    )
    logger.info(
        "Found %d observed station-year target(s) for sensor-matched assembly.",
        len(targets),
    )

    logger.info("Loading land-cover data from %s", land_cover_path)
    land_cover_path = Path(land_cover_path)
    land_cover_df = (
        pd.read_feather(land_cover_path)
        if land_cover_path.suffix == ".feather"
        else pd.read_parquet(land_cover_path)
    )
    lc_columns = land_cover_assembly_columns(land_cover_df)
    land_cover_by_year = {
        int(year): year_frame.drop(columns=[YEAR_COLUMN]).reset_index(drop=True)
        for year, year_frame in land_cover_df.groupby(YEAR_COLUMN, sort=False)
    }

    network = _load_network(river_network_path)
    rivers = network.trenches
    (
        system_trench_id_arrays,
        system_trench_positions,
        system_valid_positions,
    ) = _build_system_trench_lookup(rivers)
    trench_system_position_lookup = _build_trench_system_position_lookup(rivers)
    trench_lengths = build_trench_length_lookup(rivers)

    target_trench_ids = (
        station_trenches[TRENCH_ID_COLUMN].drop_duplicates().astype(np.int64).tolist()
    )
    logger.info(
        "Resolving shifted upstream distances for %d target trench(es) with %s thread(s).",
        len(target_trench_ids),
        n_jobs,
    )

    def resolve_target_trench(trench_id):
        upstream_distances = _resolve_upstream_trench_distances(
            int(trench_id),
            network,
            system_trench_id_arrays,
            system_valid_positions,
            trench_system_position_lookup,
        )
        return (
            int(trench_id),
            _shift_upstream_distances(upstream_distances, trench_lengths),
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

    station_upstream_distance_cache = {
        str(station_code): _combine_station_upstream_distances(
            station_rows[TRENCH_ID_COLUMN].astype(np.int64).tolist(),
            upstream_distance_cache,
        )
        for station_code, station_rows in station_trenches.groupby(STATION_CODE_COLUMN)
    }

    logger.info(
        "Aggregating %d station-year target(s) into long-format 25 km buckets with %s thread(s).",
        len(targets),
        n_jobs,
    )

    def aggregate_target(target):
        station_code = str(getattr(target, STATION_CODE_COLUMN))
        year = int(getattr(target, YEAR_COLUMN))
        return _aggregate_sensor_station_year(
            (station_code, year),
            station_upstream_distance_cache.get(
                station_code,
                pd.DataFrame(
                    columns=[
                        TRENCH_ID_COLUMN,
                        UPSTREAM_DISTANCE_COLUMN,
                        "trench_length_km",
                        ADJUSTED_DISTANCE_COLUMN,
                    ]
                ),
            ),
            land_cover_by_year.get(year),
            lc_columns,
        )

    target_records = list(targets.itertuples(index=False))
    if n_jobs == 1:
        nested_records = [
            aggregate_target(target)
            for target in tqdm(target_records, desc="Sensor station-years")
        ]
    else:
        nested_records = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(aggregate_target)(target)
            for target in tqdm(target_records, desc="Sensor station-years")
        )

    records = [record for target_rows in nested_records for record in target_rows]
    result_columns = [
        STATION_CODE_COLUMN,
        YEAR_COLUMN,
        DISTANCE_BUCKET_COLUMN,
        LAND_COVER_CLASS_COLUMN,
        BUCKET_REACHABLE_COUNT_COLUMN,
        BUCKET_COUNT_COLUMN,
        BUCKET_SHARE_COLUMN,
    ]
    result_df = pd.DataFrame.from_records(records, columns=result_columns)
    if not result_df.empty:
        result_df = result_df.sort_values(
            [
                STATION_CODE_COLUMN,
                YEAR_COLUMN,
                DISTANCE_BUCKET_COLUMN,
                LAND_COVER_CLASS_COLUMN,
            ]
        ).reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(output_path, index=False)
    logger.info("Saved sensor-matched upstream land cover to %s", output_path)
    logger.info("Output shape: %s", result_df.shape)
    return result_df


def assemble_land_cover(
    self,
    variant=SENSOR_ASSEMBLY_VARIANT,
    land_cover_path=DEFAULT_ASSEMBLY_LAND_COVER_PATH,
    water_quality_path=DEFAULT_WATER_QUALITY_PATH,
    stations_rivers_path=DEFAULT_STATIONS_TRENCHES_PATH,
    river_network_path=DEFAULT_RIVER_NETWORK_PATH,
    output_path=None,
    n_jobs=None,
):
    """Assemble analysis-ready land-cover outputs for the requested variant."""
    variant = _normalize_assembly_variant(variant)
    output_path = output_path or _default_output_path_for_variant(variant)
    if n_jobs is None:
        n_jobs = resolve_n_jobs()

    if variant == ADM2_ASSEMBLY_VARIANT:
        from .aggregation import aggregate_along_rivers

        return aggregate_along_rivers(
            self,
            land_cover_path=land_cover_path,
            river_network_path=river_network_path,
            drainage_polygons_path=None,
            n_jobs=n_jobs,
            output_path=output_path,
        )

    return _assemble_sensor_land_cover(
        land_cover_path=land_cover_path,
        water_quality_path=water_quality_path,
        stations_rivers_path=stations_rivers_path,
        river_network_path=river_network_path,
        output_path=output_path,
        n_jobs=n_jobs,
    )
