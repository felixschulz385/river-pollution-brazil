import logging
from multiprocessing import cpu_count
from pathlib import Path
import shutil
import tempfile

import duckdb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from .constants import (
    ADJUSTED_DISTANCE_COLUMN,
    ADM2_ID_COLUMN,
    ADM2_UPSTREAM_YEARLY_VARIANT,
    CLIMATE_ASSEMBLE_VARIANTS,
    CLIMATE_VARIABLE_COLUMN,
    DATE_COLUMN,
    DATETIME_COLUMN,
    DEFAULT_ADM2_UPSTREAM_YEARLY_OUTPUT_PATH,
    DEFAULT_ERA5_LAND_TRENCH_DAY_PATH,
    DEFAULT_RIVER_NETWORK_PATH,
    DEFAULT_SENSOR_UPSTREAM_OUTPUT_PATH,
    DEFAULT_STATIONS_RIVERS_PATH,
    DEFAULT_WATER_QUALITY_PATH,
    DISTANCE_BUCKET_COLUMN,
    MONTH_COLUMN,
    REACHABLE_TRENCH_COUNT_COLUMN,
    SENSOR_DISTANCE_BUCKETS,
    SENSOR_UPSTREAM_DISTANCE_BUCKETS_VARIANT,
    STATION_CODE_COLUMN,
    TRENCH_ID_COLUMN,
    UPSTREAM_DISTANCE_COLUMN,
    YEAR_COLUMN,
)
from .schema import (
    ANNUAL_MAX_VARIABLES,
    ANNUAL_MEAN_VARIABLES,
    ANNUAL_MIN_VARIABLES,
    SENSOR_WINDOW_LABELS,
)
from src.data.sources.land_cover.aggregation import (
    _apply_shifted_origin,
    _assign_distance_bucket,
)
from src.data.sources.land_cover.schema import build_trench_length_lookup as _build_trench_length_lookup
from src.data.sources.river_network import RiverNetwork
import src.data.sources.river_network as rn_module
from src.data.shared.sensor_upstream import (
    BUCKET_INTERSECTS_ADM2_COLUMN,
    assign_distance_buckets,
    bucket_label,
    build_group_index_lookup,
    build_system_trench_lookup,
    build_trench_system_position_lookup,
    combine_station_upstream_distances,
    normalize_network_frame,
    prepare_entity_links,
    prepare_observation_targets,
    prepare_trench_adm2_matches,
    resolve_multi_seed_reachable_distances,
    resolve_upstream_trench_distances,
    shift_upstream_distances,
    validate_network_index_tables,
)
from src.data.shared.spatial_tabular import deduplicate_drainage_polygons


logger = logging.getLogger(__name__)

SENSOR_ASSEMBLY_MAX_MONTHS_PER_BATCH = 12
SENSOR_ASSEMBLY_MIN_TARGETS_PER_BATCH = 2000


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _sql_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _sql_string_list(values: list[str]) -> str:
    return "[" + ", ".join(_sql_literal(value) for value in values) + "]"


def _resolve_path(root_dir, path, default_path):
    candidate = Path(path or default_path)
    if not candidate.is_absolute():
        candidate = Path(root_dir) / candidate
    return candidate


def _resolve_output_path(root_dir, output_path, variant):
    default_path = (
        DEFAULT_SENSOR_UPSTREAM_OUTPUT_PATH
        if variant == SENSOR_UPSTREAM_DISTANCE_BUCKETS_VARIANT
        else DEFAULT_ADM2_UPSTREAM_YEARLY_OUTPUT_PATH
    )
    return _resolve_path(root_dir, output_path, default_path)


def _load_trench_day_climate_columns(path, connection):
    logger.info("Inspecting preprocessed climate trench/day data from %s", path)
    describe = connection.execute(
        f"DESCRIBE SELECT * FROM read_parquet({_sql_literal(str(path))})"
    ).fetchdf()
    columns = describe["column_name"].astype(str).tolist()
    required_columns = {TRENCH_ID_COLUMN, DATE_COLUMN}
    missing_columns = required_columns.difference(columns)
    if missing_columns:
        raise ValueError(
            "Climate trench/day data is missing required columns: "
            f"{sorted(missing_columns)}."
        )

    climate_columns = [
        column
        for column in columns
        if column not in {TRENCH_ID_COLUMN, DATE_COLUMN, YEAR_COLUMN, MONTH_COLUMN}
    ]
    if not climate_columns:
        raise ValueError("Climate trench/day data does not include any climate variables.")
    return climate_columns


def _partitioned_trench_day_paths(
    climate_path: Path,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> list[Path]:
    if not climate_path.is_dir():
        return [climate_path]

    partition_paths = []
    for period in pd.period_range(start=start_date, end=end_date, freq="M"):
        partition_dir = (
            climate_path
            / f"{YEAR_COLUMN}={period.year:04d}"
            / f"{MONTH_COLUMN}={period.month:02d}"
        )
        if partition_dir.exists():
            partition_paths.append(partition_dir)

    if not partition_paths:
        # None of the requested months have a partition on disk. Correctness
        # is unaffected -- the caller's own `WHERE date BETWEEN` filter still
        # applies -- but this silently turns what might be a real data gap
        # (missing preprocessing, wrong `climate_path`) into an
        # undifferentiated full-directory scan with no signal that anything
        # unusual happened.
        logger.warning(
            "No climate partitions found under %s for %s to %s; falling back to a full "
            "directory scan.",
            climate_path,
            start_date.date(),
            end_date.date(),
        )
    return partition_paths or [climate_path]


def _annual_aggregate_sql(climate_columns, *, source_alias, date_column=DATE_COLUMN):
    # `COUNT(*)` only counts rows actually present in the group, so comparing
    # against it (as this used to) only catches nulls among present rows, not
    # whole calendar days missing from the group (a partial preprocessing
    # run, a trench added mid-pipeline, a store gap). Compare against the
    # true number of days in that trench-year's calendar year instead.
    year_expr = f"EXTRACT(YEAR FROM {source_alias}.{_sql_ident(date_column)})::BIGINT"
    expected_days_expr = (
        f"(CASE WHEN ({year_expr} % 4 = 0 AND ({year_expr} % 100 != 0 OR {year_expr} % 400 = 0)) "
        f"THEN 366 ELSE 365 END)"
    )
    expressions = []
    for column in climate_columns:
        identifier = f"{source_alias}.{_sql_ident(column)}"
        if column in ANNUAL_MEAN_VARIABLES:
            expressions.append(f"AVG({identifier}) AS {_sql_ident(column)}")
        elif column in ANNUAL_MIN_VARIABLES or column in ANNUAL_MAX_VARIABLES:
            # 2t_daily_min/2t_daily_max are only populated on days sourced from
            # ARCO (era5_land_hourly's fallback, era5_land_daily, never writes
            # them -- see ERA5L_VAR_CONFIG). A trench-year that mixes ARCO-backed
            # and era5_land_daily-backed days would otherwise get a MIN/MAX
            # silently computed over just the ARCO-covered subset, indistinguishable
            # from a value based on the full year. Require every calendar day in the
            # year to be present with a non-null value before reporting one; a
            # trench-year with no ARCO coverage at all still correctly aggregates to
            # NULL either way.
            fn = "MIN" if column in ANNUAL_MIN_VARIABLES else "MAX"
            expressions.append(
                f"CASE WHEN COUNT({identifier}) = {expected_days_expr} THEN {fn}({identifier}) "
                f"ELSE NULL END AS {_sql_ident(column)}"
            )
        else:
            raise ValueError(f"Unknown climate annual aggregation rule for column: {column}")
    return ",\n                ".join(expressions)


def _load_sensor_river_network(river_network_path):
    """Load the river network and validate tables needed for trench aggregation."""
    logger.info("Loading river network from %s", river_network_path)
    network = RiverNetwork()
    network.load(str(river_network_path))
    validate_network_index_tables(
        network,
        location_column=TRENCH_ID_COLUMN,
        system_column=rn_module.SYSTEM_ID_KEY,
        position_column=rn_module.TRENCH_INDEX_COLUMN,
    )
    if "distance" not in network.trenches.columns:
        raise ValueError("River trench data is missing matrix index columns: ['distance'].")
    network.trenches = normalize_network_frame(network.trenches)
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


def _empty_sensor_long_columns():
    """Return the long-format output column names for the sensor climate variant."""
    return [
        STATION_CODE_COLUMN,
        DATE_COLUMN,
        DISTANCE_BUCKET_COLUMN,
        CLIMATE_VARIABLE_COLUMN,
        REACHABLE_TRENCH_COUNT_COLUMN,
        "mean_day",
        *[f"mean_{window_label}" for window_label in SENSOR_WINDOW_LABELS],
    ]


def _iter_sensor_target_batches(targets):
    batch_dates = pd.to_datetime(targets[DATE_COLUMN]).dt.normalize()
    batch_periods = batch_dates.dt.to_period("M")
    period_counts = (
        pd.Series(1, index=batch_periods)
        .groupby(level=0)
        .sum()
        .sort_index()
    )
    unique_periods = period_counts.index.tolist()
    contiguous_runs = []
    run_start = 0
    for index in range(1, len(unique_periods) + 1):
        is_break = index == len(unique_periods) or (
            unique_periods[index].ordinal - unique_periods[index - 1].ordinal > 1
        )
        if is_break:
            contiguous_runs.append(unique_periods[run_start:index])
            run_start = index

    for run_periods in contiguous_runs:
        batch_start_index = 0
        while batch_start_index < len(run_periods):
            batch_end_index = batch_start_index
            batch_target_count = 0
            while batch_end_index < len(run_periods):
                next_period = run_periods[batch_end_index]
                batch_target_count += int(period_counts.loc[next_period])
                batch_month_count = batch_end_index - batch_start_index + 1
                batch_end_index += 1
                if (
                    batch_target_count >= SENSOR_ASSEMBLY_MIN_TARGETS_PER_BATCH
                    or batch_month_count >= SENSOR_ASSEMBLY_MAX_MONTHS_PER_BATCH
                ):
                    break

            batch_period_slice = run_periods[batch_start_index:batch_end_index]
            batch_mask = batch_periods.isin(batch_period_slice)
            batch_targets = targets.loc[batch_mask].copy()
            batch_targets[DATE_COLUMN] = pd.to_datetime(batch_targets[DATE_COLUMN]).dt.normalize()

            batch_start_period = batch_period_slice[0]
            batch_end_period = batch_period_slice[-1]
            batch_start = batch_start_period.to_timestamp(how="start")
            batch_end = batch_end_period.to_timestamp(how="end").normalize()
            if batch_start_period == batch_end_period:
                batch_label = batch_start_period.strftime("%Y-%m")
            else:
                batch_label = (
                    f"{batch_start_period.strftime('%Y-%m')}"
                    f"_to_{batch_end_period.strftime('%Y-%m')}"
                )

            yield batch_label, batch_targets, batch_start, batch_end
            batch_start_index = batch_end_index


def _build_sensor_targets(water_quality_df, stations_rivers_df):
    """Return unique station-day targets and the full station-trench map."""
    station_trenches = prepare_entity_links(
        stations_rivers_df,
        entity_column=STATION_CODE_COLUMN,
        location_column=TRENCH_ID_COLUMN,
    )
    targets = prepare_observation_targets(
        water_quality_df,
        station_trenches,
        entity_column=STATION_CODE_COLUMN,
        date_column=DATE_COLUMN,
        timestamp_column=DATETIME_COLUMN,
        location_column=TRENCH_ID_COLUMN,
    )
    targets = (
        targets[[STATION_CODE_COLUMN, DATE_COLUMN]]
        .drop_duplicates()
        .sort_values([STATION_CODE_COLUMN, DATE_COLUMN])
        .reset_index(drop=True)
    )
    return targets, station_trenches


def _build_station_upstream_bucket_lookup(station_trenches, network, n_jobs):
    """Resolve a long (station_code, source_trench_id, distance_bucket) lookup.

    Combines all of a station's own trenches into one shifted upstream
    distance table per station (min adjusted distance per reachable source
    trench), matching the land_cover long-format approach.
    """
    rivers = network.trenches
    (
        system_trench_id_arrays,
        _system_trench_positions,
        system_valid_positions,
    ) = _build_system_trench_lookup(rivers)
    trench_system_position_lookup = _build_trench_system_position_lookup(rivers)
    trench_lengths = _build_trench_length_lookup(rivers)

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

    logger.info("Combining upstream trench distances into station-level 25 km buckets.")
    lookup_frames = []
    for station_code, station_rows in station_trenches.groupby(STATION_CODE_COLUMN):
        combined = _combine_station_upstream_distances(
            station_rows[TRENCH_ID_COLUMN].astype(np.int64).tolist(),
            upstream_distance_cache,
        )
        if combined.empty:
            continue
        combined = combined.copy()
        combined[DISTANCE_BUCKET_COLUMN] = _assign_sensor_distance_buckets(
            combined[ADJUSTED_DISTANCE_COLUMN]
        )
        combined = combined.dropna(subset=[DISTANCE_BUCKET_COLUMN])
        if combined.empty:
            continue
        lookup_frames.append(
            pd.DataFrame(
                {
                    STATION_CODE_COLUMN: str(station_code),
                    "source_trench_id": combined[TRENCH_ID_COLUMN].astype(np.int64),
                    DISTANCE_BUCKET_COLUMN: combined[DISTANCE_BUCKET_COLUMN].astype(int),
                }
            )
        )

    if not lookup_frames:
        return pd.DataFrame(
            columns=[STATION_CODE_COLUMN, "source_trench_id", DISTANCE_BUCKET_COLUMN]
        )
    return pd.concat(lookup_frames, ignore_index=True)


def _assemble_sensor_upstream_duckdb(
    *,
    climate_path,
    climate_columns,
    water_quality_path,
    stations_rivers_path,
    river_network_path,
    output_path,
    n_jobs,
):
    logger.info("Loading cleaned water-quality data from %s", water_quality_path)
    water_quality_df = pd.read_parquet(water_quality_path)
    logger.info("Loading station-river matches from %s", stations_rivers_path)
    stations_rivers_df = pd.read_parquet(stations_rivers_path)
    targets, station_trenches = _build_sensor_targets(water_quality_df, stations_rivers_df)
    logger.info("Found %d climate sensor target row(s).", len(targets))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if targets.empty:
        empty_df = pd.DataFrame(columns=_empty_sensor_long_columns())
        empty_df.to_parquet(output_path, index=False)
        return empty_df

    network = _load_sensor_river_network(river_network_path)
    upstream_lookup = _build_station_upstream_bucket_lookup(station_trenches, network, n_jobs)
    logger.info(
        "Prepared %d station upstream-bucket mapping row(s) for %d station(s).",
        len(upstream_lookup),
        station_trenches[STATION_CODE_COLUMN].nunique(),
    )

    temp_dir = Path(tempfile.mkdtemp(prefix="climate_sensor_duckdb_"))
    parts_dir = temp_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    max_window_size = max(SENSOR_WINDOW_LABELS.values())
    lookback_days = max_window_size - 1
    batch_specs = list(_iter_sensor_target_batches(targets))
    logger.info(
        "Processing %d sensor target batch(es) with %d-day lookback windows.",
        len(batch_specs),
        lookback_days,
    )

    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute(f"PRAGMA threads={int(max(1, n_jobs))}")
        connection.execute(f"PRAGMA temp_directory={_sql_literal(str(temp_dir))}")
        connection.execute("PRAGMA preserve_insertion_order=false")

        # Every column (including the accumulation variables tp/sro/ssro/pev)
        # aggregates to a mean daily rate here, matching `_annual_aggregate_sql`'s
        # `ANNUAL_MEAN_VARIABLES` classification for the ADM2 panel -- see the
        # comment there for why a mean (not a cumulative sum) was chosen for
        # accumulation variables too.
        aggregate_columns_sql = ",\n            ".join(
            [
                f"AVG(c.{_sql_ident(column)}) AS {_sql_ident(f'{column}_mean_day')}"
                for column in climate_columns
            ]
        )
        window_columns = []
        for column in climate_columns:
            window_columns.append(_sql_ident(f"{column}_mean_day"))
            for window_label, window_size in SENSOR_WINDOW_LABELS.items():
                # RANGE (not ROWS) so the window spans `window_size` actual
                # calendar days regardless of gaps in the daily climate-bucket
                # series (e.g. a date with no contributing trench rows for
                # this station/bucket is simply absent, not zero-filled) --
                # ROWS BETWEEN counts physical rows, so a single missing day
                # would silently make "7d" reach back 8 calendar days.
                window_columns.append(
                    f"AVG({_sql_ident(f'{column}_mean_day')}) OVER ("
                    f"PARTITION BY {_sql_ident(STATION_CODE_COLUMN)}, {_sql_ident(DISTANCE_BUCKET_COLUMN)} "
                    f"ORDER BY {_sql_ident(DATE_COLUMN)} "
                    f"RANGE BETWEEN INTERVAL {window_size - 1} DAYS PRECEDING AND CURRENT ROW"
                    f") AS {_sql_ident(f'{column}_mean_{window_label}')}"
                )
        window_columns_sql = ",\n            ".join(window_columns)

        window_value_columns = [f"mean_{window_label}" for window_label in SENSOR_WINDOW_LABELS]
        value_columns_sql = ",\n                    ".join(
            f"w.{_sql_ident(column)}" for column in ["mean_day", *window_value_columns]
        )
        variable_branches = []
        for column in climate_columns:
            window_select_sql = ",\n                    ".join(
                f"{_sql_ident(f'{column}_mean_{window_label}')} AS {_sql_ident(f'mean_{window_label}')}"
                for window_label in SENSOR_WINDOW_LABELS
            )
            variable_branches.append(
                f"""
                SELECT
                    {_sql_ident(STATION_CODE_COLUMN)},
                    {_sql_ident(DATE_COLUMN)},
                    {_sql_ident(DISTANCE_BUCKET_COLUMN)},
                    {_sql_literal(column)} AS {_sql_ident(CLIMATE_VARIABLE_COLUMN)},
                    {_sql_ident(REACHABLE_TRENCH_COUNT_COLUMN)},
                    {_sql_ident(f'{column}_mean_day')} AS mean_day,
                    {window_select_sql}
                FROM climate_bucket_windowed
                """
            )
        climate_long_sql = " UNION ALL ".join(variable_branches)

        distance_buckets_df = pd.DataFrame(
            {DISTANCE_BUCKET_COLUMN: [_bucket_label(lower) for lower, _ in SENSOR_DISTANCE_BUCKETS]}
        )
        climate_variables_df = pd.DataFrame({CLIMATE_VARIABLE_COLUMN: list(climate_columns)})
        connection.register("distance_buckets_df", distance_buckets_df)
        connection.register("climate_variables_df", climate_variables_df)

        part_paths = []

        for batch_index, (batch_label, batch_targets, batch_start, batch_end) in enumerate(batch_specs):
            batch_station_codes = batch_targets[STATION_CODE_COLUMN].drop_duplicates()
            batch_upstream_lookup = upstream_lookup.loc[
                upstream_lookup[STATION_CODE_COLUMN].isin(batch_station_codes)
            ].copy()
            if batch_upstream_lookup.empty:
                logger.info(
                    "Skipping empty sensor target batch %s (%d/%d).",
                    batch_label,
                    batch_index + 1,
                    len(batch_specs),
                )
                continue

            batch_lookup_source_trenches = (
                batch_upstream_lookup["source_trench_id"].drop_duplicates().astype(np.int64)
            )
            climate_start = batch_start - pd.Timedelta(days=lookback_days)
            climate_batch_paths = _partitioned_trench_day_paths(
                climate_path,
                start_date=climate_start,
                end_date=batch_end,
            )
            climate_sql_path = _sql_string_list([str(path) for path in climate_batch_paths])
            part_path = parts_dir / f"part-{batch_index:04d}-{batch_label}.parquet"

            logger.info(
                "Processing sensor climate batch %s (%d/%d): %d targets, %d station upstream-bucket links, %d source trenches, %d climate partition(s), dates %s to %s",
                batch_label,
                batch_index + 1,
                len(batch_specs),
                len(batch_targets),
                len(batch_upstream_lookup),
                len(batch_lookup_source_trenches),
                len(climate_batch_paths),
                climate_start.date(),
                batch_end.date(),
            )

            connection.register("sensor_targets_batch_df", batch_targets)
            connection.register("upstream_lookup_batch_df", batch_upstream_lookup)
            connection.register(
                "source_trench_ids_batch_df",
                pd.DataFrame({TRENCH_ID_COLUMN: batch_lookup_source_trenches}),
            )

            connection.execute("DROP TABLE IF EXISTS climate_bucket_daily")
            connection.execute("DROP TABLE IF EXISTS climate_bucket_windowed")
            connection.execute(
                f"""
                CREATE TEMP TABLE climate_bucket_daily AS
                SELECT
                    u.{STATION_CODE_COLUMN},
                    CAST(c.{DATE_COLUMN} AS DATE) AS {DATE_COLUMN},
                    u.{DISTANCE_BUCKET_COLUMN},
                    COUNT(DISTINCT c.{TRENCH_ID_COLUMN}) AS {REACHABLE_TRENCH_COUNT_COLUMN},
                    {aggregate_columns_sql}
                FROM read_parquet({climate_sql_path}) AS c
                INNER JOIN source_trench_ids_batch_df AS s
                    ON c.{TRENCH_ID_COLUMN} = s.{TRENCH_ID_COLUMN}
                INNER JOIN upstream_lookup_batch_df AS u
                    ON c.{TRENCH_ID_COLUMN} = u.source_trench_id
                WHERE CAST(c.{DATE_COLUMN} AS DATE) BETWEEN DATE {_sql_literal(str(climate_start.date()))}
                    AND DATE {_sql_literal(str(batch_end.date()))}
                GROUP BY 1, 2, 3
                """
            )

            connection.execute(
                f"""
                CREATE TEMP TABLE climate_bucket_windowed AS
                SELECT
                    {STATION_CODE_COLUMN},
                    {DATE_COLUMN},
                    {DISTANCE_BUCKET_COLUMN},
                    {REACHABLE_TRENCH_COUNT_COLUMN},
                    {window_columns_sql}
                FROM climate_bucket_daily
                """
            )

            part_sql_path = _sql_literal(str(part_path))

            connection.execute(
                f"""
                COPY (
                    SELECT
                        g.{STATION_CODE_COLUMN},
                        g.{DATE_COLUMN},
                        g.{DISTANCE_BUCKET_COLUMN},
                        g.{CLIMATE_VARIABLE_COLUMN},
                        COALESCE(w.{REACHABLE_TRENCH_COUNT_COLUMN}, 0) AS {REACHABLE_TRENCH_COUNT_COLUMN},
                        {value_columns_sql}
                    FROM (
                        SELECT
                            t.{STATION_CODE_COLUMN},
                            CAST(t.{DATE_COLUMN} AS DATE) AS {DATE_COLUMN},
                            b.{DISTANCE_BUCKET_COLUMN},
                            v.{CLIMATE_VARIABLE_COLUMN}
                        FROM sensor_targets_batch_df AS t
                        CROSS JOIN distance_buckets_df AS b
                        CROSS JOIN climate_variables_df AS v
                    ) AS g
                    LEFT JOIN ({climate_long_sql}) AS w
                        ON g.{STATION_CODE_COLUMN} = w.{STATION_CODE_COLUMN}
                       AND g.{DATE_COLUMN} = w.{DATE_COLUMN}
                       AND g.{DISTANCE_BUCKET_COLUMN} = w.{DISTANCE_BUCKET_COLUMN}
                       AND g.{CLIMATE_VARIABLE_COLUMN} = w.{CLIMATE_VARIABLE_COLUMN}
                    ORDER BY 1, 2, 3, 4
                ) TO {part_sql_path} (FORMAT PARQUET)
                """
            )
            part_paths.append(part_path)

        if not part_paths:
            empty_df = pd.DataFrame(columns=_empty_sensor_long_columns())
            empty_df.to_parquet(output_path, index=False)
            return empty_df

        output_sql_path = _sql_literal(str(output_path))
        part_glob_sql = _sql_literal(str(parts_dir / "part-*.parquet"))
        connection.execute(
            f"""
            COPY (
                SELECT *
                FROM read_parquet({part_glob_sql})
                ORDER BY 1, 2, 3, 4
            ) TO {output_sql_path} (FORMAT PARQUET)
            """
        )
    finally:
        close = getattr(connection, "close", None)
        if callable(close):
            close()
        shutil.rmtree(temp_dir, ignore_errors=True)

    logger.info("Saved climate sensor assembly to %s", output_path)
    return output_path


def _build_adm2_upstream_buckets(
    *,
    network,
    n_jobs,
):
    """Bin each ADM2 unit's upstream trenches into discrete 25 km distance buckets.

    Mirrors `land_cover.aggregation.aggregate_along_rivers`'s ADM2 binning exactly
    (same shifted-origin + bucket-width scheme) so climate and land-cover ADM2
    outputs share one upstream-distance representation; any distance weighting
    across buckets happens downstream, at assembly time.
    """
    if not network.trench_reachability_matrices:
        raise ValueError("River network must have trench reachability data computed.")
    if network.trenches is None:
        raise ValueError("River network must include trench data.")
    if network.drainage_areas is None:
        raise ValueError("River network must include drainage polygon data.")

    trench_adm2_matches = prepare_trench_adm2_matches(
        network,
        rn_module=rn_module,
        trench_id_column=TRENCH_ID_COLUMN,
    )
    drainage_polygons = deduplicate_drainage_polygons(
        network.drainage_areas.reset_index(drop=True).copy()
    ).reset_index(drop=True)
    trench_lookup = drainage_polygons[[TRENCH_ID_COLUMN]].merge(
        trench_adm2_matches[[TRENCH_ID_COLUMN, "adm2", rn_module.SYSTEM_ID_KEY]].drop_duplicates(),
        on=TRENCH_ID_COLUMN,
        how="left",
        validate="one_to_many",
    ).dropna(subset=[rn_module.SYSTEM_ID_KEY])

    adm2_units = trench_lookup["adm2"].dropna().unique()
    validate_network_index_tables(
        network,
        location_column=TRENCH_ID_COLUMN,
        system_column=rn_module.SYSTEM_ID_KEY,
        position_column=rn_module.TRENCH_INDEX_COLUMN,
    )
    system_location_arrays, system_positions = build_group_index_lookup(
        network.trenches,
        location_column=TRENCH_ID_COLUMN,
        system_column=rn_module.SYSTEM_ID_KEY,
        position_column=rn_module.TRENCH_INDEX_COLUMN,
    )
    trench_lengths = _build_trench_length_lookup(network.trenches)

    def process_adm2(adm2_id):
        adm2_trenches = trench_lookup.loc[
            trench_lookup["adm2"] == adm2_id,
            [TRENCH_ID_COLUMN, rn_module.SYSTEM_ID_KEY],
        ].drop_duplicates()
        if adm2_trenches.empty:
            return None
        intersecting_trench_ids = set(adm2_trenches[TRENCH_ID_COLUMN])

        trench_distance_lookup = resolve_multi_seed_reachable_distances(
            network,
            adm2_trenches,
            location_column=TRENCH_ID_COLUMN,
            distance_column=UPSTREAM_DISTANCE_COLUMN,
            system_column=rn_module.SYSTEM_ID_KEY,
            position_column=rn_module.TRENCH_INDEX_COLUMN,
            system_location_arrays=system_location_arrays,
            system_positions=system_positions,
        )
        if trench_distance_lookup.empty:
            return None
        trench_distance_lookup = trench_distance_lookup.set_index(TRENCH_ID_COLUMN)[
            UPSTREAM_DISTANCE_COLUMN
        ]
        trench_distance_lookup = _apply_shifted_origin(trench_distance_lookup, trench_lengths)

        buckets = trench_distance_lookup.reset_index()[[TRENCH_ID_COLUMN]]
        buckets[ADM2_ID_COLUMN] = adm2_id
        buckets[DISTANCE_BUCKET_COLUMN] = _assign_distance_bucket(
            trench_distance_lookup[ADJUSTED_DISTANCE_COLUMN].to_numpy()
        )
        buckets[BUCKET_INTERSECTS_ADM2_COLUMN] = buckets[TRENCH_ID_COLUMN].isin(
            intersecting_trench_ids
        )
        return buckets[[ADM2_ID_COLUMN, TRENCH_ID_COLUMN, DISTANCE_BUCKET_COLUMN, BUCKET_INTERSECTS_ADM2_COLUMN]]

    logger.info("Preparing ADM2 upstream buckets for %d ADM2 unit(s) with %s worker(s).", len(adm2_units), n_jobs)
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(process_adm2)(adm2_id)
        for adm2_id in tqdm(adm2_units, desc="Climate ADM2 buckets")
    )
    bucket_frames = [result for result in results if result is not None and not result.empty]
    if not bucket_frames:
        return pd.DataFrame(
            columns=[ADM2_ID_COLUMN, TRENCH_ID_COLUMN, DISTANCE_BUCKET_COLUMN, BUCKET_INTERSECTS_ADM2_COLUMN]
        )
    buckets_df = pd.concat(bucket_frames, ignore_index=True)
    logger.info("Prepared %d ADM2 upstream bucket row(s).", len(buckets_df))
    return buckets_df


def _assemble_adm2_upstream_duckdb(
    *,
    climate_path,
    climate_columns,
    river_network_path,
    output_path,
    n_jobs,
):
    """Bin climate into 25 km upstream distance buckets per ADM2 unit/year/variable.

    Long output, structurally aligned with `land_cover.aggregation.aggregate_along_rivers`'s
    ADM2 output (same bucket scheme, plus a `bucket_intersects_adm2` flag); any
    distance-kernel weighting across buckets happens downstream, at assembly time,
    via `src.data.assembly`.
    """
    logger.info("Loading river network from %s", river_network_path)
    network = RiverNetwork()
    network.load(str(river_network_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    buckets_df = _build_adm2_upstream_buckets(network=network, n_jobs=n_jobs)
    empty_columns = [
        ADM2_ID_COLUMN,
        YEAR_COLUMN,
        DISTANCE_BUCKET_COLUMN,
        CLIMATE_VARIABLE_COLUMN,
        "mean_value",
        REACHABLE_TRENCH_COUNT_COLUMN,
        BUCKET_INTERSECTS_ADM2_COLUMN,
    ]
    if buckets_df.empty:
        pd.DataFrame(columns=empty_columns).to_parquet(output_path, index=False)
        logger.info("Saved climate ADM2 assembly to %s", output_path)
        return output_path

    temp_dir = Path(tempfile.mkdtemp(prefix="climate_adm2_duckdb_"))
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute(f"PRAGMA threads={int(max(1, n_jobs))}")
        connection.execute(f"PRAGMA temp_directory={_sql_literal(str(temp_dir))}")
        connection.register("adm2_upstream_buckets_df", buckets_df)

        climate_sql_path = _sql_literal(str(climate_path))
        annual_aggregate_sql = _annual_aggregate_sql(climate_columns, source_alias="c")
        connection.execute(
            f"""
            CREATE TEMP TABLE climate_by_trench_year AS
            SELECT
                c.{TRENCH_ID_COLUMN} AS {TRENCH_ID_COLUMN},
                EXTRACT(YEAR FROM c.{DATE_COLUMN})::BIGINT AS {YEAR_COLUMN},
                {annual_aggregate_sql}
            FROM read_parquet({climate_sql_path}) AS c
            GROUP BY 1, 2
            """
        )

        long_branches_sql = " UNION ALL ".join(
            f"""
            SELECT
                {TRENCH_ID_COLUMN}, {YEAR_COLUMN},
                {_sql_literal(column)} AS {_sql_ident(CLIMATE_VARIABLE_COLUMN)},
                {_sql_ident(column)} AS value
            FROM climate_by_trench_year
            """
            for column in climate_columns
        )

        output_sql_path = _sql_literal(str(output_path))
        connection.execute(
            f"""
            COPY (
                SELECT
                    b.{ADM2_ID_COLUMN},
                    y.{YEAR_COLUMN},
                    b.{DISTANCE_BUCKET_COLUMN},
                    y.{CLIMATE_VARIABLE_COLUMN},
                    AVG(y.value) AS mean_value,
                    COUNT(*) AS {REACHABLE_TRENCH_COUNT_COLUMN},
                    BOOL_OR(b.{BUCKET_INTERSECTS_ADM2_COLUMN}) AS {BUCKET_INTERSECTS_ADM2_COLUMN}
                FROM ({long_branches_sql}) AS y
                INNER JOIN adm2_upstream_buckets_df AS b
                    ON y.{TRENCH_ID_COLUMN} = b.{TRENCH_ID_COLUMN}
                GROUP BY 1, 2, 3, 4
                ORDER BY 1, 2, 3, 4
            ) TO {output_sql_path} (FORMAT PARQUET)
            """
        )
    finally:
        close = getattr(connection, "close", None)
        if callable(close):
            close()
        shutil.rmtree(temp_dir, ignore_errors=True)

    logger.info("Saved climate ADM2 assembly to %s", output_path)
    return output_path


def assemble_climate(
    self,
    variant=SENSOR_UPSTREAM_DISTANCE_BUCKETS_VARIANT,
    climate_path=DEFAULT_ERA5_LAND_TRENCH_DAY_PATH,
    water_quality_path=DEFAULT_WATER_QUALITY_PATH,
    stations_rivers_path=DEFAULT_STATIONS_RIVERS_PATH,
    river_network_path=DEFAULT_RIVER_NETWORK_PATH,
    output_path=None,
    n_jobs=None,
):
    """Assemble preprocessed climate data into sensor or ADM2 outputs."""
    if variant not in CLIMATE_ASSEMBLE_VARIANTS:
        raise ValueError(
            f"Unsupported climate assemble variant: {variant}. "
            f"Available variants: {sorted(CLIMATE_ASSEMBLE_VARIANTS)}"
        )
    if n_jobs is None:
        n_jobs = cpu_count()

    climate_path = _resolve_path(self.root_dir, climate_path, DEFAULT_ERA5_LAND_TRENCH_DAY_PATH)
    river_network_path = _resolve_path(self.root_dir, river_network_path, DEFAULT_RIVER_NETWORK_PATH)
    output_path = _resolve_output_path(self.root_dir, output_path, variant)

    connection = duckdb.connect(database=":memory:")
    try:
        climate_columns = _load_trench_day_climate_columns(climate_path, connection)
    finally:
        connection.close()

    if variant == SENSOR_UPSTREAM_DISTANCE_BUCKETS_VARIANT:
        water_quality_path = _resolve_path(self.root_dir, water_quality_path, DEFAULT_WATER_QUALITY_PATH)
        stations_rivers_path = _resolve_path(
            self.root_dir,
            stations_rivers_path,
            DEFAULT_STATIONS_RIVERS_PATH,
        )
        return _assemble_sensor_upstream_duckdb(
            climate_path=climate_path,
            climate_columns=climate_columns,
            water_quality_path=water_quality_path,
            stations_rivers_path=stations_rivers_path,
            river_network_path=river_network_path,
            output_path=output_path,
            n_jobs=n_jobs,
        )

    return _assemble_adm2_upstream_duckdb(
        climate_path=climate_path,
        climate_columns=climate_columns,
        river_network_path=river_network_path,
        output_path=output_path,
        n_jobs=n_jobs,
    )
