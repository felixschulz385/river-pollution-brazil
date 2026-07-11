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
    ADM2_ID_COLUMN,
    ADM2_UPSTREAM_YEARLY_VARIANT,
    ANNUAL_MAX_VARIABLES,
    ANNUAL_MEAN_VARIABLES,
    ANNUAL_MIN_VARIABLES,
    ANNUAL_SUM_VARIABLES,
    CLIMATE_ASSEMBLE_VARIANTS,
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
    SENSOR_WINDOW_LABELS,
    STATION_CODE_COLUMN,
    TOTAL_WEIGHT_COLUMN,
    TRENCH_ID_COLUMN,
    UPSTREAM_DISTANCE_COLUMN,
    YEAR_COLUMN,
)
from land_cover.aggregation import AVAILABLE_KERNELS, distance_weights
from land_cover.preprocess import deduplicate_drainage_polygons
from river_network import RiverNetwork
import river_network as rn_module
from shared.sensor_upstream import (
    build_target_reachability_lookup,
    label_values_by_intervals,
    prepare_entity_links,
    prepare_observation_targets,
    prepare_trench_adm2_matches,
    resolve_multi_seed_reachable_distances,
)


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

    return partition_paths or [climate_path]


def _annual_aggregate_sql(climate_columns, *, source_alias):
    expressions = []
    for column in climate_columns:
        identifier = f"{source_alias}.{_sql_ident(column)}"
        if column in ANNUAL_SUM_VARIABLES:
            expressions.append(f"SUM({identifier}) AS {_sql_ident(column)}")
        elif column in ANNUAL_MEAN_VARIABLES:
            expressions.append(f"AVG({identifier}) AS {_sql_ident(column)}")
        elif column in ANNUAL_MIN_VARIABLES:
            expressions.append(f"MIN({identifier}) AS {_sql_ident(column)}")
        elif column in ANNUAL_MAX_VARIABLES:
            expressions.append(f"MAX({identifier}) AS {_sql_ident(column)}")
        else:
            raise ValueError(f"Unknown climate annual aggregation rule for column: {column}")
    return ",\n                ".join(expressions)


def _assign_distance_buckets(distances):
    return label_values_by_intervals(distances, SENSOR_DISTANCE_BUCKETS)


def _sensor_bucket_count_column(bucket_name):
    return f"cl_{bucket_name}_n"


def _sensor_bucket_mean_column(bucket_name, variable_name, window_label):
    return f"cl_{bucket_name}_{variable_name}_mean_{window_label}"


def _empty_sensor_feature_row(climate_columns):
    result = {}
    for bucket_name, _, _ in SENSOR_DISTANCE_BUCKETS:
        result[_sensor_bucket_count_column(bucket_name)] = 0
        for variable_name in climate_columns:
            result[_sensor_bucket_mean_column(bucket_name, variable_name, "day")] = np.nan
            for window_label in SENSOR_WINDOW_LABELS:
                result[_sensor_bucket_mean_column(bucket_name, variable_name, window_label)] = np.nan
    return result


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
    logger.info("Found %d climate sensor target row(s).", len(targets))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if targets.empty:
        empty_df = pd.DataFrame(
            columns=[
                STATION_CODE_COLUMN,
                DATE_COLUMN,
                TRENCH_ID_COLUMN,
                *_empty_sensor_feature_row(climate_columns).keys(),
            ]
        ).set_index([STATION_CODE_COLUMN, DATE_COLUMN])
        empty_df.to_parquet(output_path, index=True)
        return empty_df

    logger.info("Loading river network from %s", river_network_path)
    network = RiverNetwork()
    network.load(str(river_network_path))

    target_trench_ids = (
        targets[TRENCH_ID_COLUMN].drop_duplicates().astype(np.int64).sort_values().tolist()
    )
    upstream_lookup = build_target_reachability_lookup(
        network,
        target_trench_ids,
        location_column=TRENCH_ID_COLUMN,
        distance_column=UPSTREAM_DISTANCE_COLUMN,
        category_column=DISTANCE_BUCKET_COLUMN,
        system_column=rn_module.SYSTEM_ID_KEY,
        position_column=rn_module.TRENCH_INDEX_COLUMN,
        categorize_distances=_assign_distance_buckets,
    )
    upstream_lookup = upstream_lookup.rename(
        columns={
            "target_location_id": "target_trench_id",
            "source_location_id": "source_trench_id",
        }
    )
    logger.info(
        "Prepared %d upstream trench mapping row(s) for %d target trench(es).",
        len(upstream_lookup),
        len(target_trench_ids),
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
                window_columns.append(
                    f"AVG({_sql_ident(f'{column}_mean_day')}) OVER ("
                    f"PARTITION BY target_trench_id, {_sql_ident(DISTANCE_BUCKET_COLUMN)} "
                    f"ORDER BY {_sql_ident(DATE_COLUMN)} "
                    f"ROWS BETWEEN {window_size - 1} PRECEDING AND CURRENT ROW"
                    f") AS {_sql_ident(f'{column}_mean_{window_label}')}"
                )
        window_columns_sql = ",\n            ".join(window_columns)
        part_paths = []

        for batch_index, (batch_label, batch_targets, batch_start, batch_end) in enumerate(batch_specs):
            batch_target_trench_ids = (
                batch_targets[TRENCH_ID_COLUMN].drop_duplicates().astype(np.int64)
            )
            batch_upstream_lookup = upstream_lookup.loc[
                upstream_lookup["target_trench_id"].isin(batch_target_trench_ids)
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
                "Processing sensor climate batch %s (%d/%d): %d targets, %d upstream trench links, %d source trenches, %d climate partition(s), dates %s to %s",
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
                    u.target_trench_id,
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
                    target_trench_id,
                    {DATE_COLUMN},
                    {DISTANCE_BUCKET_COLUMN},
                    {REACHABLE_TRENCH_COUNT_COLUMN},
                    {window_columns_sql}
                FROM climate_bucket_daily
                """
            )

            feature_selects = []
            for bucket_name, _, _ in SENSOR_DISTANCE_BUCKETS:
                bucket_literal = _sql_literal(bucket_name)
                feature_selects.append(
                    f"COALESCE(MAX({REACHABLE_TRENCH_COUNT_COLUMN}) FILTER (WHERE {DISTANCE_BUCKET_COLUMN} = {bucket_literal}), 0) AS {_sensor_bucket_count_column(bucket_name)}"
                )
                for variable_name in climate_columns:
                    feature_selects.append(
                        f"MAX({_sql_ident(f'{variable_name}_mean_day')}) FILTER (WHERE {DISTANCE_BUCKET_COLUMN} = {bucket_literal}) AS {_sensor_bucket_mean_column(bucket_name, variable_name, 'day')}"
                    )
                    for window_label in SENSOR_WINDOW_LABELS:
                        feature_selects.append(
                            f"MAX({_sql_ident(f'{variable_name}_mean_{window_label}')}) FILTER (WHERE {DISTANCE_BUCKET_COLUMN} = {bucket_literal}) AS {_sensor_bucket_mean_column(bucket_name, variable_name, window_label)}"
                        )
            feature_selects_sql = ",\n                ".join(feature_selects)
            part_sql_path = _sql_literal(str(part_path))

            connection.execute(
                f"""
                COPY (
                    SELECT
                        t.{STATION_CODE_COLUMN},
                        CAST(t.{DATE_COLUMN} AS DATE) AS {DATE_COLUMN},
                        t.{TRENCH_ID_COLUMN},
                        {feature_selects_sql}
                    FROM sensor_targets_batch_df AS t
                    LEFT JOIN climate_bucket_windowed AS w
                        ON t.{TRENCH_ID_COLUMN} = w.target_trench_id
                       AND CAST(t.{DATE_COLUMN} AS DATE) = w.{DATE_COLUMN}
                    GROUP BY 1, 2, 3
                    ORDER BY 1, 2, 3
                ) TO {part_sql_path} (FORMAT PARQUET)
                """
            )
            part_paths.append(part_path)

        if not part_paths:
            empty_df = pd.DataFrame(
                columns=[
                    STATION_CODE_COLUMN,
                    DATE_COLUMN,
                    TRENCH_ID_COLUMN,
                    *_empty_sensor_feature_row(climate_columns).keys(),
                ]
            ).set_index([STATION_CODE_COLUMN, DATE_COLUMN])
            empty_df.to_parquet(output_path, index=True)
            return empty_df

        output_sql_path = _sql_literal(str(output_path))
        part_glob_sql = _sql_literal(str(parts_dir / "part-*.parquet"))
        connection.execute(
            f"""
            COPY (
                SELECT *
                FROM read_parquet({part_glob_sql})
                ORDER BY 1, 2, 3
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


def _build_adm2_upstream_weights(
    *,
    network,
    kernel,
    h,
    n_jobs,
):
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
    def process_adm2(adm2_id):
        adm2_trenches = trench_lookup.loc[
            trench_lookup["adm2"] == adm2_id,
            [TRENCH_ID_COLUMN, rn_module.SYSTEM_ID_KEY],
        ].drop_duplicates()
        if adm2_trenches.empty:
            return None

        trench_distance_lookup = resolve_multi_seed_reachable_distances(
            network,
            adm2_trenches,
            location_column=TRENCH_ID_COLUMN,
            distance_column=UPSTREAM_DISTANCE_COLUMN,
            system_column=rn_module.SYSTEM_ID_KEY,
            position_column=rn_module.TRENCH_INDEX_COLUMN,
        )
        if trench_distance_lookup.empty:
            return None
        trench_distance_lookup["weight"] = distance_weights(
            trench_distance_lookup[UPSTREAM_DISTANCE_COLUMN].to_numpy(),
            kernel=kernel,
            h=h,
        )
        trench_distance_lookup[ADM2_ID_COLUMN] = adm2_id
        return trench_distance_lookup[[ADM2_ID_COLUMN, TRENCH_ID_COLUMN, "weight"]]

    logger.info("Preparing ADM2 upstream weights for %d ADM2 unit(s) with %s worker(s).", len(adm2_units), n_jobs)
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(process_adm2)(adm2_id)
        for adm2_id in tqdm(adm2_units, desc="Climate ADM2 weights")
    )
    weight_frames = [result for result in results if result is not None and not result.empty]
    if not weight_frames:
        return pd.DataFrame(columns=[ADM2_ID_COLUMN, TRENCH_ID_COLUMN, "weight"])
    weights_df = pd.concat(weight_frames, ignore_index=True)
    logger.info("Prepared %d ADM2 upstream weight row(s).", len(weights_df))
    return weights_df


def _assemble_adm2_upstream_duckdb(
    *,
    climate_path,
    climate_columns,
    river_network_path,
    output_path,
    kernel,
    h,
    n_jobs,
):
    if kernel not in AVAILABLE_KERNELS:
        raise ValueError(f"Unknown kernel: {kernel}. Available: {AVAILABLE_KERNELS}")

    logger.info("Loading river network from %s", river_network_path)
    network = RiverNetwork()
    network.load(str(river_network_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    weights_df = _build_adm2_upstream_weights(
        network=network,
        kernel=kernel,
        h=h,
        n_jobs=n_jobs,
    )
    if weights_df.empty:
        pd.DataFrame(
            columns=[
                ADM2_ID_COLUMN,
                YEAR_COLUMN,
                *climate_columns,
                REACHABLE_TRENCH_COUNT_COLUMN,
                TOTAL_WEIGHT_COLUMN,
            ]
        ).to_parquet(output_path, index=False)
        logger.info("Saved climate ADM2 assembly to %s", output_path)
        return output_path

    temp_dir = Path(tempfile.mkdtemp(prefix="climate_adm2_duckdb_"))
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute(f"PRAGMA threads={int(max(1, n_jobs))}")
        connection.execute(f"PRAGMA temp_directory={_sql_literal(str(temp_dir))}")
        connection.register("adm2_upstream_weights_df", weights_df)

        climate_sql_path = _sql_literal(str(climate_path))
        annual_aggregate_sql = _annual_aggregate_sql(climate_columns, source_alias="c")
        weighted_columns_sql = ",\n                    ".join(
            [
                f"SUM(y.{_sql_ident(column)} * w.weight) AS {_sql_ident(column)}"
                for column in climate_columns
            ]
        )

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

        output_sql_path = _sql_literal(str(output_path))
        connection.execute(
            f"""
            COPY (
                SELECT
                    w.{ADM2_ID_COLUMN},
                    y.{YEAR_COLUMN},
                    {weighted_columns_sql},
                    COUNT(*) AS {REACHABLE_TRENCH_COUNT_COLUMN},
                    SUM(w.weight) AS {TOTAL_WEIGHT_COLUMN}
                FROM climate_by_trench_year AS y
                INNER JOIN adm2_upstream_weights_df AS w
                    ON y.{TRENCH_ID_COLUMN} = w.{TRENCH_ID_COLUMN}
                GROUP BY 1, 2
                ORDER BY 1, 2
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
    kernel="gaussian",
    h=1000000,
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
        kernel=kernel,
        h=h,
        n_jobs=n_jobs,
    )
