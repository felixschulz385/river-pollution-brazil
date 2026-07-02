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
from land_cover.aggregation import AVAILABLE_KERNELS, _explode_trench_adm2_matches, distance_weights
from land_cover.preprocess import deduplicate_drainage_polygons
from river_network import RiverNetwork
import river_network as rn_module


logger = logging.getLogger(__name__)


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _sql_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


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
        column for column in columns if column not in {TRENCH_ID_COLUMN, DATE_COLUMN, YEAR_COLUMN}
    ]
    if not climate_columns:
        raise ValueError("Climate trench/day data does not include any climate variables.")
    return climate_columns


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


def _prepare_station_trenches(stations_rivers_df):
    required_columns = {STATION_CODE_COLUMN, TRENCH_ID_COLUMN}
    missing_columns = required_columns.difference(stations_rivers_df.columns)
    if missing_columns:
        raise ValueError(
            "Stations-rivers data is missing required columns: "
            f"{sorted(missing_columns)}."
        )

    station_trenches = stations_rivers_df[[STATION_CODE_COLUMN, TRENCH_ID_COLUMN]].dropna().copy()
    station_trenches[STATION_CODE_COLUMN] = station_trenches[STATION_CODE_COLUMN].astype(str)
    station_trenches[TRENCH_ID_COLUMN] = station_trenches[TRENCH_ID_COLUMN].astype(np.int64)
    return station_trenches.drop_duplicates(
        subset=[STATION_CODE_COLUMN, TRENCH_ID_COLUMN],
        keep="first",
    )


def _collapse_same_day_targets(targets):
    if targets.empty:
        return targets

    collapsed = targets.sort_values(
        [STATION_CODE_COLUMN, DATE_COLUMN, DATETIME_COLUMN],
        kind="mergesort",
    )
    group_columns = [STATION_CODE_COLUMN, DATE_COLUMN]
    duplicate_mask = collapsed.duplicated(subset=group_columns, keep=False)
    if not duplicate_mask.any():
        return collapsed.reset_index(drop=True)

    duplicate_rows = collapsed.loc[duplicate_mask].copy()
    fill_columns = [
        column for column in duplicate_rows.columns if column not in group_columns
    ]
    duplicate_rows.loc[:, fill_columns] = (
        duplicate_rows.groupby(group_columns, sort=False, observed=True)[fill_columns]
        .bfill()
    )
    duplicate_rows = duplicate_rows.drop_duplicates(
        subset=group_columns,
        keep="first",
    )

    collapsed = pd.concat(
        [collapsed.loc[~duplicate_mask], duplicate_rows],
        ignore_index=True,
    ).sort_values(
        [STATION_CODE_COLUMN, DATE_COLUMN, DATETIME_COLUMN],
        kind="mergesort",
    )
    return collapsed.loc[:, targets.columns].reset_index(drop=True)


def _prepare_sensor_targets(water_quality_df, station_trenches_df):
    if STATION_CODE_COLUMN not in water_quality_df.columns:
        raise ValueError("Water-quality data must include `station_code` for climate assembly.")
    if DATETIME_COLUMN in water_quality_df.columns:
        datetime_column = DATETIME_COLUMN
    elif DATE_COLUMN in water_quality_df.columns:
        datetime_column = DATE_COLUMN
    else:
        raise ValueError("Water-quality data must include either `datetime` or `date`.")

    targets = water_quality_df[[STATION_CODE_COLUMN, datetime_column]].copy()
    targets = targets.rename(columns={datetime_column: DATETIME_COLUMN})
    targets[STATION_CODE_COLUMN] = targets[STATION_CODE_COLUMN].astype(str)
    targets[DATETIME_COLUMN] = pd.to_datetime(targets[DATETIME_COLUMN], errors="coerce")
    targets[DATE_COLUMN] = targets[DATETIME_COLUMN].dt.normalize()
    targets = targets.dropna(subset=[STATION_CODE_COLUMN, DATETIME_COLUMN, DATE_COLUMN])
    targets = _collapse_same_day_targets(targets)

    station_trench_lookup = station_trenches_df.drop_duplicates(
        subset=[STATION_CODE_COLUMN],
        keep="first",
    )
    targets = targets.merge(
        station_trench_lookup,
        on=STATION_CODE_COLUMN,
        how="inner",
        validate="many_to_one",
    )
    targets[TRENCH_ID_COLUMN] = targets[TRENCH_ID_COLUMN].astype(np.int64)
    return targets.drop_duplicates(
        subset=[STATION_CODE_COLUMN, DATE_COLUMN],
        keep="first",
    ).reset_index(drop=True)


def _assign_distance_buckets(distances):
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


def _build_sensor_upstream_lookup(network, target_trench_ids):
    upstream_frames = []
    for trench_id in target_trench_ids:
        upstream = network.get_upstream_trenches(int(trench_id))[
            [TRENCH_ID_COLUMN, UPSTREAM_DISTANCE_COLUMN]
        ].copy()
        upstream[DISTANCE_BUCKET_COLUMN] = _assign_distance_buckets(upstream[UPSTREAM_DISTANCE_COLUMN])
        upstream = upstream.dropna(subset=[DISTANCE_BUCKET_COLUMN])
        if upstream.empty:
            continue
        upstream["target_trench_id"] = int(trench_id)
        upstream = upstream.rename(columns={TRENCH_ID_COLUMN: "source_trench_id"})
        upstream_frames.append(
            upstream[["target_trench_id", "source_trench_id", DISTANCE_BUCKET_COLUMN]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

    if not upstream_frames:
        return pd.DataFrame(columns=["target_trench_id", "source_trench_id", DISTANCE_BUCKET_COLUMN])
    return pd.concat(upstream_frames, ignore_index=True)


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
    station_trenches = _prepare_station_trenches(stations_rivers_df)
    targets = _prepare_sensor_targets(water_quality_df, station_trenches)
    logger.info("Found %d climate sensor target row(s).", len(targets))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if targets.empty:
        empty_df = pd.DataFrame(
            columns=[
                STATION_CODE_COLUMN,
                DATETIME_COLUMN,
                DATE_COLUMN,
                TRENCH_ID_COLUMN,
                *_empty_sensor_feature_row(climate_columns).keys(),
            ]
        ).set_index([STATION_CODE_COLUMN, DATETIME_COLUMN])
        empty_df.to_parquet(output_path, index=True)
        return empty_df

    logger.info("Loading river network from %s", river_network_path)
    network = RiverNetwork()
    network.load(str(river_network_path))

    target_trench_ids = (
        targets[TRENCH_ID_COLUMN].drop_duplicates().astype(np.int64).sort_values().tolist()
    )
    upstream_lookup = _build_sensor_upstream_lookup(network, target_trench_ids)
    logger.info(
        "Prepared %d upstream trench mapping row(s) for %d target trench(es).",
        len(upstream_lookup),
        len(target_trench_ids),
    )

    temp_dir = Path(tempfile.mkdtemp(prefix="climate_sensor_duckdb_"))
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute(f"PRAGMA threads={int(max(1, n_jobs))}")
        connection.execute(f"PRAGMA temp_directory={_sql_literal(str(temp_dir))}")
        connection.register("sensor_targets_df", targets)
        connection.register("upstream_lookup_df", upstream_lookup)

        climate_sql_path = _sql_literal(str(climate_path))
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

        connection.execute(
            f"""
            CREATE TEMP TABLE climate_bucket_daily AS
            SELECT
                u.target_trench_id,
                c.{DATE_COLUMN} AS {DATE_COLUMN},
                u.{DISTANCE_BUCKET_COLUMN},
                COUNT(DISTINCT c.{TRENCH_ID_COLUMN}) AS {REACHABLE_TRENCH_COUNT_COLUMN},
                {aggregate_columns_sql}
            FROM read_parquet({climate_sql_path}) AS c
            INNER JOIN upstream_lookup_df AS u
                ON c.{TRENCH_ID_COLUMN} = u.source_trench_id
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
        output_sql_path = _sql_literal(str(output_path))

        connection.execute(
            f"""
            COPY (
                SELECT
                    t.{STATION_CODE_COLUMN},
                    t.{DATETIME_COLUMN},
                    t.{DATE_COLUMN},
                    t.{TRENCH_ID_COLUMN},
                    {feature_selects_sql}
                FROM sensor_targets_df AS t
                LEFT JOIN climate_bucket_windowed AS w
                    ON t.{TRENCH_ID_COLUMN} = w.target_trench_id
                   AND t.{DATE_COLUMN} = w.{DATE_COLUMN}
                GROUP BY 1, 2, 3, 4
                ORDER BY 1, 2, 4
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

    trench_adm2_matches = _explode_trench_adm2_matches(network.trenches)
    drainage_polygons = deduplicate_drainage_polygons(network.drainage_areas.copy())
    trench_lookup = drainage_polygons[[TRENCH_ID_COLUMN]].merge(
        trench_adm2_matches[[TRENCH_ID_COLUMN, "adm2", rn_module.SYSTEM_ID_KEY]].drop_duplicates(),
        on=TRENCH_ID_COLUMN,
        how="left",
        validate="one_to_many",
    ).dropna(subset=[rn_module.SYSTEM_ID_KEY])

    adm2_units = trench_lookup["adm2"].dropna().unique()
    rivers = network.trenches
    system_trench_tables = {
        int(system_id): system_trenches[[TRENCH_ID_COLUMN, rn_module.TRENCH_INDEX_COLUMN]]
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

    def process_adm2(adm2_id):
        adm2_trenches = trench_lookup.loc[
            trench_lookup["adm2"] == adm2_id,
            [TRENCH_ID_COLUMN, rn_module.SYSTEM_ID_KEY],
        ].drop_duplicates()
        if adm2_trenches.empty:
            return None

        trench_distance_frames = []
        for system_id, system_adm2_trenches in adm2_trenches.groupby(rn_module.SYSTEM_ID_KEY):
            system_id = int(system_id)
            system_trench_ids = system_trench_id_arrays.get(system_id, np.asarray([], dtype=np.int64))
            if len(system_trench_ids) == 0:
                continue

            trench_position_lookup = system_trench_positions[system_id]
            seed_positions = np.asarray(
                [
                    trench_position_lookup[trench_id]
                    for trench_id in system_adm2_trenches[TRENCH_ID_COLUMN]
                    if trench_id in trench_position_lookup
                ],
                dtype=np.int64,
            )
            if len(seed_positions) == 0:
                continue

            system_reachability = network.trench_reachability_matrices[system_id][seed_positions, :].tocsr()
            system_distance = network.trench_distance_matrices[system_id][seed_positions, :].tocsr()

            min_distances = np.full(len(system_trench_ids), np.inf)
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

            trench_distance_frames.append(
                pd.DataFrame(
                    {
                        TRENCH_ID_COLUMN: system_trench_ids[reachable_mask],
                        UPSTREAM_DISTANCE_COLUMN: min_distances[reachable_mask],
                    }
                )
            )

        if not trench_distance_frames:
            return None

        trench_distance_lookup = (
            pd.concat(trench_distance_frames, ignore_index=True)
            .groupby(TRENCH_ID_COLUMN, as_index=False)[UPSTREAM_DISTANCE_COLUMN]
            .min()
        )
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
