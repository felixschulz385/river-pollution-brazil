import logging
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..fetch.paths import get_water_quality_dir
from .settings import (
    ASSEMBLED_SENSOR_DATA_PARQUET,
    CLEAN_STREAMFLOW_PARQUET,
    CLEAN_WATER_QUALITY_PARQUET,
    DATETIME_COLUMN,
    STATIONS_RIVERS_PARQUET,
    STREAMFLOW_MATCH_RADIUS_M,
    STREAMFLOW_ROLLING_WINDOWS,
)

try:
    from ...river_network import RiverNetwork
    from ... import river_network as rn_module
except ImportError:
    from river_network import RiverNetwork
    import river_network as rn_module


logger = logging.getLogger(__name__)

STATION_CODE_COLUMN = "station_code"
DATE_COLUMN = "date"
TRENCH_ID_COLUMN = "trench_id"
DISCHARGE_COLUMN = "discharge"
STREAMFLOW_DAY_COLUMN = "streamflow_discharge_day"
STREAMFLOW_FEATURE_COLUMNS = (
    STREAMFLOW_DAY_COLUMN,
    *(
        f"streamflow_discharge_mean_{window}d"
        for window in STREAMFLOW_ROLLING_WINDOWS
    ),
)
STREAMFLOW_DIAGNOSTIC_COLUMNS = (
    "streamflow_match_count",
    "streamflow_nonnull_day_count",
    "streamflow_total_weight",
    "streamflow_nearest_distance_m",
)


def _resolve_path(root_dir, path, default_filename):
    if path is not None:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = Path(root_dir) / candidate
        return candidate
    return get_water_quality_dir(root_dir) / default_filename


def _resolve_project_path(root_dir, path, default_path):
    candidate = Path(path or default_path)
    if not candidate.is_absolute():
        candidate = Path(root_dir) / candidate
    return candidate


def _validate_columns(frame, columns, frame_name):
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(
            f"Missing required {frame_name} column(s): " + ", ".join(missing)
        )


def _prepare_station_trenches(stations_rivers):
    _validate_columns(
        stations_rivers,
        [STATION_CODE_COLUMN, TRENCH_ID_COLUMN],
        "stations-rivers",
    )
    station_trenches = stations_rivers[
        [STATION_CODE_COLUMN, TRENCH_ID_COLUMN]
    ].dropna().copy()
    station_trenches[STATION_CODE_COLUMN] = station_trenches[
        STATION_CODE_COLUMN
    ].astype(str)
    station_trenches[TRENCH_ID_COLUMN] = station_trenches[TRENCH_ID_COLUMN].astype(
        np.int64
    )
    return station_trenches.drop_duplicates(
        subset=[STATION_CODE_COLUMN, TRENCH_ID_COLUMN],
        keep="first",
    )


def _prepare_streamflow_features(streamflow):
    _validate_columns(
        streamflow,
        [STATION_CODE_COLUMN, DATE_COLUMN, DISCHARGE_COLUMN],
        "streamflow",
    )
    features = streamflow[[STATION_CODE_COLUMN, DATE_COLUMN, DISCHARGE_COLUMN]].copy()
    features[STATION_CODE_COLUMN] = features[STATION_CODE_COLUMN].astype(str)
    features[DATE_COLUMN] = pd.to_datetime(
        features[DATE_COLUMN],
        errors="coerce",
    ).dt.normalize()
    features[DISCHARGE_COLUMN] = pd.to_numeric(
        features[DISCHARGE_COLUMN],
        errors="coerce",
    )
    features = features.dropna(subset=[STATION_CODE_COLUMN, DATE_COLUMN])
    features = features.sort_values([STATION_CODE_COLUMN, DATE_COLUMN], kind="mergesort")
    features[STREAMFLOW_DAY_COLUMN] = features[DISCHARGE_COLUMN]

    for window in STREAMFLOW_ROLLING_WINDOWS:
        column = f"streamflow_discharge_mean_{window}d"
        features[column] = (
            features.groupby(STATION_CODE_COLUMN, observed=True)[DISCHARGE_COLUMN]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    return features[
        [STATION_CODE_COLUMN, DATE_COLUMN, *STREAMFLOW_FEATURE_COLUMNS]
    ].reset_index(drop=True)


def _validate_network(network):
    if network.trenches is None:
        raise ValueError("River network must include trench data.")
    if not network.trench_reachability_matrices:
        raise ValueError("River network must include trench reachability matrices.")
    if not network.trench_distance_matrices:
        raise ValueError("River network must include trench distance matrices.")
    required_columns = {
        rn_module.TRENCH_ID_COLUMN,
        rn_module.SYSTEM_ID_KEY,
        rn_module.TRENCH_INDEX_COLUMN,
    }
    missing_columns = required_columns.difference(network.trenches.columns)
    if missing_columns:
        raise ValueError(
            "River network trench data is missing required column(s): "
            + ", ".join(sorted(missing_columns))
        )


def _build_trench_metadata(network):
    trenches = network.trenches[
        [
            rn_module.TRENCH_ID_COLUMN,
            rn_module.SYSTEM_ID_KEY,
            rn_module.TRENCH_INDEX_COLUMN,
        ]
    ].dropna().copy()
    trenches[rn_module.TRENCH_ID_COLUMN] = trenches[
        rn_module.TRENCH_ID_COLUMN
    ].astype(np.int64)
    trenches[rn_module.SYSTEM_ID_KEY] = trenches[rn_module.SYSTEM_ID_KEY].astype(int)
    trenches[rn_module.TRENCH_INDEX_COLUMN] = trenches[
        rn_module.TRENCH_INDEX_COLUMN
    ].astype(int)

    trench_lookup = trenches.drop_duplicates(
        subset=[rn_module.TRENCH_ID_COLUMN],
        keep="first",
    ).set_index(rn_module.TRENCH_ID_COLUMN)
    system_trench_ids = {
        int(system_id): system_trenches.sort_values(rn_module.TRENCH_INDEX_COLUMN)[
            rn_module.TRENCH_ID_COLUMN
        ].to_numpy(dtype=np.int64)
        for system_id, system_trenches in trenches.groupby(rn_module.SYSTEM_ID_KEY)
    }
    return trench_lookup, system_trench_ids


def _sparse_distance_lookup(sparse_row_or_col):
    matrix = sparse_row_or_col.tocoo()
    index_values = matrix.col if matrix.shape[0] == 1 else matrix.row
    return {
        int(index): float(distance)
        for index, distance in zip(index_values, matrix.data)
    }


def _sparse_indices(sparse_row_or_col):
    matrix = sparse_row_or_col.tocoo()
    index_values = matrix.col if matrix.shape[0] == 1 else matrix.row
    return [int(index) for index in index_values.tolist()]


def _sparse_row(sparse_matrix, row_index):
    if hasattr(sparse_matrix, "getrow"):
        return sparse_matrix.getrow(row_index)
    return sparse_matrix[row_index : row_index + 1, :]


def _sparse_col(sparse_matrix, col_index):
    if hasattr(sparse_matrix, "getcol"):
        return sparse_matrix.getcol(col_index)
    return sparse_matrix[:, col_index : col_index + 1]


def _candidate_trench_distances(
    target_trench_id,
    trench_lookup,
    system_trench_ids,
    network,
    match_radius_m,
):
    if target_trench_id not in trench_lookup.index:
        return pd.DataFrame(columns=[TRENCH_ID_COLUMN, "distance_m"])

    target = trench_lookup.loc[target_trench_id]
    system_id = int(target[rn_module.SYSTEM_ID_KEY])
    target_position = int(target[rn_module.TRENCH_INDEX_COLUMN])
    system_ids = system_trench_ids.get(system_id)
    if system_ids is None:
        return pd.DataFrame(columns=[TRENCH_ID_COLUMN, "distance_m"])

    reachability = network.trench_reachability_matrices.get(system_id)
    distances = network.trench_distance_matrices.get(system_id)
    if reachability is None or distances is None:
        return pd.DataFrame(columns=[TRENCH_ID_COLUMN, "distance_m"])

    upstream_reach = _sparse_row(reachability, target_position)
    upstream_distances = _sparse_distance_lookup(_sparse_row(distances, target_position))
    downstream_reach = _sparse_col(reachability, target_position)
    downstream_distances = _sparse_distance_lookup(_sparse_col(distances, target_position))

    candidate_distances = {target_position: 0.0}
    for candidate_position in _sparse_indices(upstream_reach):
        candidate_distances[int(candidate_position)] = abs(
            upstream_distances.get(int(candidate_position), 0.0)
        )
    for candidate_position in _sparse_indices(downstream_reach):
        candidate_distances[int(candidate_position)] = min(
            candidate_distances.get(int(candidate_position), np.inf),
            abs(downstream_distances.get(int(candidate_position), 0.0)),
        )

    records = [
        {
            TRENCH_ID_COLUMN: int(system_ids[candidate_position]),
            "distance_m": float(distance_m),
        }
        for candidate_position, distance_m in candidate_distances.items()
        if distance_m <= match_radius_m
    ]
    if not records:
        return pd.DataFrame(columns=[TRENCH_ID_COLUMN, "distance_m"])
    return pd.DataFrame(records).drop_duplicates(
        subset=[TRENCH_ID_COLUMN],
        keep="first",
    )


def _triangular_weight(distance_m, match_radius_m):
    return max(0.0, 1.0 - (float(distance_m) / float(match_radius_m)))


def _build_station_matches(
    water_quality_stations,
    streamflow_stations,
    station_trenches,
    network,
    match_radius_m,
    n_jobs=None,
):
    _validate_network(network)
    if n_jobs is None:
        n_jobs = cpu_count()

    trench_lookup, system_trench_ids = _build_trench_metadata(network)

    wq_trenches = station_trenches.loc[
        station_trenches[STATION_CODE_COLUMN].isin(water_quality_stations)
    ].rename(columns={STATION_CODE_COLUMN: "wq_station_code"})
    sf_trenches = station_trenches.loc[
        station_trenches[STATION_CODE_COLUMN].isin(streamflow_stations)
    ].rename(columns={STATION_CODE_COLUMN: "streamflow_station_code"})

    streamflow_by_trench = sf_trenches.groupby(TRENCH_ID_COLUMN, observed=True)[
        "streamflow_station_code"
    ].apply(list)

    def build_station_records(wq_row):
        wq_station_code = str(wq_row.wq_station_code)
        target_trench_id = int(getattr(wq_row, TRENCH_ID_COLUMN))
        candidate_trenches = _candidate_trench_distances(
            target_trench_id,
            trench_lookup,
            system_trench_ids,
            network,
            match_radius_m,
        )
        station_records = []
        for candidate in candidate_trenches.itertuples(index=False):
            candidate_trench_id = int(getattr(candidate, TRENCH_ID_COLUMN))
            distance_m = float(candidate.distance_m)
            streamflow_station_codes = streamflow_by_trench.get(candidate_trench_id, [])
            for streamflow_station_code in streamflow_station_codes:
                station_records.append(
                    {
                        "wq_station_code": wq_station_code,
                        "streamflow_station_code": str(streamflow_station_code),
                        "streamflow_distance_m": distance_m,
                        "streamflow_weight": _triangular_weight(
                            distance_m,
                            match_radius_m,
                        ),
                    }
                )
        return station_records

    wq_rows = list(wq_trenches.itertuples(index=False))
    if n_jobs == 1:
        nested_records = [build_station_records(wq_row) for wq_row in wq_rows]
    else:
        nested_records = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(build_station_records)(wq_row) for wq_row in wq_rows
        )
    records = [
        record
        for station_records in nested_records
        for record in station_records
    ]

    if not records:
        return pd.DataFrame(
            columns=[
                "wq_station_code",
                "streamflow_station_code",
                "streamflow_distance_m",
                "streamflow_weight",
            ]
        )
    matches = pd.DataFrame(records)
    return matches.loc[matches["streamflow_weight"] > 0].drop_duplicates(
        subset=["wq_station_code", "streamflow_station_code"],
        keep="first",
    )


def _weighted_mean(group, value_column):
    valid = group[value_column].notna()
    if not valid.any():
        return np.nan
    weights = group.loc[valid, "streamflow_weight"]
    total_weight = weights.sum()
    if total_weight <= 0:
        return np.nan
    values = group.loc[valid, value_column]
    return float((values * weights).sum() / total_weight)


def _aggregate_streamflow_matches(water_quality_keys, station_matches, streamflow_features):
    if station_matches.empty:
        return pd.DataFrame(
            columns=[
                "wq_station_code",
                DATE_COLUMN,
                *STREAMFLOW_FEATURE_COLUMNS,
                *STREAMFLOW_DIAGNOSTIC_COLUMNS,
            ]
        )

    matched = station_matches.merge(
        streamflow_features,
        left_on="streamflow_station_code",
        right_on=STATION_CODE_COLUMN,
        how="inner",
        validate="many_to_many",
    ).drop(columns=[STATION_CODE_COLUMN])
    matched = matched.merge(
        water_quality_keys,
        on=["wq_station_code", DATE_COLUMN],
        how="inner",
        validate="many_to_many",
    )
    if matched.empty:
        return pd.DataFrame(
            columns=[
                "wq_station_code",
                DATE_COLUMN,
                *STREAMFLOW_FEATURE_COLUMNS,
                *STREAMFLOW_DIAGNOSTIC_COLUMNS,
            ]
        )

    group_columns = ["wq_station_code", DATE_COLUMN]
    aggregated = matched.groupby(group_columns, observed=True).apply(
        lambda group: pd.Series(
            {
                STREAMFLOW_DAY_COLUMN: _weighted_mean(group, STREAMFLOW_DAY_COLUMN),
                **{
                    f"streamflow_discharge_mean_{window}d": _weighted_mean(
                        group,
                        f"streamflow_discharge_mean_{window}d",
                    )
                    for window in STREAMFLOW_ROLLING_WINDOWS
                },
                "streamflow_match_count": int(
                    group["streamflow_station_code"].nunique()
                ),
                "streamflow_nonnull_day_count": int(
                    group.loc[group[STREAMFLOW_DAY_COLUMN].notna(), "streamflow_station_code"]
                    .nunique()
                ),
                "streamflow_total_weight": float(
                    group[["streamflow_station_code", "streamflow_weight"]]
                    .drop_duplicates()["streamflow_weight"]
                    .sum()
                ),
                "streamflow_nearest_distance_m": float(
                    group["streamflow_distance_m"].min()
                ),
            }
        )
    ).reset_index()
    return aggregated


def assemble_sensor_data(
    root_dir=".",
    water_quality_path=None,
    streamflow_path=None,
    stations_rivers_path=None,
    river_network_path="data/river_network",
    output_path=None,
    match_radius_m=STREAMFLOW_MATCH_RADIUS_M,
    n_jobs=None,
):
    """Assemble cleaned water-quality observations with nearby streamflow data."""
    if n_jobs is None:
        n_jobs = cpu_count()

    water_quality_path = _resolve_path(
        root_dir,
        water_quality_path,
        CLEAN_WATER_QUALITY_PARQUET,
    )
    streamflow_path = _resolve_path(root_dir, streamflow_path, CLEAN_STREAMFLOW_PARQUET)
    stations_rivers_path = _resolve_path(
        root_dir,
        stations_rivers_path,
        STATIONS_RIVERS_PARQUET,
    )
    river_network_path = _resolve_project_path(
        root_dir,
        river_network_path,
        "data/river_network",
    )
    output_path = _resolve_path(root_dir, output_path, ASSEMBLED_SENSOR_DATA_PARQUET)

    logger.info("Loading cleaned water-quality data from %s.", water_quality_path)
    water_quality = pd.read_parquet(water_quality_path)
    _validate_columns(water_quality, [STATION_CODE_COLUMN, DATETIME_COLUMN], "water-quality")
    assembled = water_quality.copy()
    assembled[STATION_CODE_COLUMN] = assembled[STATION_CODE_COLUMN].astype(str)
    assembled[DATETIME_COLUMN] = pd.to_datetime(
        assembled[DATETIME_COLUMN],
        errors="coerce",
    )
    assembled[DATE_COLUMN] = assembled[DATETIME_COLUMN].dt.normalize()

    logger.info("Loading cleaned streamflow data from %s.", streamflow_path)
    streamflow = pd.read_parquet(streamflow_path)
    streamflow_features = _prepare_streamflow_features(streamflow)

    logger.info("Loading station-river matches from %s.", stations_rivers_path)
    stations_rivers = pd.read_parquet(stations_rivers_path)
    station_trenches = _prepare_station_trenches(stations_rivers)
    station_trench_lookup = station_trenches.drop_duplicates(
        subset=[STATION_CODE_COLUMN],
        keep="first",
    )
    assembled = assembled.merge(
        station_trench_lookup,
        on=STATION_CODE_COLUMN,
        how="left",
        validate="many_to_one",
    )

    logger.info("Loading river network from %s.", river_network_path)
    network = RiverNetwork()
    network.load(str(river_network_path))

    water_quality_stations = assembled[STATION_CODE_COLUMN].dropna().unique()
    streamflow_stations = streamflow_features[STATION_CODE_COLUMN].dropna().unique()
    station_matches = _build_station_matches(
        water_quality_stations,
        streamflow_stations,
        station_trenches,
        network,
        match_radius_m,
        n_jobs=n_jobs,
    )
    logger.info(
        "Built %s water-quality to streamflow station match(es) with %s job(s).",
        len(station_matches),
        n_jobs,
    )

    water_quality_keys = assembled[
        [STATION_CODE_COLUMN, DATE_COLUMN]
    ].dropna().drop_duplicates().rename(
        columns={STATION_CODE_COLUMN: "wq_station_code"}
    )
    streamflow_aggregates = _aggregate_streamflow_matches(
        water_quality_keys,
        station_matches,
        streamflow_features,
    )

    assembled = assembled.merge(
        streamflow_aggregates,
        left_on=[STATION_CODE_COLUMN, DATE_COLUMN],
        right_on=["wq_station_code", DATE_COLUMN],
        how="left",
        validate="many_to_one",
    ).drop(columns=["wq_station_code"], errors="ignore")

    for column in ("streamflow_match_count", "streamflow_nonnull_day_count"):
        assembled[column] = assembled[column].fillna(0).astype(int)
    assembled["streamflow_total_weight"] = assembled["streamflow_total_weight"].fillna(0.0)

    assembled = assembled.sort_values(
        [STATION_CODE_COLUMN, DATETIME_COLUMN],
        kind="mergesort",
    ).set_index([STATION_CODE_COLUMN, DATETIME_COLUMN])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    assembled.to_parquet(output_path, index=True)
    logger.info("Saved assembled sensor-data parquet to %s.", output_path)
    logger.info("Output shape: %s", assembled.shape)
    return assembled
