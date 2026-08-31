import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.data.shared.slurm import resolve_n_jobs

from ..constants import get_processed_dir
from ..schema import (
    ASSEMBLED_SENSOR_DATA_PARQUET,
    CLEAN_STREAMFLOW_PARQUET,
    CLEAN_WATER_QUALITY_PARQUET,
    DATETIME_COLUMN,
    STATIONS_FILTERED_PARQUET,
    STATIONS_TRENCHES_COLUMNS,
    STATIONS_TRENCHES_PARQUET,
    STREAMFLOW_MATCH_RADIUS_M,
    STREAMFLOW_ROLLING_WINDOWS,
)

from src.data.sources.river_network import RiverNetwork
from src.data.sources import river_network as rn_module
from src.data.sources.river_network.constants import PROCESSED_DIR as RIVER_NETWORK_PROCESSED_DIR
from src.data.sources.gadm.constants import DEFAULT_SIMPLIFIED_GADM_PATH
from src.data.shared.sensor_upstream import prepare_entity_links, sparse_row


logger = logging.getLogger(__name__)

STATION_CODE_COLUMN = "station_code"
DATE_COLUMN = "date"
TRENCH_ID_COLUMN = "trench_id"
DEFAULT_BRAZIL_BOUNDARY_LAYER = "ADM_ADM_0"
BRAZIL_PROJECTED_CRS = 5641
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


def _resolve_path(root_dir, path, default_filename, stage="extract"):
    if path is not None:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = Path(root_dir) / candidate
        return candidate
    return get_processed_dir(root_dir, stage=stage) / default_filename


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


def _filter_stations_to_brazil(stations_geo, brazil_boundary_path):
    """Precise Brazil-boundary filter (GADM), applied here rather than at
    fetch time since only assembly requires GADM as a dependency."""
    brazil = gpd.read_file(
        brazil_boundary_path,
        layer=DEFAULT_BRAZIL_BOUNDARY_LAYER,
        engine="pyogrio",
    )
    brazil_geometry = brazil.union_all()
    in_bounds = stations_geo.within(brazil_geometry)
    return stations_geo.loc[in_bounds].copy()


def _join_stations_to_trenches(stations_geo, network):
    """Match each station to its nearest river trench."""
    trenches = network.trenches[[TRENCH_ID_COLUMN, "geometry"]].copy()
    station_matches = gpd.sjoin_nearest(
        stations_geo[[STATION_CODE_COLUMN, "geometry"]].to_crs(BRAZIL_PROJECTED_CRS),
        trenches.to_crs(BRAZIL_PROJECTED_CRS),
        how="left",
        distance_col="distance_to_river",
    )
    station_matches = pd.DataFrame(
        station_matches[[STATION_CODE_COLUMN, TRENCH_ID_COLUMN, "distance_to_river"]]
    ).sort_values([STATION_CODE_COLUMN, "distance_to_river"]).drop_duplicates(
        subset=[STATION_CODE_COLUMN], keep="first"
    )
    return stations_geo.merge(
        station_matches.drop(columns=["distance_to_river"]),
        on=STATION_CODE_COLUMN,
        how="left",
    )


def _prepare_station_trenches(stations_rivers):
    return prepare_entity_links(
        stations_rivers,
        entity_column=STATION_CODE_COLUMN,
        location_column=TRENCH_ID_COLUMN,
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
    # A source station-date can appear more than once (e.g. multiple consistency
    # levels/re-imports in the raw export); collapse to one row per station-date
    # before rolling so a duplicate doesn't double-weight that date in the rolling
    # window or in the cross-station weighted average computed downstream.
    features = features.drop_duplicates(subset=[STATION_CODE_COLUMN, DATE_COLUMN], keep="first")
    features[STREAMFLOW_DAY_COLUMN] = features[DISCHARGE_COLUMN]

    for window in STREAMFLOW_ROLLING_WINDOWS:
        column = f"streamflow_discharge_mean_{window}d"
        # A row-count window (`.rolling(window=int)`) counts *rows*, not
        # calendar days -- a station with date gaps (outages, partial
        # monthly imports) would get a "7-day"/"31-day" mean that silently
        # spans however many calendar days those rows actually cover. Use a
        # date-offset window (`f"{window}D"` with `on=DATE_COLUMN`) instead,
        # so the window is anchored to actual elapsed time.
        #
        # This changes the rolling result's index from `features`'s own row
        # index to a (station, date) MultiIndex, so it can't be assigned
        # back by index alignment; `.to_numpy()` + positional assignment is
        # safe here because `features` is already sorted by
        # `[STATION_CODE_COLUMN, DATE_COLUMN]` above, which is exactly the
        # row order `groupby(..., observed=True)` (default `sort=True`,
        # stable within-group order) produces.
        rolled = (
            features.groupby(STATION_CODE_COLUMN, observed=True)
            .rolling(f"{window}D", on=DATE_COLUMN, min_periods=1)[DISCHARGE_COLUMN]
            .mean()
        )
        features[column] = rolled.to_numpy()

    return features[
        [STATION_CODE_COLUMN, DATE_COLUMN, *STREAMFLOW_FEATURE_COLUMNS]
    ].reset_index(drop=True)


def _collapse_same_day_observations(water_quality):
    _validate_columns(
        water_quality,
        [STATION_CODE_COLUMN, DATETIME_COLUMN, DATE_COLUMN],
        "water-quality",
    )
    if water_quality.empty:
        return water_quality

    collapsed = water_quality.sort_values(
        [STATION_CODE_COLUMN, DATE_COLUMN, DATETIME_COLUMN],
        kind="mergesort",
    )
    group_columns = [STATION_CODE_COLUMN, DATE_COLUMN]
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
        [STATION_CODE_COLUMN, DATE_COLUMN, DATETIME_COLUMN],
        kind="mergesort",
    )
    return collapsed.loc[:, water_quality.columns].reset_index(drop=True)


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

    upstream_reach = sparse_row(reachability, target_position)
    upstream_distances = _sparse_distance_lookup(sparse_row(distances, target_position))
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
        n_jobs = resolve_n_jobs()

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


def _aggregate_streamflow_matches(water_quality_keys, station_matches, streamflow_features):
    empty_columns = [
        "wq_station_code",
        DATE_COLUMN,
        *STREAMFLOW_FEATURE_COLUMNS,
        *STREAMFLOW_DIAGNOSTIC_COLUMNS,
    ]
    if station_matches.empty:
        return pd.DataFrame(columns=empty_columns)

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
        return pd.DataFrame(columns=empty_columns)

    group_columns = ["wq_station_code", DATE_COLUMN]

    # Collapse to one row per (group, streamflow_station_code) *before* weighting.
    # A duplicate row for the same matched station -- e.g. from a many-to-many
    # join, or duplicate source data upstream -- must contribute once to the
    # weighted average across *distinct* matched stations, not scale with its
    # duplicate count: summing raw (possibly fanned-out) rows would inflate that
    # station's weight relative to the other matched stations in the group.
    unique_station_rows = matched.drop_duplicates(
        subset=[*group_columns, "streamflow_station_code"]
    ).copy()

    # Weighted-mean numerator/denominator per feature column, summed per group
    # over the deduplicated per-station rows.
    sum_columns = []
    for feature_column in STREAMFLOW_FEATURE_COLUMNS:
        valid = unique_station_rows[feature_column].notna()
        weight = unique_station_rows["streamflow_weight"].where(valid, 0.0)
        unique_station_rows[f"_{feature_column}_num"] = (
            unique_station_rows[feature_column].fillna(0.0) * weight
        )
        unique_station_rows[f"_{feature_column}_den"] = weight
        sum_columns.extend([f"_{feature_column}_num", f"_{feature_column}_den"])

    sums = unique_station_rows.groupby(group_columns, observed=True)[sum_columns].sum()
    feature_frame = pd.DataFrame(index=sums.index)
    for feature_column in STREAMFLOW_FEATURE_COLUMNS:
        denominator = sums[f"_{feature_column}_den"]
        feature_frame[feature_column] = (sums[f"_{feature_column}_num"] / denominator).where(
            denominator > 0
        )

    diagnostics = pd.DataFrame(
        {
            "streamflow_match_count": matched.groupby(group_columns, observed=True)[
                "streamflow_station_code"
            ].nunique(),
            "streamflow_nonnull_day_count": (
                matched.loc[matched[STREAMFLOW_DAY_COLUMN].notna()]
                .groupby(group_columns, observed=True)["streamflow_station_code"]
                .nunique()
            ),
            "streamflow_total_weight": unique_station_rows.groupby(
                group_columns, observed=True
            )["streamflow_weight"].sum(),
            "streamflow_nearest_distance_m": matched.groupby(group_columns, observed=True)[
                "streamflow_distance_m"
            ].min(),
        }
    )
    diagnostics["streamflow_match_count"] = diagnostics["streamflow_match_count"].astype(int)
    diagnostics["streamflow_nonnull_day_count"] = (
        diagnostics["streamflow_nonnull_day_count"].fillna(0).astype(int)
    )

    aggregated = pd.concat([feature_frame, diagnostics], axis=1).reset_index()
    return aggregated[empty_columns]


def assemble_sensor_data(
    root_dir=".",
    water_quality_path=None,
    water_quality_frame=None,
    streamflow_path=None,
    streamflow_frame=None,
    stations_path=None,
    stations_frame=None,
    stations_rivers_path=None,
    river_network_path=RIVER_NETWORK_PROCESSED_DIR,
    brazil_boundary_path=None,
    output_path=None,
    match_radius_m=STREAMFLOW_MATCH_RADIUS_M,
    n_jobs=None,
):
    """Assemble cleaned water-quality observations with nearby streamflow data.

    This is the stage where GADM (Brazil-boundary filter) and river_network
    (station-to-trench join, reachability matching) are required -- fetch and
    extract-stage preprocessing have no such dependency.

    `water_quality_frame`/`streamflow_frame`/`stations_frame` let a caller
    that just cleaned this data in memory (the normal
    `SensorData.preprocess()` flow) pass it straight through instead of
    writing then re-reading a parquet file -- none of the three cleaned
    inputs have a consumer of their own outside this function (the output
    this function writes, `sensor_data.parquet`, is the canonical file
    land_cover/climate read), so none are written to disk
    separately. When not given, the corresponding `*_path` is read from disk
    instead (e.g. for standalone reuse against a manually-placed file).
    """
    if n_jobs is None:
        n_jobs = resolve_n_jobs()

    if water_quality_frame is None:
        water_quality_path = _resolve_path(
            root_dir,
            water_quality_path,
            CLEAN_WATER_QUALITY_PARQUET,
            stage="extract",
        )
    if streamflow_frame is None:
        streamflow_path = _resolve_path(
            root_dir, streamflow_path, CLEAN_STREAMFLOW_PARQUET, stage="extract"
        )
    if stations_frame is None:
        stations_path = _resolve_path(
            root_dir,
            stations_path,
            STATIONS_FILTERED_PARQUET,
            stage="extract",
        )
    # Unlike the other paths above, `stations_rivers_path` names an *output*:
    # the station-to-trench join is computed here (it needs GADM and
    # river_network), not read from a pre-built file.
    stations_rivers_output_path = _resolve_path(
        root_dir,
        stations_rivers_path,
        STATIONS_TRENCHES_PARQUET,
        stage="aggregate",
    )
    river_network_path = _resolve_project_path(
        root_dir,
        river_network_path,
        RIVER_NETWORK_PROCESSED_DIR,
    )
    brazil_boundary_path = _resolve_project_path(
        root_dir,
        brazil_boundary_path,
        DEFAULT_SIMPLIFIED_GADM_PATH,
    )
    output_path = _resolve_path(
        root_dir, output_path, ASSEMBLED_SENSOR_DATA_PARQUET, stage="aggregate"
    )

    if water_quality_frame is not None:
        logger.info("Using in-memory cleaned water-quality data (%s row(s)).", len(water_quality_frame))
        water_quality = water_quality_frame.copy()
    else:
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
    raw_observation_count = len(assembled)
    assembled = _collapse_same_day_observations(assembled)
    collapsed_observation_count = raw_observation_count - len(assembled)
    if collapsed_observation_count > 0:
        logger.info(
            "Collapsed %s duplicate water-quality observation row(s) within station-day groups.",
            collapsed_observation_count,
        )

    if streamflow_frame is not None:
        logger.info("Using in-memory cleaned streamflow data (%s row(s)).", len(streamflow_frame))
        streamflow = streamflow_frame.copy()
    else:
        logger.info("Loading cleaned streamflow data from %s.", streamflow_path)
        streamflow = pd.read_parquet(streamflow_path)
    streamflow_features = _prepare_streamflow_features(streamflow)

    logger.info("Loading river network from %s.", river_network_path)
    network = RiverNetwork()
    network.load(str(river_network_path))
    _validate_network(network)

    if stations_frame is not None:
        logger.info("Using in-memory station inventory (%s row(s)).", len(stations_frame))
        stations_geo = stations_frame.copy()
    else:
        logger.info("Loading station inventory from %s.", stations_path)
        stations_geo = gpd.read_parquet(stations_path)
    _validate_columns(stations_geo, [STATION_CODE_COLUMN], "stations")
    stations_geo[STATION_CODE_COLUMN] = stations_geo[STATION_CODE_COLUMN].astype(str)

    logger.info("Filtering stations to within Brazil using %s.", brazil_boundary_path)
    stations_geo = _filter_stations_to_brazil(stations_geo, brazil_boundary_path)
    logger.info("Joining %s in-bounds station(s) to their nearest river trench.", len(stations_geo))
    stations_rivers = _join_stations_to_trenches(stations_geo, network)

    stations_rivers_output_path.parent.mkdir(parents=True, exist_ok=True)
    stations_rivers.loc[:, STATIONS_TRENCHES_COLUMNS].to_parquet(
        stations_rivers_output_path, index=False
    )
    logger.info("Wrote station-to-trench join to %s.", stations_rivers_output_path)

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
        [STATION_CODE_COLUMN, DATE_COLUMN, DATETIME_COLUMN],
        kind="mergesort",
    ).set_index([STATION_CODE_COLUMN, DATE_COLUMN])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    assembled.to_parquet(output_path, index=True)
    logger.info("Saved assembled sensor-data parquet to %s.", output_path)
    logger.info("Output shape: %s", assembled.shape)
    return assembled
