import logging
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from .constants import (
    ADM2_ID_COLUMN,
    REACHABLE_TRENCH_COUNT_COLUMN,
    TOTAL_WEIGHT_COLUMN,
    TRENCH_ID_COLUMN,
    YEAR_COLUMN,
)
from .preprocess import deduplicate_drainage_polygons
from .river_network_import import rn_module
from .schema import validate_land_cover_output_columns


logger = logging.getLogger(__name__)


def distance_weights(distances, kernel="gaussian", h=1000):
    """Compute distance-based kernel weights."""
    d = np.asarray(distances, dtype=float)
    if kernel == "uniform":
        w = (d <= h).astype(float)
    elif kernel == "triangular":
        w = np.clip(1 - d / h, 0, None)
    elif kernel == "epanechnikov":
        w = np.clip(1 - (d / h) ** 2, 0, None)
    elif kernel == "gaussian":
        w = np.exp(-((d / h) ** 2))
    elif kernel == "exponential":
        w = np.exp(-d / h)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    return w


AVAILABLE_KERNELS = ["uniform", "triangular", "epanechnikov", "gaussian", "exponential"]


def _explode_trench_adm2_matches(rivers):
    """Expand trench-level ADM2 match lists into one row per trench-ADM2 pair."""
    adm2_list_column = getattr(rn_module, "ADM2_LIST_COLUMN", None)
    intersection_lengths_column = getattr(
        rn_module,
        "ADM2_INTERSECTION_LENGTHS_COLUMN",
        None,
    )
    if adm2_list_column is None or adm2_list_column not in rivers.columns:
        if "adm2" not in rivers.columns:
            raise ValueError(
                "River trench data must include either `adm2` or saved ADM2 list columns "
                "for upstream aggregation."
            )
        trench_adm2 = rivers[[TRENCH_ID_COLUMN, rn_module.SYSTEM_ID_KEY, "adm2"]].copy()
        trench_adm2 = trench_adm2.dropna(subset=["adm2"])
        trench_adm2["intersection_length"] = np.nan
        return trench_adm2

    required_columns = {
        TRENCH_ID_COLUMN,
        rn_module.SYSTEM_ID_KEY,
        adm2_list_column,
        intersection_lengths_column,
    }
    missing_columns = required_columns.difference(rivers.columns)
    if missing_columns:
        raise ValueError(
            "River trench data is missing ADM2 match columns: "
            f"{sorted(missing_columns)}."
        )

    trench_adm2 = rivers[
        [
            TRENCH_ID_COLUMN,
            rn_module.SYSTEM_ID_KEY,
            adm2_list_column,
            intersection_lengths_column,
        ]
    ].copy()
    trench_adm2 = trench_adm2.rename(
        columns={
            adm2_list_column: "adm2",
            intersection_lengths_column: "intersection_length",
        }
    )
    trench_adm2["adm2"] = trench_adm2["adm2"].apply(
        lambda values: values if isinstance(values, list) else []
    )
    trench_adm2["intersection_length"] = trench_adm2["intersection_length"].apply(
        lambda values: values if isinstance(values, list) else []
    )
    trench_adm2 = trench_adm2.explode(
        ["adm2", "intersection_length"],
        ignore_index=True,
    )
    return trench_adm2.dropna(subset=["adm2"])


def aggregate_along_rivers(
    self,
    land_cover_path,
    river_network_path,
    drainage_polygons_path,
    kernel="gaussian",
    h=1000000,
    years=None,
    n_jobs=None,
    output_path="land_cover_river_aggregated.feather",
):
    """Aggregate land cover variables upstream of each ADM2 unit."""
    if n_jobs is None:
        n_jobs = cpu_count()

    if kernel not in AVAILABLE_KERNELS:
        raise ValueError(f"Unknown kernel: {kernel}. Available: {AVAILABLE_KERNELS}")

    logger.info("Loading land cover data from %s", land_cover_path)
    land_cover_df = pd.read_feather(land_cover_path)
    validate_land_cover_output_columns(land_cover_df)

    logger.info("Loading river network from %s", river_network_path)
    network = rn_module.RiverNetwork()
    network.load(str(Path(river_network_path)))

    if not network.trench_reachability_matrices:
        raise ValueError("River network must have trench reachability data computed.")
    if network.trenches is None:
        raise ValueError("River network must include trench data.")

    if network.drainage_areas is not None:
        drainage_polygons = deduplicate_drainage_polygons(network.drainage_areas.copy())
    else:
        raise ValueError("River network must include drainage polygon data.")

    rivers = network.trenches
    missing_drainage_columns = {TRENCH_ID_COLUMN}.difference(drainage_polygons.columns)
    if missing_drainage_columns:
        raise ValueError(
            "Drainage polygons are missing required columns: "
            f"{sorted(missing_drainage_columns)}."
        )

    trench_adm2_matches = _explode_trench_adm2_matches(rivers)
    trench_columns = [TRENCH_ID_COLUMN, rn_module.SYSTEM_ID_KEY, "adm2"]
    trench_lookup = drainage_polygons[[TRENCH_ID_COLUMN]].merge(
        trench_adm2_matches[trench_columns].drop_duplicates(),
        on=TRENCH_ID_COLUMN,
        how="left",
        validate="one_to_many",
    )
    trench_lookup = trench_lookup.dropna(subset=[rn_module.SYSTEM_ID_KEY])

    lc_columns = [
        column
        for column in land_cover_df.columns
        if column not in [TRENCH_ID_COLUMN, YEAR_COLUMN]
    ]
    logger.info("Land cover columns: %s", lc_columns)

    land_cover_by_trench_year = land_cover_df.groupby(
        [TRENCH_ID_COLUMN, YEAR_COLUMN]
    )[lc_columns].sum().sort_index()

    if years is None:
        years = land_cover_by_trench_year.index.get_level_values(YEAR_COLUMN).unique().tolist()

    logger.info("Processing years: %s", years)

    adm2_units = trench_lookup["adm2"].dropna().unique()
    logger.info("Processing %d ADM2 units", len(adm2_units))

    trench_index_columns = {
        TRENCH_ID_COLUMN,
        rn_module.SYSTEM_ID_KEY,
        rn_module.TRENCH_INDEX_COLUMN,
    }
    missing_trench_index_columns = trench_index_columns.difference(rivers.columns)
    if missing_trench_index_columns:
        raise ValueError(
            "River trench data is missing matrix index columns: "
            f"{sorted(missing_trench_index_columns)}. "
            "Recompute river matrices with RiverNetwork.compute_distance_matrices()."
        )

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
        """Process a single ADM2 unit for all years."""
        try:
            adm2_trenches = trench_lookup.loc[
                trench_lookup["adm2"] == adm2_id,
                [TRENCH_ID_COLUMN, rn_module.SYSTEM_ID_KEY],
            ].drop_duplicates()

            if adm2_trenches.empty:
                return None

            trench_distance_frames = []
            for system_id, system_adm2_trenches in adm2_trenches.groupby(
                rn_module.SYSTEM_ID_KEY
            ):
                system_id = int(system_id)
                system_trench_ids = system_trench_id_arrays.get(
                    system_id,
                    np.asarray([], dtype=np.int64),
                )
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

                system_reachability = network.trench_reachability_matrices[system_id][
                    seed_positions, :
                ].tocsr()
                system_distance = network.trench_distance_matrices[system_id][
                    seed_positions, :
                ].tocsr()

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
                            "upstream_distance": min_distances[reachable_mask],
                        }
                    )
                )

            if not trench_distance_frames:
                return None

            trench_distance_lookup = (
                pd.concat(trench_distance_frames, ignore_index=True)
                .groupby(TRENCH_ID_COLUMN, as_index=False)["upstream_distance"]
                .min()
                .set_index(TRENCH_ID_COLUMN)["upstream_distance"]
            )

            matched_mask = land_cover_by_trench_year.index.get_level_values(
                TRENCH_ID_COLUMN
            ).isin(trench_distance_lookup.index)
            df_matched = land_cover_by_trench_year.loc[matched_mask].reset_index()
            if len(df_matched) == 0:
                return None

            df_matched["upstream_distance"] = df_matched[TRENCH_ID_COLUMN].map(
                trench_distance_lookup
            )
            weights = distance_weights(
                df_matched["upstream_distance"].to_numpy(),
                kernel=kernel,
                h=h,
            )

            results = []
            for year in years:
                try:
                    year_mask = df_matched[YEAR_COLUMN] == year
                    df_year = df_matched.loc[year_mask]
                    w_year = weights[year_mask.to_numpy()]

                    if len(df_year) == 0 or np.sum(w_year) == 0:
                        continue

                    weighted_sum = np.sum(
                        df_year[lc_columns].to_numpy() * w_year[:, None],
                        axis=0,
                    )
                    result = {col: val for col, val in zip(lc_columns, weighted_sum)}
                    result[ADM2_ID_COLUMN] = adm2_id
                    result[YEAR_COLUMN] = int(year)
                    result[REACHABLE_TRENCH_COUNT_COLUMN] = len(df_year)
                    result[TOTAL_WEIGHT_COLUMN] = float(np.sum(w_year))
                    results.append(result)
                except Exception as e:
                    logger.warning("Error processing ADM2 %s, year %s: %s", adm2_id, year, e)
                    continue

            return results
        except Exception as e:
            logger.warning("Error processing ADM2 %s: %s", adm2_id, e)
            return None

    logger.info("Processing %d ADM2 units with %s workers", len(adm2_units), n_jobs)
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_adm2)(adm2_id)
        for adm2_id in tqdm(adm2_units, desc="ADM2 units")
    )

    all_results = []
    for result in results:
        if result is not None:
            all_results.extend(result)

    if not all_results:
        logger.warning("No results produced")
        return pd.DataFrame()

    result_df = pd.DataFrame(all_results)
    result_df = result_df.sort_values([ADM2_ID_COLUMN, YEAR_COLUMN]).reset_index(drop=True)

    output_path = Path(output_path)
    result_df.to_feather(output_path)
    logger.info("Results saved to %s", output_path)
    logger.info("Output shape: %s", result_df.shape)

    return result_df
