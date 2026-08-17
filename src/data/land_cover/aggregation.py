import logging
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from .constants import (
    BUCKET_COUNT_COLUMN,
    BUCKET_REACHABLE_COUNT_COLUMN,
    BUCKET_SHARE_COLUMN,
    DEFAULT_ASSEMBLY_LAND_COVER_PATH,
    DEFAULT_RIVER_NETWORK_PATH,
    DISTANCE_BUCKET_COLUMN,
    LAND_COVER_CLASS_COLUMN,
    LAND_COVER_CLASS_PREFIX,
    LAND_COVER_TOTAL_COLUMN,
    MUN_ID_COLUMN,
    SENSOR_DISTANCE_BUCKET_WIDTH_KM,
    TRENCH_ID_COLUMN,
    YEAR_COLUMN,
    derive_mun_id_from_adm2_id,
)
from src.data import river_network as rn_module
from .schema import validate_land_cover_output_columns
from src.data.shared.sensor_upstream import (
    BUCKET_INTERSECTS_ADM2_COLUMN,
    build_group_index_lookup,
    prepare_trench_adm2_matches,
    resolve_multi_seed_reachable_distances,
    validate_network_index_tables,
)
from src.data.shared.spatial_tabular import deduplicate_drainage_polygons


logger = logging.getLogger(__name__)


def _normalize_network_frame(frame):
    """Return a copy with a simple RangeIndex to avoid index/column ambiguity."""
    if frame is None:
        return None
    return frame.reset_index(drop=True).copy()


def _build_trench_length_lookup(rivers):
    """Return trench lengths keyed by trench id."""
    required_columns = {TRENCH_ID_COLUMN, "distance"}
    missing_columns = required_columns.difference(rivers.columns)
    if missing_columns:
        raise ValueError(
            "River trench data is missing required length columns: "
            f"{sorted(missing_columns)}."
        )
    return (
        rivers[[TRENCH_ID_COLUMN, "distance"]]
        .drop_duplicates(subset=[TRENCH_ID_COLUMN], keep="first")
        .rename(columns={"distance": "trench_length_km"})
    )


def _apply_shifted_origin(trench_distance_lookup, trench_lengths):
    """Shift distances so zero is the upstream end of the ADM2-touching trench."""
    shifted = trench_distance_lookup.reset_index().merge(
        trench_lengths,
        on=TRENCH_ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
    if shifted["trench_length_km"].isna().any():
        missing_ids = shifted.loc[
            shifted["trench_length_km"].isna(),
            TRENCH_ID_COLUMN,
        ].tolist()
        raise ValueError(
            "Missing trench length(s) for shifted upstream-distance calculation: "
            f"{missing_ids[:10]}"
        )
    shifted["adjusted_distance"] = (
        shifted["upstream_distance"] - shifted["trench_length_km"]
    )
    return shifted.set_index(TRENCH_ID_COLUMN, drop=True)


def _assign_distance_bucket(distances):
    """Return 25 km lower-bound bucket labels on the shifted distance scale."""
    distances = np.asarray(distances, dtype=float)
    return (
        np.floor(distances / SENSOR_DISTANCE_BUCKET_WIDTH_KM) * SENSOR_DISTANCE_BUCKET_WIDTH_KM
    ).astype(int)


def _land_cover_feature_stem(lc_column):
    """Return the integer-coded land-cover class id used in long outputs."""
    if lc_column == LAND_COVER_TOTAL_COLUMN:
        return -1
    if lc_column.startswith(LAND_COVER_CLASS_PREFIX):
        return int(lc_column.removeprefix(LAND_COVER_CLASS_PREFIX))
    raise ValueError(f"Unsupported land-cover column for long output: {lc_column}")


def aggregate_along_rivers(
    self,
    land_cover_path=DEFAULT_ASSEMBLY_LAND_COVER_PATH,
    river_network_path=DEFAULT_RIVER_NETWORK_PATH,
    drainage_polygons_path=None,
    years=None,
    n_jobs=None,
    output_path="land_cover_river_aggregated.parquet",
):
    """Aggregate land cover variables upstream of each ADM2 unit."""
    if n_jobs is None:
        n_jobs = cpu_count()

    logger.info("Loading land cover data from %s", land_cover_path)
    land_cover_df = pd.read_feather(land_cover_path)
    validate_land_cover_output_columns(land_cover_df)

    logger.info("Loading river network from %s", river_network_path)
    network = rn_module.RiverNetwork()
    network.load(str(Path(river_network_path)))
    network.trenches = _normalize_network_frame(network.trenches)
    if network.drainage_areas is not None:
        network.drainage_areas = _normalize_network_frame(network.drainage_areas)
    if getattr(network, "trench_adm2_table", None) is not None:
        network.trench_adm2_table = _normalize_network_frame(network.trench_adm2_table)

    if not network.trench_reachability_matrices:
        raise ValueError("River network must have trench reachability data computed.")
    if network.trenches is None:
        raise ValueError("River network must include trench data.")

    if network.drainage_areas is not None:
        drainage_polygons = deduplicate_drainage_polygons(network.drainage_areas.copy())
    else:
        raise ValueError("River network must include drainage polygon data.")

    rivers = network.trenches
    trench_lengths = _build_trench_length_lookup(rivers)
    missing_drainage_columns = {TRENCH_ID_COLUMN}.difference(drainage_polygons.columns)
    if missing_drainage_columns:
        raise ValueError(
            "Drainage polygons are missing required columns: "
            f"{sorted(missing_drainage_columns)}."
        )

    trench_adm2_matches = prepare_trench_adm2_matches(
        network,
        rn_module=rn_module,
        trench_id_column=TRENCH_ID_COLUMN,
    )
    trench_columns = [TRENCH_ID_COLUMN, rn_module.SYSTEM_ID_KEY, "adm2"]
    trench_lookup = drainage_polygons[[TRENCH_ID_COLUMN]].merge(
        trench_adm2_matches[trench_columns].drop_duplicates(),
        on=TRENCH_ID_COLUMN,
        how="left",
        validate="one_to_many",
    )
    trench_lookup = trench_lookup.dropna(subset=[rn_module.SYSTEM_ID_KEY])
    adm2_groups = {
        adm2_id: adm2_rows[[TRENCH_ID_COLUMN, rn_module.SYSTEM_ID_KEY]].drop_duplicates()
        for adm2_id, adm2_rows in trench_lookup.groupby("adm2", sort=False)
    }

    lc_columns = [
        column
        for column in land_cover_df.columns
        if column not in [TRENCH_ID_COLUMN, YEAR_COLUMN]
    ]
    logger.info("Land cover columns: %s", lc_columns)

    land_cover_by_trench_year = land_cover_df.groupby(
        [TRENCH_ID_COLUMN, YEAR_COLUMN]
    )[lc_columns].sum().sort_index()
    land_cover_by_trench_year_reset = land_cover_by_trench_year.reset_index()
    land_cover_by_trench_year_indexed = land_cover_by_trench_year_reset.set_index(
        TRENCH_ID_COLUMN,
        drop=False,
    )

    if years is None:
        years = land_cover_by_trench_year.index.get_level_values(YEAR_COLUMN).unique().tolist()

    logger.info("Processing years: %s", years)

    adm2_units = list(adm2_groups.keys())
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

    def process_adm2(adm2_id):
        """Process a single ADM2 unit for all years."""
        try:
            adm2_trenches = adm2_groups.get(adm2_id)

            if adm2_trenches is None or adm2_trenches.empty:
                return None
            intersecting_trench_ids = set(adm2_trenches[TRENCH_ID_COLUMN])

            trench_distance_lookup = resolve_multi_seed_reachable_distances(
                network,
                adm2_trenches,
                location_column=TRENCH_ID_COLUMN,
                distance_column="upstream_distance",
                system_column=rn_module.SYSTEM_ID_KEY,
                position_column=rn_module.TRENCH_INDEX_COLUMN,
                system_location_arrays=system_location_arrays,
                system_positions=system_positions,
            )
            if trench_distance_lookup.empty:
                return None
            trench_distance_lookup = trench_distance_lookup.set_index(TRENCH_ID_COLUMN)[
                "upstream_distance"
            ]
            trench_distance_lookup = _apply_shifted_origin(
                trench_distance_lookup,
                trench_lengths,
            )

            matched_trench_ids = trench_distance_lookup.index.intersection(
                land_cover_by_trench_year_indexed.index
            )
            df_matched = (
                land_cover_by_trench_year_indexed.loc[matched_trench_ids]
                .copy()
                .reset_index(drop=True)
            )
            if len(df_matched) == 0:
                return None

            df_matched["upstream_distance"] = df_matched[TRENCH_ID_COLUMN].map(
                trench_distance_lookup["upstream_distance"]
            )
            df_matched["adjusted_distance"] = df_matched[TRENCH_ID_COLUMN].map(
                trench_distance_lookup["adjusted_distance"]
            )
            df_matched[DISTANCE_BUCKET_COLUMN] = _assign_distance_bucket(
                df_matched["adjusted_distance"].to_numpy()
            )

            results = []
            for (year, bucket), df_bucket in df_matched.groupby(
                [YEAR_COLUMN, DISTANCE_BUCKET_COLUMN],
                sort=False,
            ):
                try:
                    if len(df_bucket) == 0:
                        continue

                    bucket_sums = df_bucket[lc_columns].sum()
                    bucket_total = float(bucket_sums.get(LAND_COVER_TOTAL_COLUMN, 0.0))
                    bucket_intersects_adm2 = bool(
                        df_bucket[TRENCH_ID_COLUMN].isin(intersecting_trench_ids).any()
                    )
                    for lc_column in lc_columns:
                        count_value = float(bucket_sums.get(lc_column, 0.0))
                        results.append(
                            {
                                MUN_ID_COLUMN: derive_mun_id_from_adm2_id(adm2_id),
                                YEAR_COLUMN: int(year),
                                DISTANCE_BUCKET_COLUMN: int(bucket),
                                LAND_COVER_CLASS_COLUMN: _land_cover_feature_stem(lc_column),
                                BUCKET_REACHABLE_COUNT_COLUMN: int(len(df_bucket)),
                                BUCKET_COUNT_COLUMN: count_value,
                                BUCKET_SHARE_COLUMN: (
                                    count_value / bucket_total if bucket_total > 0 else np.nan
                                ),
                                BUCKET_INTERSECTS_ADM2_COLUMN: bucket_intersects_adm2,
                            }
                        )
                except Exception as e:
                    logger.warning(
                        "Error processing ADM2 %s, year %s, bucket %s: %s",
                        adm2_id,
                        year,
                        bucket,
                        e,
                    )
                    continue

            return results
        except Exception as e:
            logger.warning("Error processing ADM2 %s: %s", adm2_id, e)
            return None

    logger.info("Processing %d ADM2 units with %s workers", len(adm2_units), n_jobs)
    results = Parallel(n_jobs=n_jobs, backend="threading", verbose=10)(
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
    ordered_columns = [
        MUN_ID_COLUMN,
        YEAR_COLUMN,
        DISTANCE_BUCKET_COLUMN,
        LAND_COVER_CLASS_COLUMN,
        BUCKET_REACHABLE_COUNT_COLUMN,
        BUCKET_COUNT_COLUMN,
        BUCKET_SHARE_COLUMN,
        BUCKET_INTERSECTS_ADM2_COLUMN,
    ]
    result_df = result_df.loc[:, ordered_columns]
    result_df = result_df.sort_values(
        [MUN_ID_COLUMN, YEAR_COLUMN, DISTANCE_BUCKET_COLUMN, LAND_COVER_CLASS_COLUMN]
    ).reset_index(drop=True)

    output_path = Path(output_path)
    if output_path.suffix == ".feather":
        result_df.to_feather(output_path)
    else:
        result_df.to_parquet(output_path, index=False)
    logger.info("Results saved to %s", output_path)
    logger.info("Output shape: %s", result_df.shape)

    return result_df
