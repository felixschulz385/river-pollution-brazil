import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.shared.slurm import resolve_n_jobs

from .constants import (
    BUCKET_COUNT_COLUMN,
    BUCKET_REACHABLE_COUNT_COLUMN,
    BUCKET_SHARE_COLUMN,
    DEFAULT_ADM2_UPSTREAM_OUTPUT_PATH,
    DEFAULT_ASSEMBLY_LAND_COVER_PATH,
    DEFAULT_RIVER_NETWORK_PATH,
    DISTANCE_BUCKET_COLUMN,
    LAND_COVER_CLASS_COLUMN,
    LAND_COVER_TOTAL_COLUMN,
    MUN_ID_COLUMN,
    SENSOR_DISTANCE_BUCKET_STARTS_KM,
    SENSOR_DISTANCE_BUCKET_WIDTH_KM,
    TRENCH_ID_COLUMN,
    YEAR_COLUMN,
    derive_mun_id_from_adm2_id,
)
from src.data.sources import river_network as rn_module
from .schema import (
    land_cover_feature_stem,
    validate_land_cover_output_columns,
)
from src.data.shared.adm2_upstream_buckets import build_adm2_upstream_bucket_parts
from src.data.shared.sensor_upstream import BUCKET_INTERSECTS_ADM2_COLUMN


logger = logging.getLogger(__name__)


def aggregate_along_rivers(
    self,
    land_cover_path=DEFAULT_ASSEMBLY_LAND_COVER_PATH,
    river_network_path=DEFAULT_RIVER_NETWORK_PATH,
    drainage_polygons_path=None,
    years=None,
    n_jobs=None,
    output_path=DEFAULT_ADM2_UPSTREAM_OUTPUT_PATH,
):
    """Aggregate land cover variables upstream of each ADM2 unit.

    The per-ADM2 upstream distance-bucket binning is delegated to the shared
    `build_adm2_upstream_bucket_parts` driver (same code path as climate's ADM2
    panel); this module only supplies the land-cover-specific reduction -- summed
    class counts and shares per (year, bucket) -- via its `reduce_adm2` callback.
    The driver streams each chunk of ADM2 units to a Parquet part file, so peak
    memory no longer grows with the number of ADM2 units.
    """
    if n_jobs is None:
        n_jobs = resolve_n_jobs()

    logger.info("Loading land cover data from %s", land_cover_path)
    land_cover_path = Path(land_cover_path)
    land_cover_df = (
        pd.read_feather(land_cover_path)
        if land_cover_path.suffix == ".feather"
        else pd.read_parquet(land_cover_path)
    )
    validate_land_cover_output_columns(land_cover_df)

    logger.info("Loading river network from %s", river_network_path)
    network = rn_module.RiverNetwork()
    network.load(str(Path(river_network_path)))

    lc_columns = [
        column
        for column in land_cover_df.columns
        if column not in [TRENCH_ID_COLUMN, YEAR_COLUMN]
    ]
    if LAND_COVER_TOTAL_COLUMN not in lc_columns:
        raise ValueError(
            f"Land-cover input is missing the `{LAND_COVER_TOTAL_COLUMN}` column required "
            "to compute bucket shares."
        )
    logger.info("Land cover columns: %s", lc_columns)

    land_cover_by_trench_year = (
        land_cover_df.groupby([TRENCH_ID_COLUMN, YEAR_COLUMN])[lc_columns].sum().sort_index()
    )
    # Indexed by trench id (non-unique: one row per trench-year) so each ADM2
    # unit's `reduce_adm2` can slice its reachable trenches with `.loc[...]`
    # instead of hash-joining the whole all-trenches x all-years table per unit.
    land_cover_by_trench_year_indexed = land_cover_by_trench_year.reset_index().set_index(
        TRENCH_ID_COLUMN, drop=False
    )

    if years is None:
        years = land_cover_by_trench_year.index.get_level_values(YEAR_COLUMN).unique().tolist()
    logger.info("Processing years: %s", years)

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

    def reduce_adm2(adm2_id, membership):
        """Sum land-cover class counts/shares per (year, distance bucket) for one ADM2 unit."""
        matched_trench_ids = land_cover_by_trench_year_indexed.index.intersection(
            membership[TRENCH_ID_COLUMN]
        )
        if matched_trench_ids.empty:
            return None

        df_matched = land_cover_by_trench_year_indexed.loc[matched_trench_ids].copy()
        bucket_by_trench = membership.set_index(TRENCH_ID_COLUMN)
        df_matched[DISTANCE_BUCKET_COLUMN] = df_matched[TRENCH_ID_COLUMN].map(
            bucket_by_trench[DISTANCE_BUCKET_COLUMN]
        )
        df_matched[BUCKET_INTERSECTS_ADM2_COLUMN] = df_matched[TRENCH_ID_COLUMN].map(
            bucket_by_trench[BUCKET_INTERSECTS_ADM2_COLUMN]
        )

        mun_id = derive_mun_id_from_adm2_id(adm2_id)
        rows = []
        for (year, bucket), df_bucket in df_matched.groupby(
            [YEAR_COLUMN, DISTANCE_BUCKET_COLUMN],
            sort=False,
        ):
            if df_bucket.empty:
                continue
            bucket_sums = df_bucket[lc_columns].sum()
            bucket_total = float(bucket_sums[LAND_COVER_TOTAL_COLUMN])
            bucket_intersects_adm2 = bool(df_bucket[BUCKET_INTERSECTS_ADM2_COLUMN].any())
            bucket_reachable = int(len(df_bucket))
            for lc_column in lc_columns:
                count_value = float(bucket_sums.get(lc_column, 0.0))
                rows.append(
                    {
                        MUN_ID_COLUMN: mun_id,
                        YEAR_COLUMN: int(year),
                        DISTANCE_BUCKET_COLUMN: int(bucket),
                        LAND_COVER_CLASS_COLUMN: land_cover_feature_stem(lc_column),
                        BUCKET_REACHABLE_COUNT_COLUMN: bucket_reachable,
                        BUCKET_COUNT_COLUMN: count_value,
                        BUCKET_SHARE_COLUMN: (
                            count_value / bucket_total if bucket_total > 0 else np.nan
                        ),
                        BUCKET_INTERSECTS_ADM2_COLUMN: bucket_intersects_adm2,
                    }
                )
        if not rows:
            return None
        return pd.DataFrame(rows, columns=ordered_columns)

    with tempfile.TemporaryDirectory(prefix="land_cover_adm2_buckets_") as temp_dir:
        part_paths = build_adm2_upstream_bucket_parts(
            network=network,
            rn_module=rn_module,
            parts_dir=Path(temp_dir) / "parts",
            n_jobs=n_jobs,
            trench_id_column=TRENCH_ID_COLUMN,
            adm2_id_column="adm2_id",
            distance_bucket_column=DISTANCE_BUCKET_COLUMN,
            bucket_width_km=SENSOR_DISTANCE_BUCKET_WIDTH_KM,
            max_bucket_start_km=SENSOR_DISTANCE_BUCKET_STARTS_KM[-1],
            reduce_adm2=reduce_adm2,
            progress_desc="ADM2 units",
        )
        if not part_paths:
            logger.warning("No results produced")
            return pd.DataFrame()
        result_df = pd.concat(
            (pd.read_parquet(part_path) for part_path in part_paths),
            ignore_index=True,
        )

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
