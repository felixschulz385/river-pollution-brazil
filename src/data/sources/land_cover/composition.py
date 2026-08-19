import duckdb

from src.data.shared.sensor_upstream import INV_SQRT_DISTANCE_KERNEL, bucket_kernel_weights

from .constants import (
    LAND_COVER_ALR_CLASSES,
    LAND_COVER_CLASS_CODE_AGRICULTURE,
    LAND_COVER_CLASS_CODE_FARMING_PARENT,
    LAND_COVER_CLASS_CODE_FOREST,
    LAND_COVER_CLASS_CODE_MINING,
    LAND_COVER_CLASS_CODE_NONFOREST_NAT,
    LAND_COVER_CLASS_CODE_OTHER_RAW,
    LAND_COVER_CLASS_CODE_PASTURE,
    LAND_COVER_CLASS_CODE_URBAN,
    LAND_COVER_CLASS_CODE_URBAN_PARENT,
    LAND_COVER_CLASS_CODE_WATER,
    LAND_COVER_COMPOSITION_BUCKET_MAP,
    LAND_COVER_COMPOSITION_PSEUDOCOUNT,
    LAND_COVER_LEAF_CLASSES,
)


def _build_bucket_map_table(con, bucket_map, raw_weights):
    """Load the per-bucket kernel weight table used to weight buckets by distance."""
    bucket_rows = ", ".join(
        f"({bucket}, '{label}', {raw_weights[bucket]})"
        for bucket, (label, _midpoint) in bucket_map.items()
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE bucket_map AS
        SELECT * FROM (VALUES {bucket_rows}) AS t(bucket, bucket_label, kernel_weight)
        """
    )


def compute_kernel_weighted_composition(
    bucket_df,
    *,
    entity_columns,
    kernel=INV_SQRT_DISTANCE_KERNEL,
    bandwidth=None,
    pseudocount=LAND_COVER_COMPOSITION_PSEUDOCOUNT,
    bucket_map=None,
):
    """Collapse a long, distance-bucketed land-cover table into per-entity composition shares.

    Turns a `[*entity_columns, bucket, land_cover_class, n, cnt, share]` table (as
    produced by both the sensor and ADM2 land-cover assembly variants) into one row
    per entity with kernel-weighted `lc_*`/`lc_nat` composition shares and
    additive-log-ratio (`alr_*`) transforms relative to natural land. The c3/c4
    mismatch resolution and ALR transform are ported from
    `src/analysis/notebooks/pollution.ipynb`, which used the `inv_sqrt_distance`
    default kernel; pass `kernel`/`bandwidth` to instead use one of
    `shared.sensor_upstream.DISTANCE_KERNELS` (e.g. to align with climate's ADM2
    aggregation, which weights the same way).
    """
    entity_columns = list(entity_columns)
    bucket_map = LAND_COVER_COMPOSITION_BUCKET_MAP if bucket_map is None else bucket_map
    raw_weights = dict(
        zip(
            bucket_map,
            bucket_kernel_weights(
                [midpoint for _label, midpoint in bucket_map.values()],
                kernel=kernel,
                bandwidth=bandwidth,
            ),
        )
    )

    entity_cols_sql = ", ".join(entity_columns)
    unpivot_sql = "\n            UNION ALL\n            ".join(
        f"SELECT {entity_cols_sql}, bucket, '{leaf_class}' AS class_short, {leaf_class} AS share "
        "FROM bucket_resolved"
        for leaf_class in LAND_COVER_LEAF_CLASSES
    )
    wide_cols_sql = ",\n                ".join(
        f"MAX(dw_share) FILTER (WHERE class_short = '{leaf_class}') AS lc_{leaf_class}"
        for leaf_class in LAND_COVER_LEAF_CLASSES
    )
    alr_cols_sql = ", ".join(
        f"LN((lc_{alr_class} + {pseudocount}) / (lc_forest + lc_nonforest_nat + {pseudocount})) AS alr_{alr_class}"
        for alr_class in LAND_COVER_ALR_CLASSES
    )
    select_lc_cols = ", ".join(f"lc_{leaf_class}" for leaf_class in LAND_COVER_LEAF_CLASSES)

    con = duckdb.connect()
    con.register("lc_long", bucket_df)
    _build_bucket_map_table(con, bucket_map, raw_weights)

    result = con.sql(
        f"""
        WITH bucket_pivot AS (
            SELECT
                {entity_cols_sql}, bucket,
                MAX(share) FILTER (WHERE land_cover_class = {LAND_COVER_CLASS_CODE_FOREST})  AS forest,
                MAX(share) FILTER (WHERE land_cover_class = {LAND_COVER_CLASS_CODE_NONFOREST_NAT})  AS nonforest_nat,
                MAX(share) FILTER (WHERE land_cover_class = {LAND_COVER_CLASS_CODE_FARMING_PARENT})  AS c3,
                MAX(share) FILTER (WHERE land_cover_class = {LAND_COVER_CLASS_CODE_PASTURE}) AS pasture,
                MAX(share) FILTER (WHERE land_cover_class = {LAND_COVER_CLASS_CODE_AGRICULTURE}) AS agriculture,
                MAX(share) FILTER (WHERE land_cover_class = {LAND_COVER_CLASS_CODE_URBAN_PARENT})  AS c4,
                MAX(share) FILTER (WHERE land_cover_class = {LAND_COVER_CLASS_CODE_URBAN}) AS urban,
                MAX(share) FILTER (WHERE land_cover_class = {LAND_COVER_CLASS_CODE_MINING}) AS mining,
                MAX(share) FILTER (WHERE land_cover_class = {LAND_COVER_CLASS_CODE_OTHER_RAW}) AS other_raw,
                MAX(share) FILTER (WHERE land_cover_class = {LAND_COVER_CLASS_CODE_WATER})  AS water
            FROM lc_long
            GROUP BY {entity_cols_sql}, bucket
        ),
        -- resolve c3 -> pasture/agriculture and c4 -> urban/mining/other mismatches
        -- by keeping unattributed mass explicit rather than dropping or guessing a split.
        bucket_resolved AS (
            SELECT
                {entity_cols_sql}, bucket,
                forest, nonforest_nat, pasture, agriculture,
                CASE WHEN c3 IS NULL THEN NULL
                     ELSE GREATEST(c3 - COALESCE(pasture, 0) - COALESCE(agriculture, 0), 0)
                END AS farming_unclassified,
                urban, mining,
                CASE WHEN c4 IS NULL THEN NULL
                     ELSE COALESCE(other_raw, 0)
                          + GREATEST(COALESCE(c4, 0) - COALESCE(urban, 0) - COALESCE(mining, 0) - COALESCE(other_raw, 0), 0)
                END AS other,
                water
            FROM bucket_pivot
        ),
        leaf AS (
            {unpivot_sql}
        ),
        -- total leaf-class share per entity/bucket (< 1 to the extent c0/no-data
        -- pixels exist); NULL if the bucket has no valid area at all.
        leaf_totals AS (
            SELECT {entity_cols_sql}, bucket, SUM(share) AS leaf_total
            FROM leaf GROUP BY {entity_cols_sql}, bucket
        ),
        leaf_renorm AS (
            SELECT l.*, CASE WHEN l.share IS NULL THEN NULL
                        ELSE l.share / NULLIF(t.leaf_total, 0) END AS share_renorm
            FROM leaf l JOIN leaf_totals t USING ({entity_cols_sql}, bucket)
        ),
        bucket_avail AS (
            SELECT lt.{entity_cols_sql.replace(", ", ", lt.")}, lt.bucket, bm.kernel_weight, lt.leaf_total
            FROM leaf_totals lt
            JOIN bucket_map bm USING (bucket)
        ),
        bucket_weighted AS (
            SELECT *, CASE WHEN leaf_total IS NOT NULL AND leaf_total > 0
                          THEN kernel_weight ELSE 0 END AS raw_weight
            FROM bucket_avail
        ),
        -- renormalize weights over only the buckets that exist for this entity
        -- (small headwater catchments won't have distant rings).
        bucket_weight_norm AS (
            SELECT {entity_cols_sql}, bucket,
                   raw_weight / NULLIF(SUM(raw_weight) OVER (PARTITION BY {entity_cols_sql}), 0) AS weight
            FROM bucket_weighted
        ),
        weighted_class AS (
            SELECT lr.{entity_cols_sql.replace(", ", ", lr.")}, lr.class_short,
                   SUM(lr.share_renorm * w.weight) AS dw_share
            FROM leaf_renorm lr
            JOIN bucket_weight_norm w USING ({entity_cols_sql}, bucket)
            WHERE lr.share_renorm IS NOT NULL
            GROUP BY {entity_cols_sql}, lr.class_short
        ),
        wide AS (
            SELECT {entity_cols_sql},
                {wide_cols_sql}
            FROM weighted_class GROUP BY {entity_cols_sql}
        )
        SELECT {entity_cols_sql},
               {select_lc_cols},
               (lc_forest + lc_nonforest_nat) AS lc_nat,
               {alr_cols_sql}
        FROM wide
        ORDER BY {entity_cols_sql}
        """
    ).df()
    return result
