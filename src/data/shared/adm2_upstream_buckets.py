"""Per-ADM2 upstream river-distance bucket binning, shared by the climate and
land-cover ADM2 aggregations.

Both sources bin every trench reachable upstream of an ADM2 unit into fixed-width
distance buckets on the shifted-origin scale (0 = the upstream end of the
ADM2-touching trench; negative buckets = that trench's own body). The only
source-specific part is what gets aggregated onto the buckets afterwards -- annual
climate means (SQL, in ``climate.assembly``) vs. land-cover class counts/shares
(pandas, in ``land_cover.aggregation``) -- so that stays in each source; the
seed -> reachable distance -> shifted origin -> bucket-label pipeline, plus the
`(adm2_id, trench_id, distance_bucket, bucket_intersects_adm2)` membership table
it produces, lives here.

`build_adm2_upstream_bucket_table` returns the whole membership table as one
concatenated DataFrame. Units are resolved in parallel work chunks and the
per-chunk frames are concatenated once at the end; the observed full table is a
few tens of millions of narrow rows, small enough to hold in memory and hand to
DuckDB directly. (An earlier revision streamed each chunk to a Parquet part and
had the consumer glob them back -- climate registered the glob with DuckDB,
land_cover `pd.concat`-ed it straight back into a frame -- which added a disk
round-trip for no measured memory benefit.)
"""

from __future__ import annotations

import logging
import math
import threading

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.data.shared.sensor_upstream import (
    BUCKET_INTERSECTS_ADM2_COLUMN,
    build_group_index_lookup,
    normalize_network_frame,
    prepare_trench_adm2_matches,
    resolve_multi_seed_reachable_distances,
    validate_network_index_tables,
)
from src.data.shared.spatial_tabular import deduplicate_drainage_polygons


logger = logging.getLogger(__name__)

UPSTREAM_DISTANCE_COLUMN = "upstream_distance"
ADJUSTED_DISTANCE_COLUMN = "adjusted_distance"
TRENCH_LENGTH_COLUMN = "trench_length_km"

# Fixed 25 km bins, clamped at a 500 km lower bound -- matches
# `SENSOR_DISTANCE_BUCKETS` / `LAND_COVER_COMPOSITION_BUCKET_MAP`, which are both
# keyed only up to 500; an uncapped label beyond that is silently dropped by the
# downstream composition join.
DEFAULT_BUCKET_WIDTH_KM = 25.0
DEFAULT_MAX_BUCKET_START_KM = 500.0

# ADM2 units are split into this many parallel work chunks (one joblib task
# each). Purely a parallelism-granularity knob -- the per-chunk frames are
# concatenated at the end regardless of how many chunks there are.
DEFAULT_CHUNK_COUNT = 256


def trench_length_lookup(rivers, *, trench_id_column="trench_id"):
    """Return trench lengths keyed by trench id (the ``distance`` trench column)."""
    required_columns = {trench_id_column, "distance"}
    missing_columns = required_columns.difference(rivers.columns)
    if missing_columns:
        raise ValueError(
            "River trench data is missing required length columns: "
            f"{sorted(missing_columns)}."
        )
    return (
        rivers[[trench_id_column, "distance"]]
        .drop_duplicates(subset=[trench_id_column], keep="first")
        .rename(columns={"distance": TRENCH_LENGTH_COLUMN})
    )


def apply_shifted_origin(
    trench_distance_lookup,
    trench_lengths,
    *,
    trench_id_column="trench_id",
    upstream_distance_column=UPSTREAM_DISTANCE_COLUMN,
    adjusted_distance_column=ADJUSTED_DISTANCE_COLUMN,
):
    """Shift distances so zero is the upstream end of the ADM2-touching trench.

    `trench_distance_lookup` is a Series of upstream distances indexed by trench
    id; the return is a frame indexed by trench id with the original
    `upstream_distance_column`, the joined `trench_length_km`, and the shifted
    `adjusted_distance_column` (``upstream_distance - trench_length_km``).
    """
    shifted = trench_distance_lookup.reset_index().merge(
        trench_lengths,
        on=trench_id_column,
        how="left",
        validate="one_to_one",
    )
    if shifted[TRENCH_LENGTH_COLUMN].isna().any():
        missing_ids = shifted.loc[
            shifted[TRENCH_LENGTH_COLUMN].isna(),
            trench_id_column,
        ].tolist()
        raise ValueError(
            "Missing trench length(s) for shifted upstream-distance calculation: "
            f"{missing_ids[:10]}"
        )
    shifted[adjusted_distance_column] = (
        shifted[upstream_distance_column] - shifted[TRENCH_LENGTH_COLUMN]
    )
    return shifted.set_index(trench_id_column, drop=True)


def assign_adm2_distance_bucket(
    distances,
    *,
    bucket_width_km=DEFAULT_BUCKET_WIDTH_KM,
    max_bucket_start_km=DEFAULT_MAX_BUCKET_START_KM,
):
    """Return fixed-width lower-bound bucket labels on the shifted distance scale.

    Negative buckets are kept on purpose -- they represent the ADM2-touching
    trench's own downstream portion, behind the shifted-distance zero point -- so
    this deliberately does not delegate to
    `shared.sensor_upstream.assign_distance_buckets`, whose bucket list starts at
    0. The upper end is clamped to `max_bucket_start_km`.
    """
    distances = np.asarray(distances, dtype=float)
    buckets = np.floor(distances / bucket_width_km) * bucket_width_km
    buckets = np.minimum(buckets, max_bucket_start_km)
    return buckets.astype(int)


def _chunk_bounds(n_items, chunk_count):
    if n_items <= 0:
        return
    chunk_size = max(1, math.ceil(n_items / max(1, chunk_count)))
    for start in range(0, n_items, chunk_size):
        yield start, min(start + chunk_size, n_items)


def build_adm2_upstream_bucket_table(
    *,
    network,
    rn_module,
    n_jobs,
    trench_id_column="trench_id",
    adm2_id_column="adm2_id",
    distance_bucket_column="distance_bucket",
    bucket_width_km=DEFAULT_BUCKET_WIDTH_KM,
    max_bucket_start_km=DEFAULT_MAX_BUCKET_START_KM,
    reduce_adm2=None,
    chunk_count=DEFAULT_CHUNK_COUNT,
):
    """Bin every ADM2 unit's upstream trenches into distance buckets.

    Returns one concatenated DataFrame. Each row is
    ``[adm2_id_column, trench_id_column, distance_bucket_column,
    bucket_intersects_adm2, upstream_distance, adjusted_distance]`` unless
    `reduce_adm2(adm2_id, bucket_frame)` is given, in which case that callback's
    returned frame -- the source-specific per-ADM2 aggregation -- is used instead
    (return ``None``/empty to skip a unit).

    A unit whose resolution or `reduce_adm2` call raises is logged and skipped
    rather than aborting the whole (multi-thousand-unit) run; each work chunk is
    logged at INFO once its units are resolved. Returns an empty DataFrame if
    nothing produced rows. `network.trenches` / `network.drainage_areas` are
    normalized in place, matching the previous per-source behaviour.
    """
    if not network.trench_reachability_matrices:
        raise ValueError("River network must have trench reachability data computed.")
    if network.trenches is None:
        raise ValueError("River network must include trench data.")
    if network.drainage_areas is None:
        raise ValueError("River network must include drainage polygon data.")

    network.trenches = normalize_network_frame(network.trenches)
    network.drainage_areas = normalize_network_frame(network.drainage_areas)

    system_column = rn_module.SYSTEM_ID_KEY
    position_column = rn_module.TRENCH_INDEX_COLUMN

    trench_adm2_matches = prepare_trench_adm2_matches(
        network,
        rn_module=rn_module,
        trench_id_column=trench_id_column,
    )
    drainage_polygons = deduplicate_drainage_polygons(
        network.drainage_areas.reset_index(drop=True).copy()
    ).reset_index(drop=True)
    missing_drainage_columns = {trench_id_column}.difference(drainage_polygons.columns)
    if missing_drainage_columns:
        raise ValueError(
            "Drainage polygons are missing required columns: "
            f"{sorted(missing_drainage_columns)}."
        )

    trench_lookup = (
        drainage_polygons[[trench_id_column]]
        .merge(
            trench_adm2_matches[[trench_id_column, "adm2", system_column]].drop_duplicates(),
            on=trench_id_column,
            how="left",
            validate="one_to_many",
        )
        .dropna(subset=[system_column])
    )
    adm2_groups = [
        (adm2_id, adm2_rows[[trench_id_column, system_column]].drop_duplicates())
        for adm2_id, adm2_rows in trench_lookup.groupby("adm2", sort=False)
    ]

    validate_network_index_tables(
        network,
        location_column=trench_id_column,
        system_column=system_column,
        position_column=position_column,
    )
    system_location_arrays, system_positions = build_group_index_lookup(
        network.trenches,
        location_column=trench_id_column,
        system_column=system_column,
        position_column=position_column,
    )
    trench_lengths = trench_length_lookup(network.trenches, trench_id_column=trench_id_column)

    def resolve_membership(adm2_id, adm2_trenches):
        if adm2_trenches is None or adm2_trenches.empty:
            return None
        intersecting_trench_ids = set(adm2_trenches[trench_id_column])

        trench_distance_lookup = resolve_multi_seed_reachable_distances(
            network,
            adm2_trenches,
            location_column=trench_id_column,
            distance_column=UPSTREAM_DISTANCE_COLUMN,
            system_column=system_column,
            position_column=position_column,
            system_location_arrays=system_location_arrays,
            system_positions=system_positions,
        )
        if trench_distance_lookup.empty:
            return None
        trench_distance_lookup = trench_distance_lookup.set_index(trench_id_column)[
            UPSTREAM_DISTANCE_COLUMN
        ]
        trench_distance_lookup = apply_shifted_origin(
            trench_distance_lookup,
            trench_lengths,
            trench_id_column=trench_id_column,
        )

        frame = trench_distance_lookup.reset_index()[
            [trench_id_column, UPSTREAM_DISTANCE_COLUMN, ADJUSTED_DISTANCE_COLUMN]
        ].copy()
        frame.insert(0, adm2_id_column, adm2_id)
        frame[distance_bucket_column] = assign_adm2_distance_bucket(
            frame[ADJUSTED_DISTANCE_COLUMN].to_numpy(),
            bucket_width_km=bucket_width_km,
            max_bucket_start_km=max_bucket_start_km,
        )
        frame[BUCKET_INTERSECTS_ADM2_COLUMN] = frame[trench_id_column].isin(
            intersecting_trench_ids
        )
        return frame[
            [
                adm2_id_column,
                trench_id_column,
                distance_bucket_column,
                BUCKET_INTERSECTS_ADM2_COLUMN,
                UPSTREAM_DISTANCE_COLUMN,
                ADJUSTED_DISTANCE_COLUMN,
            ]
        ]

    failure_lock = threading.Lock()
    failure_state = {"failed": 0}

    def process_chunk(chunk_index, n_chunks, chunk_groups):
        frames = []
        for adm2_id, adm2_trenches in chunk_groups:
            try:
                membership = resolve_membership(adm2_id, adm2_trenches)
                if membership is None or membership.empty:
                    continue
                part = (
                    membership
                    if reduce_adm2 is None
                    else reduce_adm2(adm2_id, membership)
                )
            except Exception:
                # Tolerate isolated unit failures; a systematic one (majority of
                # units) is re-raised after the run.
                logger.warning(
                    "Skipping ADM2 unit %r: failed to resolve upstream buckets.",
                    adm2_id,
                    exc_info=True,
                )
                with failure_lock:
                    failure_state["failed"] += 1
                continue
            if part is None or len(part) == 0:
                continue
            frames.append(part)
        if not frames:
            return None
        chunk_frame = pd.concat(frames, ignore_index=True)
        logger.info(
            "Resolved ADM2 upstream bucket chunk %d/%d: %d unit(s), %d row(s).",
            chunk_index + 1,
            n_chunks,
            len(frames),
            len(chunk_frame),
        )
        return chunk_frame

    chunks = list(_chunk_bounds(len(adm2_groups), chunk_count))
    logger.info(
        "Resolving ADM2 upstream buckets for %d ADM2 unit(s) in %d chunk(s) with %s worker(s).",
        len(adm2_groups),
        len(chunks),
        n_jobs,
    )
    chunk_frames = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(process_chunk)(chunk_index, len(chunks), adm2_groups[start:end])
        for chunk_index, (start, end) in enumerate(chunks)
    )

    failed = failure_state["failed"]
    unit_count = len(adm2_groups)
    if failed and failed == unit_count:
        raise RuntimeError(
            f"Failed to resolve upstream buckets for all {unit_count} ADM2 "
            "unit(s); aborting rather than emitting an empty table. This usually "
            "means a systematic input problem (e.g. an incomplete trench-length "
            "table)."
        )
    if failed:
        logger.warning(
            "Resolved ADM2 upstream buckets with %d/%d unit(s) skipped after errors.",
            failed,
            unit_count,
        )

    chunk_frames = [frame for frame in chunk_frames if frame is not None]
    if not chunk_frames:
        logger.info("Built ADM2 upstream bucket table: 0 row(s).")
        return pd.DataFrame()
    table = pd.concat(chunk_frames, ignore_index=True)
    logger.info(
        "Built ADM2 upstream bucket table: %d row(s) from %d chunk(s).",
        len(table),
        len(chunk_frames),
    )
    return table
