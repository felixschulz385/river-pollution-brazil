"""SGB occurrences upstream of each municipality along the river network.

Each occurrence is placed in the river trench whose drainage polygon contains
it. Two exposures follow:

- counts of occurrences in the trenches upstream of the municipality within
  50 km, 200 km and without limit, on the shifted distance scale of the
  land-cover ADM2 aggregation (0 = the upstream end of the trenches touching
  the municipality; the trenches' own bodies, the negative buckets, are not
  upstream);
- Moreno-Louzada's binaries: any occurrence in an Otto level-4 sub-basin
  upstream (or downstream) of a sub-basin the municipality intersects, within
  the same level-3 basin, by Pfafstetter coding.
"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from src.data.shared.adm2_upstream_buckets import ADJUSTED_DISTANCE_COLUMN, build_adm2_upstream_bucket_table
from src.data.shared.slurm import resolve_n_jobs
from src.data.shared.spatial_tabular import deduplicate_drainage_polygons
from src.data.sources import river_network as rn_module
from src.data.sources.biomes.constants import BRAZIL_PROJECTED_CRS
from src.data.sources.land_cover.constants import derive_mun_id_from_adm2_id
from src.data.sources.protected_areas.preprocess import load_protected_areas, protected_mask
from src.data.sources.river_network.constants import PROCESSED_DIR as RIVER_NETWORK_PROCESSED_DIR

from .constants import (
    ALLUVIAL_HOSTS,
    DEFAULT_ADM2_UPSTREAM_OUTPUT_PATH,
    MUN_ID_COLUMN,
    NEAREST_MUNICIPALITY_MAX_DISTANCE_M,
    OTTO_GROUPS,
    OTTO_LEVEL,
    UPSTREAM_COLUMNS,
    UPSTREAM_GROUPS,
    UPSTREAM_REACHES_KM,
    gpkg_path,
    otto_column,
    upstream_count_column,
)


logger = logging.getLogger(__name__)

TRENCH_ID_COLUMN = "trench_id"
OTTO_COLUMN = f"otto_basin_l{OTTO_LEVEL}"


def load_occurrence_groups(path, protected_areas):
    """SGB occurrences with one boolean column per `UPSTREAM_GROUPS` group."""
    occurrences = gpd.read_file(path, layer="occurrences")
    substances = occurrences["SUBSTANCIAS"].fillna("")
    alluvial = occurrences["ROCHAS_HOSPEDEIRAS"].fillna("").str.contains(ALLUVIAL_HOSTS)
    protected = protected_mask(occurrences, protected_areas)
    for group, (substance, status, alluvial_only, protected_only) in UPSTREAM_GROUPS.items():
        member = substances.str.contains(substance)
        if status is not None:
            member &= occurrences["STATUS_ECONOMICO"].eq(status)
        if alluvial_only:
            member &= alluvial
        if protected_only:
            member &= protected
        occurrences[group] = member
    return occurrences[occurrences[list(UPSTREAM_GROUPS)].any(axis=1)].reset_index(drop=True)


def assign_trench(points, drainage_polygons):
    """Trench whose drainage polygon contains each point; points just outside every polygon get the nearest one."""
    polygons = drainage_polygons[[TRENCH_ID_COLUMN, "geometry"]].to_crs(BRAZIL_PROJECTED_CRS)
    points = points[["geometry"]].to_crs(BRAZIL_PROJECTED_CRS)
    joined = gpd.sjoin(points, polygons, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    unmatched = joined.index[joined[TRENCH_ID_COLUMN].isna()]
    if len(unmatched):
        nearest = gpd.sjoin_nearest(
            points.loc[unmatched], polygons, how="left", max_distance=NEAREST_MUNICIPALITY_MAX_DISTANCE_M
        )
        nearest = nearest[~nearest.index.duplicated(keep="first")]
        joined.loc[unmatched, TRENCH_ID_COLUMN] = nearest[TRENCH_ID_COLUMN]
    dropped = int(joined[TRENCH_ID_COLUMN].isna().sum())
    if dropped:
        logger.info("Dropped %d occurrence(s) more than %d m from any drainage polygon.",
                    dropped, NEAREST_MUNICIPALITY_MAX_DISTANCE_M)
    return joined[TRENCH_ID_COLUMN]


def trench_counts(occurrences, trench_ids):
    """Occurrences per trench and group (trenches with none are absent)."""
    frame = occurrences[list(UPSTREAM_GROUPS)].astype("int64").assign(**{TRENCH_ID_COLUMN: trench_ids})
    frame = frame.dropna(subset=[TRENCH_ID_COLUMN])
    frame[TRENCH_ID_COLUMN] = frame[TRENCH_ID_COLUMN].astype("int64")
    return frame.groupby(TRENCH_ID_COLUMN)[list(UPSTREAM_GROUPS)].sum()


def reach_counts(membership, counts):
    """Occurrence counts in one municipality's upstream trenches, per group and reach."""
    distances = membership.set_index(TRENCH_ID_COLUMN)[ADJUSTED_DISTANCE_COLUMN]
    matched = counts.index.intersection(distances.index)
    distance = distances.loc[matched].groupby(level=0).min()
    row = {}
    for reach, limit in UPSTREAM_REACHES_KM.items():
        within = distance.index[(distance >= 0) & (distance < (np.inf if limit is None else limit))]
        sums = counts.loc[within].sum()
        for group in UPSTREAM_GROUPS:
            row[upstream_count_column(group, reach)] = int(sums.get(group, 0))
    return row


def pfafstetter_upstream(code_a, code_b):
    """True if basin `code_a` drains into basin `code_b` (same parent basin, Pfafstetter digits).

    Within a parent, odd digits are main-stem interbasins numbered from the
    outlet (1) to the headwater (9), even digits tributaries. A basin is
    upstream of a main-stem interbasin with a lower digit; nothing in the
    parent is upstream of a tributary.
    """
    if len(code_a) != len(code_b) or code_a[:-1] != code_b[:-1] or code_a == code_b:
        return False
    last_a, last_b = int(code_a[-1]), int(code_b[-1])
    return last_b % 2 == 1 and last_a > last_b


def otto_exposure(municipality_basins, occurrence_basins):
    """Binary upstream/downstream exposure per municipality and group.

    `municipality_basins` maps `mun_id` to the set of level-4 codes it
    intersects; `occurrence_basins` maps each group to the set of level-4
    codes holding at least one of its occurrences.
    """
    rows = {}
    for mun_id, basins in municipality_basins.items():
        row = {}
        for group in OTTO_GROUPS:
            deposits = occurrence_basins.get(group, set())
            row[otto_column(group, "upstream")] = int(
                any(pfafstetter_upstream(deposit, basin) for deposit in deposits for basin in basins)
            )
            row[otto_column(group, "downstream")] = int(
                any(pfafstetter_upstream(basin, deposit) for deposit in deposits for basin in basins)
            )
        rows[mun_id] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def trench_otto_basins(river_network_path):
    """Otto level-`OTTO_LEVEL` code of each trench, as strings indexed by `trench_id`."""
    otto = pd.read_parquet(Path(river_network_path) / "river_network_otto_basins.parquet",
                           columns=[TRENCH_ID_COLUMN, OTTO_COLUMN])
    return otto.dropna().set_index(TRENCH_ID_COLUMN)[OTTO_COLUMN].astype(str)


def municipality_otto_basins(river_network_path, otto=None):
    """Map each `mun_id` to the set of level-`OTTO_LEVEL` codes of the trenches it intersects."""
    otto = trench_otto_basins(river_network_path) if otto is None else otto
    matches = pd.read_parquet(Path(river_network_path) / "river_network_trench_adm2_matches.parquet")
    matches[MUN_ID_COLUMN] = matches["adm2"].map(derive_mun_id_from_adm2_id)
    matches[OTTO_COLUMN] = matches[TRENCH_ID_COLUMN].map(otto)
    return matches.dropna(subset=[OTTO_COLUMN]).groupby(MUN_ID_COLUMN)[OTTO_COLUMN].agg(set).to_dict()


def build_adm2_upstream_geology(root_dir=".", gpkg=None, river_network_path=None, n_jobs=None, output_path=None,
                                protected_areas_path=None):
    """Write `geology_adm2_upstream.parquet`: upstream occurrence counts and Otto-4 binaries per municipality."""
    gpkg = Path(gpkg) if gpkg else gpkg_path(root_dir)
    river_network_path = Path(root_dir) / (river_network_path or RIVER_NETWORK_PROCESSED_DIR)
    n_jobs = n_jobs or resolve_n_jobs()

    occurrences = load_occurrence_groups(gpkg, load_protected_areas(root_dir, protected_areas_path))
    logger.info("SGB occurrences by group: %s", occurrences[list(UPSTREAM_GROUPS)].sum().to_dict())

    network = rn_module.RiverNetwork()
    network.load(str(river_network_path))
    drainage = deduplicate_drainage_polygons(network.drainage_areas.reset_index(drop=True))
    trench_ids = assign_trench(occurrences, drainage)
    counts = trench_counts(occurrences, trench_ids)
    logger.info("Occurrences placed in %d trenches.", len(counts))

    def reduce_adm2(adm2_id, membership):
        return pd.DataFrame([{MUN_ID_COLUMN: derive_mun_id_from_adm2_id(adm2_id), **reach_counts(membership, counts)}])

    upstream = build_adm2_upstream_bucket_table(
        network=network, rn_module=rn_module, n_jobs=n_jobs, reduce_adm2=reduce_adm2
    ).set_index(MUN_ID_COLUMN)

    otto = trench_otto_basins(river_network_path)
    municipality_basins = municipality_otto_basins(river_network_path, otto)
    occurrence_basins = trench_ids.map(otto)
    basins_by_group = {
        group: set(occurrence_basins[occurrences[group] & occurrence_basins.notna()]) for group in OTTO_GROUPS
    }
    exposure = otto_exposure(municipality_basins, basins_by_group)

    municipalities = sorted(set(upstream.index) | set(municipality_basins))
    result = upstream.reindex(municipalities).join(exposure.reindex(municipalities))
    result = result.fillna(0).astype("int64").rename_axis(MUN_ID_COLUMN).reset_index()[list(UPSTREAM_COLUMNS)]

    resolved_output_path = Path(root_dir) / (output_path or DEFAULT_ADM2_UPSTREAM_OUTPUT_PATH)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(resolved_output_path, index=False)
    logger.info("Wrote upstream geology to %s (shape=%s)", resolved_output_path, result.shape)
    return result
