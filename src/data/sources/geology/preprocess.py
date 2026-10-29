import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

from src.data.sources.biomes.constants import BRAZIL_PROJECTED_CRS
from src.data.sources.biomes.preprocess import _load_adm2_boundaries
from src.data.sources.land_cover.constants import derive_mun_id_from_adm2_id
from src.data.sources.protected_areas.preprocess import covered_share, load_protected_areas, protected_mask

from .constants import (
    ADM2_COLUMNS,
    ALLUVIAL_COLUMN,
    ALLUVIAL_HOSTS,
    AREA_COLUMN,
    COUNT_COLUMNS,
    DEFAULT_ADM2_OUTPUT_PATH,
    DISTRICT_SHARE_COLUMN,
    GOLD_OCCURRENCE_COLUMN,
    GOLD_PROTECTED_COLUMN,
    GOLD_SUBSTANCE,
    GOLD_UNEXPLOITED_PROTECTED_COLUMN,
    MUN_ID_COLUMN,
    NEAREST_MUNICIPALITY_MAX_DISTANCE_M,
    PROVINCE_SHARE_COLUMN,
    STATUS_COLUMNS,
    TIN_OCCURRENCE_COLUMN,
    TIN_PROTECTED_COLUMN,
    TIN_SUBSTANCE,
    TIN_UNEXPLOITED_COLUMN,
    gpkg_path,
)


logger = logging.getLogger(__name__)


def load_gold_occurrences(path):
    """Gold occurrences with an `alluvial` flag from their host material."""
    occurrences = gpd.read_file(path, layer="occurrences")
    gold = occurrences[occurrences["SUBSTANCIAS"].fillna("").str.contains(GOLD_SUBSTANCE)].copy()
    gold["alluvial"] = gold["ROCHAS_HOSPEDEIRAS"].fillna("").str.contains(ALLUVIAL_HOSTS)
    unknown = set(gold["STATUS_ECONOMICO"].dropna()) - set(STATUS_COLUMNS)
    if unknown:
        logger.warning("Gold occurrences with unmapped economic status (total only): %s", sorted(unknown))
    return gold


def load_tin_occurrences(path):
    """Tin (cassiterite) occurrences, the other-minerals placebo."""
    occurrences = gpd.read_file(path, layer="occurrences")
    return occurrences[occurrences["SUBSTANCIAS"].fillna("").str.contains(TIN_SUBSTANCE)].copy()


def assign_municipality(points, boundaries):
    """Municipality of each point; points just outside every polygon get the nearest one."""
    joined = gpd.sjoin(points, boundaries, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    unmatched = joined.index[joined[MUN_ID_COLUMN].isna()]
    if len(unmatched):
        nearest = gpd.sjoin_nearest(
            points.loc[unmatched],
            boundaries,
            how="left",
            max_distance=NEAREST_MUNICIPALITY_MAX_DISTANCE_M,
        )
        nearest = nearest[~nearest.index.duplicated(keep="first")]
        joined.loc[unmatched, MUN_ID_COLUMN] = nearest[MUN_ID_COLUMN]
    dropped = int(joined[MUN_ID_COLUMN].isna().sum())
    if dropped:
        logger.info(
            "Dropped %d occurrence(s) more than %d m from any municipality.",
            dropped,
            NEAREST_MUNICIPALITY_MAX_DISTANCE_M,
        )
    return joined[MUN_ID_COLUMN]


def build_adm2_geology(
    root_dir=".",
    gpkg=None,
    gadm_path=None,
    layer=None,
    adm2_id_column=None,
    output_path=None,
    protected_areas_path=None,
):
    """Gold occurrence counts and gold province/district area shares per municipality."""
    gpkg = Path(gpkg) if gpkg else gpkg_path(root_dir)
    protected_areas = load_protected_areas(root_dir, protected_areas_path)
    boundaries = _load_adm2_boundaries(
        root_dir, gadm_path=gadm_path, layer=layer, adm2_id_column=adm2_id_column
    ).to_crs(BRAZIL_PROJECTED_CRS)
    boundaries[MUN_ID_COLUMN] = boundaries["adm2_id"].map(derive_mun_id_from_adm2_id)
    boundaries = boundaries[[MUN_ID_COLUMN, "geometry"]]
    area = boundaries.geometry.area.groupby(boundaries[MUN_ID_COLUMN]).sum()

    gold = load_gold_occurrences(gpkg).to_crs(BRAZIL_PROJECTED_CRS)
    gold[MUN_ID_COLUMN] = assign_municipality(gold, boundaries)
    gold = gold.dropna(subset=[MUN_ID_COLUMN])
    by_municipality = gold.groupby(MUN_ID_COLUMN)

    result = pd.DataFrame({AREA_COLUMN: area / 1e6})
    result[GOLD_OCCURRENCE_COLUMN] = by_municipality.size()
    for status, column in STATUS_COLUMNS.items():
        result[column] = gold[gold["STATUS_ECONOMICO"] == status].groupby(MUN_ID_COLUMN).size()
    result[ALLUVIAL_COLUMN] = gold[gold["alluvial"]].groupby(MUN_ID_COLUMN).size()
    gold_protected = gold[protected_mask(gold, protected_areas)]
    result[GOLD_PROTECTED_COLUMN] = gold_protected.groupby(MUN_ID_COLUMN).size()
    result[GOLD_UNEXPLOITED_PROTECTED_COLUMN] = (
        gold_protected[gold_protected["STATUS_ECONOMICO"] == "Não explotado"].groupby(MUN_ID_COLUMN).size()
    )

    tin = load_tin_occurrences(gpkg).to_crs(BRAZIL_PROJECTED_CRS)
    tin[MUN_ID_COLUMN] = assign_municipality(tin, boundaries)
    tin = tin.dropna(subset=[MUN_ID_COLUMN])
    result[TIN_OCCURRENCE_COLUMN] = tin.groupby(MUN_ID_COLUMN).size()
    result[TIN_UNEXPLOITED_COLUMN] = tin[tin["STATUS_ECONOMICO"] == "Não explotado"].groupby(MUN_ID_COLUMN).size()
    result[TIN_PROTECTED_COLUMN] = tin[protected_mask(tin, protected_areas)].groupby(MUN_ID_COLUMN).size()
    result[list(COUNT_COLUMNS)] = result[list(COUNT_COLUMNS)].fillna(0).astype("int64")

    for column, polygon_layer in (
        (DISTRICT_SHARE_COLUMN, "gold_districts"),
        (PROVINCE_SHARE_COLUMN, "gold_provinces"),
    ):
        polygons = gpd.read_file(gpkg, layer=polygon_layer).to_crs(BRAZIL_PROJECTED_CRS)
        result[column] = covered_share(boundaries, polygons, area)

    result = result.rename_axis(MUN_ID_COLUMN).reset_index()[list(ADM2_COLUMNS)]
    result = result.sort_values(MUN_ID_COLUMN).reset_index(drop=True)

    resolved_output_path = Path(root_dir) / (output_path or DEFAULT_ADM2_OUTPUT_PATH)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(resolved_output_path, index=False)
    logger.info(
        "Wrote ADM2 gold geology to %s (shape=%s, %d gold occurrences assigned)",
        resolved_output_path,
        result.shape,
        int(result[GOLD_OCCURRENCE_COLUMN].sum()),
    )
    return result
