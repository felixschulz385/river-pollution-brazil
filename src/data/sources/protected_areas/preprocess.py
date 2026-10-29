import logging
from pathlib import Path

import geopandas as gpd
import shapely
import pandas as pd

from src.data.sources.biomes.constants import BRAZIL_PROJECTED_CRS
from src.data.sources.biomes.preprocess import _load_adm2_boundaries
from src.data.sources.land_cover.constants import derive_mun_id_from_adm2_id

from .constants import (
    ADM2_COLUMNS,
    AREA_COLUMNS,
    BARRED_CATEGORIES,
    BARRED_GROUPS,
    BARRED_STAGES,
    CONSERVATION,
    CONSERVATION_SHARE_COLUMN,
    CUTOFF_YEAR,
    DEFAULT_ADM2_OUTPUT_PATH,
    DEFAULT_AREAS_OUTPUT_PATH,
    ESTABLISHED_STAGES,
    FUNAI_FIELDS,
    INDIGENOUS,
    INDIGENOUS_SHARE_COLUMN,
    LAYERS,
    MUN_ID_COLUMN,
    PROTECTED_2012_SHARE_COLUMN,
    PROTECTED_SHARE_COLUMN,
    funai_path,
    gpkg_path,
    layer_name,
)


logger = logging.getLogger(__name__)


def read_kind(gpkg, kind):
    """All TerraBrasilis layers of one kind, stacked; an area clipped by several regions appears once per region."""
    frames = [gpd.read_file(gpkg, layer=layer_name(type_name)) for type_name in LAYERS[kind]]
    return pd.concat(frames, ignore_index=True)


def _proper_ring(ring):
    return len(set(ring.coords)) >= 3


def _polygon_parts(geometry):
    """The polygons of a geometry, without rings of fewer than three distinct points (GEOS rejects those)."""
    polygons = []
    for part in shapely.get_parts(geometry):
        if part.geom_type == "MultiPolygon":
            polygons += _polygon_parts(part)
        elif part.geom_type == "Polygon" and _proper_ring(part.exterior):
            polygons.append(shapely.Polygon(part.exterior, [ring for ring in part.interiors if _proper_ring(ring)]))
    return polygons


def _polygonal(geometry):
    """A geometry's polygons, repaired: some pieces carry degenerate rings, stray lines or points."""
    if geometry is None or geometry.is_empty:
        return None
    polygons = _polygon_parts(geometry)
    if not polygons:
        return None
    repaired = shapely.make_valid(shapely.MultiPolygon(polygons))
    polygons = _polygon_parts(repaired)
    return shapely.MultiPolygon(polygons) if polygons else None


def _merge(frame, keys):
    """One row per area: the union of its pieces, with the attributes of its first piece."""
    frame = frame.assign(geometry=frame.geometry.map(_polygonal))
    frame = frame[frame.geometry.notna()]
    return frame.dissolve(by=keys, aggfunc="first", as_index=False, dropna=False)


def read_funai(path):
    """FUNAI's land polygons with their stage dates, under the names of `FUNAI_FIELDS` (EPSG:4326)."""
    frame = gpd.read_file(f"zip://{path}", columns=list(FUNAI_FIELDS)).rename(columns=FUNAI_FIELDS)
    return frame.to_crs(4326)


def indigenous_lands(frame):
    """Indigenous lands in the `AREA_COLUMNS` layout, barred from their first declaration, homologation or registry."""
    lands = _merge(frame.dropna(subset=["code"]), "code")
    stage = lands["stage"].str.strip()
    dates = pd.DataFrame({
        column: pd.to_datetime(lands[column], errors="coerce") for column in ("declared", "homologated", "registered")
    })
    first = dates.min(axis=1)
    # FUNAI's stage lags its dates for a few lands (a declaration date on a land still marked Delimitada).
    barred = stage.isin(BARRED_STAGES) | first.notna()
    year_barred = first.dt.year.astype("Int64")
    in_force = year_barred.le(CUTOFF_YEAR).fillna(False).astype(bool) | (first.isna() & stage.isin(ESTABLISHED_STAGES))
    return gpd.GeoDataFrame({
        "area_id": "ti:" + lands["code"].astype("int64").astype(str),
        "kind": INDIGENOUS,
        "name": lands["name"],
        "category": lands["modality"],
        "group": None,
        "sphere": "federal",
        "stage": stage,
        "year_created": pd.array([pd.NA] * len(lands), dtype="Int64"),
        "date_declared": dates["declared"],
        "date_homologated": dates["homologated"],
        "year_barred": year_barred,
        "mining_barred": barred,
        "in_force_2012": in_force & barred,
    }, geometry=lands.geometry, crs=lands.crs)


def creation_year(values):
    """The first year in `ano_cria` (mostly a plain year; a few carry the act's text)."""
    return pd.to_numeric(values.astype(str).str.extract(r"((?:18|19|20)\d\d)")[0], errors="coerce").astype("Int64")


def conservation_units(frame):
    """Conservation units in the `AREA_COLUMNS` layout; pieces share name, category, sphere and year."""
    frame = frame.assign(
        key=frame["nome"].str.strip().str.casefold(), year_created=creation_year(frame["ano_cria"])
    )
    units = _merge(frame, ["key", "categoria", "esfera", "year_created"]).sort_values("key").reset_index(drop=True)
    barred = units["grupo"].isin(BARRED_GROUPS) | units["categoria"].isin(BARRED_CATEGORIES)
    return gpd.GeoDataFrame({
        "area_id": [f"uc:{index:04d}" for index in range(len(units))],
        "kind": CONSERVATION,
        "name": units["nome"],
        "category": units["categoria"],
        "group": units["grupo"],
        "sphere": units["esfera"],
        "stage": None,
        "year_created": units["year_created"].astype("Int64"),
        "date_declared": pd.NaT,
        "date_homologated": pd.NaT,
        "year_barred": units["year_created"].astype("Int64").where(barred),
        "mining_barred": barred,
        "in_force_2012": units["year_created"].le(CUTOFF_YEAR).fillna(False).astype(bool),
    }, geometry=units.geometry, crs=units.crs)


def build_areas(gpkg, funai):
    """FUNAI's Indigenous lands and TerraBrasilis' conservation units in one GeoDataFrame (EPSG:4326)."""
    areas = pd.concat(
        [indigenous_lands(read_funai(funai)), conservation_units(read_kind(gpkg, CONSERVATION))],
        ignore_index=True,
    )
    return gpd.GeoDataFrame(areas, geometry="geometry", crs=4326)[list(AREA_COLUMNS)]


def protected_mask(points, areas):
    """Whether each point lies in an area that bars mining and was in force by `CUTOFF_YEAR`."""
    selected = areas[areas["mining_barred"] & areas["in_force_2012"]].to_crs(points.crs)
    joined = gpd.sjoin(points[["geometry"]], selected[["geometry"]], how="left", predicate="within")
    return joined["index_right"].notna().groupby(level=0).any().reindex(points.index, fill_value=False)


def load_protected_areas(root_dir=".", path=None):
    """The processed protected areas, or an error pointing at the preprocess step."""
    path = Path(root_dir) / (path or DEFAULT_AREAS_OUTPUT_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run `python -m src.cli data preprocess --source protected_areas` first."
        )
    return gpd.read_parquet(path)


def covered_share(boundaries, polygons, area):
    """Share of each municipality's area inside the union of `polygons`."""
    if polygons.empty:
        return pd.Series(0.0, index=area.index)
    union = gpd.GeoDataFrame(geometry=[polygons.geometry.make_valid().union_all()], crs=polygons.crs)
    overlay = gpd.overlay(boundaries, union, how="intersection")
    covered = overlay.geometry.area.groupby(overlay[MUN_ID_COLUMN]).sum()
    return (covered / area).reindex(area.index).fillna(0.0).clip(upper=1.0)


def adm2_shares(areas, boundaries):
    """Per municipality, the area shares in Indigenous lands, barred conservation units and both."""
    areas = areas.to_crs(BRAZIL_PROJECTED_CRS)
    area = boundaries.geometry.area.groupby(boundaries[MUN_ID_COLUMN]).sum()
    barred = areas[areas["mining_barred"]]
    result = pd.DataFrame(index=area.index)
    result[INDIGENOUS_SHARE_COLUMN] = covered_share(boundaries, barred[barred["kind"] == INDIGENOUS], area)
    result[CONSERVATION_SHARE_COLUMN] = covered_share(boundaries, barred[barred["kind"] == CONSERVATION], area)
    result[PROTECTED_SHARE_COLUMN] = covered_share(boundaries, barred, area)
    result[PROTECTED_2012_SHARE_COLUMN] = covered_share(boundaries, barred[barred["in_force_2012"]], area)
    return result.rename_axis(MUN_ID_COLUMN).reset_index()[list(ADM2_COLUMNS)]


def build_protected_areas(
    root_dir=".",
    gpkg=None,
    funai=None,
    gadm_path=None,
    layer=None,
    adm2_id_column=None,
    areas_output_path=None,
    adm2_output_path=None,
):
    """Write the merged protected areas and their area shares per municipality."""
    gpkg = Path(gpkg) if gpkg else gpkg_path(root_dir)
    funai = Path(funai) if funai else funai_path(root_dir)
    if funai is None:
        raise FileNotFoundError(
            "No FUNAI snapshot (raw/funai_<date>/tis_poligonais_portarias.zip) under data/protected_areas; "
            "it is placed by hand."
        )
    areas = build_areas(gpkg, funai)
    logger.info(
        "Protected areas: %d Indigenous lands, %d conservation units; %d bar mining, %d of those in force by %d",
        int((areas["kind"] == INDIGENOUS).sum()), int((areas["kind"] == CONSERVATION).sum()),
        int(areas["mining_barred"].sum()), int((areas["mining_barred"] & areas["in_force_2012"]).sum()), CUTOFF_YEAR,
    )

    boundaries = _load_adm2_boundaries(
        root_dir, gadm_path=gadm_path, layer=layer, adm2_id_column=adm2_id_column
    ).to_crs(BRAZIL_PROJECTED_CRS)
    boundaries[MUN_ID_COLUMN] = boundaries["adm2_id"].map(derive_mun_id_from_adm2_id)
    shares = adm2_shares(areas, boundaries[[MUN_ID_COLUMN, "geometry"]])

    outputs = []
    for frame, path in ((areas, areas_output_path or DEFAULT_AREAS_OUTPUT_PATH),
                        (shares, adm2_output_path or DEFAULT_ADM2_OUTPUT_PATH)):
        resolved = Path(root_dir) / path
        resolved.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(resolved, index=False)
        outputs.append(resolved)
    logger.info("Wrote %d protected areas to %s and ADM2 shares %s to %s", len(areas), outputs[0], shares.shape,
                outputs[1])
    return areas, shares
