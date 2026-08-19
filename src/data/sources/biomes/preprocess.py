import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

from src.data.sources.land_cover.constants import derive_mun_id_from_adm2_id

from .constants import (
    BIOME_COLUMN,
    BIOME_NAME_COLUMN_CANDIDATES,
    BIOME_SHAPEFILE_EXCLUDE_HINT,
    BIOME_SHAPEFILE_NAME_HINT,
    BRAZIL_PROJECTED_CRS,
    DEFAULT_ADM2_ID_COLUMN,
    DEFAULT_ADM2_LAYER,
    DEFAULT_ADM2_OUTPUT_PATH,
    DEFAULT_GADM_PATH,
    DEFAULT_SENSOR_OUTPUT_PATH,
    MUN_ID_COLUMN,
    STATION_CODE_COLUMN,
    raw_dir,
)


logger = logging.getLogger(__name__)


def _find_biome_shapefile(root_dir="."):
    """Locate the terrestrial biomes shapefile within the extracted archive."""
    candidates = sorted(raw_dir(root_dir).rglob("*.shp"))
    matches = [
        path
        for path in candidates
        if BIOME_SHAPEFILE_NAME_HINT in path.stem.lower()
        and BIOME_SHAPEFILE_EXCLUDE_HINT not in path.stem.lower()
    ]
    if not matches:
        raise FileNotFoundError(
            f"No biome shapefile found under {raw_dir(root_dir)} "
            f"(looked for '*{BIOME_SHAPEFILE_NAME_HINT}*.shp', excluding "
            f"'*{BIOME_SHAPEFILE_EXCLUDE_HINT}*')."
        )
    return matches[0]


def _biome_name_column(frame):
    for candidate in BIOME_NAME_COLUMN_CANDIDATES:
        if candidate in frame.columns:
            return candidate
    raise KeyError(
        f"None of the expected biome name columns {BIOME_NAME_COLUMN_CANDIDATES} "
        f"were found; available columns: {list(frame.columns)}."
    )


def load_biome_polygons(root_dir=".", shapefile_path=None):
    """Load and standardize the terrestrial biome polygons."""
    shapefile_path = shapefile_path or _find_biome_shapefile(root_dir)
    logger.info("Loading biome polygons from %s", shapefile_path)
    polygons = gpd.read_file(shapefile_path)
    name_column = _biome_name_column(polygons)
    polygons = polygons.rename(columns={name_column: BIOME_COLUMN})[[BIOME_COLUMN, "geometry"]]
    return polygons.dissolve(by=BIOME_COLUMN, as_index=False)


def _load_adm2_boundaries(root_dir=".", gadm_path=None, layer=None, adm2_id_column=None):
    gadm_path = Path(root_dir) / (gadm_path or DEFAULT_GADM_PATH)
    layer = layer or DEFAULT_ADM2_LAYER
    adm2_id_column = adm2_id_column or DEFAULT_ADM2_ID_COLUMN
    boundaries = gpd.read_file(gadm_path, layer=layer)
    boundaries["adm2_id"] = boundaries[adm2_id_column]
    return boundaries[["adm2_id", "geometry"]]


def build_adm2_biomes(
    root_dir=".",
    shapefile_path=None,
    gadm_path=None,
    layer=None,
    adm2_id_column=None,
    output_path=None,
):
    """Assign each ADM2 municipality its dominant biome by intersecting area."""
    biome_polygons = load_biome_polygons(root_dir, shapefile_path=shapefile_path)
    adm2_boundaries = _load_adm2_boundaries(
        root_dir, gadm_path=gadm_path, layer=layer, adm2_id_column=adm2_id_column
    )

    biome_projected = biome_polygons.to_crs(BRAZIL_PROJECTED_CRS)
    adm2_projected = adm2_boundaries.to_crs(BRAZIL_PROJECTED_CRS)

    overlay = gpd.overlay(adm2_projected, biome_projected, how="intersection")
    overlay["_area"] = overlay.geometry.area
    dominant = (
        overlay.sort_values("_area", ascending=False)
        .drop_duplicates(subset=["adm2_id"], keep="first")[["adm2_id", BIOME_COLUMN]]
        .reset_index(drop=True)
    )
    dominant[MUN_ID_COLUMN] = dominant["adm2_id"].map(derive_mun_id_from_adm2_id)
    result = dominant[[MUN_ID_COLUMN, BIOME_COLUMN]].sort_values(MUN_ID_COLUMN).reset_index(drop=True)

    resolved_output_path = Path(root_dir) / (output_path or DEFAULT_ADM2_OUTPUT_PATH)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(resolved_output_path, index=False)
    logger.info("Wrote ADM2-to-biome mapping to %s (shape=%s)", resolved_output_path, result.shape)
    return result


STATION_LONGITUDE_COLUMN_CANDIDATES = ("Longitude", "longitude", "LONGITUDE")
STATION_LATITUDE_COLUMN_CANDIDATES = ("Latitude", "latitude", "LATITUDE")


def _first_present_column(frame, candidates):
    return next((candidate for candidate in candidates if candidate in frame.columns), None)


def _load_station_points(root_dir="."):
    from src.data.sources.sensor_data.fetch.database import (
        STATIONS_TABLE,
        read_dataframe_table,
        read_geodataframe_table,
    )

    try:
        stations = read_geodataframe_table(root_dir, STATIONS_TABLE)
    except KeyError:
        # Some `stations` tables were written without geometry metadata (e.g.
        # by an older pipeline run); fall back to building points from raw
        # latitude/longitude columns in that case.
        logger.warning(
            "'%s' table has no recorded geometry column; deriving station "
            "points from latitude/longitude columns instead.",
            STATIONS_TABLE,
        )
        raw_stations = read_dataframe_table(root_dir, STATIONS_TABLE)
        longitude_column = _first_present_column(raw_stations, STATION_LONGITUDE_COLUMN_CANDIDATES)
        latitude_column = _first_present_column(raw_stations, STATION_LATITUDE_COLUMN_CANDIDATES)
        if longitude_column is None or latitude_column is None:
            raise ValueError(
                f"'{STATIONS_TABLE}' table has neither a recorded geometry column nor "
                f"latitude/longitude columns among {STATION_LATITUDE_COLUMN_CANDIDATES} / "
                f"{STATION_LONGITUDE_COLUMN_CANDIDATES}; available columns: "
                f"{list(raw_stations.columns)}."
            ) from None
        longitude = pd.to_numeric(raw_stations[longitude_column], errors="coerce")
        latitude = pd.to_numeric(raw_stations[latitude_column], errors="coerce")
        stations = gpd.GeoDataFrame(
            raw_stations,
            geometry=gpd.points_from_xy(longitude, latitude),
            crs=4326,
        )

    station_code_column = _first_present_column(stations, ("Codigo", "codigo", STATION_CODE_COLUMN))
    if station_code_column is None:
        raise ValueError(
            f"'{STATIONS_TABLE}' table has no station-code column among "
            f"('Codigo', 'codigo', '{STATION_CODE_COLUMN}'); available columns: "
            f"{list(stations.columns)}."
        )
    stations = stations.rename(columns={station_code_column: STATION_CODE_COLUMN})
    return stations[[STATION_CODE_COLUMN, "geometry"]]


def build_station_biomes(root_dir=".", shapefile_path=None, output_path=None):
    """Assign each monitoring station the biome its point location falls within."""
    biome_polygons = load_biome_polygons(root_dir, shapefile_path=shapefile_path)
    stations = _load_station_points(root_dir)

    biome_projected = biome_polygons.to_crs(BRAZIL_PROJECTED_CRS)
    stations_projected = stations.to_crs(BRAZIL_PROJECTED_CRS)

    joined = gpd.sjoin(
        stations_projected, biome_projected, how="left", predicate="within"
    ).drop_duplicates(subset=[STATION_CODE_COLUMN], keep="first")

    unmatched = joined[joined[BIOME_COLUMN].isna()]
    if len(unmatched) > 0:
        # Stations that fall just outside every polygon (e.g. coastline
        # simplification) get the nearest biome instead of a missing value.
        nearest = gpd.sjoin_nearest(
            stations_projected.loc[unmatched.index],
            biome_projected,
            how="left",
            distance_col="_distance",
        ).drop_duplicates(subset=[STATION_CODE_COLUMN], keep="first")
        joined.loc[unmatched.index, BIOME_COLUMN] = nearest[BIOME_COLUMN].to_numpy()

    result = (
        pd.DataFrame(joined[[STATION_CODE_COLUMN, BIOME_COLUMN]])
        .sort_values(STATION_CODE_COLUMN)
        .reset_index(drop=True)
    )

    resolved_output_path = Path(root_dir) / (output_path or DEFAULT_SENSOR_OUTPUT_PATH)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(resolved_output_path, index=False)
    logger.info("Wrote station-to-biome mapping to %s (shape=%s)", resolved_output_path, result.shape)
    return result
