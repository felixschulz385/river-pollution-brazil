import pandas as pd
import geopandas as gpd
import requests
from xml.etree import ElementTree as ET

from ..database import (
    RAW_STATIONS_TABLE,
    STATIONS_TABLE,
    STATION_RIVERS_TABLE,
    read_geodataframe_table,
    write_geodataframe_table,
)
from ..paths import (
    ensure_water_quality_dirs,
    get_root_path,
)


STATION_INVENTORY_URL = (
    "http://telemetriaws1.ana.gov.br/ServiceANA.asmx/HidroInventario?"
    "codEstDE=&codEstATE=&tpEst=&nmEst=&nmRio=&codSubBacia=&codBacia=&"
    "nmMunicipio=&nmEstado=&sgResp=&sgOper=&telemetrica="
)
DEFAULT_BRAZIL_BOUNDARY_PATH = "data/misc/gadm41_BRA.gpkg"
DEFAULT_BRAZIL_BOUNDARY_LAYER = "ADM_ADM_0"
DEFAULT_RIVER_NETWORK_DIR = "data/river_network"
BRAZIL_PROJECTED_CRS = 5641


def parse_station_inventory_xml(xml_content):
    """Convert the ANA XML response into a plain tabular DataFrame."""
    root = ET.fromstring(xml_content)
    station_rows = []

    for table in root.findall(".//Table"):
        row = {}
        for child in table:
            row[child.tag] = child.text
        station_rows.append(row)

    return pd.DataFrame(station_rows)


def fetch_station_inventory(root_dir="."):
    """Fetch the raw ANA station inventory and cache it in DuckDB."""
    ensure_water_quality_dirs(root_dir)
    response = requests.get(url=STATION_INVENTORY_URL, timeout=60)
    response.raise_for_status()

    stations = parse_station_inventory_xml(response.content)
    stations_geo = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(
            pd.to_numeric(stations["Longitude"], errors="coerce"),
            pd.to_numeric(stations["Latitude"], errors="coerce"),
        ),
        crs=4326,
    )
    # Persist the raw inventory immediately so the preprocess step can be rerun
    # without touching the remote API again.
    write_geodataframe_table(root_dir, RAW_STATIONS_TABLE, stations_geo)
    return stations_geo


def resolve_brazil_boundary_path(root_dir=".", brazil_boundary_path=None):
    if brazil_boundary_path is None:
        return get_root_path(root_dir) / DEFAULT_BRAZIL_BOUNDARY_PATH
    return get_root_path(root_dir) / brazil_boundary_path


def resolve_river_network_dir(root_dir=".", river_network_dir=None):
    if river_network_dir is None:
        return get_root_path(root_dir) / DEFAULT_RIVER_NETWORK_DIR
    return get_root_path(root_dir) / river_network_dir


def _column_name(frame, *candidates):
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise KeyError(f"None of these columns exist in the station inventory: {', '.join(candidates)}")


def filter_station_inventory(stations_geo, brazil_boundary_path):
    """Apply the geographic sanity checks used by the downstream pipeline."""
    stations_geo = stations_geo.copy()
    longitude_column = _column_name(stations_geo, "Longitude", "longitude")
    latitude_column = _column_name(stations_geo, "Latitude", "latitude")
    stations_geo[longitude_column] = pd.to_numeric(stations_geo[longitude_column], errors="coerce")
    stations_geo[latitude_column] = pd.to_numeric(stations_geo[latitude_column], errors="coerce")
    stations_geo = stations_geo.loc[stations_geo[longitude_column] > -100].copy()
    brazil = gpd.read_file(
        brazil_boundary_path,
        layer=DEFAULT_BRAZIL_BOUNDARY_LAYER,
        engine="pyogrio",
    )
    brazil_geometry = brazil.union_all()
    stations_geo["in_bounds"] = stations_geo.within(brazil_geometry)
    return stations_geo.loc[stations_geo["in_bounds"]].copy()


def preprocess_station_inventory(
    root_dir=".",
    brazil_boundary_path=None,
    river_network_dir=None,
):
    try:
        from ...river_network import RiverNetwork
    except ImportError:
        from river_network import RiverNetwork

    # The fetch step populates the raw station inventory in DuckDB. Preprocess
    # narrows that raw feed down to a single curated station-to-trench table.
    stations_geo = read_geodataframe_table(root_dir, RAW_STATIONS_TABLE)
    stations_geo = filter_station_inventory(
        stations_geo,
        resolve_brazil_boundary_path(root_dir, brazil_boundary_path),
    )

    network = RiverNetwork()
    network.load(resolve_river_network_dir(root_dir, river_network_dir))
    if network.trenches is None:
        raise ValueError("River network trenches are required before preprocessing stations.")

    trenches = network.trenches[["trench_id", "geometry"]].copy()
    station_code_column = _column_name(stations_geo, "Codigo", "codigo")
    station_matches = gpd.sjoin_nearest(
        stations_geo[[station_code_column, "geometry"]].to_crs(BRAZIL_PROJECTED_CRS),
        trenches.to_crs(BRAZIL_PROJECTED_CRS),
        how="left",
        distance_col="distance_to_river",
    )
    station_matches = pd.DataFrame(
        station_matches[[station_code_column, "trench_id", "distance_to_river"]]
    ).sort_values([station_code_column, "distance_to_river"]).drop_duplicates(
        subset=[station_code_column], keep="first"
    )
    stations_with_trench = stations_geo.merge(
        station_matches.drop(columns=["distance_to_river"]),
        on=station_code_column,
        how="left",
    )

    # `stations` captures the filtered station inventory with original source
    # fields. The companion `stations_rivers` table adds only the trench join
    # key used by downstream scraping and analysis tasks.
    write_geodataframe_table(
        root_dir,
        STATIONS_TABLE,
        stations_geo.reset_index(drop=True),
    )
    write_geodataframe_table(root_dir, STATION_RIVERS_TABLE, stations_with_trench.reset_index(drop=True))
    return stations_with_trench
