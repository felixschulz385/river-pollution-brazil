import pandas as pd
import geopandas as gpd
import requests
from xml.etree import ElementTree as ET

from ..database import (
    RAW_STATIONS_TABLE,
    STATIONS_TABLE,
    read_geodataframe_table,
    write_geodataframe_table,
)
from ...constants import (
    ensure_water_quality_dirs,
)


STATION_INVENTORY_URL = (
    "http://telemetriaws1.ana.gov.br/ServiceANA.asmx/HidroInventario?"
    "codEstDE=&codEstATE=&tpEst=&nmEst=&nmRio=&codSubBacia=&codBacia=&"
    "nmMunicipio=&nmEstado=&sgResp=&sgOper=&telemetrica="
)


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


def _column_name(frame, *candidates):
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise KeyError(f"None of these columns exist in the station inventory: {', '.join(candidates)}")


def filter_station_inventory(stations_geo):
    """Apply a cheap geographic sanity check to the raw station inventory.

    This is intentionally not a precise Brazil-boundary filter (that requires
    GADM, which is only available at assembly time) -- just a bounds check to
    drop obviously bad coordinates before scraping station data.
    """
    stations_geo = stations_geo.copy()
    longitude_column = _column_name(stations_geo, "Longitude", "longitude")
    latitude_column = _column_name(stations_geo, "Latitude", "latitude")
    stations_geo[longitude_column] = pd.to_numeric(stations_geo[longitude_column], errors="coerce")
    stations_geo[latitude_column] = pd.to_numeric(stations_geo[latitude_column], errors="coerce")
    return stations_geo.loc[stations_geo[longitude_column] > -100].copy()


def preprocess_station_inventory(root_dir="."):
    # The fetch step populates the raw station inventory in DuckDB. Preprocess
    # narrows that raw feed down with a cheap sanity filter -- the precise
    # Brazil-boundary filter and river-trench join happen later, at assembly
    # time, since those require GADM and river_network.
    stations_geo = read_geodataframe_table(root_dir, RAW_STATIONS_TABLE)
    stations_geo = filter_station_inventory(stations_geo)
    write_geodataframe_table(
        root_dir,
        STATIONS_TABLE,
        stations_geo.reset_index(drop=True),
    )
    return stations_geo
