import logging
import time
from xml.etree import ElementTree as ET

import pandas as pd
import geopandas as gpd
import requests

from ..database import (
    RAW_STATIONS_TABLE,
    STATIONS_TABLE,
    read_geodataframe_table,
    table_exists,
    write_geodataframe_table,
)
from ...constants import (
    ensure_water_quality_dirs,
)


logger = logging.getLogger(__name__)


STATION_INVENTORY_URL = (
    "http://telemetriaws1.ana.gov.br/ServiceANA.asmx/HidroInventario?"
    "codEstDE=&codEstATE=&tpEst=&nmEst=&nmRio=&codSubBacia=&codBacia=&"
    "nmMunicipio=&nmEstado=&sgResp=&sgOper=&telemetrica="
)

# The ANA endpoint is frequently overloaded and answers with HTTP 500 (or drops
# the connection) for minutes at a time. Retry transient failures with an
# exponential backoff before giving up.
STATION_INVENTORY_MAX_ATTEMPTS = 5
STATION_INVENTORY_BACKOFF_SECONDS = 5.0


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


def _is_retryable_http_error(error):
    """Retry 5xx and 429 (server-side / rate limit); a 4xx won't fix itself."""
    status_code = getattr(error.response, "status_code", None)
    return status_code is None or status_code >= 500 or status_code == 429


def _download_station_inventory_frame():
    """Fetch and parse the ANA station inventory, retrying transient failures.

    Returns a non-empty DataFrame, or re-raises the last error once the retry
    budget is exhausted.
    """
    last_error = None
    for attempt in range(1, STATION_INVENTORY_MAX_ATTEMPTS + 1):
        try:
            response = requests.get(url=STATION_INVENTORY_URL, timeout=60)
            response.raise_for_status()
            stations = parse_station_inventory_xml(response.content)
            if stations.empty:
                raise ValueError("ANA station inventory response contained no station rows.")
            return stations
        except requests.HTTPError as error:
            if not _is_retryable_http_error(error):
                raise
            last_error = error
        except (requests.RequestException, ET.ParseError, ValueError) as error:
            last_error = error
        if attempt == STATION_INVENTORY_MAX_ATTEMPTS:
            break
        wait_seconds = STATION_INVENTORY_BACKOFF_SECONDS * (2 ** (attempt - 1))
        logger.warning(
            "Station inventory fetch attempt %s/%s failed (%s); retrying in %.0fs.",
            attempt,
            STATION_INVENTORY_MAX_ATTEMPTS,
            last_error,
            wait_seconds,
        )
        time.sleep(wait_seconds)
    raise last_error


def fetch_station_inventory(root_dir=".", *, allow_stale_cache=True):
    """Fetch the raw ANA station inventory and cache it in DuckDB.

    Transient failures against the ANA endpoint are retried with an exponential
    backoff. If every attempt still fails and a previously fetched inventory is
    already cached in DuckDB, that stale copy is reused (with a warning) so the
    rest of the fetch pipeline can proceed instead of aborting on an ANA outage.
    Pass ``allow_stale_cache=False`` to require a fresh download.
    """
    ensure_water_quality_dirs(root_dir)
    try:
        stations = _download_station_inventory_frame()
    except (requests.RequestException, ET.ParseError, ValueError) as error:
        if allow_stale_cache and table_exists(root_dir, RAW_STATIONS_TABLE):
            logger.warning(
                "Could not refresh the ANA station inventory (%s); "
                "falling back to the cached copy already in DuckDB.",
                error,
            )
            return read_geodataframe_table(root_dir, RAW_STATIONS_TABLE)
        raise

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
