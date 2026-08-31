from __future__ import annotations

import geopandas as gpd
import pandas as pd

from src.data.sources.sensor_data.fetch.database import (
    RAW_STATIONS_TABLE,
    STATIONS_TABLE,
    read_geodataframe_table,
    write_geodataframe_table,
)
from src.data.sources.sensor_data.fetch.stations.inventory import (
    filter_station_inventory,
    preprocess_station_inventory,
)


def _write_raw_station_inventory(root_dir):
    stations = pd.DataFrame(
        {
            "Codigo": ["11111111", "22222222", "33333333"],
            "Latitude": [-10.0, -12.0, -20.0],
            # The third station has an implausible longitude (outside the
            # `> -100` sanity bound) and must be dropped.
            "Longitude": [-45.0, -46.0, -150.0],
        }
    )
    stations_geo = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations["Longitude"], stations["Latitude"]),
        crs=4326,
    )
    write_geodataframe_table(root_dir, RAW_STATIONS_TABLE, stations_geo)


def test_preprocess_station_inventory_applies_bbox_filter_with_no_gadm_or_river_network(tmp_path):
    """fetch's station-inventory step must not require GADM or river_network
    -- only the cheap longitude sanity bound. The precise Brazil-boundary
    filter and river-trench join happen later, at assembly time."""
    root_dir = str(tmp_path)
    _write_raw_station_inventory(root_dir)

    result = preprocess_station_inventory(root_dir=root_dir)

    assert sorted(result["Codigo"].tolist()) == ["11111111", "22222222"]

    written = read_geodataframe_table(root_dir, STATIONS_TABLE)
    assert sorted(written["Codigo"].tolist()) == ["11111111", "22222222"]


def test_filter_station_inventory_drops_implausible_longitude():
    stations = pd.DataFrame(
        {
            "Codigo": ["11111111", "22222222"],
            "Latitude": [-10.0, -12.0],
            "Longitude": [-45.0, -150.0],
        }
    )
    stations_geo = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations["Longitude"], stations["Latitude"]),
        crs=4326,
    )

    result = filter_station_inventory(stations_geo)

    assert result["Codigo"].tolist() == ["11111111"]
