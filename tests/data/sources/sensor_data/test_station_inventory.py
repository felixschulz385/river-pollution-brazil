from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from src.data.sources.sensor_data.fetch.database import (
    RAW_STATIONS_TABLE,
    write_geodataframe_table,
)
from src.data.sources.sensor_data.fetch.stations import inventory as inventory_module
from src.data.sources.sensor_data.preprocess.preprocess import preprocess_stations_rivers
from src.data.sources.sensor_data.schema import STATIONS_RIVERS_COLUMNS


class _FakeRiverNetwork:
    """Stand-in for RiverNetwork that skips the real on-disk trench index."""

    def __init__(self, trenches):
        self._trenches = trenches
        self.trenches = None

    def load(self, _river_network_dir):
        self.trenches = self._trenches


def _write_raw_station_inventory(root_dir):
    stations = pd.DataFrame(
        {
            "Codigo": ["11111111", "22222222"],
            "Latitude": [-10.0, -12.0],
            "Longitude": [-45.0, -46.0],
        }
    )
    stations_geo = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations["Longitude"], stations["Latitude"]),
        crs=4326,
    )
    write_geodataframe_table(root_dir, RAW_STATIONS_TABLE, stations_geo)


def test_preprocess_station_inventory_output_satisfies_stations_rivers_schema(tmp_path, monkeypatch):
    """`preprocess_station_inventory`'s stations_rivers table must satisfy every
    column `preprocess_stations_rivers` requires, and the latter must be able to
    parse the geometry it wrote back out. This is an end-to-end regression test
    for a bug where the write side produced `Codigo`/`geometry` (a live
    GeoSeries) while the read side required `station_code`/`geometry_wkt`
    (WKT text) and `operator_agency_code` (a column nothing ever produced).
    """
    root_dir = str(tmp_path)
    _write_raw_station_inventory(root_dir)

    # Skip the real geographic in-bounds filter (needs a GADM boundary file);
    # every synthetic station is treated as inside Brazil.
    def _fake_filter(stations_geo, _brazil_boundary_path):
        result = stations_geo.copy()
        result["in_bounds"] = True
        return result

    monkeypatch.setattr(inventory_module, "filter_station_inventory", _fake_filter)

    trenches = gpd.GeoDataFrame(
        {"trench_id": [1, 2]},
        geometry=[
            LineString([(-45.0, -10.0), (-45.1, -10.1)]),
            LineString([(-46.0, -12.0), (-46.1, -12.1)]),
        ],
        crs=4326,
    )
    fake_network = _FakeRiverNetwork(trenches)
    # `preprocess_station_inventory` does `from src.data.sources.river_network
    # import RiverNetwork` locally inside the function, resolving the name
    # against the package each call, so patch it there.
    import src.data.sources.river_network as river_network_module

    monkeypatch.setattr(river_network_module, "RiverNetwork", lambda: fake_network)

    inventory_module.preprocess_station_inventory(
        root_dir=root_dir,
        brazil_boundary_path="unused.gpkg",
        river_network_dir="unused",
    )

    # preprocess_stations_rivers() reads the stations_rivers DuckDB table and
    # hard-requires STATIONS_RIVERS_COLUMNS; it must not raise.
    output_path = preprocess_stations_rivers(root_dir)

    result = gpd.read_parquet(output_path)
    assert set(STATIONS_RIVERS_COLUMNS).issubset(result.columns)
    assert sorted(result["station_code"].tolist()) == ["11111111", "22222222"]
    assert sorted(result["trench_id"].tolist()) == [1, 2]
    assert result.geometry.notna().all()


def test_operator_agency_code_is_not_a_required_stations_rivers_column():
    # The raw ANA HidroInventario feed has no operating-agency field, so this
    # column must not be silently fabricated at preprocess time.
    assert "operator_agency_code" not in STATIONS_RIVERS_COLUMNS
