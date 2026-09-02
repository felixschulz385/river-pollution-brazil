from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
import requests

from src.data.sources.sensor_data.fetch.database import (
    RAW_STATIONS_TABLE,
    STATIONS_TABLE,
    read_geodataframe_table,
    write_geodataframe_table,
)
from src.data.sources.sensor_data.fetch.stations import inventory as inventory_module
from src.data.sources.sensor_data.fetch.stations.inventory import (
    fetch_station_inventory,
    filter_station_inventory,
    preprocess_station_inventory,
)


_SAMPLE_INVENTORY_XML = b"""<?xml version="1.0" encoding="utf-8"?>
<NewDataSet>
  <Table><Codigo>11111111</Codigo><Latitude>-10.0</Latitude><Longitude>-45.0</Longitude></Table>
  <Table><Codigo>22222222</Codigo><Latitude>-12.0</Latitude><Longitude>-46.0</Longitude></Table>
</NewDataSet>
"""


class _FakeResponse:
    def __init__(self, *, content=b"", status_code=200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.HTTPError(f"{self.status_code} Server Error")
            error.response = self
            raise error


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


def test_fetch_station_inventory_retries_transient_server_errors(tmp_path, monkeypatch):
    """A run of HTTP 500s from ANA must be retried, not fatal, as long as a
    later attempt succeeds."""
    root_dir = str(tmp_path)
    responses = [
        _FakeResponse(status_code=500),
        _FakeResponse(status_code=500),
        _FakeResponse(content=_SAMPLE_INVENTORY_XML),
    ]
    calls = []

    def fake_get(url, timeout=None):
        calls.append(url)
        return responses[len(calls) - 1]

    monkeypatch.setattr(inventory_module.requests, "get", fake_get)
    monkeypatch.setattr(inventory_module.time, "sleep", lambda _seconds: None)

    result = fetch_station_inventory(root_dir=root_dir)

    assert len(calls) == 3
    assert sorted(result["Codigo"].tolist()) == ["11111111", "22222222"]
    written = read_geodataframe_table(root_dir, RAW_STATIONS_TABLE)
    assert sorted(written["Codigo"].tolist()) == ["11111111", "22222222"]


def test_fetch_station_inventory_falls_back_to_cached_table_when_ana_is_down(tmp_path, monkeypatch):
    """If every attempt fails but a previous inventory is cached in DuckDB,
    reuse the cache so the rest of the fetch pipeline can still run."""
    root_dir = str(tmp_path)
    _write_raw_station_inventory(root_dir)

    def always_500(url, timeout=None):
        return _FakeResponse(status_code=500)

    monkeypatch.setattr(inventory_module.requests, "get", always_500)
    monkeypatch.setattr(inventory_module.time, "sleep", lambda _seconds: None)

    result = fetch_station_inventory(root_dir=root_dir)

    assert sorted(result["Codigo"].tolist()) == ["11111111", "22222222", "33333333"]


def test_fetch_station_inventory_raises_when_down_and_no_cache(tmp_path, monkeypatch):
    """With no cached inventory to fall back on, an ANA outage is still fatal."""
    root_dir = str(tmp_path)

    def always_500(url, timeout=None):
        return _FakeResponse(status_code=500)

    monkeypatch.setattr(inventory_module.requests, "get", always_500)
    monkeypatch.setattr(inventory_module.time, "sleep", lambda _seconds: None)

    with pytest.raises(requests.HTTPError):
        fetch_station_inventory(root_dir=root_dir)


def test_fetch_station_inventory_does_not_retry_client_errors(tmp_path, monkeypatch):
    """A 4xx won't fix itself -- fail fast instead of burning the retry budget."""
    root_dir = str(tmp_path)
    calls = []

    def fake_get(url, timeout=None):
        calls.append(url)
        return _FakeResponse(status_code=404)

    monkeypatch.setattr(inventory_module.requests, "get", fake_get)
    monkeypatch.setattr(inventory_module.time, "sleep", lambda _seconds: None)

    with pytest.raises(requests.HTTPError):
        fetch_station_inventory(root_dir=root_dir, allow_stale_cache=False)
    assert len(calls) == 1


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
