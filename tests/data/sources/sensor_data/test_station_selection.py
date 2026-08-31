from __future__ import annotations

import pandas as pd

from src.data.sources.sensor_data.fetch.data.station_selection import load_queryable_stations
from src.data.sources.sensor_data.fetch.database import STATIONS_TABLE, write_dataframe_table


def test_load_queryable_stations_sources_codes_from_stations_table_alone(tmp_path):
    """Station selection must not require the (now assembly-stage-only)
    stations_rivers join -- every station in the bbox-filtered inventory is
    eligible for scraping."""
    root_dir = str(tmp_path)
    stations = pd.DataFrame(
        {
            "Codigo": ["11111111", "22222222"],
            "Nome": ["Station One", "Station Two"],
        }
    )
    write_dataframe_table(root_dir, STATIONS_TABLE, stations)

    result = load_queryable_stations(root_dir=root_dir)

    assert sorted(result.index.tolist()) == ["11111111", "22222222"]
    assert result.loc["11111111", "station_name"] == "Station One"
