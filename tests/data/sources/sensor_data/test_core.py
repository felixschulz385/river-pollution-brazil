from __future__ import annotations

from src.data.sources.sensor_data.core import SensorData


def test_sensor_data_initializes_all_constructor_options():
    agent = SensorData(
        root_dir="/tmp/example",
        brazil_boundary_path="boundary.gpkg",
        river_network_dir="data/river_network",
        download_dir="/tmp/downloads",
        headless=True,
        keep_browser_on_error=True,
        single_station="12345678",
        fetch_mode="missing-only",
        preprocess_workers=4,
        source_tables=["water_quality"],
        preprocess_backend="process",
        log_every_tables=10,
    )

    assert agent.root_dir == "/tmp/example"
    assert agent.brazil_boundary_path == "boundary.gpkg"
    assert agent.river_network_dir == "data/river_network"
    assert agent.download_dir == "/tmp/downloads"
    assert agent.headless is True
    assert agent.keep_browser_on_error is True
    assert agent.single_station == "12345678"
    assert agent.fetch_mode == "missing-only"
    assert agent.preprocess_workers == 4
    assert agent.source_tables == ["water_quality"]
    assert agent.preprocess_backend == "process"
    assert agent.log_every_tables == 10
