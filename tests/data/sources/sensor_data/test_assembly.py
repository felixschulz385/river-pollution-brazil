from __future__ import annotations

import math

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon

from src.data.sources.sensor_data.preprocess import assembly as assembly_module
from src.data.sources.sensor_data.preprocess.assembly import (
    DISCHARGE_COLUMN,
    STATION_CODE_COLUMN,
    DATE_COLUMN,
    _aggregate_streamflow_matches,
    _filter_stations_to_brazil,
    _join_stations_to_trenches,
    _prepare_station_trenches,
    _prepare_streamflow_features,
)


class _FakeRiverNetwork:
    """Stand-in for RiverNetwork that skips the real on-disk trench index."""

    def __init__(self, trenches):
        self.trenches = trenches


def test_filter_stations_to_brazil_keeps_only_within_boundary(monkeypatch):
    boundary = gpd.GeoDataFrame(
        {"geometry": [Polygon([(-50.0, -15.0), (-40.0, -15.0), (-40.0, -5.0), (-50.0, -5.0)])]},
        crs=4326,
    )
    monkeypatch.setattr(assembly_module.gpd, "read_file", lambda *args, **kwargs: boundary)

    stations = gpd.GeoDataFrame(
        {
            "station_code": ["in_bounds", "out_of_bounds"],
            "geometry": [Point(-45.0, -10.0), Point(10.0, 10.0)],
        },
        crs=4326,
    )

    result = _filter_stations_to_brazil(stations, "unused.gpkg")

    assert result["station_code"].tolist() == ["in_bounds"]


def test_join_stations_to_trenches_matches_nearest():
    stations = gpd.GeoDataFrame(
        {
            "station_code": ["11111111", "22222222"],
            "geometry": [Point(-45.0, -10.0), Point(-46.0, -12.0)],
        },
        crs=4326,
    )
    trenches = gpd.GeoDataFrame(
        {"trench_id": [1, 2]},
        geometry=[
            LineString([(-45.0, -10.0), (-45.1, -10.1)]),
            LineString([(-46.0, -12.0), (-46.1, -12.1)]),
        ],
        crs=4326,
    )
    network = _FakeRiverNetwork(trenches)

    result = _join_stations_to_trenches(stations, network)

    assert sorted(result["station_code"].tolist()) == ["11111111", "22222222"]
    assert sorted(result["trench_id"].tolist()) == [1, 2]


def test_prepare_station_trenches_dedupes_and_casts_types():
    stations_rivers = pd.DataFrame(
        {
            "station_code": ["12345678", "12345678", "87654321", None],
            "trench_id": [101, 101, 202, 303],
        }
    )

    result = _prepare_station_trenches(stations_rivers)

    assert result["station_code"].tolist() == ["12345678", "87654321"]
    assert result["trench_id"].tolist() == [101, 202]


def test_prepare_station_trenches_requires_expected_columns():
    stations_rivers = pd.DataFrame({"station_code": ["12345678"]})

    with pytest.raises(ValueError):
        _prepare_station_trenches(stations_rivers)


def test_aggregate_streamflow_matches_weights_and_dedupes():
    date = pd.Timestamp("2020-01-01")
    station_matches = pd.DataFrame(
        {
            "wq_station_code": ["wq1", "wq1"],
            "streamflow_station_code": ["sf1", "sf2"],
            "streamflow_distance_m": [100.0, 200.0],
            "streamflow_weight": [0.5, 0.3],
        }
    )
    # sf1's row is duplicated (e.g. from a many-to-many join upstream) to exercise
    # the dedup path. Duplication must NOT change the weighted mean across
    # distinct stations: sf1 (weight 0.5, value 20) and sf2 (weight 0.3, value 5)
    # must each contribute once, giving (20*0.5 + 5*0.3) / (0.5 + 0.3) = 14.375 for
    # mean_7d -- not 21.5/1.3, which is what you'd get if the duplicate row
    # doubled sf1's weight relative to sf2.
    streamflow_features = pd.DataFrame(
        {
            STATION_CODE_COLUMN: ["sf1", "sf1", "sf2"],
            DATE_COLUMN: [date, date, date],
            "streamflow_discharge_day": [10.0, 10.0, float("nan")],
            "streamflow_discharge_mean_7d": [20.0, 20.0, 5.0],
            "streamflow_discharge_mean_31d": [30.0, 30.0, float("nan")],
        }
    )
    water_quality_keys = pd.DataFrame({"wq_station_code": ["wq1"], DATE_COLUMN: [date]})

    result = _aggregate_streamflow_matches(water_quality_keys, station_matches, streamflow_features)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["wq_station_code"] == "wq1"
    assert row["streamflow_discharge_day"] == pytest.approx(10.0)
    assert row["streamflow_discharge_mean_7d"] == pytest.approx(11.5 / 0.8)
    assert row["streamflow_discharge_mean_31d"] == pytest.approx(30.0)
    assert row["streamflow_match_count"] == 2
    assert row["streamflow_nonnull_day_count"] == 1
    assert row["streamflow_total_weight"] == pytest.approx(0.8)
    assert row["streamflow_nearest_distance_m"] == pytest.approx(100.0)


def test_aggregate_streamflow_matches_returns_nan_when_all_values_missing():
    date = pd.Timestamp("2020-01-01")
    station_matches = pd.DataFrame(
        {
            "wq_station_code": ["wq1"],
            "streamflow_station_code": ["sf1"],
            "streamflow_distance_m": [100.0],
            "streamflow_weight": [0.5],
        }
    )
    streamflow_features = pd.DataFrame(
        {
            STATION_CODE_COLUMN: ["sf1"],
            DATE_COLUMN: [date],
            "streamflow_discharge_day": [float("nan")],
            "streamflow_discharge_mean_7d": [float("nan")],
            "streamflow_discharge_mean_31d": [float("nan")],
        }
    )
    water_quality_keys = pd.DataFrame({"wq_station_code": ["wq1"], DATE_COLUMN: [date]})

    result = _aggregate_streamflow_matches(water_quality_keys, station_matches, streamflow_features)

    row = result.iloc[0]
    assert math.isnan(row["streamflow_discharge_day"])
    assert row["streamflow_nonnull_day_count"] == 0
    assert row["streamflow_match_count"] == 1


def test_prepare_streamflow_features_rolling_window_is_calendar_day_based():
    """A station with a date gap must get a 7-day mean anchored to actual
    elapsed time, not to the 7 most recent *rows* (which could span far more
    than 7 calendar days if readings are missing in between)."""
    streamflow = pd.DataFrame(
        {
            STATION_CODE_COLUMN: ["12345678"] * 4,
            DATE_COLUMN: [
                "2020-01-01",
                "2020-01-02",
                # a gap: no readings for 2020-01-03 through 2020-01-09
                "2020-01-10",
                "2020-01-11",
            ],
            DISCHARGE_COLUMN: [10.0, 20.0, 30.0, 40.0],
        }
    )

    features = _prepare_streamflow_features(streamflow)

    last_row = features.iloc[-1]
    # A row-count-based 7-row window would have averaged all 4 rows (10, 20,
    # 30, 40) = 25.0. A calendar-day 7-day window only includes rows within
    # 7 days of 2020-01-11, i.e. 2020-01-10 and 2020-01-11 -> (30 + 40) / 2.
    assert last_row["streamflow_discharge_mean_7d"] == pytest.approx(35.0)
