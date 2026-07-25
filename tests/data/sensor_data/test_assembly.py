from __future__ import annotations

import pandas as pd
import pytest

from src.data.sensor_data.preprocess.assembly import _prepare_station_trenches


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
