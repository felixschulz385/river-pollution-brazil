"""Tests for the raid-intensity design: raid windows by source, rings and the distributed-lag estimates."""

from __future__ import annotations

from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.garimpo import intensity as I
from src.analysis.garimpo import raids as R


def test_window_and_neighbour_sums():
    matrix = pd.DataFrame([[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 5.0, 0.0]], index=["a", "b"], columns=range(10, 14))
    lagged = I.window_sum(matrix, -2, -1)
    assert lagged.loc["a"].tolist()[2:] == [3.0, 5.0] and lagged.loc["a"].isna().sum() == 2
    assert I.window_sum(matrix, 1, 1).loc["b", 11] == 5.0  # the next month
    border = I.neighbour_sum(matrix, {"a": {"b", "outside"}})
    assert border.loc["a"].tolist() == [0.0, 0.0, 5.0, 0.0] and (border.loc["b"] == 0).all()


def test_rings_leave_out_the_municipality_and_its_borders():
    boundaries = gpd.GeoDataFrame({"mun_id": list("abcd")},
                                  geometry=[box(-55 + i, -6, -54 + i, -5) for i in (0, 1, 2, 6)], crs=4326)
    adjacency = {"a": {"b"}, "b": {"a", "c"}, "c": {"b"}}
    ring = I.rings(boundaries, adjacency, radius_km=250)  # neighbouring centroids are about 110 km apart
    assert ring["a"] == {"c"} and ring["b"] == set() and ring["d"] == set()


def _inputs(n=60, seed=11):
    """Municipalities in a row. Raids hit a random third of them in random months; garimpo against other malaria
    falls 0.3 per log raid term in the municipality over the next three months and rises 0.2 per log raid term
    next door over the same lags; raids do not respond to the outcome."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("2018-01-01", "2022-12-01", freq="MS")
    ids = [f"15{i:04d}" for i in range(n)]
    adjacency = {mun: {ids[j] for j in (i - 1, i + 1) if 0 <= j < n} for i, mun in enumerate(ids)}
    ring = {mun: {ids[j] for j in (i - 2, i + 2) if 0 <= j < n} for i, mun in enumerate(ids)}
    raided = rng.choice(ids, n // 3, replace=False)
    raids = pd.DataFrame([{"mun_id": mun, "month": month, "raid_terms": int(rng.integers(1, 6))}
                          for mun in raided for month in months if rng.random() < 0.25])
    panel = pd.DataFrame([{"mun_id": mun, "month": month} for mun in ids for month in months])
    panel["t"] = R.month_index(panel["month"])
    panel["garimpo_vs_other"] = 0.0
    basins = pd.Series({mun: f"b{i % 20}" for i, mun in enumerate(ids)})
    data = I.build_panel(panel, raids, adjacency, ring, basins)
    data["garimpo_vs_other"] = (-0.3 * data["own_lag1_3"] + 0.2 * data["border_lag1_3"]
                                + rng.normal(0, 0.05, len(data)))
    return data


def test_build_panel_keeps_months_with_observed_windows():
    data = _inputs()
    assert data["month"].min() == pd.Timestamp(I.FIRST_MONTH)
    assert data["month"].max() == pd.Timestamp("2022-09-01")  # the last month with a three-month lead
    assert set(I.terms()) <= set(data.columns) and data[I.terms()].notna().all().all()


def test_distributed_lags_recover_the_effects_and_a_null_lead():
    table = I.distributed(_inputs(), "garimpo_vs_other").set_index(["source", "window"])
    assert table.loc[("own", "lag1_3"), "estimate"] == pytest.approx(-0.3, abs=0.03)
    assert table.loc[("border", "lag1_3"), "estimate"] == pytest.approx(0.2, abs=0.03)
    assert abs(table.loc[("own", "lead"), "estimate"]) < 0.03
    assert abs(table.loc[("ring", "year sum"), "estimate"]) < 0.05
    assert table.loc[("own", "year sum"), "estimate"] == pytest.approx(-0.3, abs=0.05)
    rows = {(row["source"], row["window"]): row for row in I.pooled(_inputs(), "garimpo_vs_other", bootstrap=False)}
    assert rows[("own", "past")]["estimate"] < -0.1 and rows[("own", "past")]["p"] < 0.01
