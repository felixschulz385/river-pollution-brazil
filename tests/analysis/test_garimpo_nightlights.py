"""Tests for night lights as a garimpo activity measure: brightness classes, composite shocks and tracking."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import garimpo_laundering as L
from src.analysis import garimpo_nightlights as N
from src.analysis import garimpo_shift as G


def _frame(n_municipalities=100, years=range(2012, 2025), seed=7):
    """Municipality-years. Light rises 0.5 log points with each log garimpo case; from 2023 the composite adds 0.6
    to the two dimmest brightness classes, where IGD lies (every fourth municipality there). Only every other
    municipality has gold pixels."""
    rng = np.random.default_rng(seed)
    price = pd.Series(np.cumsum(rng.normal(0.08, 0.15, len(years))), index=list(years))
    base = np.sort(rng.normal(5.0, 1.5, n_municipalities))  # dimmest first
    rows = []
    for i in range(n_municipalities):
        for year in years:
            cases = np.log1p(rng.poisson(5))
            log_light = base[i] + 0.5 * cases + rng.normal(0, 0.1)
            radiance = np.expm1(log_light)
            row = {
                "mun_id": f"15{i:04d}", "year": year, G.CLUSTER: f"b{i % 20}", "i": i,
                G.PRESENCE: rng.normal(), L.DEPOSITS: float(i % 3 == 0), L.S.ALL_GOLD_EXPOSURE: 1.0,
                L.PRICE: price[year], "log1p_garimpo_cases": cases,
            }
            for zone in N.ZONES:
                in_zone = zone == "" or i % 2 == 0
                row[f"viirs_pixels{zone}"] = 100 * in_zone
                row[f"viirs_lit_pixels{zone}"] = 50 * in_zone
                row[f"viirs_radiance{zone}"] = radiance * in_zone
            rows.append(row)
    frame = N.add_lights(pd.DataFrame(rows))  # classes from 2012-2014, before the jump
    dim = frame["brightness"] <= 1
    frame[L.IGD] = (dim & frame["i"].mod(4).eq(0)).astype(float)
    jump = np.where(dim & (frame["year"] >= 2023), np.exp(0.6), 1.0)
    for zone in N.ZONES:
        frame[f"viirs_radiance{zone}"] *= jump
    return N.add_lights(frame)


def test_brightness_classes_are_quintiles_of_early_radiance_constant_within_municipalities():
    frame = _frame()
    per_municipality = frame.groupby("mun_id")["brightness"]

    assert (per_municipality.nunique() == 1).all()
    assert per_municipality.first().value_counts().tolist() == [20] * 5
    dimmest = frame.loc[frame["mun_id"] == "150000", "brightness_year"].iloc[0]
    assert dimmest == "0_2012"
    assert frame.groupby("mun_id")[N.has_pixels("_gold5km")].first().mean() == pytest.approx(0.5)
    assert [N.zone_of(outcome) for outcome in ("log1p_radiance", "log1p_lit_protected",
                                               "log1p_radiance_gold5km_protected")] == ["", "_protected",
                                                                                         "_gold5km_protected"]


def test_brightness_by_year_fe_absorb_a_composite_jump_in_dim_municipalities():
    table = N.run_table(_frame(), outcomes=("log1p_radiance_gold5km",), bootstrap=False)
    igd = table[(table["design"] == "IGD") & (table["term"] == "post2023")].set_index("fe")["estimate"]

    assert igd["year FE"] > 0.3  # the jump loads on IGD, which lies in dim places
    assert abs(igd["brightness x year FE"]) < 0.1
    assert set(table["n_municipalities"]) == {50}  # municipalities with pixels near gold


def test_tracking_recovers_the_light_case_slope():
    row = N.tracking(_frame(), "log1p_radiance_gold5km", "log1p_garimpo_cases", fe=N.BRIGHTNESS_FE)

    assert row["estimate"] == pytest.approx(0.5, abs=0.05)
    assert row["n_municipalities"] == 50
