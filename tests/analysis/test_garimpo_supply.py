"""Tests for lead 1's supply test: spec rows, deposit trends, randomization with controls and the event study."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import garimpo_supply as S
from src.analysis import moreno_louzada as M


def _frame(n_municipalities=40, years=range(2004, 2025), seed=1):
    """Municipality-years with a deposit exposure, a trending price and an outcome on both and a deposit trend."""
    rng = np.random.default_rng(seed)
    price = pd.Series(np.cumsum(rng.normal(0.08, 0.15, len(years))), index=list(years))
    rows = []
    for i in range(n_municipalities):
        exposed = float(i % 4 == 0)
        for year in years:
            rows.append({
                "mun_id": f"15{i:04d}", "year": year, S.CLUSTER: f"b{i % 9}", "state_year": f"15_{year}",
                S.EXPOSURE: exposed, S.TIN_EXPOSURE: float(i % 5 == 0), S.PRICE: price[year],
                "trend": float(year - S.TREND_BASE_YEAR),
                "log1p_garimpo_cases": 1.5 * exposed * price[year] + 0.05 * exposed * (year - 2004)
                + rng.normal(0, 0.5) + i % 3,
            })
    return pd.DataFrame(rows)


def test_spec_data_keeps_the_outcome_years_and_adds_trend_terms():
    data = S.spec_data(_frame(), "log1p_garimpo_cases", years=(2006, 2010), trend=True)

    assert data["year"].between(2006, 2010).all()
    term = M.interaction(S.EXPOSURE)
    assert data[term].equals(data[S.EXPOSURE] * data[S.PRICE])
    assert data[S.trend_term(S.EXPOSURE)].equals(data[S.EXPOSURE] * data["trend"])
    assert (data["weight"] == 1.0).all()


def test_random_walk_with_controls_reproduces_the_trend_spec_coefficient():
    data = S.spec_data(_frame(), "log1p_garimpo_cases", trend=True)
    term, trend = M.interaction(S.EXPOSURE), S.trend_term(S.EXPOSURE)
    model = M.estimate(data, [term, trend], weighted=False)

    observed, p = M.random_walk_p(data, S.EXPOSURE, S.PRICE, weighted=False, draws=199, controls=[trend])

    assert observed == pytest.approx(model.coef()[term], rel=1e-6)
    assert 1 / 200 <= p <= 1


def test_run_spec_reports_each_exposure_and_randomizes_single_exposure_specs():
    frame = _frame()

    single = S.run_spec(frame, "log1p_garimpo_cases", trend=True, randomization=True)
    joint = S.run_spec(frame, "log1p_garimpo_cases", exposures=(S.EXPOSURE, S.TIN_EXPOSURE), randomization=True)

    assert [row["exposure"] for row in joint] == [S.EXPOSURE, S.TIN_EXPOSURE]
    assert "p_random_walk" in single[0] and "p_random_walk" not in joint[0]
    assert single[0]["n_exposed"] == 10


def test_event_study_omits_the_base_year():
    study = S.event_study(_frame(), "log1p_garimpo_cases")

    assert study["year"].tolist() == list(range(2004, 2025))
    assert study.loc[study["year"] == 2004, ["estimate", "se"]].to_numpy().tolist() == [[0.0, 0.0]]
    assert study["estimate"].iloc[-1] > study["estimate"].iloc[1]  # the price rises with the trend
