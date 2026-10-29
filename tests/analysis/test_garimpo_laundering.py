"""Tests for lead 1's laundering test: exposure terms by window, the permutation inference and the event study."""

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
from src.analysis import moreno_louzada as M


def _frame(n_municipalities=60, years=range(2004, 2025), effect=1.0, seed=2):
    """Municipality-years with IGD in every sixth municipality and an outcome that rises with IGD from 2013."""
    rng = np.random.default_rng(seed)
    price = pd.Series(np.cumsum(rng.normal(0.08, 0.15, len(years))), index=list(years))
    rows = []
    for i in range(n_municipalities):
        igd = float(i % 6 == 0)
        for year in years:
            rows.append({
                "mun_id": f"15{i:04d}", "year": year, L.CLUSTER: f"b{i % 12}", "state_year": f"15_{year}",
                L.IGD: igd, L.TIN_IGD: float(i % 7 == 0), L.DEPOSITS: float(i % 3 == 0), "gold_any": 1.0,
                L.PRICE: price[year], "log1p_garimpo_enforcement": 0.0,
                "log1p_garimpo_cases": effect * igd * (year >= 2013) + rng.normal(0, 0.3) + i % 4,
                "c11_garimpo_gold_pp": effect * igd * (year >= 2013) + rng.normal(0, 0.3),
            })
    return pd.DataFrame(rows)


def test_spec_data_drops_the_2023_term_outside_the_window_and_adds_requested_terms():
    frame = _frame()

    _, c11_terms, controls, _ = L.spec_data(frame, "c11_garimpo_gold_pp")
    _, case_terms, _, _ = L.spec_data(frame, "log1p_garimpo_cases", collapse=True, trend=True, placebo=True,
                                      enforcement=True)

    assert c11_terms == [L.term(L.IGD, "post2013")]  # C11 stops in 2019
    assert case_terms == [L.term(L.IGD, period) for period in ("post2013", "post2023", "post2016", "trend")]
    assert controls == [M.interaction(L.DEPOSITS, L.PRICE)]


def test_run_spec_recovers_the_effect_and_the_permutation_matches_the_regression():
    frame = _frame()

    rows = {row["term"]: row for row in L.run_spec(frame, "log1p_garimpo_cases", randomization=True)}
    data, terms, controls, periods = L.spec_data(frame, "log1p_garimpo_cases")
    observed, p_values = L.permutation_p(data, L.IGD, terms, controls, periods, draws=99)

    assert rows["post2013"]["estimate"] == pytest.approx(1.0, abs=0.15)
    assert observed[0] == pytest.approx(rows["post2013"]["estimate"], rel=1e-6)
    assert observed[1] == pytest.approx(rows["post2023"]["estimate"], rel=1e-6)
    assert p_values[0] == pytest.approx(1 / 100)  # no permutation comes close
    assert rows["post2013"]["n_exposed"] == 10


def test_permutation_p_is_not_small_without_an_effect():
    frame = _frame(effect=0.0, seed=3)
    data, terms, controls, periods = L.spec_data(frame, "log1p_garimpo_cases")

    _, p_values = L.permutation_p(data, L.IGD, terms, controls, periods, draws=199)

    assert (p_values > 0.05).all()


def test_event_study_is_zero_in_the_base_year_and_steps_up_from_2013():
    study = L.event_study(_frame(), "log1p_garimpo_cases").set_index("year")

    assert study.loc[L.BASE_YEAR, "estimate"] == 0.0
    assert study.loc[2013:, "estimate"].mean() > 0.8 > abs(study.loc[:2011, "estimate"].mean())
