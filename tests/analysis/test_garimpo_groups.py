"""Tests for lead 2's group responses: the shocks and the local-minus-incoming difference."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import garimpo_groups as P
from src.analysis import garimpo_laundering as L
from src.analysis import garimpo_shift as G


def _frame(n_municipalities=60, years=range(2000, 2025), seed=11):
    """Municipality-years. Local cases rise 0.3 per SD of last year's precipitation deficit; incoming cases rise
    0.5 per log point of the price x deposits; both rise 0.2 with this year's deficit (transmission). One
    municipality in ten never has a garimpo case."""
    rng = np.random.default_rng(seed)
    price = pd.Series(np.cumsum(rng.normal(0.08, 0.15, len(years))), index=list(years))
    rows = []
    for i in range(n_municipalities):
        deposits = float(i % 3 == 0)
        wet = rng.normal(5.0, 0.5)
        rain = wet + rng.normal(0, 0.5, len(years))
        for k, year in enumerate(years):
            rows.append({"mun_id": f"15{i:04d}", "year": year, G.CLUSTER: f"b{i % 15}",
                         "state_year": f"{15 + i % 2}_{year}", L.DEPOSITS: deposits, L.PRICE: price[year],
                         G.PRESENCE: 0.0, "local_tp": rain[k], "tp": rain[k] + rng.normal(0, 0.1),
                         "c11_garimpo_pp": 0.0, "garimpo_site": i % 10 != 0})
            for reach in ("50", "200"):
                for group in ("gold", "unknown", "other"):
                    rows[-1][f"c11_garimpo_{group}_{reach}"] = 0.0
    frame = pd.DataFrame(rows).sort_values(["mun_id", "year"]).reset_index(drop=True)
    dry = P.deficit(frame, "local_tp")
    dry_lag = dry.groupby(frame["mun_id"]).shift().fillna(0.0)
    noise = lambda: rng.normal(0, 0.1, len(frame))
    site = frame["garimpo_site"]
    frame["log1p_garimpo_local"] = site * (2.0 + 0.3 * dry_lag + 0.2 * dry + noise())
    frame["log1p_garimpo_incoming"] = site * (2.0 + 0.5 * frame[L.DEPOSITS] * frame[L.PRICE] + 0.2 * dry + noise())
    frame["log1p_other_malaria"] = 3.0 + 0.2 * dry + noise()
    frame["inf_garimpo"] = np.expm1(frame["log1p_garimpo_local"]) + np.expm1(frame["log1p_garimpo_incoming"])
    return P.add_shocks(frame)


def test_drought_is_a_within_municipality_deficit_and_the_sample_has_garimpo():
    frame = _frame()
    window = frame[frame["year"].between(*P.CLIMATE_YEARS)].groupby("mun_id")["drought"]

    assert window.mean().abs().max() == pytest.approx(0.0, abs=1e-9)
    assert window.std().to_numpy() == pytest.approx(1.0)
    assert (frame["drought"] * (frame["local_tp"] - frame.groupby("mun_id")["local_tp"].transform("mean")) <= 1e-12).all()
    assert frame.groupby("mun_id")[P.SAMPLE].first().mean() == pytest.approx(0.9)


def test_the_difference_separates_residents_from_crews_and_drops_the_common_shock():
    frame = _frame()
    table = P.run_table(frame, spec_table={name: spec for name, spec in P.specs().items()
                                           if name in ("drought", "gold price x deposits")}, bootstrap=False)
    estimate = table.set_index(["spec", "outcome", "term"])["estimate"]

    assert estimate[("drought", "log1p_garimpo_local", "drought_lag")] == pytest.approx(0.3, abs=0.05)
    assert estimate[("drought", "log1p_garimpo_incoming", "drought_lag")] == pytest.approx(0.0, abs=0.05)
    assert estimate[("drought", "local_minus_incoming", "drought")] == pytest.approx(0.0, abs=0.05)
    assert estimate[("drought", "local_minus_incoming", "drought_lag")] == pytest.approx(0.3, abs=0.05)
    price = P.M.interaction(L.DEPOSITS, L.PRICE)
    assert estimate[("gold price x deposits", "local_minus_incoming", price)] == pytest.approx(-0.5, abs=0.1)
    assert set(table["n_municipalities"]) == {54}
