"""Tests for Moreno-Louzada's design: the sample and the price-exposure and random-walk inferences."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.core.estimation import estimate, interaction
from src.analysis.core.inference import demean, price_exposure_se, random_walk_p
from src.analysis.health import moreno_louzada as M


def _panel(n_municipalities=40, years=range(2005, 2022), seed=0):
    """Municipality-year rows with a binary exposure, a price path and an outcome responding to both."""
    rng = np.random.default_rng(seed)
    price = pd.Series(np.cumsum(rng.normal(0.1, 0.2, len(years))), index=list(years))
    rows = []
    for i in range(n_municipalities):
        exposed = i % 3 == 0
        births = 50 + 10 * i
        for year in years:
            rows.append({
                "mun_id": f"15{i:04d}", "year": year, "sample": i != 1, M.CLUSTER: f"b{i % 8}",
                M.EXPOSURE: float(exposed), M.PRICE: price[year], "births": births + (year % 3),
                "y_raw": 2.0 * exposed * price[year] + rng.normal(0, 1) + i % 5,
            })
    return pd.DataFrame(rows)


def test_spec_data_drops_own_deposits_and_weights_by_mean_births():
    frame = _panel().assign(preterm=1.0)

    data = M.spec_data(frame, "preterm", years=(2007, 2021))

    assert "150001" not in set(data["mun_id"])
    assert data["year"].min() == M.PRETERM_FIRST_YEAR
    first = data[data["mun_id"] == "150000"]
    assert first["weight"].iloc[0] == pytest.approx(first["births"].mean())
    assert data[interaction(M.EXPOSURE, M.PRICE)].equals(data[M.EXPOSURE] * data[M.PRICE])


def test_price_exposure_se_matches_the_plug_in_formula():
    frame = _panel().rename(columns={"y_raw": "outcome"})
    data = M.spec_data(frame, "outcome", years=(2007, 2021))
    term = interaction(M.EXPOSURE, M.PRICE)
    model = estimate(data, [term])

    se, n_years = price_exposure_se(model, term, M.EXPOSURE, M.PRICE)

    # Two-way weighted demeaning of z and the residuals by hand.
    w = data["weight"].to_numpy()
    z = demean(data[[term]].to_numpy(), data, M.FE, w)[:, 0]
    y = demean(data[["y"]].to_numpy(), data, M.FE, w)[:, 0]
    beta = (w * z * y).sum() / (w * z ** 2).sum()
    u = y - beta * z
    price = data.groupby("year")[M.PRICE].first()
    sums = pd.Series(data[M.EXPOSURE].to_numpy() * w * u).groupby(data["year"].to_numpy()).sum()
    expected = np.sqrt(((price - price.mean()) ** 2 * sums ** 2).sum() / (w * z ** 2).sum() ** 2)

    assert model.coef()[term] == pytest.approx(beta, rel=1e-6)
    assert n_years == 15
    assert se == pytest.approx(expected, rel=1e-5)


def test_random_walk_observed_coefficient_equals_the_fit_and_p_is_a_share():
    frame = _panel().rename(columns={"y_raw": "outcome"})
    data = M.spec_data(frame, "outcome", years=(2007, 2021))
    model = estimate(data, [interaction(M.EXPOSURE, M.PRICE)])

    observed, p = random_walk_p(data, M.EXPOSURE, M.PRICE, draws=199)

    assert observed == pytest.approx(model.coef().iloc[0], rel=1e-6)
    assert 1 / 200 <= p <= 1
