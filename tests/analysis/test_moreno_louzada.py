"""Tests for Moreno-Louzada's design: sample, upstream municipalities and the price-exposure inference."""

from __future__ import annotations

from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import moreno_louzada as M


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


def test_legal_amazon_takes_full_states_and_maranhao_west_of_44(tmp_path):
    path = tmp_path / "gadm.gpkg"
    gpd.GeoDataFrame(
        {"CC_2": ["1500107", "2100055", "2100105", "2200053"],
         "geometry": [box(-50, -5, -49, -4), box(-44.5, -5, -43.5, -4), box(-43.9, -5, -43, -4),
                      box(-45, -5, -44.5, -4)]},
        crs=4326,
    ).to_file(path, layer=M.GADM_LAYER, driver="GPKG")

    assert M.legal_amazon(path) == {"150010", "210005"}


def test_upstream_municipalities_follow_pfafstetter_within_the_parent(tmp_path):
    pd.DataFrame({"trench_id": [1, 2, 3, 4, 5], "otto_basin_l4": ["4583", "4587", "4582", "4589", "4687"]}).to_parquet(
        tmp_path / "river_network_otto_basins.parquet")
    pd.DataFrame({"trench_id": [1, 2, 3, 4, 5],
                  "adm2": ["1500011", "1500021", "1500031", "1500011", "1500041"]}).to_parquet(
        tmp_path / "river_network_trench_adm2_matches.parquet")
    a, b, c = "150001", "150002", "150003"

    upstream = M.upstream_municipalities([a, b, c], river_network_path=tmp_path)

    assert upstream[a] == {b}  # 4587 drains into 4583; 4582 enters below it; 4687 is another parent
    assert upstream[b] == {a}  # a also holds 4589, upstream of 4587
    assert upstream[c] == set()  # nothing drains into a tributary


def test_spec_data_drops_own_deposits_and_weights_by_mean_births():
    frame = _panel().assign(preterm=1.0)

    data = M.spec_data(frame, "preterm", years=(2007, 2021))

    assert "150001" not in set(data["mun_id"])
    assert data["year"].min() == M.PRETERM_FIRST_YEAR
    first = data[data["mun_id"] == "150000"]
    assert first["weight"].iloc[0] == pytest.approx(first["births"].mean())
    assert data[M.interaction(M.EXPOSURE)].equals(data[M.EXPOSURE] * data[M.PRICE])


def test_price_exposure_se_matches_the_plug_in_formula():
    frame = _panel().rename(columns={"y_raw": "outcome"})
    data = M.spec_data(frame, "outcome", years=(2007, 2021))
    term = M.interaction(M.EXPOSURE)
    model = M.estimate(data, [term])

    se, n_years = M.price_exposure_se(model, term, M.EXPOSURE, M.PRICE)

    # Two-way weighted demeaning of z and the residuals by hand.
    w = data["weight"].to_numpy()
    z = M._demean(data[[term]].to_numpy(), data, M.FE, w)[:, 0]
    y = M._demean(data[["y"]].to_numpy(), data, M.FE, w)[:, 0]
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
    model = M.estimate(data, [M.interaction(M.EXPOSURE)])

    observed, p = M.random_walk_p(data, M.EXPOSURE, M.PRICE, draws=199)

    assert observed == pytest.approx(model.coef().iloc[0], rel=1e-6)
    assert 1 / 200 <= p <= 1
