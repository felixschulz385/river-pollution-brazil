"""Tests for the raid design: episodes, borders, the stacked event study and the displacement ratio."""

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

from src.analysis import garimpo_raids as R


RAIDED = {"150005": "2020-06-01", "150020": "2022-03-01"}


def _municipality(i):
    return f"15{i:04d}"


def _inputs(n=30, seed=3):
    """Municipalities in a row, each bordering the next. Garimpo cases fall 1.0 log point in a raided
    municipality from the raid month, and rise 0.5 in its two neighbours; other malaria does not move. The first
    raided municipality is raided again three months later."""
    rng = np.random.default_rng(seed)
    months = pd.date_range(*R.MONTHS, freq="MS")
    raids = pd.DataFrame({
        "mun_id": [*RAIDED, "150005"],
        "month": pd.to_datetime([*RAIDED.values(), "2020-09-01"]),
        "raid_terms": [3, 2, 1],
    })
    adjacency = {_municipality(i): {_municipality(j) for j in (i - 1, i + 1) if 0 <= j < n} for i in range(n)}
    rows = []
    for i in range(n):
        mun = _municipality(i)
        level = rng.normal(2.0, 0.5)
        for month in months:
            effect = 0.0
            for raided, start in RAIDED.items():
                after = month >= pd.Timestamp(start)
                effect += -1.0 * after * (mun == raided) + 0.5 * after * (mun in adjacency[raided])
            garimpo = level + effect + rng.normal(0, 0.1)
            rows.append({"mun_id": mun, "month": month, "inf_garimpo": np.expm1(garimpo),
                         "inf_cases": np.expm1(garimpo) + 20})
    cases = pd.DataFrame(rows)
    for column in ("inf_garimpo_local", "inf_garimpo_incoming"):
        cases[column] = cases["inf_garimpo"] / 2
    panel = R.monthly_panel(cases, {_municipality(i) for i in range(n)})
    # monthly_panel takes logs of counts; replace with the exact synthetic log levels.
    panel["log1p_garimpo"] = np.log1p(panel["inf_garimpo"])
    return panel, raids, adjacency


def test_episodes_need_clean_months_before_them():
    _, raids, _ = _inputs()
    table = R.episodes(raids)

    assert table["mun_id"].tolist() == list(RAIDED)  # the repeat raid 3 months later is not a new episode
    assert table["month"].tolist() == [pd.Timestamp(month) for month in RAIDED.values()]


def test_borders_find_touching_municipalities():
    boundaries = gpd.GeoDataFrame({"mun_id": ["a", "b", "c"],
                                   "geometry": [box(-56, -6, -55, -5), box(-55, -6, -54, -5), box(-53, -6, -52, -5)]},
                                  crs=4326)
    adjacency = R.borders(boundaries)
    assert adjacency["a"] == {"b"} and adjacency["b"] == {"a"} and "c" not in adjacency


def test_stacked_event_study_finds_the_fall_at_the_raid_and_the_rise_next_door():
    panel, raids, adjacency = _inputs()
    table = R.episodes(raids)
    own = R.stack(panel, table, raids, adjacency, "own")
    neighbours = R.stack(panel, table, raids, adjacency, "neighbours")

    assert set(own.loc[own["treated"] > 0, "mun_id"]) == set(RAIDED)
    assert set(neighbours.loc[neighbours["treated"] > 0, "mun_id"]) == {"150004", "150006", "150019", "150021"}
    first = own[(own["episode"] == 0) & (own["treated"] == 0)]
    assert not first["mun_id"].isin({"150004", "150006"}).any()  # the raid's neighbours are not its controls
    assert first["mun_id"].isin({"150019", "150021"}).any()  # the other raid's are, outside its window

    study = R.event_study(own, "log1p_garimpo").set_index("rel")["estimate"]
    assert abs(study.loc[-6]) < 0.15 and study.loc[3] == pytest.approx(-1.0, abs=0.15)
    assert R.pooled(own, "log1p_garimpo")["estimate"] == pytest.approx(-1.0, abs=0.15)
    assert R.pooled(neighbours, "log1p_garimpo")["estimate"] == pytest.approx(0.5, abs=0.15)
    assert abs(R.pooled(own, "log1p_other_malaria")["estimate"]) < 0.1


def test_displacement_ratio_scales_the_neighbours_by_their_number():
    own = {"estimate": -4.0}
    neighbours = {"estimate": 1.0, "n_treated_units": 6, "n_episodes": 2}
    assert R.displacement_ratio(own, neighbours) == pytest.approx(0.75)  # 3 neighbours x 1 case per 4 lost
    assert np.isnan(R.displacement_ratio({"estimate": 1.0}, neighbours))  # no loss at the raid


def test_episode_samples_select_large_and_malaria_visible_raids():
    panel, raids, _ = _inputs()
    table = R.episodes(raids)
    panel.loc[panel["mun_id"] == "150020", "inf_garimpo"] = 0  # no garimpo cases before the second raid
    samples = R.episode_samples(table, panel)

    assert len(samples["all raids"]) == 2
    assert samples[f"large raids ({R.LARGE_TERMS}+ terms)"]["mun_id"].tolist() == ["150005"]
    assert samples[f"malaria-visible ({R.MIN_PRE_CASES}+ cases before)"]["mun_id"].tolist() == ["150005"]
