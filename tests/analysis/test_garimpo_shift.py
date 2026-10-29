"""Tests for the 2019-20 shift analysis: enforcement presence, the presence spec and the split event study."""

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
from src.analysis import garimpo_shift as G


def _frame(n_municipalities=80, years=range(2010, 2025), seed=5):
    """Municipality-years. Cases rise 0.3 per unit of `presence` from 2019 and fall back 0.2 from 2023; local
    cases step up 1.0 in IGD municipalities of Roraima from 2019 and 0.5 in IGD elsewhere from 2020."""
    rng = np.random.default_rng(seed)
    price = pd.Series(np.cumsum(rng.normal(0.08, 0.15, len(years))), index=list(years))
    fines = rng.lognormal(1.0, 1.0, n_municipalities)
    rows = []
    for i in range(n_municipalities):
        state = "14" if i % 4 == 0 else "15"
        igd = float(i % 5 == 0)
        for year in years:
            rows.append({
                "mun_id": f"{state}{i:04d}", "year": year, G.CLUSTER: f"b{i % 16}", L.IGD: igd,
                L.DEPOSITS: float(i % 3 == 0), L.S.ALL_GOLD_EXPOSURE: 1.0, L.PRICE: price[year],
                "ibama_fines_all": rng.poisson(fines[i]),
            })
    frame = pd.DataFrame(rows)
    frame[G.PRESENCE] = G.standardised_presence(frame, "ibama_fines_all", G.PRESENCE_YEARS)
    roraima = frame["mun_id"].str[:2].eq(G.RORAIMA)
    frame[G.IGD_RORAIMA] = frame[L.IGD] * roraima
    frame[G.IGD_OTHER] = frame[L.IGD] * ~roraima
    noise = rng.normal(0, 0.2, len(frame))
    frame["log1p_garimpo_cases"] = (
        0.3 * frame[G.PRESENCE] * (frame["year"] >= 2019) - 0.2 * frame[G.PRESENCE] * (frame["year"] >= 2023) + noise
    )
    frame["log1p_garimpo_local"] = (
        1.0 * frame[G.IGD_RORAIMA] * (frame["year"] >= 2019) + 0.5 * frame[G.IGD_OTHER] * (frame["year"] >= 2020)
        + noise
    )
    return frame


def test_presence_is_standardised_and_constant_within_municipalities():
    frame = _frame()
    per_municipality = frame.groupby("mun_id")[G.PRESENCE]

    assert (per_municipality.nunique() == 1).all()
    assert per_municipality.first().mean() == pytest.approx(0.0, abs=1e-9)
    assert per_municipality.first().std() == pytest.approx(1.0)


def test_run_spec_recovers_the_shift_and_the_recovery_on_2015_2024():
    rows = {row["term"]: row for row in G.run_spec(_frame(), "log1p_garimpo_cases", bootstrap=False)}
    data, terms, _ = G.spec_data(_frame(), "log1p_garimpo_cases")

    assert data["year"].min() == 2015  # after the presence window
    assert terms == [G.term(G.PRESENCE, "post2019"), G.term(G.PRESENCE, "post2023")]
    assert rows["post2019"]["estimate"] == pytest.approx(0.3, abs=0.05)
    assert rows["post2023"]["estimate"] == pytest.approx(-0.2, abs=0.05)


def test_split_event_study_separates_igd_in_roraima_from_igd_elsewhere():
    study = G.event_study(_frame(), "log1p_garimpo_local", exposures=(G.IGD_RORAIMA, G.IGD_OTHER))
    wide = study.pivot(index="year", columns="exposure", values="estimate")

    assert (wide.loc[G.BASE_YEAR] == 0.0).all()
    assert wide.loc[2019, G.IGD_RORAIMA] == pytest.approx(1.0, abs=0.25)
    assert abs(wide.loc[2019, G.IGD_OTHER]) < 0.25  # elsewhere the step comes in 2020
    assert wide.loc[2021, G.IGD_OTHER] == pytest.approx(0.5, abs=0.25)
