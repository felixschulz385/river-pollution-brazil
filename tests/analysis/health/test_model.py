"""Tests for the health model-setup rules in src.analysis.health.model."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.health import model as H
from src.analysis.core import panels


def _upstream(mun_ids, value, suffix=""):
    frame = pd.DataFrame({"mun_id": mun_ids, "year": 2010})
    for column in H.UPSTREAM_REGRESSORS:
        frame[f"{column}{suffix}"] = value
    return frame


def test_upstream_exposure_is_capped_and_zero_for_headwater_and_missing_bands(monkeypatch):
    calls = []

    def fake_upstream_alr(min_km=0, max_km=H.MAX_UPSTREAM_KM, suffix=""):
        calls.append((min_km, max_km))
        # "a" has rings in every band, "b" only near ones; "h" is a headwater municipality.
        mun_ids = ["a"] if suffix == "__150_500" else ["a", "b"]
        return _upstream(mun_ids, 1.5, suffix)

    monkeypatch.setattr(H, "upstream_alr", fake_upstream_alr)
    frame = pd.DataFrame({"mun_id": ["a", "b", "h"], "year": 2010})
    for column in panels.LAND_COVER_ALR_COLUMNS:
        frame[column] = 9.0  # the assembled (uncapped) terms must be replaced

    out = H.add_upstream_exposure(frame, distance_bands=True).set_index("mun_id")

    assert calls[0] == (0, H.MAX_UPSTREAM_KM)
    assert out.loc[["a", "b"], "alr_pasture"].tolist() == [1.5, 1.5]
    assert out.loc["h", H.UPSTREAM_REGRESSORS].eq(0.0).all()
    assert out.loc["a", "alr_pasture__150_500"] == 1.5
    assert out.loc["b", "alr_pasture__150_500"] == 0.0
    assert "alr_mining" not in out.columns


def test_add_outcomes_masks_early_preterm_and_builds_offsets():
    frame = pd.DataFrame(
        {
            "year": [2011, 2012],
            "population": [1000.0, 1000.0],
            "births": [0.0, 20.0],
            "births_weight_known": [0.0, 19.0],
            "births_gestation_known": [0.0, 18.0],
            "deaths_under_1": [1.0, 2.0],
            "deaths_1_to_4": [0.0, 1.0],
        }
    )
    for count, _ in H.OUTCOMES.values():
        if count not in frame:
            frame[count] = 1.0

    H.add_outcomes(frame)

    assert np.isnan(frame.loc[0, "y_preterm"])
    assert frame.loc[1, "y_preterm"] == 1.0
    assert frame["y_under5_deaths"].tolist() == [1.0, 3.0]
    # No births: the birth-based offsets are missing, which drops the row from those outcomes.
    assert np.isnan(frame.loc[0, H.offset_column("infant_deaths")])
    assert frame.loc[1, H.offset_column("infant_deaths")] == np.log(20.0)
    assert H.offset_column("digestive") == "log_population"
    assert frame.loc[0, "log_population"] == np.log(1000.0)


def test_mining_exposure_is_zero_without_near_rings_and_flags_mining_municipalities(monkeypatch):
    near = pd.DataFrame({"mun_id": ["a", "a", "b", "b"], "year": [2010, 2011, 2010, 2011],
                         "lc_mining": [0.0, 0.002, 0.0, 0.0]})
    monkeypatch.setattr(H.panels, "adm2_upstream_land_cover", lambda **kwargs: near)
    # "c" has no rings within the mining reach.
    frame = pd.DataFrame({"mun_id": ["a", "a", "b", "b", "c", "c"], "year": [2010, 2011] * 3,
                          "local_lc_mining": [0.0, 0.0, 0.001, 0.001, 0.0, 0.0]})

    out = H.add_mining_exposure(frame)

    assert out[H.UPSTREAM_MINING_PP].tolist() == [0.0, 0.2, 0.0, 0.0, 0.0, 0.0]
    assert out[H.UPSTREAM_MINING_PRESENT].tolist() == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    assert out[H.LOCAL_MINING_PRESENT].tolist() == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    assert out[H.MINING_MUNICIPALITY].tolist() == [True, True, False, False, False, False]
