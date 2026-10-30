"""Tests for lead 4's demarcation design: apportioning garimpo to lands and the stacked event study."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import garimpo_demarcation as D


def test_land_garimpo_apportions_trench_pixels_by_overlap_share():
    overlaps = pd.DataFrame({"area_id": ["ti:1", "ti:1", "ti:2"], "trench_id": ["a", "b", "b"],
                             "trench_share": [1.0, 0.5, 0.5]})
    mining = pd.DataFrame({"trench_id": ["a", "b"], "year": [2000, 2000], "land_cover_total": [100, 200],
                           "land_cover_class_1": [10, 4], "land_cover_class_2": [0, 2], "land_cover_class_3": [0, 2]})
    result = D.land_garimpo(overlaps, mining).set_index("area_id")

    assert result.loc["ti:1", "land_pixels"] == pytest.approx(200)  # 100 + half of 200
    assert result.loc["ti:1", "garimpo_pixels"] == pytest.approx(14)  # 10 + half of 8
    assert result.loc["ti:2", "garimpo_pixels"] == pytest.approx(4)


def _panel(n_lands=60, seed=2):
    """Lands homologated in 1995, 2000 or 2005 (a third each of the first 45) or never. Garimpo share rises 0.4 pp
    from the homologation year, on a common trend."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_lands):
        homologated = (1995, 2000, 2005)[i % 3] if i < 45 else np.nan
        level = rng.normal(1.0, 0.3)
        for year in range(1985, 2020):
            after = not np.isnan(homologated) and year >= homologated
            rows.append({"area_id": f"ti:{i}", "year": year, "year_homologated": homologated,
                         "year_declared": np.nan, D.GOLD: i % 2,
                         "garimpo_pp": level + 0.02 * (year - 1985) + 0.4 * after + rng.normal(0, 0.05)})
    return pd.DataFrame(rows)


def test_stacked_homologation_study_recovers_the_step_with_clean_controls():
    panel = _panel()
    stacked = D.stack(panel, "homologation")

    first = stacked[stacked["episode"] == 1995]
    assert set(first.loc[first["treated"] > 0, "area_id"]) == {f"ti:{i}" for i in range(0, 45, 3)}
    assert not first.loc[first["treated"] == 0, "area_id"].isin({f"ti:{i}" for i in range(1, 45, 3)}).any()  # 2000
    study = D.event_study(stacked, "garimpo_pp").set_index("rel")["estimate"]
    assert abs(study.loc[-4]) < 0.05 and study.loc[3] == pytest.approx(0.4, abs=0.05)
    assert D.pooled(stacked, "garimpo_pp")["estimate"] == pytest.approx(0.4, abs=0.05)
    assert D.pooled(D.stack(panel, "homologation", sample=D.GOLD), "garimpo_pp")["n_treated_lands"] < 45
