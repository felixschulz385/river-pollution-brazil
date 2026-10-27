"""Tests for the garimpo-gold gate's exposure, samples and switcher definitions."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import garimpo_gold as G
from src.analysis import health as H
from src.analysis import panels


def _panel_rows(mun_id, years, gold_50, biome="Amazônia", local=0.0):
    rows = []
    for year, gold in zip(years, gold_50):
        row = {"mun_id": mun_id, "year": year, "biome": biome, "local_c11_garimpo_gold": local}
        for radius in G.GOLD_RADII:
            for group in G.C11_GROUPS:
                row[f"c11_{group}_{radius}"] = 0.0
            row[f"c11_garimpo_gold_active_{radius}"] = 0.0
        row["c11_garimpo_gold_50"] = gold
        row["c11_garimpo_unknown_50"] = 0.001
        rows.append(row)
    return rows


def test_c11_exposure_is_in_pp_with_missing_reach_as_zero():
    frame = pd.DataFrame(_panel_rows("a", [2000, 2001], [np.nan, 0.002], local=0.003))

    G.add_c11_exposure(frame)

    assert frame["gold_50"].tolist() == pytest.approx([0.0, 0.2])
    assert frame["other_mining_50"].tolist() == pytest.approx([0.1, 0.1])
    assert frame[G.LOCAL_GOLD].tolist() == pytest.approx([0.3, 0.3])
    assert frame["amazon"].tolist() == [1.0, 1.0]


def test_gold_sample_counts_only_gold_up_to_the_last_gate_year():
    frame = pd.DataFrame(
        _panel_rows("early", [2005, 2020], [0.001, 0.0])
        + _panel_rows("late", [2005, 2020], [0.0, 0.001])
        + _panel_rows("before_window", [1999, 2005], [0.001, 0.0])
    )
    G.add_c11_exposure(frame)

    assert G.gold_sample(frame, 50) == {"early"}
    assert G.gold_sample(frame, 50, last_year=2024) == {"early", "late"}


def test_gold_arm_specs_split_regions_and_restrict_years():
    frame = pd.DataFrame(
        _panel_rows("a", [2010, 2022], [0.001, 0.002]) + _panel_rows("o", [2010, 2022], [0.004, 0.004], biome="Cerrado")
    )
    G.add_c11_exposure(frame)

    specs = G.gold_arm_specs(frame, 50)
    early, terms = specs["b by region 2000-19"]
    assert terms == ["gold_50__amazon", "gold_50__other"]
    assert early["year"].max() <= G.GATE_LAST_YEAR
    assert early.set_index("mun_id")["gold_50__amazon"].to_dict() == pytest.approx({"a": 0.1, "o": 0.0})
    assert specs["d pooled 2000-24"][0]["year"].max() == 2022

    with_local = G.gold_arm_specs(frame, 50, local_gold=True)
    assert [terms[-1] for _, terms in with_local.values()] == [G.LOCAL_GOLD, G.LOCAL_GOLD]


def test_presence_switchers_and_onsets_are_amazon_upstream_switch_ons():
    rows = []
    for mun_id, biome, present in [
        ("switch", "Amazônia", [0, 1, 1]),
        ("always", "Amazônia", [1, 1, 1]),
        ("elsewhere", "Cerrado", [0, 0, 1]),
        ("never", "Amazônia", [0, 0, 0]),
    ]:
        for year, value in zip([2018, 2019, 2020], present):
            rows.append({
                "mun_id": mun_id, "year": year, "biome": biome,
                H.UPSTREAM_MINING_PRESENT: float(value), H.LOCAL_MINING_PRESENT: 0.0,
                H.MINING_MUNICIPALITY: mun_id != "never",
            })
    frame = pd.DataFrame(rows)

    mining, switchers = G.presence_frame(frame)

    assert set(mining["mun_id"]) == {"switch", "always", "elsewhere"}
    assert list(switchers) == ["switch"]
    assert G.switcher_onsets(mining, switchers).to_dict() == {"switch": 2019}
    assert mining.loc[mining["mun_id"] == "elsewhere", G.PRESENCE_TERM].sum() == 0


def test_local_mining_loader_averages_own_trench_rings(tmp_path):
    rows = []
    for bucket, gold, flagged in [(-50, 10.0, True), (-25, 30.0, True), (0, 90.0, False)]:
        counts = {-1: 100.0, 0: 100.0 - gold, 1: gold, 11: gold}
        for class_id in (-1, 0, 1, 2, 3, 4, 5, 11, 12, 21, 22, 31, 32, 41, 42, 51, 52):
            cnt = counts.get(class_id, 0.0)
            rows.append({"mun_id": "150553", "year": 2020, "bucket": bucket, "land_cover_class": class_id,
                         "n": 1, "cnt": cnt, "share": cnt / 100.0, "bucket_intersects_adm2": flagged})
    path = tmp_path / "mining.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)

    local = panels.adm2_local_mining(years=(2020, 2020), path=path).iloc[0]

    assert local["local_mining_garimpo_gold"] == pytest.approx(0.20)
    assert local["local_mining_garimpo_gold_active"] == pytest.approx(0.20)
