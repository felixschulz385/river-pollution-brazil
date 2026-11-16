"""Tests for the shared fitting helpers: event studies on a panel and on stacked episodes, and spec tables."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.core.defaults import CLUSTER
from src.analysis.core.estimation import event_study, interaction, run_table, stacked_event_study, term_row
from src.analysis.core.estimation import estimate


def _panel(n=40, years=range(2010, 2021), seed=0):
    """Municipality-years; exposed municipalities' outcome steps up by 1 from 2015."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        exposed = float(i % 2 == 0)
        for year in years:
            rows.append({"mun_id": f"m{i}", "year": year, CLUSTER: f"b{i % 10}", "exposed": exposed,
                         "y": exposed * (year >= 2015) + 0.1 * i + 0.05 * year + rng.normal(0, 0.01)})
    return pd.DataFrame(rows)


def test_event_study_puts_the_base_at_zero_and_recovers_the_step():
    result = event_study(_panel(), ["exposed"], range(2010, 2021), 2014).set_index("year")
    assert result.loc[2014, "estimate"] == 0.0 and result.loc[2014, "se"] == 0.0
    assert result.loc[2010:2013, "estimate"].abs().max() < 0.05
    assert result.loc[2015:, "estimate"].to_numpy() == pytest.approx(1.0, abs=0.05)
    assert result["exposure"].eq("exposed").all() and len(result) == 11


def test_stacked_event_study_reads_treated_by_relative_period():
    rows = []
    for episode in range(3):
        for unit in range(10):
            treated = float(unit < 3)
            for rel in range(-3, 4):
                rows.append({"episode": episode, "unit": f"{episode}_{unit}", "episode_t": f"{episode}_{rel}",
                             "mun": f"m{unit}", "treated": treated, "rel": rel,
                             "y": 2.0 * treated * (rel >= 0) + 0.1 * unit + 0.01 * (episode + rel)})
    result = stacked_event_study(pd.DataFrame(rows), "y", -1, "unit + episode_t", "mun").set_index("rel")
    assert result.loc[-1, "estimate"] == 0.0
    assert result.loc[0:, "estimate"].to_numpy() == pytest.approx(2.0)
    assert result.loc[[-3, -2], "estimate"].to_numpy() == pytest.approx(0.0, abs=1e-8)


def test_term_row_reports_the_clustered_fit_and_optionally_the_bootstrap():
    data = _panel()
    data["post"] = data["exposed"] * (data["year"] >= 2015)
    model = estimate(data, ["post"], weighted=False)
    row = term_row(model, "post", bootstrap=False)
    assert set(row) == {"estimate", "se_basin", "p_basin"} and row["estimate"] == pytest.approx(1.0, abs=0.02)
    assert term_row(model, "post", reps=99)["p_wcb"] < 0.05
    assert interaction("a", "b") == "a__x__b"


def test_run_table_tags_rows_with_the_spec_and_skips_failures_only_when_asked():
    def run_spec(frame, outcome, scale=1.0):
        if outcome == "bad":
            raise ValueError("no rows")
        return [{"outcome": outcome, "estimate": scale}]

    table = run_table(run_spec, None, ["y", "bad"], {"one": {}, "two": {"scale": 2.0}}, skip_failures=True)
    assert table[["spec", "outcome", "estimate"]].values.tolist() == [["one", "y", 1.0], ["two", "y", 2.0]]
    with pytest.raises(ValueError):
        run_table(run_spec, None, ["y", "bad"], {"one": {}})
