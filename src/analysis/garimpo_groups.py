"""Who responds to what (lead 2, prediction 5): local and incoming garimpo workers by shock.

The two-group model predicts that outside crews respond to the returns to a site
(the gold price received, enforcement) and local residents to the erosion of
their traditional economy (drought, mining upstream that degrades the river).
SIVEP splits garimpo cases by infection municipality into local workers (living
there) and incoming ones (living elsewhere in Brazil).

- *Sample.* Legal Amazon municipalities with a garimpo case in 2004-2024.
- *Outcomes.* log(1 + local) and log(1 + incoming) garimpo cases, their
  difference (local minus incoming: the coefficient is the difference of the two
  responses, with its SE), and non-garimpo malaria for comparison. Infections
  of both groups happen in the same place, so a shock to transmission that
  scales both alike drops out of the difference.
- *Residents' shocks.*
  - Drought: the deficit of annual precipitation (ERA5-Land, the municipality's
    own catchment), standardised within the municipality over 2000-2024, this
    year and the year before; a variant uses precipitation upstream.
  - Upstream garimpo: the C11 garimpo share (all substances) of the river
    reaches within 200 km (or 50 km) upstream, in pp, with the municipality's own
    garimpo share as a control, 2004-2019 (C11 steps in 2020).
- *Crews' shocks.* The log BRL gold price x never-exploited deposits (E10; also
  without Roraima and before the 2019 break), and IBAMA presence x post-2019 and
  x post-2023 on 2015-2024 (E12).

Municipality and year FE (state x year FE as a check), unweighted, Otto-3
clusters and the wild-cluster bootstrap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis import garimpo_laundering as L
from src.analysis import garimpo_shift as G
from src.analysis import moreno_louzada as M
from src.analysis.inference import wild_cluster_bootstrap

YEARS = (2004, 2024)
C11_YEARS = (2004, 2019)
CLIMATE_YEARS = (2000, 2024)
FE, CLUSTER = M.FE, M.CLUSTER
STATE_YEAR_FE = "mun_id + state_year"
SAMPLE = "garimpo_municipality"

OUTCOMES = {
    "log1p_garimpo_local": "log(1 + local garimpo cases)",
    "log1p_garimpo_incoming": "log(1 + incoming garimpo cases)",
    "local_minus_incoming": "difference: local minus incoming",
    "log1p_other_malaria": "log(1 + other malaria cases)",
}


def deficit(frame, column, years=CLIMATE_YEARS):
    """Minus the within-municipality z-score of `column` over `years` (higher is drier)."""
    window = frame[frame["year"].between(*years)].groupby("mun_id")[column]
    mean, sd = frame["mun_id"].map(window.mean()), frame["mun_id"].map(window.std())
    return -(frame[column] - mean) / sd


def add_shocks(frame):
    """The group outcomes, the drought and upstream-garimpo shocks, and the garimpo sample flag."""
    frame = frame.sort_values(["mun_id", "year"]).reset_index(drop=True)
    frame["local_minus_incoming"] = frame["log1p_garimpo_local"] - frame["log1p_garimpo_incoming"]
    frame["drought"] = deficit(frame, "local_tp")
    frame["drought_upstream"] = deficit(frame, "tp")
    for column in ("drought", "drought_upstream"):
        frame[f"{column}_lag"] = frame.groupby("mun_id")[column].shift()
    for reach in ("50", "200"):
        frame[f"upstream_garimpo_{reach}"] = 100 * sum(
            frame[f"c11_garimpo_{group}_{reach}"].fillna(0.0) for group in ("gold", "unknown", "other"))
    cases = frame[frame["year"].between(*YEARS)].groupby("mun_id")["inf_garimpo"].sum()
    frame[SAMPLE] = frame["mun_id"].map(cases.gt(0)).fillna(False).astype(float)
    return frame


def build_frame(path=None):
    """E12's Legal Amazon panel with the group outcomes and the shocks."""
    return add_shocks(G.build_frame(path))


def spec(terms, controls=(), years=YEARS, fe=FE, drop_states=()):
    return {"terms": list(terms), "controls": list(controls), "years": years, "fe": fe, "drop_states": drop_states}


def specs():
    """Spec name -> `spec_data`/`run_spec` arguments. Terms are built by `spec_data`."""
    price = M.interaction(L.DEPOSITS, L.PRICE)
    presence = [G.term(G.PRESENCE, period) for period in G.PERIODS]
    drought = ["drought", "drought_lag"]
    return {
        "drought": spec(drought, [price]),
        "drought, state x year FE": spec(drought, [price], fe=STATE_YEAR_FE),
        "drought, without Roraima": spec(drought, [price], drop_states=(G.RORAIMA,)),
        "drought upstream": spec(["drought_upstream", "drought_upstream_lag"], [price]),
        "upstream garimpo 200 km": spec(["upstream_garimpo_200"], ["c11_garimpo_pp", price], C11_YEARS),
        "upstream garimpo 50 km": spec(["upstream_garimpo_50"], ["c11_garimpo_pp", price], C11_YEARS),
        "gold price x deposits": spec([price]),
        "gold price x deposits, without Roraima": spec([price], drop_states=(G.RORAIMA,)),
        "gold price x deposits, 2004-2018": spec([price], years=(2004, 2018)),
        "enforcement presence": spec(presence, [price], G.YEARS),
    }


def spec_data(frame, outcome, terms, controls, years, drop_states=()):
    """Estimation rows of one spec: the garimpo sample in `years`, with the interaction terms built."""
    data = frame[frame["year"].between(*years) & frame[SAMPLE].gt(0)
                 & ~frame["mun_id"].str[:2].isin(drop_states)].copy()
    data[M.interaction(L.DEPOSITS, L.PRICE)] = data[L.DEPOSITS] * data[L.PRICE]
    for period, active in G.PERIODS.items():
        data[G.term(G.PRESENCE, period)] = data[G.PRESENCE] * active(data["year"])
    data["y"] = data[outcome]
    data["weight"] = 1.0
    return data.dropna(subset=["y", CLUSTER, *terms, *controls]).reset_index(drop=True)


def run_spec(frame, outcome, terms, controls, years, fe=FE, drop_states=(), bootstrap=True):
    """One row per term: estimate, Otto-3 SE and p, and the wild-cluster bootstrap p."""
    data = spec_data(frame, outcome, terms, controls, years, drop_states)
    model = M.estimate(data, terms + controls, fe=fe, weighted=False)
    tidy = model.tidy()
    rows = []
    for name in terms:
        row = {"outcome": outcome, "term": name, "estimate": tidy.loc[name, "Estimate"],
               "se_basin": tidy.loc[name, "Std. Error"], "p_basin": tidy.loc[name, "Pr(>|t|)"]}
        if bootstrap:
            row["p_wcb"] = wild_cluster_bootstrap(model, name, CLUSTER, reps=M.BOOT_REPS, seed=M.SEED)["p_wcb"]
        row.update(n_obs=int(model._N), n_municipalities=int(data["mun_id"].nunique()),
                   n_basins=int(data[CLUSTER].nunique()))
        rows.append(row)
    return rows


def run_table(frame, outcomes=tuple(OUTCOMES), spec_table=None, bootstrap=True):
    """Every spec for every outcome."""
    rows = []
    for name, arguments in (spec_table or specs()).items():
        for outcome in outcomes:
            rows += [{"spec": name, **row} for row in run_spec(frame, outcome, bootstrap=bootstrap, **arguments)]
    return pd.DataFrame(rows)


def drought_years(frame, years=YEARS, threshold=1.0):
    """Per year, the share of sample municipalities with a precipitation deficit above `threshold` SD."""
    data = frame[frame["year"].between(*years) & frame[SAMPLE].gt(0)]
    return data.groupby("year")["drought"].agg(mean="mean", share_dry=lambda d: (d > threshold).mean())
