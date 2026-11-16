"""Raid intensity (lead 3): garimpo malaria under sustained raiding, by month.

E14's episodes need 12 months without a raid before them, so the campaigns of
2023-25, which raid the same municipalities month after month, give few
episodes. Here raid terms are a continuous treatment on the monthly panel of
E14's sample municipalities, from 2019 until three months before SIVEP's last
month:

    Y_mt = sum_w b_w R_m,w(t) + sum_w n_w B_m,w(t) + sum_w f_w F_m,w(t)
         + municipality x calendar-month FE + state x month FE.

- *Treatment.* R is log(1 + garimpo destruction and seizure terms) in the
  municipality, summed over a window w of months relative to t: the three months
  after t (a lead, which tests whether raids follow the outcome), the month
  itself, and lags 1-3, 4-6 and 7-12. B sums the terms of the bordering
  municipalities and F those of the other municipalities whose centroid is
  within `RING_KM` (displacement beyond the next municipality).
- *Effect of a year of raiding.* The sum of the month and lag coefficients,
  with its SE from the coefficient covariance. A pooled spec puts the month and
  the 12 months before in one window (`past`), next to the lead, and gets the
  wild-cluster bootstrap.
- *Outcomes.* As in E14. Raids follow local malaria peaks, so garimpo against
  non-garimpo malaria in the same municipality-month is the main outcome.
- *Inference.* Clustered by Otto-3 basin; the restricted wild-cluster
  bootstrap for the pooled terms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.core import panels
from src.analysis.core.defaults import BOOT_REPS, CLUSTER, SEED
from src.analysis.core.estimation import estimate
from src.analysis.core.geography import legal_amazon, municipality_boundaries, rings
from src.analysis.core.inference import wild_cluster_bootstrap
from src.analysis.garimpo import raids as R

FIRST_MONTH = "2019-01"
LEAD_MONTHS = 3
RING_KM = 150
FE = "mun_calendar + state_month"
SOURCES = {"own": "the municipality", "border": "bordering municipalities", "ring": f"others within {RING_KM} km"}
# Window name -> (first, last) month of the raids relative to t (positive = after t).
WINDOWS = {"lead": (1, LEAD_MONTHS), "now": (0, 0), "lag1_3": (-3, -1), "lag4_6": (-6, -4), "lag7_12": (-12, -7)}
POOLED_WINDOWS = {"lead": (1, LEAD_MONTHS), "past": (-12, 0)}
YEAR_WINDOWS = ("now", "lag1_3", "lag4_6", "lag7_12")
OUTCOMES = {key: R.OUTCOMES[key] for key in ("log1p_garimpo", "log1p_garimpo_local", "log1p_garimpo_incoming",
                                             "log1p_other_malaria", "garimpo_vs_other", "incoming_vs_other")}


def raid_matrix(raids, municipalities, first_t, last_t):
    """Raid terms by municipality (rows) and month index (columns), zero where there are none."""
    raids = raids.assign(t=R.month_index(raids["month"]))
    table = raids.pivot_table(index="mun_id", columns="t", values="raid_terms", aggfunc="sum")
    return table.reindex(index=sorted(municipalities), columns=range(first_t, last_t + 1)).fillna(0.0)


def neighbour_sum(matrix, groups):
    """Per municipality, the summed rows of the municipalities in `groups[municipality]`."""
    return pd.DataFrame([matrix.reindex(sorted(groups.get(mun, ()))).fillna(0.0).sum(axis=0) for mun in matrix.index],
                        index=matrix.index)


def window_sum(matrix, first, last):
    """For each month t, the sum over months t + first .. t + last (missing where a month is outside the matrix)."""
    total = 0.0
    for offset in range(first, last + 1):
        total = total + matrix.shift(-offset, axis=1)
    return total


def build_panel(panel, raids, adjacency, ring, basins, windows=None, first_month=FIRST_MONTH):
    """The monthly panel with log(1 + raid terms) by source and window, the FE keys and the basin.

    Rows run from `first_month` to the last month whose lead window is observed. `windows` defaults to those of
    the distributed and the pooled specs.
    """
    windows = windows or {**WINDOWS, **POOLED_WINDOWS}
    first_t = int(R.month_index(pd.Series([pd.Timestamp(first_month)])).iloc[0])
    lowest = min(first for first, _ in windows.values())
    last_t = int(panel["t"].max())
    municipalities = set(adjacency) | set(panel["mun_id"]) | set(raids["mun_id"])
    own = raid_matrix(raids, municipalities, first_t + lowest, last_t)
    sources = {"own": own, "border": neighbour_sum(own, adjacency), "ring": neighbour_sum(own, ring)}

    data = panel[panel["t"].between(first_t, last_t - max(last for _, last in windows.values()))].copy()
    for source, matrix in sources.items():
        for window, (first, last) in windows.items():
            values = np.log1p(window_sum(matrix, first, last)).stack().rename(f"{source}_{window}")
            values.index.names = ["mun_id", "t"]
            data = data.join(values, on=["mun_id", "t"])
    data["state_month"] = data["mun_id"].str[:2] + "_" + data["t"].astype(str)
    data["mun_calendar"] = data["mun_id"] + "_" + (data["t"] % 12).astype(str)
    data[CLUSTER] = data["mun_id"].map(basins)
    return data.dropna(subset=[*terms(windows), CLUSTER]).reset_index(drop=True)


def terms(windows=WINDOWS, sources=tuple(SOURCES)):
    return [f"{source}_{window}" for source in sources for window in windows]


def fit(data, outcome, columns, fe=FE):
    return estimate(data, columns, fe=fe, weighted=False, outcome=outcome)


def linear_sum(model, columns):
    """Sum of `columns`' coefficients, its clustered SE and a t-test with G - 1 degrees of freedom."""
    names = list(model._coefnames)
    index = [names.index(column) for column in columns]
    estimate = float(np.asarray(model.coef())[index].sum())
    se = float(np.sqrt(np.asarray(model._vcov)[np.ix_(index, index)].sum()))
    clusters = int(model._data[CLUSTER].nunique())
    return estimate, se, float(2 * stats.t.sf(abs(estimate / se), clusters - 1))


def distributed(data, outcome, fe=FE):
    """Each source x window coefficient, and per source the year sum (month and lags 1-12)."""
    model = fit(data, outcome, terms(), fe)
    tidy = model.tidy()
    rows = []
    for source in SOURCES:
        for window in WINDOWS:
            name = f"{source}_{window}"
            rows.append({"outcome": outcome, "source": source, "window": window, "estimate": tidy.loc[name, "Estimate"],
                         "se": tidy.loc[name, "Std. Error"], "p": tidy.loc[name, "Pr(>|t|)"]})
        estimate, se, p = linear_sum(model, [f"{source}_{window}" for window in YEAR_WINDOWS])
        rows.append({"outcome": outcome, "source": source, "window": "year sum", "estimate": estimate, "se": se, "p": p})
    return pd.DataFrame(rows).assign(n_obs=int(model._N), n_municipalities=int(data["mun_id"].nunique()),
                                     n_basins=int(data[CLUSTER].nunique()))


def pooled(data, outcome, fe=FE, bootstrap=True):
    """Lead and past-year windows per source, with Otto-3 and wild-cluster bootstrap p-values."""
    model = fit(data, outcome, terms(POOLED_WINDOWS), fe)
    tidy = model.tidy()
    rows = []
    for name in terms(POOLED_WINDOWS):
        source, window = name.split("_", 1)
        row = {"outcome": outcome, "source": source, "window": window, "estimate": tidy.loc[name, "Estimate"],
               "se": tidy.loc[name, "Std. Error"], "p": tidy.loc[name, "Pr(>|t|)"]}
        if bootstrap:
            row["p_wcb"] = wild_cluster_bootstrap(model, name, CLUSTER, reps=BOOT_REPS, seed=SEED)["p_wcb"]
        rows.append(row)
    return rows


def load_inputs(root_dir=panels.PROJECT_ROOT):
    """E14's monthly panel, raid months and border map, the rings, and each municipality's Otto-3 basin."""
    panel, raids, adjacency = R.load_inputs(root_dir)
    boundaries = municipality_boundaries(root_dir)
    boundaries = boundaries[boundaries["mun_id"].isin(legal_amazon())]
    ring = rings(boundaries[["mun_id", "geometry"]], adjacency, RING_KM)
    basins = pd.read_parquet(panels.ADM2_PANEL_PATH, columns=["mun_id", CLUSTER]).drop_duplicates("mun_id")
    return panel, raids, adjacency, ring, basins.set_index("mun_id")[CLUSTER]
