"""The 2019-20 shift in garimpo: the Roraima rush, and the collapse of IBAMA enforcement.

Garimpo cases, gold PLG requests and non-garimpo malaria all break around
2019-20 in the IGD municipalities of lead 1 (E11). This module separates two
candidate causes.

- *The Roraima rush.* The IGD event study is split into IGD municipalities in
  Roraima (the Yanomami territory) and elsewhere, each against the other Legal
  Amazon municipalities.
- *The enforcement collapse* (lead 3). IBAMA's fines fell from 2019 and
  recovered in 2023. Municipalities where IBAMA was present before lost the
  most deterrence. Presence is the log of the municipality's mean yearly IBAMA
  fines in 2010-2014 (all records, any infraction), standardised across the
  Legal Amazon. On 2015-2024, after the presence window,

      Y_mt = b19 (presence_m x post-2019_t) + b23 (presence_m x post-2023_t)
           + g (log gold price in BRL_t x deposits_m) + municipality FE + year FE.

  The fines themselves are the first stage (b19 < 0, b23 > 0); garimpo cases
  and gold PLG requests are the reduced forms; non-garimpo malaria is the
  placebo. Presence covers most municipalities, so Otto-3 clustering and the
  wild-cluster bootstrap apply without the few-treated problem of IGD.

Presence marks the deforestation frontier, and clearing also surged from 2019.
The checks add the frontier (forest loss in 2010-2014 and forest cover in 2014,
each x post-2019 and x post-2023), current forest loss (a bad control, as a
bound), a presence trend, forest loss as an outcome, permutation inference
(all municipalities and within states) and presence quintiles x the periods.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis import garimpo_laundering as L
from src.analysis import moreno_louzada as M
from src.analysis.inference import wild_cluster_bootstrap

YEARS = (2015, 2024)
PRESENCE_YEARS = (2010, 2014)
FRONTIER_YEARS = (2010, 2014)
FRONTIER = ("frontier_loss", "forest_2014")  # standardised, x post-2019 and x post-2023
EARLY_PRESENCE_YEARS = (2005, 2010)
SHIFT_YEAR = 2019  # IBAMA fines fall from 2019 (new federal government, January 2019)
RECOVERY_YEAR = 2023  # and recover in 2023
BASE_YEAR = 2018
FE, CLUSTER = M.FE, M.CLUSTER
RORAIMA = L.RORAIMA

PRESENCE = "presence"
EXPOSURES = {
    PRESENCE: "IBAMA presence: log mean yearly fines 2010-2014 (SD)",
    "presence_nonmining": "... from fines other than mining",
    "presence_early": "... from fines in 2005-2010",
}
OUTCOMES = {
    "log1p_fines_all": "log(1 + IBAMA fines), first stage",
    "log1p_garimpo_cases": "log(1 + garimpo cases)",
    "log1p_garimpo_local": "log(1 + local garimpo cases)",
    "log1p_garimpo_incoming": "log(1 + incoming garimpo cases)",
    "log1p_plg_requests_gold": "log(1 + gold PLG requests)",
    "log1p_other_malaria": "log(1 + other malaria cases), placebo",
    "forest_loss": "forest loss, own trenches (pp)",
}
PERIODS = {
    "post2019": lambda year: (year >= SHIFT_YEAR).astype(float),
    "post2023": lambda year: (year >= RECOVERY_YEAR).astype(float),
}
DOSE_BINS = 5

IGD_RORAIMA = "igd_roraima"
IGD_OTHER = "igd_other_states"
GROUPS = (IGD_RORAIMA, IGD_OTHER, "rest_roraima", "rest")
GROUP_COLUMNS = {
    "inf_garimpo": "garimpo cases",
    "other_malaria": "other malaria cases",
    "plg_requests_gold": "gold PLG requests",
    "ibama_fines_all": "IBAMA fines",
    "ibama_seizures_garimpo": "garimpo seizures",
    "ibama_destructions_garimpo": "garimpo destruction terms",
}


def standardised_presence(frame, column, years):
    """Per municipality, log(1 + mean yearly `column` over `years`), standardised across municipalities."""
    pre = np.log1p(frame[frame["year"].between(*years)].groupby("mun_id")[column].mean())
    return frame["mun_id"].map((pre - pre.mean()) / pre.std())


def standardised(values):
    return (values - values.mean()) / values.std()


def build_frame(path=None):
    """Lead 1's Legal Amazon panel with enforcement presence, the extra outcomes, groups and FE keys."""
    frame = L.build_frame(path)
    frame[PRESENCE] = standardised_presence(frame, "ibama_fines_all", PRESENCE_YEARS)
    frame["nonmining_fines"] = frame["ibama_fines_all"] - frame["ibama_fines_mining"]
    frame["presence_nonmining"] = standardised_presence(frame, "nonmining_fines", PRESENCE_YEARS)
    frame["presence_early"] = standardised_presence(frame, "ibama_fines_all", EARLY_PRESENCE_YEARS)
    frame["log1p_fines_all"] = np.log1p(frame["ibama_fines_all"])
    frame["other_malaria"] = (frame["inf_cases"] - frame["inf_garimpo"]).clip(lower=0)
    frame = frame.sort_values(["mun_id", "year"]).reset_index(drop=True)
    frame["forest_loss"] = -100 * frame.groupby("mun_id")["local_lc_forest"].diff()
    early = frame[frame["year"].between(*FRONTIER_YEARS)].groupby("mun_id")
    frame["frontier_loss"] = frame["mun_id"].map(standardised(early["forest_loss"].mean()))
    frame["forest_2014"] = frame["mun_id"].map(standardised(early["local_lc_forest"].last()))

    roraima = frame["mun_id"].str[:2].eq(RORAIMA)
    igd = frame[L.IGD] > 0
    frame[IGD_RORAIMA] = (igd & roraima).astype(float)
    frame[IGD_OTHER] = (igd & ~roraima).astype(float)
    frame["group"] = np.select([igd & roraima, igd, roraima], [IGD_RORAIMA, IGD_OTHER, "rest_roraima"], "rest")

    year = frame["year"].astype(str)
    frame["state_year"] = frame["mun_id"].str[:2] + "_" + year
    frame["igd_year"] = igd.astype(int).astype(str) + "_" + year
    frame["gold_year"] = (frame[L.S.ALL_GOLD_EXPOSURE] > 0).astype(int).astype(str) + "_" + year
    return frame


def group_table(frame, column, first_year=2010):
    """Year x group totals of `column` (IGD in Roraima and elsewhere, the rest of Roraima, the rest)."""
    table = frame[frame["year"] >= first_year].pivot_table(index="year", columns="group", values=column,
                                                           aggfunc="sum")
    return table.reindex(columns=list(GROUPS))


def term(exposure, period):
    return f"{exposure}__x__{period}"


def spec_data(frame, outcome, exposure=PRESENCE, years=YEARS, drop_states=(), gold_only=False, sample=None,
              frontier=False, deforestation=False, trend=False):
    """Estimation rows of one presence spec: (data, exposure terms, other controls).

    `frontier` adds the frontier measures x the periods, `deforestation` current forest loss, and `trend` the
    exposure x a linear trend.
    """
    data = frame[frame["year"].between(*years) & ~frame["mun_id"].str[:2].isin(drop_states)].copy()
    if gold_only:
        data = data[data[L.S.ALL_GOLD_EXPOSURE] > 0]
    if sample is not None:
        data = data[data[sample] > 0]
    data["y"] = data[outcome]
    data["weight"] = 1.0
    required = ["y", exposure, CLUSTER, L.PRICE, *(FRONTIER if frontier else ()),
                *(("forest_loss",) if deforestation else ())]
    data = data.dropna(subset=required).reset_index(drop=True)
    terms = []
    for period, active in PERIODS.items():
        data[term(exposure, period)] = data[exposure] * active(data["year"])
        terms.append(term(exposure, period))
    data[M.interaction(L.DEPOSITS, L.PRICE)] = data[L.DEPOSITS] * data[L.PRICE]
    controls = [M.interaction(L.DEPOSITS, L.PRICE)]
    if frontier:
        for measure in FRONTIER:
            for period, active in PERIODS.items():
                data[term(measure, period)] = data[measure] * active(data["year"])
                controls.append(term(measure, period))
    if deforestation:
        controls.append("forest_loss")
    if trend:
        data[term(exposure, "trend")] = data[exposure] * (data["year"] - years[0])
        controls.append(term(exposure, "trend"))
    return data, terms, controls


def specs():
    """Spec name -> (`spec_data` kwargs, fixed effects)."""
    return {
        "baseline": ({}, FE),
        "presence from non-mining fines": ({"exposure": "presence_nonmining"}, FE),
        "presence 2005-2010": ({"exposure": "presence_early"}, FE),
        "state x year FE": ({}, "mun_id + state_year"),
        "IGD x year FE": ({}, "mun_id + igd_year"),
        "gold x year FE": ({}, "mun_id + gold_year"),
        "without Roraima": ({"drop_states": (RORAIMA,)}, FE),
        "gold municipalities only": ({"gold_only": True}, FE),
        "frontier x periods": ({"frontier": True}, FE),
        "frontier x periods, state x year FE": ({"frontier": True}, "mun_id + state_year"),
        "+ current forest loss": ({"deforestation": True}, FE),
        "presence trend": ({"trend": True}, FE),
    }


def run_spec(frame, outcome, fe=FE, bootstrap=True, **kwargs):
    """Fit one presence spec and return one row per term (post-2019, post-2023)."""
    data, terms, controls = spec_data(frame, outcome, **kwargs)
    model = M.estimate(data, terms + controls, fe=fe, weighted=False)
    tidy = model.tidy()
    rows = []
    for name in terms:
        row = {
            "outcome": outcome,
            "exposure": name.split("__x__")[0],
            "term": name.split("__x__")[1],
            "estimate": tidy.loc[name, "Estimate"],
            "se_basin": tidy.loc[name, "Std. Error"],
            "p_basin": tidy.loc[name, "Pr(>|t|)"],
        }
        if bootstrap:
            row["p_wcb"] = wild_cluster_bootstrap(model, name, CLUSTER, reps=M.BOOT_REPS, seed=M.SEED)["p_wcb"]
        row.update(n_obs=int(model._N), n_municipalities=int(data["mun_id"].nunique()),
                   n_basins=int(data[CLUSTER].nunique()))
        rows.append(row)
    return rows


def run_table(frame, outcomes=tuple(OUTCOMES), spec_table=None, bootstrap=True):
    """Every spec for every outcome."""
    rows = []
    for spec, (kwargs, fe) in (spec_table or specs()).items():
        for outcome in outcomes:
            rows += [{"spec": spec, **row} for row in run_spec(frame, outcome, fe=fe, bootstrap=bootstrap, **kwargs)]
    return pd.DataFrame(rows)


def event_study(frame, outcome, exposures=(PRESENCE,), years=YEARS, base_year=BASE_YEAR, controls=True,
                gold_only=False, fe=FE, sample=None):
    """Coefficients of each exposure x year (base year omitted), jointly, municipality and year FE, Otto-3 SEs.

    With `exposures=(IGD_RORAIMA, IGD_OTHER)` it splits lead 1's IGD event study by state. `sample` is a
    boolean column restricting the rows.
    """
    data = frame[frame["year"].between(*years)].copy()
    if gold_only:
        data = data[data[L.S.ALL_GOLD_EXPOSURE] > 0]
    if sample is not None:
        data = data[data[sample] > 0]
    data["y"] = data[outcome]
    data["weight"] = 1.0
    data = data.dropna(subset=["y", *exposures, CLUSTER, L.PRICE])
    terms = {}
    for exposure in exposures:
        for year in range(years[0], years[1] + 1):
            if year != base_year:
                name = f"{exposure}__d{year}"
                data[name] = data[exposure] * data["year"].eq(year)
                terms[name] = (exposure, year)
    other = []
    if controls:
        data[M.interaction(L.DEPOSITS, L.PRICE)] = data[L.DEPOSITS] * data[L.PRICE]
        other = [M.interaction(L.DEPOSITS, L.PRICE)]
    tidy = M.estimate(data, list(terms) + other, fe=fe, weighted=False).tidy()
    rows = [{"exposure": exposure, "year": base_year, "estimate": 0.0, "se": 0.0} for exposure in exposures]
    rows += [{"exposure": exposure, "year": year, "estimate": tidy.loc[name, "Estimate"],
              "se": tidy.loc[name, "Std. Error"]} for name, (exposure, year) in terms.items()]
    return pd.DataFrame(rows).sort_values(["exposure", "year"]).assign(outcome=outcome).reset_index(drop=True)


def randomization(frame, outcome, strata=None, draws=L.RI_DRAWS, **kwargs):
    """Permutation p-values of presence x post-2019 and x post-2023, presence permuted across municipalities
    (within `strata`, a municipality-level column, if given)."""
    data, terms, controls = spec_data(frame, outcome, **kwargs)
    observed, p_values = L.permutation_p(data, terms[0].split("__x__")[0], terms, controls, PERIODS, draws=draws,
                                         strata=strata)
    return [{"outcome": outcome, "term": name.split("__x__")[1], "estimate": estimate, "p_ri": p,
             "strata": strata or "none"} for name, estimate, p in zip(terms, observed, p_values)]


def dose_response(frame, outcome, bins=DOSE_BINS, years=YEARS):
    """Presence quantile x post-2019 and x post-2023 (lowest quantile omitted), municipality and year FE."""
    data, _, controls = spec_data(frame, outcome, years=years)
    quantile = data["mun_id"].map(pd.qcut(data.groupby("mun_id")[PRESENCE].first(), bins, labels=False))
    terms = []
    for q in range(1, bins):
        for period, active in PERIODS.items():
            name = f"q{q}__x__{period}"
            data[name] = (quantile == q) * active(data["year"])
            terms.append(name)
    tidy = M.estimate(data, terms + controls, fe=FE, weighted=False).tidy()
    rows = [{"quantile": 0, "term": period, "estimate": 0.0, "se": 0.0} for period in PERIODS]
    rows += [{"quantile": int(name[1:].split("__")[0]), "term": name.split("__x__")[1],
              "estimate": tidy.loc[name, "Estimate"], "se": tidy.loc[name, "Std. Error"]} for name in terms]
    return pd.DataFrame(rows).assign(outcome=outcome).sort_values(["term", "quantile"]).reset_index(drop=True)
