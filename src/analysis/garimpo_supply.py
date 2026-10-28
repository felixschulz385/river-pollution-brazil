"""Lead 1's supply test: does garimpo respond to the gold price where there is gold geology?

The municipality's own gold geology (SGB never-exploited gold occurrences, the
predetermined part of `geology_adm2`) interacted with the log gold price in BRL,
on the municipality's own garimpo: the C11 garimpo share of its own trenches
(2004-2019, before the 2020 reclassification step) and SIVEP garimpo malaria
cases by infection municipality (2004-2024), split into local and incoming
workers. Legal Amazon, municipality and year FE, unweighted, clustered by Otto-3
basin.

The BRL price trends up over the window and jumps in 2020, so the interaction is
close to deposits x trend (`04_moreno_louzada`). The checks therefore add a
deposit-specific linear trend, and the randomization over random-walk price
paths (`moreno_louzada.random_walk_p`) is the binding inference. The
price-exposure SE is left out: in this setting it is close to clustering by year
(E9 of `output/research/mining_economics_leads.md`). Tin occurrences are the
other-minerals placebo, and non-garimpo malaria the outcome placebo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis import moreno_louzada as M
from src.analysis import panels
from src.analysis.inference import wild_cluster_bootstrap

YEARS = (2004, 2024)  # SIVEP
C11_YEARS = (2004, 2019)  # before the C11 2020 reclassification step
PRE_LAW_YEARS = (2004, 2012)  # before the 2013 good-faith rule
RORAIMA = "14"  # the Yanomami garimpo rush from 2019
TREND_BASE_YEAR = 2004

EXPOSURE = "gold_unexploited_any"
ALL_GOLD_EXPOSURE = "gold_any"
COUNT_EXPOSURE = "log1p_gold_unexploited"
TIN_EXPOSURE = "tin_any"
PRICE, PRICE_USD = M.PRICE, M.PRICE_USD
FE, STATE_YEAR_FE, CLUSTER = M.FE, M.STATE_YEAR_FE, M.CLUSTER

# Outcome -> (label, years). Shares in pp of the municipality's own trench area;
# cases as log(1 + cases) by infection municipality.
OUTCOMES = {
    "c11_garimpo_gold_pp": ("C11 garimpo gold, own trenches (pp)", C11_YEARS),
    "c11_garimpo_pp": ("C11 garimpo, all substances, own trenches (pp)", C11_YEARS),
    "log1p_garimpo_cases": ("log(1 + garimpo cases)", YEARS),
    "log1p_garimpo_local": ("log(1 + local garimpo cases)", YEARS),
    "log1p_garimpo_incoming": ("log(1 + incoming garimpo cases)", YEARS),
    "any_garimpo_case": ("any garimpo case (0/1)", YEARS),
    "log1p_other_malaria": ("log(1 + other malaria cases), placebo", YEARS),
}
PLACEBO_OUTCOMES = ("log1p_other_malaria",)


def trend_term(exposure):
    return f"{exposure}__x__trend"


def build_frame(path=panels.ADM2_PANEL_PATH):
    """The Legal Amazon panel with the own-geology exposures and the supply outcomes."""
    frame = panels.load_adm2_panel(path=path, years=YEARS)
    frame = frame[frame["mun_id"].isin(M.legal_amazon())].reset_index(drop=True)
    frame["state"] = frame["mun_id"].str[:2]
    frame["state_year"] = frame["state"] + "_" + frame["year"].astype(str)
    frame["trend"] = (frame["year"] - TREND_BASE_YEAR).astype(float)

    frame[EXPOSURE] = frame["gold_occurrences_unexploited"].fillna(0).gt(0).astype(float)
    frame[ALL_GOLD_EXPOSURE] = frame["gold_occurrences"].fillna(0).gt(0).astype(float)
    frame[COUNT_EXPOSURE] = np.log1p(frame["gold_occurrences_unexploited"].fillna(0))
    frame[TIN_EXPOSURE] = frame["tin_occurrences"].fillna(0).gt(0).astype(float)

    local = {group: 100 * frame[f"local_c11_garimpo_{group}"].fillna(0.0) for group in ("gold", "unknown", "other")}
    frame["c11_garimpo_gold_pp"] = local["gold"]
    frame["c11_garimpo_pp"] = local["gold"] + local["unknown"] + local["other"]
    frame["log1p_garimpo_cases"] = np.log1p(frame["inf_garimpo"])
    frame["log1p_garimpo_local"] = np.log1p(frame["inf_garimpo_local"])
    frame["log1p_garimpo_incoming"] = np.log1p(frame["inf_garimpo_incoming"])
    frame["any_garimpo_case"] = frame["inf_garimpo"].gt(0).astype(float).where(frame["inf_garimpo"].notna())
    frame["log1p_other_malaria"] = np.log1p((frame["inf_cases"] - frame["inf_garimpo"]).clip(lower=0))
    return frame


def spec_data(frame, outcome, exposures=(EXPOSURE,), price=PRICE, years=None, trend=False, drop_states=()):
    """Estimation rows of one spec: `y`, the price interactions, deposit trends and a unit weight."""
    years = years or OUTCOMES[outcome][1]
    data = frame[frame["year"].between(*years) & ~frame["mun_id"].str[:2].isin(drop_states)].copy()
    data["y"] = data[outcome]
    for exposure in exposures:
        data[M.interaction(exposure, price)] = data[exposure] * data[price]
        if trend:
            data[trend_term(exposure)] = data[exposure] * data["trend"]
    data["weight"] = 1.0
    return data.dropna(subset=["y", CLUSTER, price, *exposures]).reset_index(drop=True)


def infer(model, data, term, exposure, price, fe=FE, controls=(), randomization=False):
    """Otto-3 SE, wild-cluster bootstrap, two-way (basin x year) and, optionally, random-walk p-values."""
    tidy = model.tidy().loc[term]
    two_way = M.estimate(data, list(model._coefnames), fe=fe, weighted=False,
                         vcov={"CRV1": f"{CLUSTER}+year"}).tidy().loc[term]
    row = {
        "term": term,
        "estimate": tidy["Estimate"],
        "se_basin": tidy["Std. Error"],
        "p_basin": tidy["Pr(>|t|)"],
        "p_wcb": wild_cluster_bootstrap(model, term, CLUSTER, reps=M.BOOT_REPS, seed=M.SEED)["p_wcb"],
        "se_two_way": two_way["Std. Error"],
        "p_two_way": two_way["Pr(>|t|)"],
    }
    if randomization:
        _, row["p_random_walk"] = M.random_walk_p(data, exposure, price, fe=fe, weighted=False, controls=controls)
    row.update(
        n_obs=int(model._N),
        n_municipalities=int(data["mun_id"].nunique()),
        n_exposed=int(data.loc[data[exposure] > 0, "mun_id"].nunique()),
        n_basins=int(data[CLUSTER].nunique()),
        mean_y=float(data["y"].mean()),
        mean_y_exposed=float(data.loc[data[exposure] > 0, "y"].mean()),
    )
    return row


def specs():
    """Spec name -> keyword arguments of `run_spec`, baseline first."""
    return {
        "baseline": {},
        "deposit trend": {"trend": True},
        "state x year FE": {"fe": STATE_YEAR_FE},
        "state x year FE + deposit trend": {"fe": STATE_YEAR_FE, "trend": True},
        "2004-2012 (before the 2013 rule)": {"years": PRE_LAW_YEARS},
        "without Roraima": {"drop_states": (RORAIMA,)},
        "without Roraima + deposit trend": {"drop_states": (RORAIMA,), "trend": True},
        "USD price": {"price": PRICE_USD},
        "all gold occurrences": {"exposures": (ALL_GOLD_EXPOSURE,)},
        "log(1 + never-exploited count)": {"exposures": (COUNT_EXPOSURE,)},
        "placebo: + tin": {"exposures": (EXPOSURE, TIN_EXPOSURE)},
    }


def run_spec(frame, outcome, exposures=(EXPOSURE,), price=PRICE, fe=FE, years=None, trend=False,
             drop_states=(), randomization=False):
    """Fit one spec and return one row per price interaction."""
    if years:
        years = (max(years[0], OUTCOMES[outcome][1][0]), min(years[1], OUTCOMES[outcome][1][1]))
    data = spec_data(frame, outcome, exposures, price, years, trend, drop_states)
    terms = [M.interaction(exposure, price) for exposure in exposures]
    controls = [trend_term(exposure) for exposure in exposures] if trend else []
    model = M.estimate(data, terms + controls, fe=fe, weighted=False)
    return [
        {"outcome": outcome, "exposure": exposure,
         **infer(model, data, term, exposure, price, fe=fe, controls=controls,
                 randomization=randomization and len(terms) == 1)}
        for exposure, term in zip(exposures, terms)
    ]


def run_table(frame, outcomes=tuple(OUTCOMES), spec_kwargs=None, randomization_specs=("baseline", "deposit trend", "without Roraima")):
    """Every spec for every outcome; an outcome that fails is reported and skipped."""
    rows = []
    for spec, kwargs in (spec_kwargs or specs()).items():
        for outcome in outcomes:
            try:
                for row in run_spec(frame, outcome, randomization=spec in randomization_specs, **kwargs):
                    rows.append({"spec": spec, **row})
            except Exception as error:
                print(f"[{spec} / {outcome}] failed: {error}")
    return pd.DataFrame(rows)


def event_study(frame, outcome, exposure=EXPOSURE, base_year=None, drop_states=()):
    """Coefficients of exposure x year (base year omitted), municipality and year FE, Otto-3 SEs."""
    years = OUTCOMES[outcome][1]
    base_year = base_year or years[0]
    data = spec_data(frame, outcome, (exposure,), years=years, drop_states=drop_states)
    terms = []
    for year in range(years[0], years[1] + 1):
        if year != base_year:
            data[f"d{year}"] = data[exposure] * data["year"].eq(year)
            terms.append(f"d{year}")
    tidy = M.estimate(data, terms, weighted=False).tidy()
    rows = [{"year": base_year, "estimate": 0.0, "se": 0.0}]
    rows += [{"year": int(term[1:]), "estimate": tidy.loc[term, "Estimate"], "se": tidy.loc[term, "Std. Error"]}
             for term in terms]
    return pd.DataFrame(rows).sort_values("year").assign(outcome=outcome).reset_index(drop=True)


def sample_table(frame):
    """Municipality counts by exposure, and garimpo cases by exposure group."""
    municipalities = frame.drop_duplicates("mun_id")
    cases = frame.groupby("mun_id")["inf_garimpo"].sum()
    exposed = municipalities.set_index("mun_id")[EXPOSURE].gt(0)
    return pd.Series({
        "Legal Amazon municipalities in the panel": len(municipalities),
        "with a never-exploited gold occurrence": int(municipalities[EXPOSURE].sum()),
        "with any gold occurrence": int(municipalities[ALL_GOLD_EXPOSURE].sum()),
        "with a tin occurrence": int(municipalities[TIN_EXPOSURE].sum()),
        "Otto-3 basins": int(municipalities[CLUSTER].nunique()),
        "garimpo cases 2004-2024": int(cases.sum()),
        "share in municipalities with never-exploited gold": float(cases[exposed[exposed].index].sum() / cases.sum()),
    })
