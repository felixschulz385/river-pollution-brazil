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
paths (`core.inference.random_walk_p`) is the binding inference. The
price-exposure SE is left out: in this setting it is close to clustering by year
(E9 of `output/research/mining_economics_leads.md`). Tin occurrences are the
other-minerals placebo, and non-garimpo malaria the outcome placebo.
"""

from __future__ import annotations

import pandas as pd

from src.analysis.core.defaults import CLUSTER, FE, PRICE, PRICE_USD, STATE_YEAR_FE
from src.analysis.core.estimation import estimate, event_study as _event_study, interaction
from src.analysis.core.estimation import run_table as _run_table, term_row
from src.analysis.core.geography import RORAIMA
from src.analysis.core.inference import random_walk_p
from src.analysis.garimpo.frame import (
    ALL_GOLD_EXPOSURE, COUNT_EXPOSURE, DEPOSITS, OUTCOMES, TIN_EXPOSURE, estimation_rows, geology_frame,
)

PRE_LAW_YEARS = (2004, 2012)  # before the 2013 good-faith rule
EXPOSURE = DEPOSITS


def trend_term(exposure):
    return interaction(exposure, "trend")


def build_frame(path=None):
    """The Legal Amazon panel with the own-geology exposures and the supply outcomes (`frame.geology_frame`)."""
    return geology_frame(path)


def spec_data(frame, outcome, exposures=(EXPOSURE,), price=PRICE, years=None, trend=False, drop_states=()):
    """Estimation rows of one spec: `y`, the price interactions, deposit trends and a unit weight."""
    years = years or OUTCOMES[outcome][1]
    data = estimation_rows(frame, outcome, years, drop_states, required=[price, *exposures])
    for exposure in exposures:
        data[interaction(exposure, price)] = data[exposure] * data[price]
        if trend:
            data[trend_term(exposure)] = data[exposure] * data["trend"]
    return data


def infer(model, data, term, exposure, price, fe=FE, controls=(), randomization=False):
    """Otto-3 SE, wild-cluster bootstrap, two-way (basin x year) and, optionally, random-walk p-values."""
    two_way = estimate(data, list(model._coefnames), fe=fe, weighted=False,
                       vcov={"CRV1": f"{CLUSTER}+year"}).tidy().loc[term]
    row = {"term": term, **term_row(model, term), "se_two_way": two_way["Std. Error"],
           "p_two_way": two_way["Pr(>|t|)"]}
    if randomization:
        _, row["p_random_walk"] = random_walk_p(data, exposure, price, fe=fe, weighted=False, controls=controls)
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
    terms = [interaction(exposure, price) for exposure in exposures]
    controls = [trend_term(exposure) for exposure in exposures] if trend else []
    model = estimate(data, terms + controls, fe=fe, weighted=False)
    return [
        {"outcome": outcome, "exposure": exposure,
         **infer(model, data, term, exposure, price, fe=fe, controls=controls,
                 randomization=randomization and len(terms) == 1)}
        for exposure, term in zip(exposures, terms)
    ]


def run_table(frame, outcomes=tuple(OUTCOMES), spec_kwargs=None, randomization_specs=("baseline", "deposit trend", "without Roraima")):
    """Every spec for every outcome; an outcome that fails is reported and skipped."""
    specs_ = {name: {**kwargs, "randomization": name in randomization_specs}
              for name, kwargs in (spec_kwargs or specs()).items()}
    return _run_table(run_spec, frame, outcomes, specs_, skip_failures=True)


def event_study(frame, outcome, exposure=EXPOSURE, base_year=None, drop_states=()):
    """Coefficients of exposure x year (base year omitted), municipality and year FE, Otto-3 SEs."""
    years = OUTCOMES[outcome][1]
    base_year = base_year or years[0]
    data = spec_data(frame, outcome, (exposure,), years=years, drop_states=drop_states)
    return _event_study(data, [exposure], range(years[0], years[1] + 1), base_year).assign(outcome=outcome)


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
