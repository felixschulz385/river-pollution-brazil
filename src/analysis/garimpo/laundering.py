"""Lead 1's laundering test: does garimpo respond to the gold-laundering rules where gold can only be mined illegally?

IGD (gold deposits inside protected areas) is the municipality's never-exploited
SGB gold occurrences inside areas that bar mining and were in force by 2012
(`protected_areas`, counted in `geology`). The 2013 good-faith presumption (Lei
12.844, July 2013) let such gold be sold through permits; its suspension in
2023 (STF injunction, April 2023) closed that route again. So

    X_mt = b13 (IGD_m x post-2013_t) + b23 (IGD_m x post-2023_t)
         + g (log gold price in BRL_t x deposits_m) + municipality FE + year FE,

on the outcomes of `supply` (C11 garimpo in the municipality's own
trenches, 2004-2019, so without the 2023 term; SIVEP garimpo cases, 2004-2024)
and on PLG requests and grants (`mining`). Legal Amazon, unweighted, clustered
by Otto-3 basin.

Few municipalities hold IGD (27 locally), so the wild-cluster bootstrap can
over-reject (E10). The randomization inference permutes the IGD assignment
across municipalities: every term built from the exposure is rebuilt with the
permuted exposure and the coefficients are recomputed by FWL, net of the FE and
of the other controls. Across all Legal Amazon municipalities the permutation
is not valid (IGD municipalities are garimpo regions with their own trends, and
it rejects for the non-garimpo malaria placebo); within municipalities with
gold occurrences ("gold municipalities only") it is the binding test.

The permit outcomes are also read by quarter (`mining`'s monthly requests): an
event study of IGD x quarter around the law (2013Q3) and the suspension
(2023Q2), and a scan of IGD x post-q over candidate break quarters q. The
least-squares break date is the q with the smallest residual sum of squares;
it is the law's quarter if the law is what moved the requests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.core.defaults import CLUSTER, FE, PRICE, STATE_YEAR_FE
from src.analysis.core.estimation import estimate, event_study as _event_study, interaction
from src.analysis.core.estimation import run_table as _run_table, term_row
from src.analysis.core.geography import RORAIMA
from src.analysis.core.inference import permutation_p
from src.analysis.core.panels import PROJECT_ROOT
from src.analysis.garimpo import frame as F
from src.analysis.garimpo.frame import (
    ALL_GOLD_EXPOSURE, DEPOSITS, IGD, PRICE_CONTROL, TIN_IGD, YEARS,
    add_price_control, estimation_rows, laundering_frame,
)
from src.data.sources.mining.constants import DEFAULT_MONTHLY_OUTPUT_PATH

LAW_YEAR = 2013  # Lei 12.844/2013, July 2013
REVERSAL_YEAR = 2023  # STF suspension of the good-faith presumption, April 2023
COLLAPSE_YEAR = 2016  # IBAMA field enforcement falls from 2016
BASE_YEAR = 2012

QUARTERS = ("2009Q1", "2025Q4")
LAW_QUARTER = "2013Q3"  # Lei 12.844 of 19 July 2013
REVERSAL_QUARTER = "2023Q2"  # STF injunction of April 2023
BASE_QUARTER = "2013Q2"
QUARTER_FE = "mun_id + quarter_id"
SCAN = ("2010Q1", "2022Q4")  # candidate break quarters, scanned on the quarters before the reversal

EXPOSURES = F.IGD_EXPOSURES
OUTCOMES = {
    **F.OUTCOMES,
    "log1p_plg_requests_gold": ("log(1 + gold PLG requests)", YEARS),
    "log1p_plg_grants_gold": ("log(1 + gold PLG grants)", YEARS),
}


def build_frame(path=None):
    """The Legal Amazon panel with the IGD exposures, PLG outcomes and enforcement counts
    (`frame.laundering_frame`)."""
    return laundering_frame(path)


def period_columns(law_year=LAW_YEAR, collapse=False, trend=False):
    """Name -> function of the year for the terms interacted with the exposure."""
    columns = {
        "post2013": lambda year: (year >= law_year).astype(float),
        "post2023": lambda year: (year >= REVERSAL_YEAR).astype(float),
    }
    if collapse:
        columns["post2016"] = lambda year: (year >= COLLAPSE_YEAR).astype(float)
    if trend:
        columns["trend"] = lambda year: (year - YEARS[0]).astype(float)
    return columns



def spec_data(frame, outcome, exposure=IGD, years=None, law_year=LAW_YEAR, collapse=False, trend=False,
              drop_states=(), deposit_only=False, placebo=False, enforcement=False):
    """Estimation rows of one spec and its terms: (data, exposure terms, other controls, periods)."""
    years = years or OUTCOMES[outcome][1]
    data = estimation_rows(frame, outcome, years, drop_states, keep=ALL_GOLD_EXPOSURE if deposit_only else None,
                           required=[PRICE, exposure])

    periods = period_columns(law_year, collapse, trend)
    exposure_terms = []
    for period, function in periods.items():
        values = function(data["year"])
        if values.nunique() > 1:  # e.g. no post-2023 year in the C11 window
            data[interaction(exposure, period)] = data[exposure] * values
            exposure_terms.append(interaction(exposure, period))
    controls = [PRICE_CONTROL]
    add_price_control(data)
    if placebo:
        for period in ("post2013", "post2023"):
            values = periods[period](data["year"])
            if values.nunique() > 1:
                data[interaction(TIN_IGD, period)] = data[TIN_IGD] * values
                controls.append(interaction(TIN_IGD, period))
    if enforcement:
        controls.append("log1p_garimpo_enforcement")
    return data, exposure_terms, controls, periods


def specs():
    """Spec name -> keyword arguments of `run_spec`, baseline first."""
    return {
        "baseline": {},
        "state x year FE": {"fe": STATE_YEAR_FE},
        "law from 2014": {"law_year": LAW_YEAR + 1},
        "+ IGD x post-2016": {"collapse": True},
        "+ IGD trend": {"trend": True},
        "without Roraima": {"drop_states": (RORAIMA,)},
        "+ garimpo enforcement": {"enforcement": True},
        "gold municipalities only": {"deposit_only": True},
        "exposure: + 200 km upstream": {"exposure": "igd_upstream_200"},
        "exposure: + all upstream": {"exposure": "igd_upstream_all"},
        "exposure: + Otto-4 upstream": {"exposure": "igd_otto"},
        "exposure: any gold in protected areas": {"exposure": "igd_all_gold"},
        "placebo: + tin in protected areas": {"placebo": True},
    }


def run_spec(frame, outcome, exposure=IGD, fe=FE, randomization=False, **kwargs):
    """Fit one spec and return one row per exposure term (post-2013, post-2023, and any extra)."""
    data, exposure_terms, controls, periods = spec_data(frame, outcome, exposure, **kwargs)
    model = estimate(data, exposure_terms + controls, fe=fe, weighted=False)
    tidy = model.tidy()
    if randomization:
        _, p_permutation = permutation_p(data, exposure, exposure_terms, controls, periods, fe=fe)
    exposed = data.loc[data[exposure] > 0]
    rows = []
    for index, name in enumerate(exposure_terms):
        row = {"outcome": outcome, "exposure": exposure, "term": name.split("__x__")[1], **term_row(model, name)}
        if randomization:
            row["p_permutation"] = p_permutation[index]
        row.update(
            n_obs=int(model._N),
            n_municipalities=int(data["mun_id"].nunique()),
            n_exposed=int(exposed["mun_id"].nunique()),
            n_exposed_basins=int(exposed[CLUSTER].nunique()),
            mean_y_exposed_pre=float(exposed.loc[exposed["year"] < LAW_YEAR, "y"].mean()),
        )
        rows.append(row)
    if kwargs.get("placebo"):
        for period in ("post2013", "post2023"):
            name = interaction(TIN_IGD, period)
            if name in tidy.index:
                rows.append({"outcome": outcome, "exposure": TIN_IGD, "term": period,
                             **term_row(model, name, bootstrap=False)})
    return rows


def run_table(frame, outcomes=tuple(OUTCOMES), spec_kwargs=None,
              randomization_specs=("baseline", "state x year FE", "without Roraima", "gold municipalities only")):
    """Every spec for every outcome; an outcome that fails is reported and skipped."""
    specs_ = {name: {**kwargs, "randomization": name in randomization_specs}
              for name, kwargs in (spec_kwargs or specs()).items()}
    return _run_table(run_spec, frame, outcomes, specs_, skip_failures=True)


def event_study(frame, outcome, exposure=IGD, base_year=BASE_YEAR, drop_states=()):
    """Coefficients of exposure x year (base year omitted), with the price x deposits control, Otto-3 SEs."""
    years = OUTCOMES[outcome][1]
    data, _, controls, _ = spec_data(frame, outcome, exposure, years=years, drop_states=drop_states)
    return _event_study(data, [exposure], range(years[0], years[1] + 1), base_year,
                        controls=controls).assign(outcome=outcome)


def sample_table(frame):
    """Municipality counts by exposure, and garimpo cases and PLG requests in IGD municipalities."""
    municipalities = frame.drop_duplicates("mun_id").set_index("mun_id")
    igd = municipalities.index[municipalities[IGD] > 0]
    cases = frame.groupby("mun_id")["inf_garimpo"].sum()
    requests = frame.groupby("mun_id")["plg_requests_gold"].sum()
    rows = {"Legal Amazon municipalities in the panel": len(municipalities)}
    rows.update({f"exposed: {label}": int(municipalities[name].sum()) for name, label in EXPOSURES.items()})
    rows.update({
        "exposed: tin in protected areas (placebo)": int(municipalities[TIN_IGD].sum()),
        "Otto-3 basins with an IGD municipality": int(municipalities.loc[igd, CLUSTER].nunique()),
        "garimpo cases 2004-2024": int(cases.sum()),
        "share in IGD municipalities": float(cases.reindex(igd).sum() / cases.sum()),
        "gold PLG requests 2004-2024": int(requests.sum()),
        "share in IGD municipalities ": float(requests.reindex(igd).sum() / requests.sum()),
    })
    return pd.Series(rows)


def quarterly_requests(frame, path=None, quarters=QUARTERS):
    """Municipality x quarter gold PLG requests (zeros filled) for `frame`'s municipalities, with their exposures."""
    path = path or PROJECT_ROOT / DEFAULT_MONTHLY_OUTPUT_PATH
    monthly = pd.read_parquet(path, columns=["mun_id", "month", "plg_requests_gold"])
    municipalities = frame.drop_duplicates("mun_id").set_index("mun_id")[
        [*EXPOSURES, TIN_IGD, DEPOSITS, ALL_GOLD_EXPOSURE, CLUSTER]
    ]
    monthly = monthly[monthly["mun_id"].isin(municipalities.index)]
    counts = monthly.groupby([monthly["mun_id"], monthly["month"].dt.to_period("Q").rename("quarter")])[
        "plg_requests_gold"].sum()
    grid = pd.MultiIndex.from_product(
        [municipalities.index, pd.period_range(*quarters, freq="Q")], names=["mun_id", "quarter"]
    )
    data = counts.reindex(grid, fill_value=0).rename("plg_requests_gold").reset_index()
    data = data.join(municipalities, on="mun_id")
    data["quarter_id"] = data["quarter"].astype(str)
    data["y"] = np.log1p(data["plg_requests_gold"])
    data["weight"] = 1.0
    return data.dropna(subset=[CLUSTER]).reset_index(drop=True)


def quarterly_event_study(data, exposure=IGD, base=BASE_QUARTER, deposit_only=False):
    """Coefficients of exposure x quarter (base quarter omitted), municipality and quarter FE, Otto-3 SEs."""
    data = data[data[ALL_GOLD_EXPOSURE] > 0] if deposit_only else data
    return _event_study(data, [exposure], sorted(data["quarter"].unique()), pd.Period(base, freq="Q"),
                        fe=QUARTER_FE, period_column="quarter")


def break_scan(data, exposure=IGD, candidates=SCAN, deposit_only=False):
    """exposure x post-q for each candidate break quarter q, on the quarters before the reversal, with its SSR."""
    data = data[data["quarter"] < pd.Period(REVERSAL_QUARTER, freq="Q")]
    if deposit_only:
        data = data[data[ALL_GOLD_EXPOSURE] > 0]
    data = data.copy()
    rows = []
    for candidate in pd.period_range(*candidates, freq="Q"):
        data["step"] = data[exposure] * (data["quarter"] >= candidate)
        model = estimate(data, ["step"], fe=QUARTER_FE, weighted=False)
        tidy = model.tidy()
        rows.append({"break": candidate, "estimate": tidy.loc["step", "Estimate"],
                     "se": tidy.loc["step", "Std. Error"], "t": tidy.loc["step", "t value"],
                     "ssr": float((model.resid() ** 2).sum())})
    return pd.DataFrame(rows).assign(exposure=exposure)
