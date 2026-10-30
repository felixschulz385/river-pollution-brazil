"""Lead 1's laundering test: does garimpo respond to the gold-laundering rules where gold can only be mined illegally?

IGD (gold deposits inside protected areas) is the municipality's never-exploited
SGB gold occurrences inside areas that bar mining and were in force by 2012
(`protected_areas`, counted in `geology`). The 2013 good-faith presumption (Lei
12.844, July 2013) let such gold be sold through permits; its suspension in
2023 (STF injunction, April 2023) closed that route again. So

    X_mt = b13 (IGD_m x post-2013_t) + b23 (IGD_m x post-2023_t)
         + g (log gold price in BRL_t x deposits_m) + municipality FE + year FE,

on the outcomes of `garimpo_supply` (C11 garimpo in the municipality's own
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

from src.analysis import garimpo_supply as S
from src.analysis import moreno_louzada as M
from src.analysis.inference import wild_cluster_bootstrap
from src.data.sources.mining.constants import DEFAULT_MONTHLY_OUTPUT_PATH

YEARS = S.YEARS
C11_YEARS = S.C11_YEARS
LAW_YEAR = 2013  # Lei 12.844/2013, July 2013
REVERSAL_YEAR = 2023  # STF suspension of the good-faith presumption, April 2023
COLLAPSE_YEAR = 2016  # IBAMA field enforcement falls from 2016
BASE_YEAR = 2012
RORAIMA = S.RORAIMA
FE, STATE_YEAR_FE, CLUSTER = M.FE, M.STATE_YEAR_FE, M.CLUSTER
PRICE = S.PRICE
DEPOSITS = S.EXPOSURE  # never-exploited gold anywhere in the municipality
RI_DRAWS = 999
RI_CHUNK = 100

QUARTERS = ("2009Q1", "2025Q4")
LAW_QUARTER = "2013Q3"  # Lei 12.844 of 19 July 2013
REVERSAL_QUARTER = "2023Q2"  # STF injunction of April 2023
BASE_QUARTER = "2013Q2"
QUARTER_FE = "mun_id + quarter_id"
SCAN = ("2010Q1", "2022Q4")  # candidate break quarters, scanned on the quarters before the reversal

IGD = "igd"
EXPOSURES = {
    IGD: "never-exploited gold in protected areas, in the municipality",
    "igd_upstream_200": "... or within 200 km upstream",
    "igd_upstream_all": "... or anywhere upstream",
    "igd_otto": "... or in an Otto-4 sub-basin upstream",
    "igd_all_gold": "any gold occurrence in protected areas, in the municipality",
}
TIN_IGD = "tin_igd"

OUTCOMES = {
    **S.OUTCOMES,
    "log1p_plg_requests_gold": ("log(1 + gold PLG requests)", YEARS),
    "log1p_plg_grants_gold": ("log(1 + gold PLG grants)", YEARS),
}
PLACEBO_OUTCOMES = S.PLACEBO_OUTCOMES


def build_frame(path=None):
    """`garimpo_supply`'s Legal Amazon panel with the IGD exposures, PLG outcomes and enforcement counts."""
    frame = S.build_frame() if path is None else S.build_frame(path)
    local = frame["gold_occurrences_unexploited_protected"].fillna(0).gt(0)
    frame[IGD] = local.astype(float)
    frame["igd_upstream_200"] = (local | frame["gold_unexploited_protected_upstream_200"].fillna(0).gt(0)).astype(float)
    frame["igd_upstream_all"] = (local | frame["gold_unexploited_protected_upstream_all"].fillna(0).gt(0)).astype(float)
    frame["igd_otto"] = (local | frame["gold_unexploited_protected_upstream_otto4"].fillna(0).gt(0)).astype(float)
    frame["igd_all_gold"] = frame["gold_occurrences_protected"].fillna(0).gt(0).astype(float)
    frame[TIN_IGD] = frame["tin_occurrences_protected"].fillna(0).gt(0).astype(float)

    frame["log1p_plg_requests_gold"] = np.log1p(frame["plg_requests_gold"])
    frame["log1p_plg_grants_gold"] = np.log1p(frame["plg_grants_gold"])
    # Garimpo fines, embargoes and seizures (destruction terms start in 2014).
    # Seizures are recorded from 2005; the missing 2004 counts as none.
    enforcement = sum(frame[f"ibama_{action}_garimpo"].fillna(0) for action in ("fines", "embargoes", "seizures"))
    frame["log1p_garimpo_enforcement"] = np.log1p(enforcement)
    return frame


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


def term(exposure, period):
    return f"{exposure}__x__{period}"


def spec_data(frame, outcome, exposure=IGD, years=None, law_year=LAW_YEAR, collapse=False, trend=False,
              drop_states=(), deposit_only=False, placebo=False, enforcement=False):
    """Estimation rows of one spec and its terms: (data, exposure terms, other controls)."""
    years = years or OUTCOMES[outcome][1]
    data = frame[frame["year"].between(*years) & ~frame["mun_id"].str[:2].isin(drop_states)].copy()
    if deposit_only:
        data = data[data[S.ALL_GOLD_EXPOSURE] > 0]
    data["y"] = data[outcome]
    data["weight"] = 1.0
    data = data.dropna(subset=["y", CLUSTER, PRICE, exposure]).reset_index(drop=True)

    periods = period_columns(law_year, collapse, trend)
    exposure_terms = []
    for period, function in periods.items():
        values = function(data["year"])
        if values.nunique() > 1:  # e.g. no post-2023 year in the C11 window
            data[term(exposure, period)] = data[exposure] * values
            exposure_terms.append(term(exposure, period))
    data[M.interaction(DEPOSITS, PRICE)] = data[DEPOSITS] * data[PRICE]
    controls = [M.interaction(DEPOSITS, PRICE)]
    if placebo:
        for period in ("post2013", "post2023"):
            values = periods[period](data["year"])
            if values.nunique() > 1:
                data[term(TIN_IGD, period)] = data[TIN_IGD] * values
                controls.append(term(TIN_IGD, period))
    if enforcement:
        controls.append("log1p_garimpo_enforcement")
    return data, exposure_terms, controls, periods


def permutation_p(data, exposure, exposure_terms, controls, periods, fe=FE, draws=RI_DRAWS, seed=M.SEED,
                  strata=None):
    """Two-sided permutation p-values of the exposure terms, permuting the exposure across municipalities
    (within the groups of `strata`, a column constant within municipalities, if given).

    Each draw reassigns the municipalities' exposure values at random, rebuilds
    every exposure term, and recomputes the coefficients by FWL net of the FE
    and the `controls`. Returns the observed coefficients and the p-values.
    """
    weights = np.ones(len(data))
    municipalities, position = np.unique(data["mun_id"].to_numpy(), return_inverse=True)
    exposure_by_mun = data.groupby("mun_id")[exposure].first().reindex(municipalities).to_numpy(dtype=np.float64)
    groups = [np.arange(len(municipalities))] if strata is None else [
        np.flatnonzero(labels == label)
        for labels in [data.groupby("mun_id")[strata].first().reindex(municipalities).to_numpy()]
        for label in np.unique(labels)
    ]

    def permute(rng):
        permuted = exposure_by_mun.copy()
        for group in groups:
            permuted[group] = rng.permutation(exposure_by_mun[group])
        return permuted
    period_values = np.column_stack([
        periods[name.split("__x__")[1]](data["year"]).to_numpy(dtype=np.float64) for name in exposure_terms
    ])

    def partial(columns):
        residuals = M._demean(columns, data, fe, weights)
        if controls:
            control_values = M._demean(data[list(controls)].to_numpy(dtype=np.float64), data, fe, weights)
            coefficients = np.linalg.lstsq(control_values, residuals, rcond=None)[0]
            residuals = residuals - control_values @ coefficients
        return residuals

    y = partial(data["y"].to_numpy(dtype=np.float64)[:, None])[:, 0]

    def betas(exposure_values):
        # exposure_values: municipalities x draws -> coefficients: draws x terms
        k = len(exposure_terms)
        stacked = np.concatenate(
            [exposure_values[position] * period_values[:, [j]] for j in range(k)], axis=1
        )
        residuals = partial(stacked)
        draws_here = exposure_values.shape[1]
        out = np.empty((draws_here, k))
        for d in range(draws_here):
            X = residuals[:, [j * draws_here + d for j in range(k)]]
            out[d] = np.linalg.lstsq(X, y, rcond=None)[0]
        return out

    observed = betas(exposure_by_mun[:, None])[0]
    rng = np.random.default_rng(seed)
    simulated = []
    for start in range(0, draws, RI_CHUNK):
        n = min(RI_CHUNK, draws - start)
        permuted = np.column_stack([permute(rng) for _ in range(n)])
        simulated.append(betas(permuted))
    simulated = np.concatenate(simulated)
    p_values = (1 + (np.abs(simulated) >= np.abs(observed)).sum(axis=0)) / (1 + draws)
    return observed, p_values


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
    model = M.estimate(data, exposure_terms + controls, fe=fe, weighted=False)
    tidy = model.tidy()
    if randomization:
        _, p_permutation = permutation_p(data, exposure, exposure_terms, controls, periods, fe=fe)
    exposed = data.loc[data[exposure] > 0]
    rows = []
    for index, name in enumerate(exposure_terms):
        row = {
            "outcome": outcome,
            "exposure": exposure,
            "term": name.split("__x__")[1],
            "estimate": tidy.loc[name, "Estimate"],
            "se_basin": tidy.loc[name, "Std. Error"],
            "p_basin": tidy.loc[name, "Pr(>|t|)"],
            "p_wcb": wild_cluster_bootstrap(model, name, CLUSTER, reps=M.BOOT_REPS, seed=M.SEED)["p_wcb"],
        }
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
            name = term(TIN_IGD, period)
            if name in tidy.index:
                rows.append({"outcome": outcome, "exposure": TIN_IGD, "term": period,
                             "estimate": tidy.loc[name, "Estimate"], "se_basin": tidy.loc[name, "Std. Error"],
                             "p_basin": tidy.loc[name, "Pr(>|t|)"]})
    return rows


def run_table(frame, outcomes=tuple(OUTCOMES), spec_kwargs=None,
              randomization_specs=("baseline", "state x year FE", "without Roraima", "gold municipalities only")):
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


def event_study(frame, outcome, exposure=IGD, base_year=BASE_YEAR, drop_states=()):
    """Coefficients of exposure x year (base year omitted), with the price x deposits control, Otto-3 SEs."""
    years = OUTCOMES[outcome][1]
    data, _, controls, _ = spec_data(frame, outcome, exposure, years=years, drop_states=drop_states)
    terms = []
    for year in range(years[0], years[1] + 1):
        if year != base_year:
            data[f"d{year}"] = data[exposure] * data["year"].eq(year)
            terms.append(f"d{year}")
    tidy = M.estimate(data, terms + controls, weighted=False).tidy()
    rows = [{"year": base_year, "estimate": 0.0, "se": 0.0}]
    rows += [{"year": int(name[1:]), "estimate": tidy.loc[name, "Estimate"], "se": tidy.loc[name, "Std. Error"]}
             for name in terms]
    return pd.DataFrame(rows).sort_values("year").assign(outcome=outcome, exposure=exposure).reset_index(drop=True)


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
    path = path or M.PROJECT_ROOT / DEFAULT_MONTHLY_OUTPUT_PATH
    monthly = pd.read_parquet(path, columns=["mun_id", "month", "plg_requests_gold"])
    municipalities = frame.drop_duplicates("mun_id").set_index("mun_id")[
        [*EXPOSURES, TIN_IGD, DEPOSITS, S.ALL_GOLD_EXPOSURE, CLUSTER]
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
    data = data[data[S.ALL_GOLD_EXPOSURE] > 0].copy() if deposit_only else data.copy()
    base = pd.Period(base, freq="Q")
    terms = {}
    for quarter in sorted(data["quarter"].unique()):
        if quarter != base:
            name = f"q{quarter}"
            data[name] = data[exposure] * data["quarter"].eq(quarter)
            terms[name] = quarter
    tidy = M.estimate(data, list(terms), fe=QUARTER_FE, weighted=False).tidy()
    rows = [{"quarter": base, "estimate": 0.0, "se": 0.0}]
    rows += [{"quarter": quarter, "estimate": tidy.loc[name, "Estimate"], "se": tidy.loc[name, "Std. Error"]}
             for name, quarter in terms.items()]
    return pd.DataFrame(rows).sort_values("quarter").assign(exposure=exposure).reset_index(drop=True)


def break_scan(data, exposure=IGD, candidates=SCAN, deposit_only=False):
    """exposure x post-q for each candidate break quarter q, on the quarters before the reversal, with its SSR."""
    data = data[data["quarter"] < pd.Period(REVERSAL_QUARTER, freq="Q")]
    if deposit_only:
        data = data[data[S.ALL_GOLD_EXPOSURE] > 0]
    data = data.copy()
    rows = []
    for candidate in pd.period_range(*candidates, freq="Q"):
        data["step"] = data[exposure] * (data["quarter"] >= candidate)
        model = M.estimate(data, ["step"], fe=QUARTER_FE, weighted=False)
        tidy = model.tidy()
        rows.append({"break": candidate, "estimate": tidy.loc["step", "Estimate"],
                     "se": tidy.loc["step", "Std. Error"], "t": tidy.loc["step", "t value"],
                     "ssr": float((model.resid() ** 2).sum())})
    return pd.DataFrame(rows).assign(exposure=exposure)
