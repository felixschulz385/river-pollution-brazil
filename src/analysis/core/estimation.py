"""Fitting and reporting helpers shared by the designs: the fixed-effects fit, term names, result rows,
event studies (on a panel or on stacked episodes) and tables of specs x outcomes."""

from __future__ import annotations

import pandas as pd
import pyfixest as pf

from src.analysis.core.defaults import BOOT_REPS, CLUSTER, FE, SEED
from src.analysis.core.inference import wild_cluster_bootstrap


def interaction(left, right):
    """Name of the product of two columns, e.g. exposure x price or exposure x period."""
    return f"{left}__x__{right}"


def estimate(data, terms, fe=FE, weighted=True, vcov=None, outcome="y"):
    """OLS of `outcome` on `terms` with fixed effects `fe`, weighted by `weight` if `weighted`, clustered by
    Otto-3 basin unless `vcov` says otherwise."""
    return pf.feols(
        f"{outcome} ~ {' + '.join(terms)} | {fe}",
        data=data,
        weights="weight" if weighted else None,
        vcov=vcov or {"CRV1": CLUSTER},
    )


def term_row(model, term, bootstrap=True, cluster=CLUSTER, reps=BOOT_REPS, seed=SEED):
    """Estimate, clustered SE and p-value of `term`, and its restricted wild-cluster bootstrap p-value."""
    tidy = model.tidy().loc[term]
    row = {"estimate": tidy["Estimate"], "se_basin": tidy["Std. Error"], "p_basin": tidy["Pr(>|t|)"]}
    if bootstrap:
        row["p_wcb"] = wild_cluster_bootstrap(model, term, cluster, reps=reps, seed=seed)["p_wcb"]
    return row


def event_study(data, exposures, periods, base, controls=(), fe=FE, period_column="year"):
    """Coefficients of each exposure x period dummy (the base period omitted), fitted jointly with the
    `controls` columns, unweighted, clustered by Otto-3 basin.

    `data` holds the outcome as `y`. Returns one row per exposure and period, with the base period at zero.
    """
    data = data.copy()
    terms = {}
    for exposure in exposures:
        for period in periods:
            if period != base:
                name = f"{exposure}__{period_column}_{period}"
                data[name] = data[exposure] * data[period_column].eq(period)
                terms[name] = (exposure, period)
    tidy = estimate(data, list(terms) + list(controls), fe=fe, weighted=False).tidy()
    rows = [{"exposure": exposure, period_column: base, "estimate": 0.0, "se": 0.0} for exposure in exposures]
    rows += [{"exposure": exposure, period_column: period, "estimate": tidy.loc[name, "Estimate"],
              "se": tidy.loc[name, "Std. Error"]} for name, (exposure, period) in terms.items()]
    return pd.DataFrame(rows).sort_values(["exposure", period_column]).reset_index(drop=True)


def stacked_event_study(stacked, outcome, base, fe, cluster):
    """Treated x period relative to the event (`rel`, the base omitted) on a stacked panel of episodes.

    `stacked` has one row per episode, unit and period, with `treated` and `rel`; `fe` names the
    episode x unit and episode x period keys.
    """
    data = stacked.copy()
    terms = []
    for rel in sorted(data["rel"].unique()):
        if rel != base:
            name = f"rel_{'m' if rel < 0 else 'p'}{abs(rel)}"
            data[name] = data["treated"] * data["rel"].eq(rel)
            terms.append((name, rel))
    model = estimate(data, [name for name, _ in terms], fe=fe, weighted=False, vcov={"CRV1": cluster},
                     outcome=outcome)
    tidy = model.tidy()
    rows = [{"rel": base, "estimate": 0.0, "se": 0.0}]
    rows += [{"rel": rel, "estimate": tidy.loc[name, "Estimate"], "se": tidy.loc[name, "Std. Error"]}
             for name, rel in terms]
    return pd.DataFrame(rows).sort_values("rel").assign(outcome=outcome).reset_index(drop=True)


def run_table(run_spec, frame, outcomes, specs, skip_failures=False):
    """`run_spec(frame, outcome, **kwargs)` for every spec and outcome, each returned row tagged with the spec.

    `specs` maps a spec name to its keyword arguments. With `skip_failures`, an outcome that fails is
    reported and skipped.
    """
    rows = []
    for spec, kwargs in specs.items():
        for outcome in outcomes:
            try:
                rows += [{"spec": spec, **row} for row in run_spec(frame, outcome, **kwargs)]
            except Exception as error:
                if not skip_failures:
                    raise
                print(f"[{spec} / {outcome}] failed: {error}")
    return pd.DataFrame(rows)
