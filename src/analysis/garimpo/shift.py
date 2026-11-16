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

The Priority List concentrated monitoring and enforcement on a few frontier
municipalities. Its checks add Priority List and monitored-list status in
2015-2018 x the periods, or drop every municipality listed in 2008-2022.

The panel stops in 2024 with land cover, while SIVEP (preliminary from 2025),
IBAMA, PLG requests and the gold price run to 2025. `extend_frame` appends
those years from the processed tables, with the municipality-level columns
carried over, so the post-2023 term can be read through 2025.
"""

from __future__ import annotations

import pandas as pd

from src.analysis.core.defaults import CLUSTER, FE, PRICE, STATE_YEAR_FE
from src.analysis.core.estimation import estimate, event_study as _event_study, interaction
from src.analysis.core.estimation import run_table as _run_table, term_row
from src.analysis.core.geography import RORAIMA
from src.analysis.core.inference import PERMUTATION_DRAWS, permutation_p
from src.analysis.core.panels import PROJECT_ROOT
from src.analysis.garimpo import frame as F
from src.analysis.garimpo.frame import (
    ALL_GOLD_EXPOSURE, FRONTIER, GROUPS, IGD_OTHER, IGD_RORAIMA, LAST_YEAR, PRESENCE, PRICE_CONTROL, PRIORITY,
    add_price_control, enforcement_frame, estimation_rows,
)

YEARS = (2015, 2024)
EXTENDED_YEARS = (YEARS[0], LAST_YEAR)
SHIFT_YEAR = 2019  # IBAMA fines fall from 2019 (new federal government, January 2019)
RECOVERY_YEAR = 2023  # and recover in 2023
BASE_YEAR = 2018

EXPOSURES = F.PRESENCE_EXPOSURES
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

GROUP_COLUMNS = {
    "inf_garimpo": "garimpo cases",
    "other_malaria": "other malaria cases",
    "plg_requests_gold": "gold PLG requests",
    "ibama_fines_all": "IBAMA fines",
    "ibama_seizures_garimpo": "garimpo seizures",
    "ibama_destructions_garimpo": "garimpo destruction terms",
}


def build_frame(path=None, last_year=LAST_YEAR, root_dir=PROJECT_ROOT):
    """Lead 1's Legal Amazon panel with enforcement presence, the extra outcomes, groups and FE keys, extended to
    `last_year` (`frame.enforcement_frame`)."""
    return enforcement_frame(path, last_year, root_dir)


def group_table(frame, column, first_year=2010):
    """Year x group totals of `column` (IGD in Roraima and elsewhere, the rest of Roraima, the rest)."""
    table = frame[frame["year"] >= first_year].pivot_table(index="year", columns="group", values=column,
                                                           aggfunc="sum")
    return table.reindex(columns=list(GROUPS))


def spec_data(frame, outcome, exposure=PRESENCE, years=YEARS, drop_states=(), gold_only=False, sample=None,
              frontier=False, deforestation=False, trend=False, priority=False, drop_listed=False):
    """Estimation rows of one presence spec: (data, exposure terms, other controls).

    `frontier` adds the frontier measures x the periods, `deforestation` current forest loss, `trend` the
    exposure x a linear trend, and `priority` the Priority List measures x the periods; `drop_listed` drops
    municipalities ever on either list. `sample` is a column restricting the rows to where it is positive.
    """
    if years[1] > frame["year"].max():
        raise ValueError(f"Years {years} beyond the frame's last year {frame['year'].max()}")
    if sample is not None:
        frame = frame[frame[sample] > 0]
    if drop_listed:
        frame = frame[frame["priority_ever"] == 0]
    measures = (*(FRONTIER if frontier else ()), *(PRIORITY if priority else ()))
    required = [exposure, PRICE, *measures, *(("forest_loss",) if deforestation else ())]
    data = estimation_rows(frame, outcome, years, drop_states, keep=ALL_GOLD_EXPOSURE if gold_only else None,
                           required=required)
    terms = []
    for period, active in PERIODS.items():
        data[interaction(exposure, period)] = data[exposure] * active(data["year"])
        terms.append(interaction(exposure, period))
    add_price_control(data)
    controls = [PRICE_CONTROL]
    for measure in measures:
        for period, active in PERIODS.items():
            data[interaction(measure, period)] = data[measure] * active(data["year"])
            controls.append(interaction(measure, period))
    if deforestation:
        controls.append("forest_loss")
    if trend:
        data[interaction(exposure, "trend")] = data[exposure] * (data["year"] - years[0])
        controls.append(interaction(exposure, "trend"))
    return data, terms, controls


def specs():
    """Spec name -> keyword arguments of `run_spec`."""
    return {
        "baseline": {},
        "presence from non-mining fines": {"exposure": "presence_nonmining"},
        "presence 2005-2010": {"exposure": "presence_early"},
        "state x year FE": {"fe": STATE_YEAR_FE},
        "IGD x year FE": {"fe": "mun_id + igd_year"},
        "gold x year FE": {"fe": "mun_id + gold_year"},
        "without Roraima": {"drop_states": (RORAIMA,)},
        "gold municipalities only": {"gold_only": True},
        "frontier x periods": {"frontier": True},
        "frontier x periods, state x year FE": {"frontier": True, "fe": STATE_YEAR_FE},
        "+ current forest loss": {"deforestation": True},
        "presence trend": {"trend": True},
        "Priority List x periods": {"priority": True},
        "Priority List x periods, state x year FE": {"priority": True, "fe": STATE_YEAR_FE},
        "without listed municipalities": {"drop_listed": True},
        "through 2025": {"years": EXTENDED_YEARS},
        "through 2025, state x year FE": {"years": EXTENDED_YEARS, "fe": STATE_YEAR_FE},
        "through 2025, without Roraima": {"years": EXTENDED_YEARS, "drop_states": (RORAIMA,)},
    }


def run_spec(frame, outcome, fe=FE, bootstrap=True, **kwargs):
    """Fit one presence spec and return one row per term (post-2019, post-2023)."""
    data, terms, controls = spec_data(frame, outcome, **kwargs)
    model = estimate(data, terms + controls, fe=fe, weighted=False)
    rows = []
    for name in terms:
        exposure, period = name.split("__x__")
        row = {"outcome": outcome, "exposure": exposure, "term": period, **term_row(model, name, bootstrap)}
        row.update(n_obs=int(model._N), n_municipalities=int(data["mun_id"].nunique()),
                   n_basins=int(data[CLUSTER].nunique()))
        rows.append(row)
    return rows


def run_table(frame, outcomes=tuple(OUTCOMES), spec_table=None, bootstrap=True):
    """Every spec for every outcome."""
    specs_ = {name: {**kwargs, "bootstrap": bootstrap} for name, kwargs in (spec_table or specs()).items()}
    return _run_table(run_spec, frame, outcomes, specs_)


def event_study(frame, outcome, exposures=(PRESENCE,), years=YEARS, base_year=BASE_YEAR, controls=True,
                gold_only=False, fe=FE, sample=None):
    """Coefficients of each exposure x year (base year omitted), jointly, municipality and year FE, Otto-3 SEs.

    With `exposures=(IGD_RORAIMA, IGD_OTHER)` it splits lead 1's IGD event study by state. `sample` is a
    column restricting the rows to where it is positive.
    """
    if sample is not None:
        frame = frame[frame[sample] > 0]
    data = estimation_rows(frame, outcome, years, keep=ALL_GOLD_EXPOSURE if gold_only else None,
                           required=[*exposures, PRICE])
    if controls:
        add_price_control(data)
    return _event_study(data, exposures, range(years[0], years[1] + 1), base_year,
                        controls=[PRICE_CONTROL] if controls else [], fe=fe).assign(outcome=outcome)


def randomization(frame, outcome, strata=None, draws=PERMUTATION_DRAWS, **kwargs):
    """Permutation p-values of presence x post-2019 and x post-2023, presence permuted across municipalities
    (within `strata`, a municipality-level column, if given)."""
    data, terms, controls = spec_data(frame, outcome, **kwargs)
    observed, p_values = permutation_p(data, terms[0].split("__x__")[0], terms, controls, PERIODS, draws=draws,
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
    tidy = estimate(data, terms + controls, fe=FE, weighted=False).tidy()
    rows = [{"quantile": 0, "term": period, "estimate": 0.0, "se": 0.0} for period in PERIODS]
    rows += [{"quantile": int(name[1:].split("__")[0]), "term": name.split("__x__")[1],
              "estimate": tidy.loc[name, "Estimate"], "se": tidy.loc[name, "Std. Error"]} for name in terms]
    return pd.DataFrame(rows).assign(outcome=outcome).sort_values(["term", "quantile"]).reset_index(drop=True)
