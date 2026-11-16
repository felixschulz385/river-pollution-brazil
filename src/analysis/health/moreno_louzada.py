"""Moreno-Louzada's design: gold deposits upstream x the gold price, and infant health in the Legal Amazon.

Moreno-Louzada (thesis, lucamlouzada.github.io/files/goldmining.pdf) interacts a
municipality's exposure, any gold deposit in an Otto level-4 sub-basin upstream of
one the municipality intersects (same level-3 basin), with the log international
gold price, and estimates its effect on birth outcomes in the Legal Amazon without
the municipalities that hold a deposit themselves, 2007-2021, with municipality and
year FE, weighted by mean births and clustered by Otto-3 basin. His headline is
+2.28 congenital anomalies per 1,000 births per log point (SE 1.16).

Here deposits are SGB gold occurrences (`geology_adm2_upstream`), the price is the
Pink Sheet gold price in BRL (USD as a robustness check), and every estimate
carries four inferences: Otto-3 clustered SEs, a restricted wild-cluster bootstrap
on Otto-3 basins, two-way clustering by basin and year, and the price-exposure
(P-E) variance of Moreno-Louzada, Figueira and Picchetti (arXiv 2512.10076), which
treats the price path as the source of randomness. The baseline adds randomization
inference over simulated random-walk price paths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.core import panels
from src.analysis.core.defaults import CLUSTER, FE, PRICE, PRICE_USD, STATE_YEAR_FE
from src.analysis.core.estimation import estimate, interaction, run_table as _run_table, term_row
from src.analysis.core.geography import legal_amazon, upstream_municipalities
from src.analysis.core.inference import price_exposure_se, random_walk_p

YEARS = (2007, 2021)
EXTENDED_YEARS = (2004, 2023)
FIRST_STAGE_C11_YEARS = (2007, 2019)  # before the C11 2020 reclassification step
DROPPED_YEARS = (2015, 2016, 2017)
PRETERM_FIRST_YEAR = 2012  # SINASC's gestation coding changes in 2011

CAPITALS = {"130260": "Manaus", "150140": "Belém"}

EXPOSURE = "gold_upstream_otto4"
DOWNSTREAM_EXPOSURE = "gold_downstream_otto4"
TIN_EXPOSURE = "tin_upstream_otto4"
UNEXPLOITED_EXPOSURE = "gold_unexploited_upstream_otto4"
OWN_DEPOSITS = "gold_occurrences"

MORENO_LOUZADA_ANOMALY = (2.28, 1.16)  # his estimate and SE, anomalies per 1,000 births

# Outcome -> (numerator, denominator, scale, first year). Rates per 1,000
# births, except the injury placebo per 1,000 population. TabNet has no mean
# birthweight, so birthweight enters as the low-birth-weight rate.
OUTCOMES = {
    "anomaly": ("births_anomaly", "births", 1000.0, None),
    "low_birth_weight": ("births_low_weight", "births_weight_known", 1000.0, None),
    "preterm": ("births_preterm", "births_gestation_known", 1000.0, PRETERM_FIRST_YEAR),
    "apgar5_below_6": ("births_apgar5_below_6", "births", 1000.0, None),
    "infant_deaths_nonexternal": ("infant_deaths_nonexternal", "births", 1000.0, None),
    "infant_deaths_congenital": ("infant_deaths_congenital", "births", 1000.0, None),
    "congenital_hospitalizations": ("hosp_icd_17", "births", 1000.0, None),
    "injury_hospitalizations": ("hosp_icd_19", "population", 1000.0, None),
}
PLACEBO_OUTCOMES = ("injury_hospitalizations",)


# --- Sample --------------------------------------------------------------------------

def add_upstream_cases(frame, upstream, column="inf_garimpo", name="upstream_garimpo_cases"):
    """Garimpo cases (by infection municipality) summed over each municipality's upstream municipalities."""
    cases = frame.set_index(["mun_id", "year"])[column]
    pairs = pd.DataFrame([(target, source) for target, sources in upstream.items() for source in sources],
                         columns=["mun_id", "source"])
    sums = (pairs.merge(cases.rename_axis(["source", "year"]).reset_index(), on="source")
            .groupby(["mun_id", "year"])[column].sum().rename(name))
    frame = frame.merge(sums, on=["mun_id", "year"], how="left")
    frame[name] = frame[name].fillna(0.0).where(frame[column].notna())
    return frame


def build_frame(years=EXTENDED_YEARS, path=panels.ADM2_PANEL_PATH):
    """The Legal Amazon panel with the exposures, rates, price interactions and keys used below.

    `sample` marks the municipalities without a gold occurrence of their own.
    """
    frame = panels.load_adm2_panel(path=path, years=years)
    frame = frame[frame["mun_id"].isin(legal_amazon())].reset_index(drop=True)
    frame["sample"] = frame[OWN_DEPOSITS].fillna(0).eq(0)
    frame["state"] = frame["mun_id"].str[:2]
    frame["state_year"] = frame["state"] + "_" + frame["year"].astype(str)
    for count in ("gold_unexploited_upstream_200", "gold_unexploited_upstream_all"):
        frame[f"log1p_{count}"] = np.log1p(frame[count])
    for name, (numerator, denominator, scale, _) in OUTCOMES.items():
        frame[name] = scale * frame[numerator] / frame[denominator].where(frame[denominator] > 0)
    for radius in (50, 200):
        frame[f"c11_gold_pp_{radius}"] = 100 * frame[f"c11_garimpo_gold_{radius}"].fillna(0.0)
    frame["log1p_local_garimpo_cases"] = np.log1p(frame["inf_garimpo"])
    upstream = upstream_municipalities(sorted(frame["mun_id"].unique()))
    frame = add_upstream_cases(frame, upstream)
    frame["log1p_upstream_garimpo_cases"] = np.log1p(frame["upstream_garimpo_cases"])
    return frame


# --- Estimation ----------------------------------------------------------------------

def spec_data(frame, outcome, exposures=(EXPOSURE,), price=PRICE, years=YEARS, drop_years=(), drop_mun_ids=()):
    """Estimation rows of one spec: the no-own-deposit sample, `y`, the interactions and the births weight.

    `outcome` is any column of `frame`; those in `OUTCOMES` start in their
    first year. The weight is each municipality's mean births over the rows kept.
    """
    first_year = OUTCOMES.get(outcome, (None, None, None, None))[3]
    first = max(years[0], first_year or years[0])
    data = frame[
        frame["sample"]
        & frame["year"].between(first, years[1])
        & ~frame["year"].isin(drop_years)
        & ~frame["mun_id"].isin(drop_mun_ids)
    ].copy()
    data["y"] = data[outcome]
    for exposure in exposures:
        data[interaction(exposure, price)] = data[exposure] * data[price]
    data = data.dropna(subset=["y", CLUSTER, "births", price, *exposures])
    data["weight"] = data.groupby("mun_id")["births"].transform("mean")
    return data[data["weight"] > 0].reset_index(drop=True)


def infer(model, data, term, exposure, price, fe=FE, weighted=True, randomization=False):
    """Estimate and the four inferences for one term of a fitted spec."""
    row = {"term": term, **term_row(model, term)}
    two_way = estimate(data, list(model._coefnames), fe=fe, weighted=weighted,
                       vcov={"CRV1": f"{CLUSTER}+year"}).tidy().loc[term]
    row["se_two_way"], row["p_two_way"] = two_way["Std. Error"], two_way["Pr(>|t|)"]
    se_pe, n_years = price_exposure_se(model, term, exposure, price, weighted=weighted)
    row["se_pe"] = se_pe
    row["p_pe"] = 2 * stats.norm.sf(abs(row["estimate"]) / se_pe)
    if randomization:
        _, row["p_random_walk"] = random_walk_p(data, exposure, price, fe=fe, weighted=weighted)
    row.update(
        n_obs=int(model._N),
        n_municipalities=int(data["mun_id"].nunique()),
        n_exposed=int(data.loc[data[exposure] > 0, "mun_id"].nunique()),
        n_basins=int(data[CLUSTER].nunique()),
        n_years=n_years,
        mean_y=float(np.average(data["y"], weights=data["weight"] if weighted else None)),
    )
    return row


# --- Specifications ------------------------------------------------------------------

def robustness_specs():
    """Spec name -> keyword arguments of `run_spec`, baseline first."""
    return {
        "baseline": {},
        "state x year FE": {"fe": STATE_YEAR_FE},
        "drop 2015-17": {"drop_years": DROPPED_YEARS},
        "unweighted": {"weighted": False},
        "without Manaus and Belém": {"drop_mun_ids": tuple(CAPITALS)},
        "2004-2023": {"years": EXTENDED_YEARS},
        "USD price": {"price": PRICE_USD},
        "never-exploited gold, Otto-4": {"exposures": (UNEXPLOITED_EXPOSURE,)},
        "never-exploited gold within 200 km, log(1+count)": {"exposures": ("log1p_gold_unexploited_upstream_200",)},
        "never-exploited gold upstream, log(1+count)": {"exposures": ("log1p_gold_unexploited_upstream_all",)},
        "placebo: + downstream gold": {"exposures": (EXPOSURE, DOWNSTREAM_EXPOSURE)},
        "placebo: + upstream tin": {"exposures": (EXPOSURE, TIN_EXPOSURE)},
    }


def run_spec(frame, outcome, exposures=(EXPOSURE,), price=PRICE, fe=FE, weighted=True, years=YEARS,
             drop_years=(), drop_mun_ids=(), randomization=False):
    """Fit one spec and return one row per exposure term."""
    data = spec_data(frame, outcome, exposures, price, years, drop_years, drop_mun_ids)
    terms = [interaction(exposure, price) for exposure in exposures]
    model = estimate(data, terms, fe=fe, weighted=weighted)
    return [
        {"outcome": outcome, "exposure": exposure,
         **infer(model, data, term, exposure, price, fe=fe, weighted=weighted,
                 randomization=randomization and len(terms) == 1)}
        for exposure, term in zip(exposures, terms)
    ]


def run_table(frame, outcomes, specs, randomization_specs=("baseline",)):
    """Every spec for every outcome; an outcome that fails is reported and skipped."""
    specs = {name: {**kwargs, "randomization": name in randomization_specs} for name, kwargs in specs.items()}
    return _run_table(run_spec, frame, outcomes, specs, skip_failures=True)


def first_stage(frame):
    """The interaction on upstream C11 garimpo gold (pp, 2007-2019) and on garimpo malaria cases (2007-2021)."""
    outcomes = {
        "C11 garimpo gold within 50 km upstream (pp)": ("c11_gold_pp_50", FIRST_STAGE_C11_YEARS),
        "C11 garimpo gold within 200 km upstream (pp)": ("c11_gold_pp_200", FIRST_STAGE_C11_YEARS),
        "log(1 + garimpo cases upstream, Otto-4)": ("log1p_upstream_garimpo_cases", YEARS),
        "log(1 + garimpo cases in the municipality)": ("log1p_local_garimpo_cases", YEARS),
    }
    rows = []
    for label, (column, years) in outcomes.items():
        data = spec_data(frame, column, years=years)
        model = estimate(data, [interaction(EXPOSURE, PRICE)])
        rows.append({"outcome": label, "years": f"{years[0]}-{years[1]}",
                     **infer(model, data, interaction(EXPOSURE, PRICE), EXPOSURE, PRICE)})
    return pd.DataFrame(rows)


def sample_table(frame):
    """Municipality counts: the Legal Amazon, without own deposits, and exposed."""
    municipalities = frame.drop_duplicates("mun_id")
    sample = municipalities[municipalities["sample"]]
    return pd.Series({
        "Legal Amazon municipalities in the panel": len(municipalities),
        "with a gold occurrence of their own (dropped)": int((~municipalities["sample"]).sum()),
        "sample": len(sample),
        "exposed: gold upstream (Otto-4)": int(sample[EXPOSURE].sum()),
        "gold downstream (Otto-4)": int(sample[DOWNSTREAM_EXPOSURE].sum()),
        "tin upstream (Otto-4)": int(sample[TIN_EXPOSURE].sum()),
        "never-exploited gold upstream (Otto-4)": int(sample[UNEXPLOITED_EXPOSURE].sum()),
        "Otto-3 basins in the sample": int(sample[CLUSTER].nunique()),
    })
