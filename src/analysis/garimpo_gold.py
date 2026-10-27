"""The garimpo-gold gate: does upstream mercury-gold mining move health downstream?

Estimates behind Evidence E2 and E4 of `output/research/mining_economics_leads.md`,
run by `output/notebooks/health/03_garimpo_gold.ipynb`. Everything sits on the
health estimation frame (`health.build_health_panel`) with the MapBiomas
Collection 11 mining shares from `adm2_panel` in percentage points:

- the gold-share arm: upstream garimpo-gold share within 50 and 200 km on the
  municipalities with gold in that reach in 2000-2019, specs a-d, with the
  local (own-trench) gold share added to a and b;
- leave-one-out on Centro do Guilherme (MA), which carries the 50 km result;
- the Amazon mining-presence result of `02_analysis.ipynb` §9.3 (C10 mining
  within 50 km), re-estimated without the municipalities that ever have gold
  upstream and without the 2020 onsets, and the C11 composition of its
  switchers.

All models: PPML with the outcome's log-exposure offset, municipality and
state x year FE, the baseline land-cover and climate terms, clustered by
Otto-3 basin, and the restricted wild-cluster score bootstrap
(`inference.poisson_score_bootstrap`) with a fixed seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyfixest as pf

from . import descriptives as desc
from . import health as H
from . import panels
from .inference import poisson_score_bootstrap


GOLD_RADII = (50, H.MAX_UPSTREAM_KM)
# C11 mining groups (`land_cover.mining.MINING_GROUPS`); all but garimpo gold
# enter as one "other mining" control.
C11_GROUPS = ("garimpo_gold", "garimpo_unknown", "garimpo_other", "industrial_gold", "industrial_other")
OTHER_GROUPS = C11_GROUPS[1:]
# Before the 2020 C10/C11 mapping step; spec d extends to the full window.
GATE_LAST_YEAR = 2019
AMAZON = "Amazônia"
CENTRO_DO_GUILHERME = "210315"

FE = "mun_id + state_year"
CLUSTER = "otto_basin_l3"
VCOV = {"CRV1": CLUSTER}
BOOT_REPS = 9_999
SEED = 20261022
BASELINE_RHS = " + ".join(H.UPSTREAM_REGRESSORS + H.LOCAL_REGRESSORS + H.CLIMATE_CONTROLS)
GATE_OUTCOMES = H.MINING_OUTCOMES + H.MINING_PLACEBO_OUTCOMES
LEAVE_ONE_OUT_OUTCOMES = ["infant_deaths", "congenital", "blood", "toxic_poisoning", "injury_poisoning", "injuries"]
PRESENCE_OUTCOMES = ["infant_deaths", "injury_poisoning"]
LOCAL_GOLD = "local_gold"


def gold_column(radius):
    return f"gold_{radius}"


def gold_active_column(radius):
    return f"gold_active_{radius}"


def other_mining_column(radius):
    return f"other_mining_{radius}"


def group_column(group, radius):
    return f"mining_{group}_{radius}"


def add_c11_exposure(frame):
    """Add the C11 shares in pp (in place): per group and radius, gold, active gold, other mining, local gold.

    `adm2_panel` leaves them missing for municipalities without rings in the
    reach; those have no mining there, so 0.
    """
    for radius in GOLD_RADII:
        for group in C11_GROUPS:
            frame[group_column(group, radius)] = 100 * frame[f"c11_{group}_{radius}"].fillna(0.0)
        frame[gold_column(radius)] = frame[group_column("garimpo_gold", radius)]
        frame[gold_active_column(radius)] = 100 * frame[f"c11_garimpo_gold_active_{radius}"].fillna(0.0)
        frame[other_mining_column(radius)] = frame[[group_column(group, radius) for group in OTHER_GROUPS]].sum(axis=1)
    frame[LOCAL_GOLD] = 100 * frame["local_c11_garimpo_gold"].fillna(0.0)
    frame["amazon"] = (frame["biome"] == AMAZON).astype(float)
    return frame


def build_gate_frame():
    """The health estimation frame with the C11 exposure (`add_c11_exposure`)."""
    return add_c11_exposure(H.build_health_panel())


# --- Estimation helpers -----------------------------------------------------------

def fit(name, data, rhs=BASELINE_RHS, fe=FE):
    """PPML of outcome `name` on `rhs` with its log-exposure offset, clustered by Otto-3 basin."""
    y, offset = H.outcome_column(name), H.offset_column(name)
    sample = data.dropna(subset=[y, offset])
    return pf.fepois(f"{y} ~ {rhs} | {fe}", data=sample, offset=offset, vcov=VCOV)


def fit_all(names, **kwargs):
    """`fit` for each outcome; an outcome that fails (e.g. too sparse) is reported and skipped."""
    fits = {}
    for name in names:
        try:
            fits[name] = fit(name, **kwargs)
        except Exception as error:
            print(f"[{name}] failed: {error}")
    return fits


def bootstrap_p(model, term):
    """Restricted wild-cluster score bootstrap p-value of `term` (Otto-3 clusters)."""
    return poisson_score_bootstrap(model, term, cluster=CLUSTER, reps=BOOT_REPS, seed=SEED)["p_wcb"]


def within_sd(frame, column):
    """SD of `column` after municipality and state x year FE."""
    return float(desc.demean(frame, [column], ["mun_id", "state_year"]).std().iloc[0])


# --- Identifying variation --------------------------------------------------------

def gold_sample(frame, radius, last_year=GATE_LAST_YEAR):
    """Municipalities with garimpo gold within `radius` km upstream in any year 2000-`last_year`."""
    window = frame[frame["year"].between(H.ANALYSIS_YEARS[0], last_year)]
    return set(window.loc[window[gold_column(radius)] > 0, "mun_id"])


def gold_variation(frame):
    """Changes and within-SD of the gold share, 2000-2019, in each radius's gold sample (all and Amazon)."""
    rows = []
    for radius in GOLD_RADII:
        gold, active = gold_column(radius), gold_active_column(radius)
        sample = frame[frame["mun_id"].isin(gold_sample(frame, radius))
                       & frame["year"].between(H.ANALYSIS_YEARS[0], GATE_LAST_YEAR)]
        residuals = desc.demean(sample, [gold, active], ["mun_id", "state_year"])
        wide = sample.pivot(index="mun_id", columns="year", values=gold)
        change = wide[GATE_LAST_YEAR] - wide[H.ANALYSIS_YEARS[0]]
        is_amazon = sample.groupby("mun_id")["amazon"].first().eq(1)
        for subset, municipalities in [("all", is_amazon.index), ("Amazon", is_amazon.index[is_amazon])]:
            rows_in = sample["mun_id"].isin(municipalities)
            part = sample[rows_in]
            rows.append({
                "R": radius,
                "subset": subset,
                "municipalities": len(municipalities),
                "otto3_basins": part["otto_basin_l3"].nunique(),
                "any change 2000-19": int(part.groupby("mun_id")[gold].nunique().gt(1).sum()),
                "rise >= 0.01 pp": int((change.loc[municipalities] >= 0.01).sum()),
                "rise >= 0.1 pp": int((change.loc[municipalities] >= 0.1).sum()),
                "fall <= -0.01 pp": int((change.loc[municipalities] <= -0.01).sum()),
                "within SD gold (pp)": residuals.loc[rows_in, gold].std(),
                "within SD gold_active (pp)": residuals.loc[rows_in, active].std(),
            })
    return pd.DataFrame(rows)


def own_vs_upstream_gold(frame, path=panels.MINING_ADM2_BUCKETED_PATH):
    """Garimpo-gold pixels in each gold sample: own trenches vs upstream rings within R, 2000-2019."""
    buckets = pd.read_parquet(
        path,
        filters=[("land_cover_class", "==", 1), ("year", ">=", H.ANALYSIS_YEARS[0]), ("year", "<=", GATE_LAST_YEAR)],
        columns=["mun_id", "year", "bucket", "cnt", "bucket_intersects_adm2"],
    )
    rows = []
    for radius in GOLD_RADII:
        part = buckets[buckets["mun_id"].isin(gold_sample(frame, radius))]
        own = part["bucket_intersects_adm2"]
        upstream = part["bucket"].between(0, radius - 1) & ~own
        per_municipality = pd.DataFrame({
            "own": part["cnt"].where(own, 0.0), "upstream": part["cnt"].where(upstream, 0.0), "mun_id": part["mun_id"],
        }).groupby("mun_id").sum()
        own_pixels, upstream_pixels = part.loc[own, "cnt"].sum(), part.loc[upstream, "cnt"].sum()
        rows.append({
            "R": radius,
            "own-trench gold px": own_pixels,
            "upstream gold px (0-R km)": upstream_pixels,
            "own share": own_pixels / (own_pixels + upstream_pixels),
            "municipalities with own gold": int((per_municipality["own"] > 0).sum()),
            "with upstream gold but no own gold": int(((per_municipality["upstream"] > 0)
                                                       & (per_municipality["own"] == 0)).sum()),
        })
    return pd.DataFrame(rows)


# --- Gold-share arm ---------------------------------------------------------------

def gold_arm_specs(frame, radius, local_gold=False):
    """Samples and gold terms of specs a-d at `radius` (or a and b plus local gold).

    a: pooled 2000-2019; b: separate Amazon and other-biome slopes; c: the
    active (exposed-soil) gold share; d: pooled 2000-2024. All on the
    municipalities with gold within `radius` in 2000-2019.
    """
    gold = gold_column(radius)
    base = frame[frame["mun_id"].isin(gold_sample(frame, radius))].copy()
    for region, indicator in [("amazon", base["amazon"]), ("other", 1 - base["amazon"])]:
        base[f"{gold}__{region}"] = base[gold] * indicator
    early = base[base["year"] <= GATE_LAST_YEAR]
    if local_gold:
        return {
            "a pooled 2000-19 + local gold": (early, [gold, LOCAL_GOLD]),
            "b by region 2000-19 + local gold": (early, [f"{gold}__amazon", f"{gold}__other", LOCAL_GOLD]),
        }
    return {
        "a pooled 2000-19": (early, [gold]),
        "b by region 2000-19": (early, [f"{gold}__amazon", f"{gold}__other"]),
        "c active 2000-19": (early, [gold_active_column(radius)]),
        "d pooled 2000-24": (base, [gold]),
    }


def _term_within_sd(data, term, radius):
    """Within-SD of a gold-arm term; region terms use the residual SD of their region's rows."""
    gold = gold_column(radius)
    if term.startswith(f"{gold}__"):
        region = term.removeprefix(f"{gold}__")
        rows = data["amazon"] == (1 if region == "amazon" else 0)
        return desc.demean(data, [gold], ["mun_id", "state_year"]).loc[rows, gold].std()
    return within_sd(data, term)


def gold_arm(frame, radius, outcomes=GATE_OUTCOMES, local_gold=False, bootstrap_specs=("a", "b")):
    """Gold-share arm at `radius`: one row per spec, outcome and reported term.

    Controls: other C11 mining within `radius`, the local C10 mining terms
    and the baseline terms. Reported terms are the gold terms and other mining;
    the gold terms (and local gold) of the specs starting with a letter in
    `bootstrap_specs` get a bootstrap p-value. Estimates are per pp of
    share; `estimate_per_sd` and `pct_per_sd` scale them by the within-SD.
    """
    other = other_mining_column(radius)
    tail = " + ".join([other, *H.LOCAL_MINING_REGRESSORS]) + " + " + BASELINE_RHS
    rows = []
    for spec, (data, terms) in gold_arm_specs(frame, radius, local_gold=local_gold).items():
        fits = fit_all(outcomes, data=data, rhs=" + ".join(terms) + " + " + tail)
        for outcome, model in fits.items():
            table = model.tidy()
            for term in [*terms, other]:
                row = {
                    "R": radius, "spec": spec, "term": term, "outcome": outcome,
                    "estimate_per_pp": table.loc[term, "Estimate"], "se_per_pp": table.loc[term, "Std. Error"],
                    "within_sd_pp": _term_within_sd(data, term, radius),
                    "p_analytic": table.loc[term, "Pr(>|t|)"], "p_wcb": np.nan,
                    "n_mun": model._data["mun_id"].nunique(), "n_clusters": model._G[0], "n_obs": model._N,
                    "n_mun_term": int((model._data[term] != 0).groupby(model._data["mun_id"]).any().sum()),
                }
                if spec[0] in bootstrap_specs and term in terms:
                    row["p_wcb"] = bootstrap_p(model, term)
                rows.append(row)
    result = pd.DataFrame(rows)
    result["estimate_per_sd"] = result["estimate_per_pp"] * result["within_sd_pp"]
    result["pct_per_sd"] = 100 * np.expm1(result["estimate_per_sd"])
    return result


def leave_one_out(frame, mun_id=CENTRO_DO_GUILHERME, outcomes=LEAVE_ONE_OUT_OUTCOMES):
    """Specs a and b on 2000-2019 without municipality `mun_id`, both radii."""
    rows = []
    for radius in GOLD_RADII:
        gold, other = gold_column(radius), other_mining_column(radius)
        base = frame[frame["mun_id"].isin(gold_sample(frame, radius) - {mun_id})
                     & (frame["year"] <= GATE_LAST_YEAR)].copy()
        base[f"{gold}__amazon"] = base[gold] * base["amazon"]
        base[f"{gold}__other"] = base[gold] * (1 - base["amazon"])
        tail = " + ".join([other, *H.LOCAL_MINING_REGRESSORS]) + " + " + BASELINE_RHS
        for spec, terms, focus in [("a pooled", [gold], gold),
                                   ("b by region", [f"{gold}__amazon", f"{gold}__other"], f"{gold}__amazon")]:
            for outcome, model in fit_all(outcomes, data=base, rhs=" + ".join(terms) + " + " + tail).items():
                table = model.tidy()
                rows.append({
                    "R": radius, "spec": spec, "term": focus, "outcome": outcome,
                    "estimate_per_pp": table.loc[focus, "Estimate"], "se": table.loc[focus, "Std. Error"],
                    "p_analytic": table.loc[focus, "Pr(>|t|)"], "p_wcb": bootstrap_p(model, focus),
                    "n_mun": model._data["mun_id"].nunique(), "n_clusters": model._G[0],
                })
    return pd.DataFrame(rows)


# --- The Amazon mining-presence result ------------------------------------------

PRESENCE_TERM = f"{H.UPSTREAM_MINING_PRESENT}__amazon"


def presence_frame(frame):
    """The mining-arm sample with Amazon/other presence slopes, and its Amazon upstream switchers.

    As `02_analysis.ipynb` §9.3: municipalities with C10 mining within 50 km
    upstream in any year; upstream and local presence split by region.
    """
    mining = frame[frame[H.MINING_MUNICIPALITY]].copy()
    presence = mining.groupby("mun_id")[H.UPSTREAM_MINING_PRESENT].agg(["min", "max"])
    switchers = presence.index[(presence["min"] == 0) & (presence["max"] == 1)]
    amazon = (mining["biome"] == AMAZON).astype(float)
    for term in [H.UPSTREAM_MINING_PRESENT, H.LOCAL_MINING_PRESENT]:
        mining[f"{term}__amazon"] = mining[term] * amazon
        mining[f"{term}__other"] = mining[term] * (1 - amazon)
    amazon_switchers = switchers.intersection(mining.loc[amazon == 1, "mun_id"].unique())
    return mining, amazon_switchers


def presence_rhs():
    region_terms = [f"{term}__{region}" for term in [H.UPSTREAM_MINING_PRESENT, H.LOCAL_MINING_PRESENT]
                    for region in ("amazon", "other")]
    return " + ".join([*region_terms, H.UPSTREAM_MINING_PP, H.LOCAL_MINING_PP, BASELINE_RHS])


def switcher_onsets(mining, amazon_switchers):
    """First year with upstream mining present, per Amazon switcher."""
    exposed = mining[mining["mun_id"].isin(amazon_switchers) & (mining[H.UPSTREAM_MINING_PRESENT] == 1)]
    return exposed.groupby("mun_id")["year"].min()


def presence_samples(frame):
    """Samples a-c (drop municipalities ever with gold upstream) and e-h (the 2020 mapping step)."""
    mining, amazon_switchers = presence_frame(frame)
    onset = switcher_onsets(mining, amazon_switchers)
    ever_gold = {radius: set(frame.loc[frame[gold_column(radius)] > 0, "mun_id"]) for radius in GOLD_RADII}
    samples = {
        "a original": mining,
        "b drop gold_50 > 0": mining[~mining["mun_id"].isin(ever_gold[50])],
        f"c drop gold_{GOLD_RADII[1]} > 0": mining[~mining["mun_id"].isin(ever_gold[GOLD_RADII[1]])],
        "e drop switchers with onset 2020": mining[~mining["mun_id"].isin(set(onset.index[onset == 2020]))],
        "f drop switchers with onset >= 2020": mining[~mining["mun_id"].isin(set(onset.index[onset >= 2020]))],
        "g years 2000-2019": mining[mining["year"] <= GATE_LAST_YEAR],
        "h drop 2020-21": mining[~mining["year"].isin(H.COVID_YEARS)],
    }
    return samples, amazon_switchers


def presence_spec(frame, outcomes=PRESENCE_OUTCOMES):
    """The Amazon upstream-presence coefficient on each of `presence_samples`."""
    samples, amazon_switchers = presence_samples(frame)
    rhs = presence_rhs()
    rows = []
    for label, data in samples.items():
        in_sample = amazon_switchers.intersection(data["mun_id"].unique())
        exposed = amazon_switchers.intersection(data.loc[data[PRESENCE_TERM] > 0, "mun_id"].unique())
        for outcome, model in fit_all(outcomes, data=data, rhs=rhs).items():
            table = model.tidy()
            rows.append({
                "sample": label, "outcome": outcome,
                "estimate": table.loc[PRESENCE_TERM, "Estimate"], "se": table.loc[PRESENCE_TERM, "Std. Error"],
                "p_analytic": table.loc[PRESENCE_TERM, "Pr(>|t|)"], "p_wcb": bootstrap_p(model, PRESENCE_TERM),
                "n_mun": model._data["mun_id"].nunique(),
                "n_amazon_mun": data.loc[data["amazon"] == 1, "mun_id"].nunique(),
                "n_amazon_switchers": len(in_sample),
                "n_amazon_switchers_exposed": len(exposed),
                "n_amazon_switcher_basins": data.loc[data["mun_id"].isin(in_sample), "otto_basin_l3"].nunique(),
                "n_clusters": model._G[0], "n_obs": model._N,
            })
    return pd.DataFrame(rows)


def switcher_composition(frame):
    """C11 shares within 50 km of each Amazon presence switcher one year after its onset (pp)."""
    mining, amazon_switchers = presence_frame(frame)
    onset = switcher_onsets(mining, amazon_switchers)
    near, far = GOLD_RADII
    groups_near = [group_column(group, near) for group in C11_GROUPS]
    groups_far = [group_column(group, far) for group in C11_GROUPS]
    by_municipality = frame.set_index(["mun_id", "year"]).sort_index()
    last_year = int(frame["year"].max())
    rows = []
    for mun_id, year in onset.items():
        at = by_municipality.loc[(mun_id, min(year + 1, last_year))]
        ever = by_municipality.loc[mun_id]
        shares = at[groups_near].astype(float)
        total = shares.sum()
        rows.append({
            "mun_id": mun_id, "onset": year, "year_used": min(year + 1, last_year), "basin": at["otto_basin_l3"],
            "c10_upstream_pp_50": at[H.UPSTREAM_MINING_PP],
            **{group: shares[group_column(group, near)] for group in C11_GROUPS},
            "c11_total_50": total,
            "largest_c11_group": shares.idxmax().removeprefix("mining_").removesuffix(f"_{near}") if total > 0 else "none",
            "c11_any_year_50": bool((ever[groups_near].sum(axis=1) > 0).any()),
            f"c11_at_year_{far}": float(at[groups_far].sum()),
            f"gold_any_year_{far}": bool((ever[gold_column(far)] > 0).any()),
        })
    return pd.DataFrame(rows).sort_values("c11_total_50", ascending=False).reset_index(drop=True)
