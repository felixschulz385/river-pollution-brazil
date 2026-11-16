"""Raids and displacement (lead 3): garimpo malaria around IBAMA's garimpo raids, by month.

IBAMA's garimpo destruction and seizure terms are dated to the day and carry the
municipality where they took place (`enforcement`). SIVEP gives garimpo cases
by infection municipality and notification month (`malaria`, with the
preliminary months from 2025). Does garimpo fall where IBAMA raids, and does it
move next door?

- *Episodes.* A municipality's first month with at least `MIN_TERMS` garimpo
  destruction or seizure terms after `CLEAN_MONTHS` months without any, from
  2019 until 12 months before SIVEP's last month (so the 12 months after it
  are observed).
- *Stacked event study.* Each episode is a stack of months -12 to +12 around the
  raid month. The treated unit is the raided municipality (`own`) or each
  bordering municipality not itself raided in the window (`neighbours`). The
  controls are the sample municipalities neither raided in the window nor
  bordering the raided one. Episode x municipality and episode x month FE; SEs
  clustered by municipality, which appears in many stacks.
- *Deterrence or displacement next door.* Neighbours where crews expect the
  next raid can lose garimpo, others gain it. The neighbour design is split by
  whether a municipality had a garimpo case in the 12 months before the raid,
  for the treated neighbours and the controls alike.
- *Sample.* Legal Amazon municipalities with a garimpo case in 2016-2024 or a
  garimpo raid. Episode samples: all, large raids (`LARGE_TERMS` terms in the
  raid month), and malaria-visible raids (`MIN_PRE_CASES` garimpo cases in the
  12 months before).
- *Outcomes.* log(1 + garimpo cases), local and incoming, and non-garimpo
  malaria as the placebo. Raids follow local malaria peaks, so garimpo is also
  measured against non-garimpo malaria in the same municipality-month (a triple
  difference). Garimpo cases in levels give the displacement ratio (cases
  gained by all neighbours per case lost at the raid).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.core.estimation import estimate, stacked_event_study
from src.analysis.core.geography import borders, legal_amazon, municipality_boundaries
from src.analysis.core.panels import PROJECT_ROOT
from src.data.sources.enforcement.constants import DEFAULT_RECORDS_OUTPUT_PATH
from src.data.sources.malaria.constants import DEFAULT_MONTHLY_OUTPUT_PATH

MONTHS = ("2018-01", "2024-12")  # defaults; `load_inputs` runs to SIVEP's last month
EPISODE_MONTHS = ("2019-01", "2023-12")
SAMPLE_YEARS = (2016, 2025)
WINDOW = (-12, 12)
BASE = -1
CLEAN_MONTHS = 12
MIN_TERMS = 1
LARGE_TERMS = 3
# SIVEP sees garimpo only where malaria is transmitted: "visible" episodes had
# at least this many garimpo cases in the 12 months before the raid.
MIN_PRE_CASES = 12
RAID_ACTIONS = ("destructions", "seizures")
DESIGNS = ("own", "neighbours")
# Municipalities split by garimpo cases in the 12 months before the raid.
GROUPS = ("all", "with garimpo", "without garimpo")

OUTCOMES = {
    "log1p_garimpo": "log(1 + garimpo cases)",
    "log1p_garimpo_local": "log(1 + local garimpo cases)",
    "log1p_garimpo_incoming": "log(1 + incoming garimpo cases)",
    "log1p_other_malaria": "log(1 + other malaria cases), placebo",
    "garimpo_vs_other": "log(1 + garimpo) - log(1 + other malaria)",
    "incoming_vs_other": "log(1 + incoming garimpo) - log(1 + other malaria)",
    "inf_garimpo": "garimpo cases",
}


def month_index(months):
    """Months as consecutive integers (year x 12 + month)."""
    months = pd.to_datetime(months)
    return months.dt.year * 12 + months.dt.month - 1


def raid_months(records, actions=RAID_ACTIONS, months=MONTHS):
    """Garimpo raid terms per municipality and month."""
    raids = records[records["garimpo"] & records["action"].isin(actions) & records["mun_id"].notna()
                    & records["date"].notna()]
    raids = raids.assign(month=pd.to_datetime(raids["date"]).dt.to_period("M").dt.to_timestamp())
    raids = raids[raids["month"].between(*map(pd.Timestamp, months))]
    return raids.groupby(["mun_id", "month"]).size().rename("raid_terms").reset_index()


def monthly_panel(cases, municipalities, months=MONTHS):
    """Every municipality x month with the case outcomes; months without a case are 0."""
    grid = pd.MultiIndex.from_product(
        [sorted(municipalities), pd.date_range(*months, freq="MS")], names=["mun_id", "month"])
    counts = ["inf_cases", "inf_garimpo", "inf_garimpo_local", "inf_garimpo_incoming"]
    panel = cases.set_index(["mun_id", "month"])[counts].reindex(grid, fill_value=0).reset_index()
    panel["log1p_garimpo"] = np.log1p(panel["inf_garimpo"])
    panel["log1p_garimpo_local"] = np.log1p(panel["inf_garimpo_local"])
    panel["log1p_garimpo_incoming"] = np.log1p(panel["inf_garimpo_incoming"])
    panel["log1p_other_malaria"] = np.log1p(panel["inf_cases"] - panel["inf_garimpo"])
    panel["garimpo_vs_other"] = panel["log1p_garimpo"] - panel["log1p_other_malaria"]
    panel["incoming_vs_other"] = panel["log1p_garimpo_incoming"] - panel["log1p_other_malaria"]
    panel["t"] = month_index(panel["month"])
    return panel


def episodes(raids, min_terms=MIN_TERMS, clean=CLEAN_MONTHS, months=EPISODE_MONTHS):
    """Each municipality's raid months with `min_terms` terms after `clean` months without any raid term."""
    raids = raids.assign(t=month_index(raids["month"])).sort_values(["mun_id", "t"])
    previous = raids.groupby("mun_id")["t"].shift()
    start = raids[(raids["t"] - previous.fillna(-10**6) > clean) & (raids["raid_terms"] >= min_terms)]
    start = start[start["month"].between(*map(pd.Timestamp, months))]
    return start[["mun_id", "month", "t", "raid_terms"]].reset_index(drop=True)


def pre_cases(episode_table, panel, months=12):
    """Garimpo cases in each episode's municipality in the `months` months before the raid month."""
    merged = episode_table[["mun_id", "t"]].reset_index().merge(panel[["mun_id", "t", "inf_garimpo"]], on="mun_id",
                                                               suffixes=("", "_panel"))
    before = merged[merged["t_panel"].between(merged["t"] - months, merged["t"] - 1)]
    return before.groupby("index")["inf_garimpo"].sum().reindex(episode_table.index, fill_value=0)


def episode_samples(episode_table, panel):
    """Sample name -> the episodes in it."""
    return {
        "all raids": episode_table,
        f"large raids ({LARGE_TERMS}+ terms)": episode_table[episode_table["raid_terms"] >= LARGE_TERMS],
        f"malaria-visible ({MIN_PRE_CASES}+ cases before)": episode_table[
            pre_cases(episode_table, panel) >= MIN_PRE_CASES],
    }


def episode_window(panel, after=WINDOW[1]):
    """Episode months: from `EPISODE_MONTHS[0]` until `after` months before the panel's last month."""
    last = panel["month"].max() - pd.DateOffset(months=after)
    return EPISODE_MONTHS[0], last.strftime("%Y-%m")


def stack(panel, episode_table, raids, adjacency, design="own", window=WINDOW, group="all"):
    """The stacked panel of one design: treated and control municipalities over each episode's window.

    With `group` "with garimpo" or "without garimpo", treated and control municipalities are restricted to those
    with or without a garimpo case in the 12 months before the raid."""
    raided = raids.assign(t=month_index(raids["month"]))
    sample = set(panel["mun_id"])
    frames = []
    for number, episode in enumerate(episode_table.itertuples()):
        low, high = episode.t + window[0], episode.t + window[1]
        raided_now = set(raided.loc[raided["t"].between(low, high), "mun_id"])
        nearby = adjacency.get(episode.mun_id, set())
        if design == "own":
            treated = {episode.mun_id}
        else:
            treated = (nearby & sample) - raided_now
        controls = sample - raided_now - nearby - {episode.mun_id}
        if group != "all":
            before = panel[panel["t"].between(episode.t - 12, episode.t - 1) & panel["inf_garimpo"].gt(0)]
            active = set(before["mun_id"])
            keep = active if group == "with garimpo" else sample - active
            treated, controls = treated & keep, controls & keep
        if not treated or not controls:
            continue
        rows = panel[panel["t"].between(low, high) & panel["mun_id"].isin(treated | controls)].copy()
        rows["episode"] = number
        rows["treated"] = rows["mun_id"].isin(treated).astype(float)
        rows["rel"] = rows["t"] - episode.t
        frames.append(rows)
    stacked = pd.concat(frames, ignore_index=True)
    stacked["unit"] = stacked["episode"].astype(str) + "_" + stacked["mun_id"]
    stacked["episode_month"] = stacked["episode"].astype(str) + "_" + stacked["t"].astype(str)
    return stacked


STACKED_FE = "unit + episode_month"
STACKED_CLUSTER = "mun_id"  # a municipality appears in many stacks


def _fit(data, terms, outcome):
    return estimate(data, terms, fe=STACKED_FE, weighted=False, vcov={"CRV1": STACKED_CLUSTER}, outcome=outcome)


def event_study(stacked, outcome, base=BASE):
    """Treated x month relative to the raid (base month omitted)."""
    return stacked_event_study(stacked, outcome, base, STACKED_FE, STACKED_CLUSTER)


def pooled(stacked, outcome, base_window=(-12, -1), post_window=(0, 12)):
    """Treated x post (months `post_window`) against the months before, with the treated pre-months as base."""
    data = stacked[stacked["rel"].between(base_window[0], post_window[1])].copy()
    data["treated_post"] = data["treated"] * data["rel"].between(*post_window)
    model = _fit(data, ["treated_post"], outcome)
    tidy = model.tidy().loc["treated_post"]
    return {"outcome": outcome, "estimate": tidy["Estimate"], "se": tidy["Std. Error"], "p": tidy["Pr(>|t|)"],
            "n_obs": int(model._N), "n_episodes": int(data["episode"].nunique()),
            "n_treated_units": int(data.loc[data["treated"] > 0, "unit"].nunique())}


def displacement_ratio(own, neighbours):
    """Cases gained by all neighbours of a raid per case lost at the raid, from the pooled level estimates.

    `own` and `neighbours` are `pooled` rows for `inf_garimpo`; the neighbours' estimate is per neighbour, so it
    is scaled by the mean number of treated neighbours per episode. Missing when the raid loses no cases."""
    if own["estimate"] >= 0:
        return np.nan
    per_episode = neighbours["n_treated_units"] / neighbours["n_episodes"]
    return -neighbours["estimate"] * per_episode / own["estimate"]


def load_inputs(root_dir=PROJECT_ROOT, sample_years=SAMPLE_YEARS):
    """The monthly case panel of the sample municipalities (from `MONTHS[0]` to SIVEP's last month), the raid
    months and the border map."""
    root = Path(root_dir)
    cases = pd.read_parquet(root / DEFAULT_MONTHLY_OUTPUT_PATH)
    months = (MONTHS[0], cases["month"].max().strftime("%Y-%m"))
    records = pd.read_parquet(root / DEFAULT_RECORDS_OUTPUT_PATH,
                              columns=["action", "date", "mun_id", "garimpo", "operation_code"])
    raids = raid_months(records, months=months)
    amazon = legal_amazon()
    in_years = cases["month"].dt.year.between(*sample_years)
    garimpo = set(cases.loc[in_years & cases["inf_garimpo"].gt(0), "mun_id"])
    sample = (garimpo | set(raids["mun_id"])) & amazon
    boundaries = municipality_boundaries(root)
    adjacency = borders(boundaries[boundaries["mun_id"].isin(amazon)])
    return monthly_panel(cases[cases["mun_id"].isin(sample)], sample, months), raids[raids["mun_id"].isin(amazon)], \
        adjacency
