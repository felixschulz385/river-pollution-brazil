"""The yearly Legal Amazon panel of the garimpo designs, built in three layers.

- `geology_frame`: the Legal Amazon municipalities of `adm2_panel`, 2004-2024,
  with their own gold geology (SGB never-exploited and all gold occurrences, tin
  as the placebo mineral), C11 garimpo in their own trenches and SIVEP garimpo
  cases by infection municipality, local and incoming (lead 1's supply test, E10).
- `laundering_frame`: adds the gold deposits inside protected areas that bar
  mining (IGD, E11), gold PLG requests and grants, and IBAMA's garimpo
  enforcement counts.
- `enforcement_frame`: adds IBAMA presence before 2015, the frontier and
  Priority List measures, the groups of E12 (IGD in Roraima and elsewhere) and
  the fixed-effect keys, and the years after the panel's last from the processed
  tables (`extend_frame`).

The designs take the layer they need; `estimation_rows` selects the rows of one
spec.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.core import panels
from src.analysis.core.defaults import CLUSTER, PRICE
from src.analysis.core.estimation import interaction
from src.analysis.core.geography import RORAIMA, legal_amazon

YEARS = (2004, 2024)  # SIVEP
C11_YEARS = (2004, 2019)  # before the C11 2020 reclassification step
TREND_BASE_YEAR = 2004
LAST_YEAR = 2025  # SIVEP, IBAMA, PLG requests and the gold price; the panel's land cover stops in 2024

# --- Geology layer -----------------------------------------------------------------------

DEPOSITS = "gold_unexploited_any"  # never-exploited gold anywhere in the municipality
ALL_GOLD_EXPOSURE = "gold_any"
COUNT_EXPOSURE = "log1p_gold_unexploited"
TIN_EXPOSURE = "tin_any"
PRICE_CONTROL = interaction(DEPOSITS, PRICE)  # the gold price x deposits, a control in the policy designs

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

# --- Laundering layer --------------------------------------------------------------------

IGD = "igd"
IGD_EXPOSURES = {
    IGD: "never-exploited gold in protected areas, in the municipality",
    "igd_upstream_200": "... or within 200 km upstream",
    "igd_upstream_all": "... or anywhere upstream",
    "igd_otto": "... or in an Otto-4 sub-basin upstream",
    "igd_all_gold": "any gold occurrence in protected areas, in the municipality",
}
TIN_IGD = "tin_igd"

# --- Enforcement layer -------------------------------------------------------------------

PRESENCE = "presence"
PRESENCE_EXPOSURES = {
    PRESENCE: "IBAMA presence: log mean yearly fines 2010-2014 (SD)",
    "presence_nonmining": "... from fines other than mining",
    "presence_early": "... from fines in 2005-2010",
}
PRESENCE_YEARS = (2010, 2014)
EARLY_PRESENCE_YEARS = (2005, 2010)
FRONTIER_YEARS = (2010, 2014)
FRONTIER = ("frontier_loss", "forest_2014")  # standardised
PRIORITY_YEARS = (2015, 2018)  # Priority List status before the shift
PRIORITY = ("priority_pre", "monitored_pre")

IGD_RORAIMA = "igd_roraima"
IGD_OTHER = "igd_other_states"
GROUPS = (IGD_RORAIMA, IGD_OTHER, "rest_roraima", "rest")

# Processed tables carrying the year-varying inputs past the panel's last year,
# zero where a municipality has no record (as assembly fills them).
EXTENSION_TABLES = {
    "data/malaria/processed/malaria_adm2.parquet": ("inf_cases", "inf_garimpo", "inf_garimpo_local",
                                                    "inf_garimpo_incoming"),
    "data/enforcement/processed/enforcement_adm2.parquet": ("ibama_fines_all", "ibama_fines_mining",
                                                            "ibama_seizures_garimpo", "ibama_destructions_garimpo"),
    "data/mining/processed/mining_permits_adm2.parquet": ("plg_requests_gold",),
}
GOLD_PRICE_TABLE = "data/commodity_prices/processed/gold_price_annual.parquet"


# --- Layers ------------------------------------------------------------------------------

def geology_frame(path=None):
    """The Legal Amazon panel with the own-geology exposures and the garimpo outcomes."""
    frame = panels.load_adm2_panel(path=path or panels.ADM2_PANEL_PATH, years=YEARS)
    frame = frame[frame["mun_id"].isin(legal_amazon())].reset_index(drop=True)
    frame["state"] = frame["mun_id"].str[:2]
    frame["state_year"] = frame["state"] + "_" + frame["year"].astype(str)
    frame["trend"] = (frame["year"] - TREND_BASE_YEAR).astype(float)

    frame[DEPOSITS] = frame["gold_occurrences_unexploited"].fillna(0).gt(0).astype(float)
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


def add_laundering(frame):
    """The IGD exposures, the PLG outcomes and the garimpo enforcement counts."""
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


def laundering_frame(path=None):
    """`geology_frame` with the IGD exposures, PLG outcomes and enforcement counts."""
    return add_laundering(geology_frame(path))


def standardised_presence(frame, column, years):
    """Per municipality, log(1 + mean yearly `column` over `years`), standardised across municipalities."""
    pre = np.log1p(frame[frame["year"].between(*years)].groupby("mun_id")[column].mean())
    return frame["mun_id"].map((pre - pre.mean()) / pre.std())


def standardised(values):
    return (values - values.mean()) / values.std()


def enforcement_frame(path=None, last_year=LAST_YEAR, root_dir=panels.PROJECT_ROOT):
    """`laundering_frame` with enforcement presence, the extra outcomes, groups and FE keys, extended to
    `last_year` (`extend_frame`)."""
    frame = laundering_frame(path)
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
    pre = frame[frame["year"].between(*PRIORITY_YEARS)].groupby("mun_id")
    frame["priority_pre"] = frame["mun_id"].map(pre["priority_list"].max())
    frame["monitored_pre"] = frame["mun_id"].map(pre["priority_monitored"].max())
    listed = frame[["priority_list", "priority_monitored"]].max(axis=1).groupby(frame["mun_id"]).max()
    frame["priority_ever"] = frame["mun_id"].map(listed)

    roraima = frame["mun_id"].str[:2].eq(RORAIMA)
    igd = frame[IGD] > 0
    frame[IGD_RORAIMA] = (igd & roraima).astype(float)
    frame[IGD_OTHER] = (igd & ~roraima).astype(float)
    frame["group"] = np.select([igd & roraima, igd, roraima], [IGD_RORAIMA, IGD_OTHER, "rest_roraima"], "rest")
    frame = extend_frame(frame, last_year, root_dir)
    igd = frame[IGD] > 0

    year = frame["year"].astype(str)
    frame["state_year"] = frame["mun_id"].str[:2] + "_" + year
    frame["igd_year"] = igd.astype(int).astype(str) + "_" + year
    frame["gold_year"] = (frame[ALL_GOLD_EXPOSURE] > 0).astype(int).astype(str) + "_" + year
    return frame


def extension_rows(frame, tables, price, last_year=LAST_YEAR):
    """Rows for the years after the frame's last, through `last_year`.

    Columns constant within municipalities in the frame's last five years are
    carried from its last year; `tables` (municipality x year frames) give the
    year-varying inputs, zero where a municipality has none, and `price`
    (indexed by year) the gold price. Every other column is left out, so it is
    missing once the rows are appended.
    """
    final = int(frame["year"].max())
    recent = frame[frame["year"] > final - 5]
    constant = recent.groupby("mun_id").nunique(dropna=False).le(1).all()
    static = [column for column in constant.index[constant] if column != "year"]
    base = frame.loc[frame["year"] == final, ["mun_id", *static]]
    rows = pd.concat([base.assign(year=year) for year in range(final + 1, last_year + 1)], ignore_index=True)
    for table in tables:
        columns = [column for column in table.columns if column not in ("mun_id", "year")]
        rows = rows.drop(columns=columns, errors="ignore").merge(table, on=["mun_id", "year"], how="left")
        rows[columns] = rows[columns].fillna(0.0)
    rows[PRICE] = rows["year"].map(price)
    rows["log1p_garimpo_cases"] = np.log1p(rows["inf_garimpo"])
    rows["log1p_garimpo_local"] = np.log1p(rows["inf_garimpo_local"])
    rows["log1p_garimpo_incoming"] = np.log1p(rows["inf_garimpo_incoming"])
    rows["other_malaria"] = (rows["inf_cases"] - rows["inf_garimpo"]).clip(lower=0)
    rows["log1p_other_malaria"] = np.log1p(rows["other_malaria"])
    rows["log1p_fines_all"] = np.log1p(rows["ibama_fines_all"])
    rows["log1p_plg_requests_gold"] = np.log1p(rows["plg_requests_gold"])
    return rows


def extend_frame(frame, last_year=LAST_YEAR, root_dir=panels.PROJECT_ROOT):
    """The frame with `extension_rows` appended from the processed tables (no-op if it already reaches
    `last_year`)."""
    if frame["year"].max() >= last_year:
        return frame
    years = range(int(frame["year"].max()) + 1, last_year + 1)
    tables = []
    for path, columns in EXTENSION_TABLES.items():
        table = pd.read_parquet(Path(root_dir) / path, columns=["mun_id", "year", *columns])
        tables.append(table[table["year"].isin(years)].astype({"mun_id": str}))
    price = pd.read_parquet(Path(root_dir) / GOLD_PRICE_TABLE).set_index("year")[PRICE]
    if not set(years) <= set(price.index):
        raise ValueError(f"No gold price for {sorted(set(years) - set(price.index))}")
    return pd.concat([frame, extension_rows(frame, tables, price, last_year)], ignore_index=True)


# --- Rows of one spec --------------------------------------------------------------------

def estimation_rows(frame, outcome, years, drop_states=(), keep=None, required=()):
    """The rows of `frame` in `years`, outside `drop_states` and, if given, where the column `keep` is positive,
    with the outcome as `y`, a unit `weight`, and no missing `y`, cluster or `required` value."""
    data = frame[frame["year"].between(*years) & ~frame["mun_id"].str[:2].isin(drop_states)]
    if keep is not None:
        data = data[data[keep] > 0]
    data = data.copy()
    data["y"] = data[outcome]
    data["weight"] = 1.0
    return data.dropna(subset=["y", CLUSTER, *required]).reset_index(drop=True)


def add_price_control(data):
    """Adds the gold price x deposits control column (`PRICE_CONTROL`) and returns `data`."""
    data[PRICE_CONTROL] = data[DEPOSITS] * data[PRICE]
    return data
