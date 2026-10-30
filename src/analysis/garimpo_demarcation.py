"""Demarcation and garimpo inside Indigenous lands (lead 4): a first pass on the stages.

Does pending protection attract garimpo, and does completing it push garimpo out?
FUNAI dates each land's declaration (by the Justice Ministry) and homologation (by
presidential decree) (`protected_areas`). C11 garimpo inside a land is measured
from the river trenches: each trench's garimpo pixels count in proportion to the
share of its drainage polygon inside the land (`protected_areas --product
trenches`), assuming garimpo is spread evenly within a trench.

- *Lands.* Indigenous lands whose centroid lies in the Legal Amazon, at any
  stage. Samples: all lands, and lands with an SGB gold occurrence inside.
- *Outcomes.* The garimpo share of the land (pp, C11 groups garimpo gold,
  unknown substance and other), whether it has any garimpo, and log(1 + garimpo
  hectares), 1985-2019 (C11 steps in 2020).
- *Stacked event studies* around each land's declaration and homologation year:
  for each event year from 1990 to 2012, the lands with the event that year
  against the lands without it in the window (years -5 to +7). Episode x land
  and episode x year FE; SEs clustered by land.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyfixest as pf

from src.analysis import moreno_louzada as M
from src.data.sources.biomes.preprocess import _load_adm2_boundaries
from src.data.sources.geology.constants import gpkg_path as geology_gpkg_path
from src.data.sources.geology.preprocess import load_gold_occurrences
from src.data.sources.land_cover.constants import derive_mun_id_from_adm2_id
from src.data.sources.protected_areas.constants import DEFAULT_TRENCHES_OUTPUT_PATH, INDIGENOUS
from src.data.sources.protected_areas.preprocess import load_protected_areas

YEARS = (1985, 2019)
WINDOW = (-5, 7)
BASE = -1
COHORTS = (1990, 2012)
EVENTS = {"declaration": "year_declared", "homologation": "year_homologated"}
MINING_PATH = "data/land_cover/processed/extract/land_cover_mining.parquet"
GARIMPO_CLASSES = ("land_cover_class_1", "land_cover_class_2", "land_cover_class_3")
PIXEL_HA = 0.09  # 30 m pixels
GOLD = "gold_inside"

OUTCOMES = {
    "garimpo_pp": "garimpo share of the land (pp)",
    "any_garimpo": "any garimpo in the land",
    "log1p_garimpo_ha": "log(1 + garimpo hectares)",
}


def land_garimpo(overlaps, mining):
    """Land pixels and garimpo pixels per land and year, apportioned from the trenches by their overlap share."""
    merged = overlaps.merge(mining, on="trench_id")
    garimpo = merged[list(GARIMPO_CLASSES)].sum(axis=1) * merged["trench_share"]
    total = merged["land_cover_total"] * merged["trench_share"]
    return (pd.DataFrame({"area_id": merged["area_id"], "year": merged["year"], "land_pixels": total,
                          "garimpo_pixels": garimpo})
            .groupby(["area_id", "year"], as_index=False).sum())


def land_table(lands, boundaries, gold):
    """Per land: its dates, current stage, centroid municipality and state, and gold occurrences inside."""
    lands = lands[lands["kind"] == INDIGENOUS].copy()
    centroids = gpd.GeoDataFrame(lands[["area_id"]], geometry=lands.geometry.representative_point(), crs=lands.crs)
    located = gpd.sjoin(centroids, boundaries[["mun_id", "geometry"]].to_crs(lands.crs), predicate="within")
    lands["mun_id"] = lands["area_id"].map(located.drop_duplicates("area_id").set_index("area_id")["mun_id"])
    inside = gpd.sjoin(gold[["geometry"]].to_crs(lands.crs), lands[["area_id", "geometry"]], predicate="within")
    lands[GOLD] = lands["area_id"].map(inside.groupby("area_id").size()).fillna(0).astype(int)
    lands["year_declared"] = pd.to_datetime(lands["date_declared"]).dt.year
    lands["year_homologated"] = pd.to_datetime(lands["date_homologated"]).dt.year
    return pd.DataFrame(lands[["area_id", "name", "stage", "mun_id", "year_declared", "year_homologated", GOLD]])


def build_panel(land_years, lands, years=YEARS, municipalities=None):
    """The land-year panel with the outcomes, restricted to lands in `municipalities` (all if None)."""
    panel = land_years[land_years["year"].between(*years)].merge(lands, on="area_id")
    if municipalities is not None:
        panel = panel[panel["mun_id"].isin(municipalities)]
    panel = panel[panel["land_pixels"] > 0].copy()
    panel["garimpo_pp"] = 100 * panel["garimpo_pixels"] / panel["land_pixels"]
    panel["any_garimpo"] = (panel["garimpo_pixels"] >= 1).astype(float)
    panel["log1p_garimpo_ha"] = np.log1p(panel["garimpo_pixels"] * PIXEL_HA)
    panel["state"] = panel["mun_id"].str[:2]
    return panel.reset_index(drop=True)


def load_panel(root_dir=M.PROJECT_ROOT, years=YEARS):
    """The Legal Amazon land-year panel from the processed areas, trench overlaps, C11 trench table and geology."""
    root = Path(root_dir)
    areas = load_protected_areas(root)
    overlaps = pd.read_parquet(root / DEFAULT_TRENCHES_OUTPUT_PATH)
    overlaps = overlaps[overlaps["area_id"].str.startswith("ti:")]
    mining = pd.read_parquet(root / MINING_PATH, columns=["trench_id", "year", "land_cover_total", *GARIMPO_CLASSES],
                             filters=[("trench_id", "in", sorted(set(overlaps["trench_id"])))])
    boundaries = _load_adm2_boundaries(root)
    boundaries["mun_id"] = boundaries["adm2_id"].map(derive_mun_id_from_adm2_id)
    lands = land_table(areas, boundaries, load_gold_occurrences(geology_gpkg_path(root)))
    return build_panel(land_garimpo(overlaps, mining), lands, years, set(M.legal_amazon()))


def stack(panel, event, window=WINDOW, cohorts=COHORTS, sample=None):
    """Lands with `event` in each cohort year against lands without it in the window, years `window` around it."""
    column = EVENTS[event]
    if sample is not None:
        panel = panel[panel[sample] > 0]
    event_year = panel.groupby("area_id")[column].first()
    frames = []
    for cohort in range(cohorts[0], cohorts[1] + 1):
        low, high = cohort + window[0], cohort + window[1]
        treated = set(event_year.index[event_year == cohort])
        controls = set(event_year.index[~event_year.between(low, high)])
        if not treated:
            continue
        rows = panel[panel["year"].between(low, high) & panel["area_id"].isin(treated | controls)].copy()
        rows["episode"] = cohort
        rows["treated"] = rows["area_id"].isin(treated).astype(float)
        rows["rel"] = rows["year"] - cohort
        frames.append(rows)
    stacked = pd.concat(frames, ignore_index=True)
    stacked["unit"] = stacked["episode"].astype(str) + "_" + stacked["area_id"]
    stacked["episode_year"] = stacked["episode"].astype(str) + "_" + stacked["year"].astype(str)
    return stacked


def _fit(data, terms, outcome):
    return pf.feols(f"{outcome} ~ {' + '.join(terms)} | unit + episode_year", data=data,
                    vcov={"CRV1": "area_id"})


def event_study(stacked, outcome, base=BASE):
    """Treated x year relative to the event (base year omitted)."""
    data = stacked.copy()
    terms = []
    for rel in sorted(data["rel"].unique()):
        if rel != base:
            name = f"rel_{'m' if rel < 0 else 'p'}{abs(rel)}"
            data[name] = data["treated"] * data["rel"].eq(rel)
            terms.append((name, rel))
    tidy = _fit(data, [name for name, _ in terms], outcome).tidy()
    rows = [{"rel": base, "estimate": 0.0, "se": 0.0}]
    rows += [{"rel": rel, "estimate": tidy.loc[name, "Estimate"], "se": tidy.loc[name, "Std. Error"]}
             for name, rel in terms]
    return pd.DataFrame(rows).sort_values("rel").assign(outcome=outcome).reset_index(drop=True)


def pooled(stacked, outcome):
    """Treated x years 0 to +7 against years -5 to -1."""
    data = stacked.copy()
    data["treated_post"] = data["treated"] * data["rel"].ge(0)
    model = _fit(data, ["treated_post"], outcome)
    tidy = model.tidy().loc["treated_post"]
    return {"outcome": outcome, "estimate": tidy["Estimate"], "se": tidy["Std. Error"], "p": tidy["Pr(>|t|)"],
            "n_obs": int(model._N), "n_cohorts": int(data["episode"].nunique()),
            "n_treated_lands": int(data.loc[data["treated"] > 0, "area_id"].nunique())}


def stage_table(panel, start=2010, end=YEARS[1]):
    """By current stage: lands, garimpo share and lands with garimpo in `start` and `end`, and the change."""
    wide = panel[panel["year"].isin([start, end])].pivot_table(index=["area_id", "stage"], columns="year",
                                                               values="garimpo_pixels").reset_index()
    wide["change_ha"] = (wide[end] - wide[start]) * PIXEL_HA
    return wide.groupby("stage").agg(
        lands=("area_id", "count"),
        **{f"garimpo_ha_{start}": (start, lambda d: d.sum() * PIXEL_HA),
           f"garimpo_ha_{end}": (end, lambda d: d.sum() * PIXEL_HA)},
        lands_with_garimpo=(end, lambda d: int((d >= 1).sum())),
        change_ha=("change_ha", "sum"),
    )
