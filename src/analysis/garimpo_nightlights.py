"""Night lights as a yearly garimpo activity measure: a measurement check, then the 2019 and 2023 dates.

C11 stops in 2019 and SIVEP counts infections, not work, so the price and policy
designs lack a measure of where garimpo is worked in a given year. VIIRS
(`nightlights`, 2012 onward) adds light per municipality, within 5 km of SGB gold
occurrences, inside the barred areas in force by 2012, and both.

- *Measurement check.* Within a municipality, does light near gold move with
  garimpo cases and C11 garimpo area? Legal Amazon municipalities with gold
  occurrences, Otto-3 clusters. Non-garimpo malaria is the comparison.
- *Composite shocks.* The annual composites jump in some years, most in dim
  municipalities (2013, 2017, and 2023 with the move to VNL v2.2, which lowers
  the background threshold). Garimpo municipalities are dim, so year FE are not
  enough: every light spec also runs with brightness class x year FE, the class
  being the quintile of radiance per pixel in 2012-2014 across the Legal Amazon.
- *Dates.* E12's designs on light: IGD in Roraima and elsewhere x year (base
  2018), IGD x post-2019 and x post-2023 on 2012-2024, and IBAMA presence x
  post-2019 and x post-2023 on 2015-2024. Each zone's outcome is estimated on
  the municipalities that have pixels in it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis import garimpo_laundering as L
from src.analysis import garimpo_shift as G
from src.analysis import moreno_louzada as M

YEARS = (2012, 2024)
C11_YEARS = (2012, 2019)
BRIGHTNESS_YEARS = (2012, 2014)
BRIGHTNESS_CLASSES = 5
FE, CLUSTER = M.FE, M.CLUSTER
BRIGHTNESS_FE = "mun_id + brightness_year"
FIXED_EFFECTS = {"year FE": FE, "brightness x year FE": BRIGHTNESS_FE}

ZONES = {
    "": "whole municipality",
    "_gold5km": "within 5 km of gold",
    "_protected": "barred areas in force by 2012",
    "_gold5km_protected": "within 5 km of gold, in barred areas",
}
LIGHT_OUTCOMES = {
    **{f"log1p_radiance{zone}": f"log(1 + radiance), {label}" for zone, label in ZONES.items()},
    **{f"log1p_lit{zone}": f"log(1 + lit pixels), {label}" for zone, label in ZONES.items()},
}
MEASURES = {
    "log1p_garimpo_cases": ("log(1 + garimpo cases)", YEARS),
    "log1p_garimpo_local": ("log(1 + local garimpo cases)", YEARS),
    "log1p_garimpo_incoming": ("log(1 + incoming garimpo cases)", YEARS),
    "c11_garimpo_pp": ("C11 garimpo, own trenches (pp)", C11_YEARS),
    "log1p_other_malaria": ("log(1 + other malaria cases), placebo", YEARS),
}
DESIGNS = {"IGD": {"exposure": L.IGD, "years": YEARS}, "presence": {"exposure": G.PRESENCE, "years": G.YEARS}}


def has_pixels(zone):
    return f"has_pixels{zone}"


def zone_of(outcome):
    """The zone suffix of a light outcome."""
    return next(zone for zone in sorted(ZONES, key=len, reverse=True) if outcome.endswith(zone))


def build_frame(path=None):
    """E12's Legal Amazon panel (IGD, presence, groups) with the light outcomes and brightness classes."""
    return add_lights(G.build_frame(path))


def add_lights(frame):
    """Log light outcomes, which zones each municipality has pixels in, and brightness classes and their FE key."""
    frame = frame.copy()
    for zone in ZONES:
        frame[f"log1p_radiance{zone}"] = np.log1p(frame[f"viirs_radiance{zone}"])
        frame[f"log1p_lit{zone}"] = np.log1p(frame[f"viirs_lit_pixels{zone}"])
        frame[has_pixels(zone)] = frame.groupby("mun_id")[f"viirs_pixels{zone}"].transform("max").gt(0).astype(float)

    pre = frame[frame["year"].between(*BRIGHTNESS_YEARS)].groupby("mun_id")[["viirs_radiance", "viirs_pixels"]].sum()
    density = pre["viirs_radiance"] / pre["viirs_pixels"]
    classes = pd.qcut(density.rank(method="first"), BRIGHTNESS_CLASSES, labels=False)
    frame["brightness"] = frame["mun_id"].map(classes)
    frame["brightness_year"] = frame["brightness"].astype("Int64").astype(str) + "_" + frame["year"].astype(str)
    return frame


def tracking(frame, light, measure, fe=FE, years=None, sample=has_pixels("_gold5km")):
    """Within-municipality association of `light` with `measure`, Otto-3 SEs."""
    years = years or MEASURES[measure][1]
    data = frame[frame["year"].between(*years) & frame[sample].gt(0)].copy()
    data["y"] = data[light]
    data["weight"] = 1.0
    data = data.dropna(subset=["y", measure, CLUSTER, "brightness"]).reset_index(drop=True)
    tidy = M.estimate(data, [measure], fe=fe, weighted=False).tidy().loc[measure]
    return {"light": light, "measure": measure, "estimate": tidy["Estimate"], "se_basin": tidy["Std. Error"],
            "p_basin": tidy["Pr(>|t|)"], "n_obs": len(data), "n_municipalities": data["mun_id"].nunique()}


def tracking_table(frame, lights=("log1p_radiance_gold5km", "log1p_lit_gold5km", "log1p_radiance", "log1p_lit"),
                   measures=tuple(MEASURES)):
    return pd.DataFrame([
        {"fe": name, **tracking(frame, light, measure, fe=fe)}
        for name, fe in FIXED_EFFECTS.items() for light in lights for measure in measures
    ])


def group_means(frame, column, years=YEARS, sample=has_pixels("_gold5km")):
    """Year x group means of `column` (IGD in Roraima and elsewhere, the rest of Roraima, the rest)."""
    data = frame[frame["year"].between(*years) & frame[sample].gt(0)]
    table = data.pivot_table(index="year", columns="group", values=column, aggfunc="mean")
    return table.reindex(columns=list(G.GROUPS))


def brightness_changes(frame, column="log1p_radiance", years=YEARS):
    """Year x brightness class mean change in `column` from the year before."""
    data = frame[frame["year"].between(years[0] - 1, years[1])].sort_values(["mun_id", "year"])
    change = data.groupby("mun_id")[column].diff()
    return change.groupby([data["year"], data["brightness"]]).mean().unstack().loc[years[0]:]


def run_table(frame, outcomes=tuple(LIGHT_OUTCOMES), bootstrap=True):
    """Each design x fixed effects x light outcome, on the municipalities with pixels in the outcome's zone."""
    rows = []
    for design, kwargs in DESIGNS.items():
        for name, fe in FIXED_EFFECTS.items():
            for outcome in outcomes:
                rows += [{"design": design, "fe": name, **row} for row in G.run_spec(
                    frame, outcome, fe=fe, bootstrap=bootstrap, sample=has_pixels(zone_of(outcome)), **kwargs)]
    return pd.DataFrame(rows)
