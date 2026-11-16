"""Model setup for the health (ADM2-panel) analysis.

Implements the decisions recorded in §9 of
`output/notebooks/health/01_descriptives.ipynb`: the 2000-2024 window, the
municipality sample, outcome counts with their exposure offsets, upstream land
cover recomputed with a capped reach next to the local block, local climate,
and the fixed-effect keys. `build_health_panel` applies all of them and returns
the estimation frame used by `output/notebooks/health/02_analysis.ipynb`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.core import panels
from src.analysis.pollution import model as pollution


# --- Window -------------------------------------------------------------------

# Population (the exposure of most outcomes) starts in 2000; land cover ends in 2024.
ANALYSIS_YEARS = (2000, 2024)
# SINASC changed how gestational age is recorded in 2011 (preterm share 7% in
# 2010, 12.5% in 2012), so preterm is used from 2012 only.
PRETERM_FIRST_YEAR = 2012
# Hospitalizations dip and mortality peaks in 2020-21; dropped as robustness.
COVID_YEARS = (2020, 2021)
SIH_FIRST_CURRENT_YEAR = panels.SIH_LAST_LEGACY_YEAR + 1

# --- Outcomes -------------------------------------------------------------------
# outcome -> (count column, exposure column). PPML uses log(exposure) as offset.

PRIMARY_OUTCOMES = {
    "infectious_parasitic": ("hosp_icd_01", "population"),
    "digestive": ("hosp_icd_11", "population"),
    "gastrointestinal": ("hosp_water_sanitation_gastrointestinal", "population"),
    "infant_deaths": ("deaths_under_1", "births"),
    "low_birth_weight": ("births_low_weight", "births_weight_known"),
}
SECONDARY_OUTCOMES = {
    "genitourinary": ("hosp_icd_14", "population"),
    "perinatal": ("hosp_icd_16", "population"),
    "congenital": ("hosp_icd_17", "population"),
    "renal_urinary": ("hosp_renal_urinary_toxic_water", "population"),
    "skin_contact": ("hosp_skin_contact", "population"),
    "leptospirosis": ("hosp_leptospirosis_water_exposure", "population"),
    "toxic_poisoning": ("hosp_toxic_poisoning", "population"),
    "vector_borne": ("hosp_vector_borne_ecological", "population"),
    "under5_deaths": ("deaths_under_5", "births"),
    "preterm": ("births_preterm", "births_gestation_known"),
}
# External causes and mental health should not respond to upstream water quality.
PLACEBO_OUTCOMES = {
    "injury_poisoning": ("hosp_icd_19", "population"),
    "injuries": ("hosp_injuries_accidents_occupational", "population"),
    "mental_health": ("hosp_mental_health_substance_use", "population"),
}
# Chapters estimated only in the mining arm: mercury is neurotoxic (ICD 06),
# lead causes anaemia (ICD 03, blood and blood-forming organs).
METAL_OUTCOMES = {
    "nervous_system": ("hosp_icd_06", "population"),
    "blood": ("hosp_icd_03", "population"),
}
OUTCOMES = {**PRIMARY_OUTCOMES, **SECONDARY_OUTCOMES, **PLACEBO_OUTCOMES, **METAL_OUTCOMES}
HOSPITALIZATION_OUTCOMES = [name for name, (count, _) in OUTCOMES.items() if count.startswith("hosp_")]

# --- Exposure -------------------------------------------------------------------

# Same reach, kernel, bands and land-cover classes as the pollution arm.
MAX_UPSTREAM_KM = pollution.MAX_UPSTREAM_KM
DISTANCE_BANDS = pollution.DISTANCE_BANDS
UPSTREAM_REGRESSORS = list(pollution.LAND_COVER_REGRESSORS)
LOCAL_REGRESSORS = [f"local_{regressor}" for regressor in UPSTREAM_REGRESSORS]

# --- Mining arm -----------------------------------------------------------------
# As in the pollution arm, mining is a local, point-like source measured within a
# short upstream reach, and the arm is estimated on municipalities with any
# mining in that reach over 2000-2024 (1,190; 216 switch presence). Local mining
# enters alongside, so the upstream terms are net of the municipality's own
# mining economy. Outcomes are those where a metal signature is expected
# (mercury, lead, cadmium); the injury placebos pick up economic spillovers.
MINING_REACH_KM = pollution.MINING_REACH_KM
UPSTREAM_MINING_PRESENT = f"upstream_mining_present_{MINING_REACH_KM}"
UPSTREAM_MINING_PP = f"upstream_mining_pp_{MINING_REACH_KM}"
LOCAL_MINING_PRESENT = "local_mining_present"
LOCAL_MINING_PP = "local_mining_pp"
MINING_MUNICIPALITY = "mining_municipality"
UPSTREAM_MINING_REGRESSORS = [UPSTREAM_MINING_PRESENT, UPSTREAM_MINING_PP]
LOCAL_MINING_REGRESSORS = [LOCAL_MINING_PRESENT, LOCAL_MINING_PP]
MINING_OUTCOMES = [
    "nervous_system", "blood", "renal_urinary", "toxic_poisoning", "congenital", "perinatal",
    "low_birth_weight", "preterm", "infant_deaths",
]
MINING_PLACEBO_OUTCOMES = ["injury_poisoning", "injuries"]

# --- Controls -------------------------------------------------------------------

# Local annual climate; the upstream block correlates 0.78-0.94 with it within
# municipality and enters only as robustness (renamed: "2t" is not formula-safe).
CLIMATE_CONTROLS = ["local_2t", "local_tp", "local_swvl1"]
UPSTREAM_CLIMATE_CONTROLS = {"2t": "upstream_2t", "tp": "upstream_tp", "swvl1": "upstream_swvl1"}
# Utilization control for hospitalization outcomes (robustness only: it contains them).
UTILIZATION_CONTROL = "log_hospitalization_rate"

# Biome heterogeneity: Pampa (160 municipalities) and Pantanal (9) are pooled.
POOLED_BIOMES = {"Pampa": "Pampa/Pantanal", "Pantanal": "Pampa/Pantanal"}


def outcome_column(name):
    return f"y_{name}"


def offset_column(name):
    return f"log_{OUTCOMES[name][1]}"


def _log_positive(values):
    values = values.astype(float)
    return np.log(values.where(values > 0))


def upstream_alr(min_km=0, max_km=MAX_UPSTREAM_KM, suffix=""):
    """Municipality upstream ALR terms recomputed from the rings in [min_km, max_km)."""
    land_cover = panels.adm2_upstream_land_cover(min_km, max_km, years=ANALYSIS_YEARS)
    land_cover = land_cover[["mun_id", "year", *UPSTREAM_REGRESSORS]]
    return land_cover.rename(columns={column: f"{column}{suffix}" for column in UPSTREAM_REGRESSORS})


def add_upstream_exposure(frame, distance_bands=False):
    """Replace the assembled upstream ALR terms with the capped ones; add band terms if asked.

    Municipalities without upstream rings (headwater) or without rings in a band
    get 0: their reach is fixed over time, so the 0 is absorbed by the
    municipality FE and the coefficient is identified from municipalities that
    have the rings.
    """
    frame = frame.drop(columns=panels.LAND_COVER_ALR_COLUMNS)
    frame = frame.merge(upstream_alr(), on=["mun_id", "year"], how="left", validate="one_to_one")
    frame[UPSTREAM_REGRESSORS] = frame[UPSTREAM_REGRESSORS].fillna(0.0)
    if distance_bands:
        for band, (lower, upper) in DISTANCE_BANDS.items():
            banded = upstream_alr(lower, upper, suffix=f"__{band}")
            frame = frame.merge(banded, on=["mun_id", "year"], how="left", validate="one_to_one")
            band_columns = [f"{column}__{band}" for column in UPSTREAM_REGRESSORS]
            frame[band_columns] = frame[band_columns].fillna(0.0)
    return frame


def add_outcomes(frame):
    """Add `y_<outcome>` counts and the `log_<exposure>` offsets (in place).

    Non-positive exposures give a missing offset, which drops the row from that
    outcome's sample (e.g. municipality-years without births).
    """
    frame["deaths_under_5"] = frame["deaths_under_1"] + frame["deaths_1_to_4"]
    for name, (count, _) in OUTCOMES.items():
        frame[outcome_column(name)] = frame[count].astype(float)
    frame.loc[frame["year"] < PRETERM_FIRST_YEAR, outcome_column("preterm")] = np.nan
    for exposure in {exposure for _, exposure in OUTCOMES.values()}:
        frame[f"log_{exposure}"] = _log_positive(frame[exposure])
    return frame


def add_mining_exposure(frame, max_km=MINING_REACH_KM):
    """Add upstream (within `max_km`) and local mining presence and share (pp), and the mining-municipality flag.

    Municipalities without rings within `max_km` have no upstream mining (0),
    unlike the ALR terms, where a missing ring set has no natural value.
    `mining_municipality` marks municipalities with upstream mining in any year.
    """
    near = panels.adm2_upstream_land_cover(max_km=max_km, years=ANALYSIS_YEARS)[["mun_id", "year", "lc_mining"]]
    frame = frame.merge(near.rename(columns={"lc_mining": "_lc_mining_near"}),
                        on=["mun_id", "year"], how="left", validate="one_to_one")
    frame[UPSTREAM_MINING_PP] = 100 * frame.pop("_lc_mining_near").fillna(0.0)
    frame[UPSTREAM_MINING_PRESENT] = (frame[UPSTREAM_MINING_PP] > 0).astype(float)
    frame[LOCAL_MINING_PP] = 100 * frame["local_lc_mining"]
    frame[LOCAL_MINING_PRESENT] = (frame[LOCAL_MINING_PP] > 0).astype(float).where(frame[LOCAL_MINING_PP].notna())
    frame[MINING_MUNICIPALITY] = frame.groupby("mun_id")[UPSTREAM_MINING_PRESENT].transform("max").eq(1)
    return frame


def build_health_panel(distance_bands=False):
    """Load `adm2_panel` and apply the §9 model-setup decisions.

    Returns the estimation frame: `y_<outcome>` counts with `log_<exposure>`
    offsets, upstream ALR capped at `MAX_UPSTREAM_KM` (`alr_*`) next to the
    local block (`local_alr_*`), local and (renamed, 0 for headwater
    municipalities) upstream climate, the
    utilization control, the mining-arm exposure (`add_mining_exposure`), and
    FE / cluster keys. With `distance_bands=True`,
    also adds `alr_*__<band>` regressors per band. Municipality-years without
    population or local climate are dropped; headwater municipalities stay in.
    """
    frame = panels.load_adm2_panel(years=ANALYSIS_YEARS)
    frame = frame.dropna(subset=["population", *CLIMATE_CONTROLS]).reset_index(drop=True)
    frame["population"] = frame["population"].astype(float)
    frame = add_upstream_exposure(frame, distance_bands=distance_bands)
    frame = add_mining_exposure(frame)
    frame = frame.rename(columns=UPSTREAM_CLIMATE_CONTROLS)
    # Headwater municipalities have no upstream climate either; 0 is absorbed by the FE.
    upstream_climate = list(UPSTREAM_CLIMATE_CONTROLS.values())
    frame.loc[~frame["has_upstream"], upstream_climate] = 0.0
    add_outcomes(frame)
    frame[UTILIZATION_CONTROL] = _log_positive(frame["hosp_hospitalizations_count"]) - frame["log_population"]

    frame["state"] = frame["mun_id"].str[:2]
    frame["state_year"] = frame["state"] + "_" + frame["year"].astype(str)
    frame["biome_year"] = frame["biome"] + "_" + frame["year"].astype(str)
    frame["biome_group"] = frame["biome"].replace(POOLED_BIOMES)
    return frame
