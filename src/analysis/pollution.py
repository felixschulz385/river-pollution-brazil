"""Model setup for the pollution (sensor-panel) analysis.

Implements the decisions recorded in §11 of
`output/notebooks/pollution/01_descriptives.ipynb`: sample rules, outcome cleaning,
the combined fecal-indicator series, CONAMA 357 exceedance outcomes, and land cover
recomputed with a capped upstream reach. `build_pollution_panel` applies all of them
and returns the estimation frame used by `output/notebooks/pollution/02_analysis.ipynb`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import panels


# --- Sample -------------------------------------------------------------------

# A station identifies a land-cover effect under station FE only with enough
# samples spread over enough years for upstream land cover to have changed.
MIN_SAMPLES_PER_STATION = 8
MIN_YEARS_PER_STATION = 3

# --- Outcomes -----------------------------------------------------------------

FECAL_OUTCOME = "fecal_indicator"
CONTINUOUS_OUTCOMES = [
    "turbidity",
    "total_suspended_solids",
    "dissolved_oxygen",
    "biochemical_oxygen_demand",
    "chemical_oxygen_demand",
    "total_phosphorus",
    "ammoniacal_nitrogen",
    "nitrates",
    FECAL_OUTCOME,
    "electrical_conductivity",
    "ph",
]
# Near-symmetric in levels (pH is already a log scale); everything else is logged.
LEVEL_OUTCOMES = {"ph", "dissolved_oxygen"}

# Physically implausible values are set to missing before anything else.
# Bounds are exclusive at the lower end: pH = 0 and DO = 0 are coding artefacts.
PLAUSIBLE_RANGES = {
    "ph": (0.0, 14.0),
    "dissolved_oxygen": (0.0, 20.0),
}
# Continuous outcomes are trimmed above this per-analyte quantile.
TRIM_QUANTILE = 0.999

# --- CONAMA 357/2005 class-2 freshwater limits --------------------------------
# Art. 15 (class 2) with the class-1 values of Art. 14 that it inherits; lotic
# limits where the resolution distinguishes lotic/lentic environments. Units are
# the panel's (mg/L unless noted). Kinds: "max" (violation if above), "min"
# (violation if below), "range" (violation if outside).
#
# Excluded on purpose: aluminum and copper (the limit applies to the dissolved
# fraction; the panel reports total), and analytes without a class-2 limit
# (TSS, COD, conductivity).
CONAMA_CLASS2_LIMITS = {
    "biochemical_oxygen_demand": ("max", 5.0),
    "dissolved_oxygen": ("min", 5.0),
    "turbidity": ("max", 100.0),  # NTU
    "ph": ("range", (6.0, 9.0)),
    "total_phosphorus": ("max", 0.1),
    "nitrates": ("max", 10.0),  # mg/L N
    "nitrites": ("max", 1.0),  # mg/L N
    "total_dissolved_solids": ("max", 500.0),
    "chlorophyll": ("max", 30.0),  # ug/L
    "color": ("max", 75.0),  # mg Pt/L
    "cadmium": ("max", 0.001),
    "lead": ("max", 0.01),
    "mercury": ("max", 0.0002),
    "zinc": ("max", 0.18),
    "nickel": ("max", 0.025),
    "total_chromium": ("max", 0.05),
    "arsenic": ("max", 0.01),
    "manganese": ("max", 0.1),
    "dissolved_iron": ("max", 0.3),
    "barium": ("max", 0.7),
}
# Ammoniacal nitrogen's limit depends on the sample's pH: (upper pH bound, limit mg/L N).
CONAMA_AMMONIACAL_NITROGEN_BY_PH = [(7.5, 3.7), (8.0, 2.0), (8.5, 1.0), (np.inf, 0.5)]
# Fecal indicator: 1,000 thermotolerant coliforms/100 mL (CONAMA 357 class 2), and
# the CONAMA 274/2000 equivalence of 800 E. coli/100 mL for samples reporting E. coli.
CONAMA_FECAL_LIMITS = {"thermotolerant": 1000.0, "e_coli": 800.0}

# A heap of at least this share at a station-period's lowest positive value marks
# a lab reporting limit (non-detects reported at the limit).
REPORTING_LIMIT_MIN_SHARE = 0.2
REPORTING_LIMIT_PERIOD_YEARS = 5

# Trace metals (limits in the ug/L range) are sometimes entered in ug/L in the
# mg/L column. A value at least this many times the limit, in a station-period
# whose lowest positive value is at or below the limit, is taken to be such an
# entry and set to missing.
UNIT_ERROR_FACTOR = 100
UNIT_ERROR_MAX_LIMIT = 0.05
UNIT_ERROR_ANALYTES = [
    column for column, (kind, limit) in CONAMA_CLASS2_LIMITS.items()
    if kind == "max" and limit <= UNIT_ERROR_MAX_LIMIT
]

# --- Regressors and controls --------------------------------------------------

# Mining is estimated separately (only ~2.6% of stations switch mining
# presence), so it is not among the main regressors; see the mining arm below.
LAND_COVER_REGRESSORS = [
    "alr_pasture",
    "alr_agriculture",
    "alr_farming_unclassified",
    "alr_urban",
    "alr_other",
]
RUNOFF_INTERACTED = ["alr_pasture", "alr_agriculture"]

# Panel climate columns -> formula-safe names (a leading digit is not a valid
# pyfixest identifier).
CLIMATE_CONTROLS = {
    "tp_mean_day": "tp_1",
    "tp_mean_7d": "tp_7",
    "sro_mean_7d": "sro_7",
    "swvl1_mean_30d": "swvl1_30",
    "swvl2_mean_90d": "swvl2_90",
    "ssro_mean_180d": "ssro_180",
    "tp_mean_365d": "tp_365",
    "pev_mean_365d": "pev_365",
    "2t_mean_365d": "t2m_365",
    "2t_mean_7d": "t2m_7",
}
RUNOFF_CONTROL = "sro_7"

# Upstream reach of the main land-cover exposure, and distance bands for the
# robustness specification (lower bound inclusive, upper exclusive, km). The
# 500 km+ catch-all ring is excluded from both.
MAX_UPSTREAM_KM = 200
DISTANCE_BANDS = {"0_50": (0, 50), "50_150": (50, 150), "150_500": (150, 500)}

# --- Mining arm -----------------------------------------------------------------
# Mining is a local, point-like source, so its exposure is measured within a
# short upstream reach. The arm is estimated on stations with any mining in
# that reach over their sampled years; presence is identified from the ~160
# stations where mining appears (or disappears), intensity from changes in the
# mined share among stations with mining.
MINING_REACH_KM = 50
MINING_PRESENT = f"mining_present_{MINING_REACH_KM}"
MINING_SHARE_PP = f"mining_pp_{MINING_REACH_KM}"
MINING_STATION = "mining_station"
MINING_REGRESSORS = [MINING_PRESENT, MINING_SHARE_PP]
MINING_CONTINUOUS_OUTCOMES = ["turbidity", "total_suspended_solids", "electrical_conductivity", "ph", "dissolved_oxygen"]
MINING_EXCEEDANCE_OUTCOMES = [
    "turbidity", "cadmium", "lead", "mercury", "zinc", "nickel",
    "total_chromium", "arsenic", "manganese", "dissolved_iron",
]

def upstream_land_cover(min_km=0, max_km=MAX_UPSTREAM_KM, years=panels.EXPOSURE_YEARS):
    """Recompute sensor land-cover composition from the rings in [min_km, max_km) (see `panels.ring_land_cover`)."""
    return panels.ring_land_cover(panels.LAND_COVER_SENSOR_BUCKETED_PATH, "station_code", min_km, max_km, years)


def add_near_mining(frame, max_km=MINING_REACH_KM):
    """Add near-upstream mining presence, share (pp) and the mining-station flag to `frame`.

    Mining is recomputed within `max_km` upstream with the same kernel as the
    main exposure. `mining_station` marks stations with mining in that reach
    in any of their sampled years.
    """
    near = upstream_land_cover(max_km=max_km)[["station_code", "year", "lc_mining"]]
    frame = frame.merge(near.rename(columns={"lc_mining": "_lc_mining_near"}),
                        on=["station_code", "year"], how="left", validate="many_to_one")
    frame[MINING_SHARE_PP] = 100 * frame.pop("_lc_mining_near")
    frame[MINING_PRESENT] = (frame[MINING_SHARE_PP] > 0).astype(float).where(frame[MINING_SHARE_PP].notna())
    frame[MINING_STATION] = frame.groupby("station_code")[MINING_PRESENT].transform("max").eq(1)
    return frame


def combine_fecal_indicator(frame):
    """Add one fecal-indicator series and its type to `frame` (in place).

    `fecal_coliforms` and `thermotolerant_coliforms` are the same measurement
    under two names (identical where both are reported) and are coalesced as
    "thermotolerant"; samples reporting only E. coli use it as "e_coli". The
    type enters the FE (station x type), so the series' different levels do not
    need rescaling.
    """
    thermotolerant = frame["fecal_coliforms"].combine_first(frame["thermotolerant_coliforms"])
    frame[FECAL_OUTCOME] = thermotolerant.combine_first(frame["escherichia"])
    frame["fecal_type"] = np.select(
        [thermotolerant.notna(), frame["escherichia"].notna()],
        ["thermotolerant", "e_coli"],
        default=None,
    )
    return frame


def apply_plausibility(frame):
    """Set physically implausible outcome values to missing (in place)."""
    for column, (lower, upper) in PLAUSIBLE_RANGES.items():
        values = frame[column]
        frame[column] = values.where((values > lower) & (values <= upper))
    return frame


def drop_unit_errors(frame):
    """Set suspected ug/L entries of trace metals to missing (in place). Expects a `period` column.

    Within a station-period whose lowest positive value is at or below the CONAMA
    limit, values at least `UNIT_ERROR_FACTOR` times the limit are implausible as
    readings from the same lab and are treated as unit errors.
    """
    keys = [frame["station_code"], frame["period"]]
    for column in UNIT_ERROR_ANALYTES:
        limit = CONAMA_CLASS2_LIMITS[column][1]
        values = frame[column]
        lowest = values.where(values > 0).groupby(keys).transform("min")
        frame[column] = values.mask((lowest <= limit) & (values >= UNIT_ERROR_FACTOR * limit))
    return frame


def transformed_outcome(frame, column):
    """Return the estimation-scale outcome: trimmed above `TRIM_QUANTILE`, then logged unless in levels."""
    values = frame[column]
    values = values.where(values <= values.quantile(TRIM_QUANTILE))
    if column in LEVEL_OUTCOMES:
        return values
    return np.log(values.where(values > 0))


def _reporting_limit_periods(frame, column, threshold):
    """Station-periods whose reporting limit is at or above `threshold` (uninformative for exceedance)."""
    observed = frame.loc[frame[column] > 0, ["station_code", "period", column]]
    keys = [observed["station_code"], observed["period"]]
    lowest = observed.groupby(keys)[column].transform("min")
    at_lowest_share = (observed[column] == lowest).groupby(keys).transform("mean")
    reporting_limit = lowest.where(at_lowest_share >= REPORTING_LIMIT_MIN_SHARE)
    by_period = reporting_limit.groupby(keys).max()
    return by_period.index[by_period >= threshold]


def conama_exceedances(frame):
    """Return 0/1 CONAMA class-2 violation indicators, one `exceed_<analyte>` column each.

    For upper limits, station-periods whose lab reporting limit is at or above
    the CONAMA limit cannot distinguish compliance from violation and are set to
    missing. Expects `period`, `fecal_type` and (for ammoniacal N) `ph` columns.
    """
    exceed = pd.DataFrame(index=frame.index)
    period_key = pd.MultiIndex.from_frame(frame[["station_code", "period"]])

    for column, (kind, limit) in CONAMA_CLASS2_LIMITS.items():
        values = frame[column]
        if kind == "max":
            violation = values > limit
            uninformative = period_key.isin(_reporting_limit_periods(frame, column, limit))
        elif kind == "min":
            violation = values < limit
            uninformative = np.zeros(len(frame), dtype=bool)
        else:
            lower, upper = limit
            violation = (values < lower) | (values > upper)
            uninformative = np.zeros(len(frame), dtype=bool)
        exceed[f"exceed_{column}"] = violation.astype(float).where(values.notna() & ~uninformative)

    upper_ph, ammonia_limits = zip(*CONAMA_AMMONIACAL_NITROGEN_BY_PH)
    ph_bin = np.searchsorted(np.asarray(upper_ph), frame["ph"].to_numpy(), side="left")
    ammonia_limit = pd.Series(
        np.where(frame["ph"].notna(), np.asarray(ammonia_limits)[np.minimum(ph_bin, len(upper_ph) - 1)], np.nan),
        index=frame.index,
    )
    ammonia = frame["ammoniacal_nitrogen"]
    exceed["exceed_ammoniacal_nitrogen"] = (ammonia > ammonia_limit).astype(float).where(
        ammonia.notna() & ammonia_limit.notna()
    )

    fecal_limit = frame["fecal_type"].map(CONAMA_FECAL_LIMITS)
    fecal = frame[FECAL_OUTCOME]
    exceed[f"exceed_{FECAL_OUTCOME}"] = (fecal > fecal_limit).astype(float).where(fecal.notna())
    return exceed


def identifying_stations(frame, column):
    """Boolean mask of rows whose station has enough non-missing `column` samples to identify under station FE."""
    observed = frame.loc[frame[column].notna(), ["station_code", "year"]]
    per_station = observed.groupby("station_code").agg(n=("year", "size"), n_years=("year", "nunique"))
    keep = per_station.index[
        (per_station["n"] >= MIN_SAMPLES_PER_STATION) & (per_station["n_years"] >= MIN_YEARS_PER_STATION)
    ]
    return frame["station_code"].isin(keep) & frame[column].notna()


def build_pollution_panel(max_km=MAX_UPSTREAM_KM, distance_bands=False):
    """Load the sensor panel and apply the §11 model-setup decisions.

    Returns the estimation frame: `y_<outcome>` (transformed continuous
    outcomes), `exceed_<analyte>` (CONAMA indicators), land cover recomputed
    within `max_km` (`lc_*`/`alr_*`), near-upstream mining exposure for the
    mining arm, formula-safe climate controls, and FE keys.
    With `distance_bands=True`, also adds `alr_*__<band>` regressors per band.
    Rows without land cover (stations not snapped to the river network) are dropped.
    """
    frame = panels.load_sensor_panel()
    assembled_land_cover = panels.LAND_COVER_SHARE_COLUMNS + panels.LAND_COVER_ALR_COLUMNS + ["lc_nat"]
    frame = frame.drop(columns=assembled_land_cover).merge(
        upstream_land_cover(max_km=max_km), on=["station_code", "year"], how="inner", validate="many_to_one"
    )
    if distance_bands:
        for band, (lower, upper) in DISTANCE_BANDS.items():
            banded = upstream_land_cover(min_km=lower, max_km=upper)
            banded = banded[["station_code", "year", *panels.LAND_COVER_ALR_COLUMNS]].rename(
                columns={column: f"{column}__{band}" for column in panels.LAND_COVER_ALR_COLUMNS}
            )
            frame = frame.merge(banded, on=["station_code", "year"], how="left", validate="many_to_one")

    frame = frame.rename(columns=CLIMATE_CONTROLS)
    frame["period"] = frame["year"] // REPORTING_LIMIT_PERIOD_YEARS * REPORTING_LIMIT_PERIOD_YEARS
    frame["biome_year"] = frame["biome"] + "_" + frame["year"].astype(str)
    frame["biome_month"] = frame["biome"] + "_" + frame["month"].astype(str)

    combine_fecal_indicator(frame)
    apply_plausibility(frame)
    drop_unit_errors(frame)
    frame = frame.join(conama_exceedances(frame))
    for column in CONTINUOUS_OUTCOMES:
        frame[f"y_{column}"] = transformed_outcome(frame, column)
    frame["log_discharge"] = np.log(frame["streamflow_discharge_day"].where(frame["streamflow_discharge_day"] > 0))
    return add_near_mining(frame)
