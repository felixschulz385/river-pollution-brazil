"""Tests for the pollution model-setup rules in src.analysis.pollution."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import pollution as P


def _frame(**columns):
    """A minimal sample frame with every column `conama_exceedances` reads."""
    n = len(next(iter(columns.values())))
    base = {column: np.full(n, np.nan) for column in P.CONAMA_CLASS2_LIMITS}
    base.update(
        station_code=["s1"] * n,
        period=[2000] * n,
        ammoniacal_nitrogen=np.full(n, np.nan),
        fecal_type=[None] * n,
        fecal_indicator=np.full(n, np.nan),
    )
    base.update(columns)
    return pd.DataFrame(base)


def test_reporting_limit_at_or_above_conama_limit_masks_station_period():
    # Mercury limit 0.0002; non-detects reported at 0.2 make the period uninformative.
    frame = _frame(mercury=[0.2, 0.2, 0.2, 0.5])
    exceed = P.conama_exceedances(frame)
    assert exceed["exceed_mercury"].isna().all()


def test_reporting_limit_below_conama_limit_keeps_values():
    # Lead limit 0.01; reporting limit 0.005 is below it, so violations are identified.
    frame = _frame(lead=[0.005, 0.005, 0.005, 0.05])
    exceed = P.conama_exceedances(frame)
    assert exceed["exceed_lead"].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_unit_errors_dropped_in_mixed_station_period():
    # Mercury limit 0.0002: 0.2 next to 0.0001 readings is a ug/L entry; 0.0005 is a real violation.
    frame = _frame(mercury=[0.0001, 0.0001, 0.0005, 0.2])
    P.drop_unit_errors(frame)
    assert frame["mercury"].isna().tolist() == [False, False, False, True]
    exceed = P.conama_exceedances(frame)
    assert exceed["exceed_mercury"].tolist()[:3] == [0.0, 0.0, 1.0]


def test_unit_errors_keep_high_reporting_limit_periods_and_moderate_values():
    # All-0.2 periods are left to the reporting-limit rule; lead 0.05 is 5x its limit, not a unit error.
    frame = _frame(mercury=[0.2, 0.2, 0.2, 0.2], lead=[0.005, 0.005, 0.005, 0.05])
    P.drop_unit_errors(frame)
    assert frame["mercury"].notna().all()
    assert frame["lead"].notna().all()


def test_unit_error_rule_covers_trace_metals_only():
    assert {"mercury", "cadmium", "lead", "arsenic"} <= set(P.UNIT_ERROR_ANALYTES)
    assert "manganese" not in P.UNIT_ERROR_ANALYTES and "turbidity" not in P.UNIT_ERROR_ANALYTES


def test_lower_and_range_limits():
    frame = _frame(dissolved_oxygen=[4.0, 6.0], ph=[5.5, 7.0])
    exceed = P.conama_exceedances(frame)
    assert exceed["exceed_dissolved_oxygen"].tolist() == [1.0, 0.0]
    assert exceed["exceed_ph"].tolist() == [1.0, 0.0]


def test_ammoniacal_nitrogen_limit_depends_on_ph():
    # 2.5 mg/L N is compliant at pH 7 (limit 3.7) but violates at pH 7.8 (limit 2.0).
    frame = _frame(ammoniacal_nitrogen=[2.5, 2.5, 2.5], ph=[7.0, 7.8, np.nan])
    exceed = P.conama_exceedances(frame)
    assert exceed["exceed_ammoniacal_nitrogen"].tolist()[:2] == [0.0, 1.0]
    assert np.isnan(exceed["exceed_ammoniacal_nitrogen"].iloc[2])


def test_combined_fecal_indicator_prefers_thermotolerant_and_sets_type():
    frame = pd.DataFrame(
        {
            "fecal_coliforms": [100.0, np.nan, np.nan, np.nan],
            "thermotolerant_coliforms": [np.nan, 200.0, np.nan, 300.0],
            "escherichia": [np.nan, np.nan, 50.0, 40.0],
        }
    )
    P.combine_fecal_indicator(frame)
    assert frame["fecal_indicator"].tolist() == [100.0, 200.0, 50.0, 300.0]
    assert frame["fecal_type"].tolist() == ["thermotolerant", "thermotolerant", "e_coli", "thermotolerant"]


def test_fecal_exceedance_uses_type_specific_limit():
    # 900/100 mL complies with the thermotolerant limit (1,000) but violates E. coli's (800).
    frame = _frame(fecal_indicator=[900.0, 900.0], fecal_type=["thermotolerant", "e_coli"])
    exceed = P.conama_exceedances(frame)
    assert exceed["exceed_fecal_indicator"].tolist() == [0.0, 1.0]


def test_plausibility_drops_zero_ph_and_supersaturated_oxygen():
    frame = pd.DataFrame({"ph": [0.0, 7.0, 14.0], "dissolved_oxygen": [0.0, 8.0, 25.0]})
    P.apply_plausibility(frame)
    assert frame["ph"].isna().tolist() == [True, False, False]
    assert frame["dissolved_oxygen"].isna().tolist() == [True, False, True]


def test_identifying_stations_requires_samples_and_years():
    frame = pd.DataFrame(
        {
            "station_code": ["a"] * 8 + ["b"] * 8,
            "year": [2000, 2000, 2001, 2001, 2002, 2002, 2003, 2003] + [2000] * 8,
            "y": 1.0,
        }
    )
    mask = P.identifying_stations(frame, "y")
    assert mask[frame["station_code"] == "a"].all()
    assert not mask[frame["station_code"] == "b"].any()
