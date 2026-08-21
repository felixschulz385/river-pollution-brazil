"""Column-set/aggregation contracts for assembled climate output tables."""

SENSOR_WINDOW_LABELS = {
    "7d": 7,
    "30d": 30,
    "90d": 90,
    "180d": 180,
    "365d": 365,
}

# All non-extreme variables aggregate to a mean daily rate over the window
# (annual for the ADM2 panel, 7/30/90/180/365-day for the sensor panel) rather
# than a cumulative total. This applies to the accumulation variables
# (tp/sro/ssro/pev, in mm/day) too, even though a raw sum would be equally
# valid hydrologically: a mean keeps different window lengths directly
# comparable (an annual mean and a 7-day mean both read as "typical daily
# rate," whereas an annual sum and a 7-day sum don't, since they scale with
# window length) and matches the sensor panel's `mean_Xd` naming and the ADM2
# panel's `mean_value` column, both of which already commit to "mean."
ANNUAL_MEAN_VARIABLES = {"tp", "sro", "ssro", "pev", "2t", "2d", "swvl1", "swvl2"}
ANNUAL_MIN_VARIABLES = {"2t_daily_min"}
ANNUAL_MAX_VARIABLES = {"2t_daily_max"}
