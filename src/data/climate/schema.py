"""Column-set/aggregation contracts for assembled climate output tables."""

SENSOR_WINDOW_LABELS = {
    "7d": 7,
    "30d": 30,
    "90d": 90,
    "180d": 180,
    "365d": 365,
}

ANNUAL_SUM_VARIABLES = {"tp", "sro", "ssro", "pev"}
ANNUAL_MEAN_VARIABLES = {"2t", "2d", "swvl1", "swvl2"}
ANNUAL_MIN_VARIABLES = {"2t_daily_min"}
ANNUAL_MAX_VARIABLES = {"2t_daily_max"}
