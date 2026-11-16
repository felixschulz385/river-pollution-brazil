"""Defaults shared by the analysis arms: clustering, fixed effects, inference and the gold price."""

from __future__ import annotations

# Otto Pfafstetter level-3 basins: errors are correlated along rivers, and level 3
# leaves enough effective clusters (output/notebooks/pollution/memo.md, Clustering).
CLUSTER = "otto_basin_l3"

# Municipality-year designs.
FE = "mun_id + year"
STATE_YEAR_FE = "mun_id + state_year"

# Wild-cluster bootstrap draws, and the seed of the bootstrap and randomization
# draws in the municipality-year designs.
BOOT_REPS = 9999
SEED = 20261028

# The log gold price (`commodity_prices`): in reais, the price garimpeiros receive,
# and in dollars as a check.
PRICE, PRICE_USD = "log_gold_brl", "log_gold_usd"
