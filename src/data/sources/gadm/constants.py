from src.data.shared.paths import GADM_PATH as RAW_GADM_PATH


PROCESSED_DIR = "data/gadm/processed"
DEFAULT_SIMPLIFIED_GADM_PATH = "data/gadm/processed/gadm_simplified.gpkg"
DEFAULT_ADM0_LAYER = "ADM_ADM_0"
DEFAULT_ADM2_LAYER = "ADM_ADM_2"
# Matches the ad hoc tolerance river_network/core.py used at its own
# simplify() call sites before this preprocessing step existed.
DEFAULT_SIMPLIFY_TOLERANCE = 0.01
