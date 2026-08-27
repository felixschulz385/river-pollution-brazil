from src.data.shared.paths import processed_dir, raw_dir

# Raw GADM boundary geopackage, manually placed (no automated fetch step) --
# shared across biomes, river_network, and sensor_data via
# DEFAULT_SIMPLIFIED_GADM_PATH below.
RAW_GADM_FILENAME = "gadm41_BRA.gpkg"
RAW_GADM_PATH = str(raw_dir(".", "gadm") / RAW_GADM_FILENAME)

PROCESSED_DIR = str(processed_dir(".", "gadm"))
DEFAULT_SIMPLIFIED_GADM_PATH = str(processed_dir(".", "gadm") / "gadm_simplified.gpkg")
DEFAULT_ADM0_LAYER = "ADM_ADM_0"
DEFAULT_ADM2_LAYER = "ADM_ADM_2"
# Matches the ad hoc tolerance river_network/core.py used at its own
# simplify() call sites before this preprocessing step existed.
DEFAULT_SIMPLIFY_TOLERANCE = 0.01
