import logging
from pathlib import Path

import geopandas as gpd

from .constants import (
    DEFAULT_ADM0_LAYER,
    DEFAULT_ADM2_LAYER,
    DEFAULT_SIMPLIFIED_GADM_PATH,
    DEFAULT_SIMPLIFY_TOLERANCE,
    RAW_GADM_PATH,
)


logger = logging.getLogger(__name__)


def simplify_gadm(
    root_dir=".",
    gadm_path=None,
    adm0_layer=None,
    adm2_layer=None,
    tolerance=None,
    output_path=None,
):
    """Simplify the raw GADM ADM0/ADM2 layers and cache the result.

    Consumers (river_network, biomes, sensor_data) each read these layers
    for coarse operations (a country-membership bounds check, an ADM2
    spatial join) that don't need full-resolution boundaries; simplifying
    once here means they no longer each pay full-resolution cost -- or, for
    river_network, redundantly re-simplify from scratch on every run.
    """
    raw_path = Path(root_dir) / (gadm_path or RAW_GADM_PATH)
    adm0_layer = adm0_layer or DEFAULT_ADM0_LAYER
    adm2_layer = adm2_layer or DEFAULT_ADM2_LAYER
    tolerance = DEFAULT_SIMPLIFY_TOLERANCE if tolerance is None else tolerance
    output_path = Path(root_dir) / (output_path or DEFAULT_SIMPLIFIED_GADM_PATH)

    logger.info("Simplifying GADM boundaries from %s (tolerance=%s)", raw_path, tolerance)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Start from a clean file: writing a layer name into an already-populated
    # gpkg has ambiguous replace/append/error semantics depending on
    # driver/version, so a stale leftover file is deleted rather than relied
    # on to behave a particular way.
    output_path.unlink(missing_ok=True)

    for layer in (adm0_layer, adm2_layer):
        boundaries = gpd.read_file(raw_path, layer=layer)
        boundaries["geometry"] = boundaries.simplify(tolerance)
        boundaries.to_file(output_path, layer=layer, driver="GPKG")
        logger.info("Wrote simplified layer '%s' (%d features) to %s", layer, len(boundaries), output_path)

    return output_path
