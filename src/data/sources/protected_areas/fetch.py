import logging
import os
import re

import geopandas as gpd
import requests

from .constants import FIELDS, LAYERS, WFS_URL, gpkg_path, layer_name


logger = logging.getLogger(__name__)


def query_layer(type_name, fields):
    """Every feature of a TerraBrasilis WFS layer in EPSG:4326, checked against the server's count."""
    params = dict(service="WFS", version="2.0.0", request="GetFeature", typeNames=type_name)
    response = requests.get(
        WFS_URL,
        params=dict(params, outputFormat="application/json", srsName="EPSG:4326", propertyName=",".join(
            (*fields, "geom"))),
        timeout=600,
    )
    response.raise_for_status()
    frame = gpd.GeoDataFrame.from_features(response.json()["features"], crs=4326)

    hits = requests.get(WFS_URL, params=dict(params, resultType="hits"), timeout=120)
    hits.raise_for_status()
    expected = int(re.search(r'numberMatched="(\d+)"', hits.text).group(1))
    if len(frame) != expected:
        raise RuntimeError(f"{type_name}: got {len(frame)} of {expected} features")
    return frame


def fetch_protected_areas(root_dir="."):
    """Download the conservation-unit layers into one GeoPackage, one layer per WFS layer."""
    destination = gpkg_path(root_dir)
    if destination.exists():
        logger.info("Protected-area GeoPackage already present at %s, skipping download.", destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Written to a temporary file first, so an interrupted download never
    # leaves a partial GeoPackage that the existence check above would skip.
    temp_path = destination.with_name(f"{destination.stem}.tmp-{os.getpid()}{destination.suffix}")
    try:
        for kind, type_names in LAYERS.items():
            for type_name in type_names:
                frame = query_layer(type_name, FIELDS[kind])
                frame.to_file(temp_path, layer=layer_name(type_name), driver="GPKG")
                logger.info("%s: %d features", type_name, len(frame))
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)
    logger.info("Wrote protected areas to %s", destination)
    return destination
