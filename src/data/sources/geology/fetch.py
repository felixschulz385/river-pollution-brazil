import logging
import os

import geopandas as gpd
import requests

from .constants import OCCURRENCE_FIELDS, SGB_LAYERS, SGB_REST_URL, gpkg_path


logger = logging.getLogger(__name__)


def query_layer(layer, out_fields="*"):
    """Return every feature of an SGB ArcGIS layer, checked against its count."""
    url = f"{SGB_REST_URL}/{layer}/query"
    response = requests.get(
        url, params=dict(where="1=1", outFields=out_fields, outSR=4326, f="geojson"), timeout=600
    )
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(f"{layer}: {payload['error']}")
    frame = gpd.GeoDataFrame.from_features(payload["features"], crs=4326)

    count = requests.get(url, params=dict(where="1=1", returnCountOnly="true", f="json"), timeout=120)
    count.raise_for_status()
    expected = count.json()["count"]
    if len(frame) != expected:
        raise RuntimeError(f"{layer}: got {len(frame)} of {expected} features")
    return frame


def fetch_sgb_gold(root_dir="."):
    """Download the SGB occurrence and gold province/district layers into one GeoPackage."""
    destination = gpkg_path(root_dir)
    if destination.exists():
        logger.info("SGB GeoPackage already present at %s, skipping download.", destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Written to a temporary file first, so an interrupted download never
    # leaves a partial GeoPackage that the existence check above would skip.
    temp_path = destination.with_name(f"{destination.stem}.tmp-{os.getpid()}{destination.suffix}")
    try:
        for name, layer in SGB_LAYERS.items():
            out_fields = ",".join(OCCURRENCE_FIELDS) if name == "occurrences" else "*"
            frame = query_layer(layer, out_fields)
            frame.to_file(temp_path, layer=name, driver="GPKG")
            logger.info("SGB %s: %d features", name, len(frame))
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)
    logger.info("Wrote SGB GeoPackage to %s", destination)
    return destination
