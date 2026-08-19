from __future__ import annotations

from pathlib import Path


# All eight pipeline data sources this module can verify. Order matters for
# the summary table's default row order.
SOURCES = (
    "river_network",
    "land_cover",
    "sensor_data",
    "climate",
    "biomes",
    "population",
    "health",
    "assembly",
)

VERIFICATION_SIDECAR_FILENAME = ".verification.json"


def sidecar_path(root_dir, source: str) -> Path:
    """Return the cache sidecar path for `source`, next to its data directory."""
    return Path(root_dir) / "data" / source / VERIFICATION_SIDECAR_FILENAME
