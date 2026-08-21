import logging
import zipfile

import requests

from src.data.shared.batches import atomic_write_bytes
from .constants import BIOMES_ARCHIVE_URL, archive_path, raw_dir


logger = logging.getLogger(__name__)


def fetch_biomes(root_dir="."):
    """Download the IBGE biomes archive and extract it under `data/biomes/raw`."""
    destination = archive_path(root_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        logger.info("Biomes archive already present at %s, skipping download.", destination)
    else:
        logger.info("Downloading IBGE biomes archive from %s", BIOMES_ARCHIVE_URL)
        response = requests.get(BIOMES_ARCHIVE_URL, timeout=300)
        response.raise_for_status()
        # Written atomically, so a process killed mid-download (network drop,
        # OOM, SLURM preemption) can never leave a partial/corrupt file at
        # `destination` -- the `destination.exists()` skip-check above would
        # otherwise treat that corrupt file as "already downloaded" forever,
        # requiring manual deletion to recover.
        atomic_write_bytes(str(destination), response.content)

    extract_dir = raw_dir(root_dir)
    with zipfile.ZipFile(destination, "r") as archive:
        archive.extractall(extract_dir)
    logger.info("Extracted biomes archive to %s", extract_dir)
    return extract_dir
