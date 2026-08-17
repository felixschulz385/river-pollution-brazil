import logging
import zipfile

import requests

from .constants import BIOMES_ARCHIVE_URL, archive_path, raw_dir


logger = logging.getLogger(__name__)


def fetch_biomes(root_dir="."):
    """Download the IBGE biomes archive and extract it under `data/biomes/raw`."""
    destination = archive_path(root_dir)
    if destination.exists():
        logger.info("Biomes archive already present at %s, skipping download.", destination)
    else:
        logger.info("Downloading IBGE biomes archive from %s", BIOMES_ARCHIVE_URL)
        response = requests.get(BIOMES_ARCHIVE_URL, timeout=300)
        response.raise_for_status()
        destination.write_bytes(response.content)

    extract_dir = raw_dir(root_dir)
    with zipfile.ZipFile(destination, "r") as archive:
        archive.extractall(extract_dir)
    logger.info("Extracted biomes archive to %s", extract_dir)
    return extract_dir
