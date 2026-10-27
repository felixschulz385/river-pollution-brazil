import hashlib
import logging
import os

import requests

from .constants import METADATA_FILENAMES, YEAR_FILENAME_TEMPLATE, YEARS, ZENODO_RECORD_URL, raw_dir


logger = logging.getLogger(__name__)


def _md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url, destination, expected_md5):
    """Stream `url` to `destination`, replacing it only once the md5 matches."""
    temp_path = destination.with_name(f"{destination.name}.tmp-{os.getpid()}")
    try:
        with requests.get(url, stream=True, timeout=600) as response:
            response.raise_for_status()
            with open(temp_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    handle.write(chunk)
        actual = _md5(temp_path)
        if actual != expected_md5:
            raise RuntimeError(f"{destination.name}: md5 {actual} != Zenodo's {expected_md5}")
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def fetch_sivep(root_dir=".", years=YEARS):
    """Download the yearly SIVEP parquet files and metadata from Zenodo.

    Files already present with Zenodo's md5 are kept, so the step is safe to
    rerun after a partial download or after moving the files in by hand.
    """
    response = requests.get(ZENODO_RECORD_URL, timeout=120)
    response.raise_for_status()
    files = {entry["key"]: entry for entry in response.json()["files"]}

    wanted = [YEAR_FILENAME_TEMPLATE.format(year=year) for year in years] + list(METADATA_FILENAMES)
    missing = [name for name in wanted if name not in files]
    if missing:
        raise RuntimeError(f"Not in Zenodo record: {missing}")

    destination_dir = raw_dir(root_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    for name in wanted:
        entry = files[name]
        expected_md5 = entry["checksum"].removeprefix("md5:")
        destination = destination_dir / name
        if destination.exists() and _md5(destination) == expected_md5:
            logger.info("%s already present and verified, skipping.", name)
            continue
        logger.info("Downloading %s (%.1f MB)", name, entry["size"] / 1e6)
        _download(entry["links"]["self"], destination, expected_md5)
    return destination_dir
