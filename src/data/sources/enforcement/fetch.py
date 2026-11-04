import hashlib
import json
import logging
import os

import requests

from .constants import PRIORITY_FILE, PRIORITY_URL, priority_path


logger = logging.getLogger(__name__)

RECORD_FILENAME = "record.json"


def _md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_priority_list(root_dir="."):
    """Download the Priority List table and its record metadata from Zenodo (the IBAMA tables are placed by hand).

    A file already present with Zenodo's md5 is kept; the download replaces it
    only once the md5 matches.
    """
    response = requests.get(PRIORITY_URL, timeout=120)
    response.raise_for_status()
    record = response.json()
    files = {entry["key"]: entry for entry in record["files"]}
    if PRIORITY_FILE not in files:
        raise RuntimeError(f"{PRIORITY_FILE} not in Zenodo record {PRIORITY_URL}")
    entry = files[PRIORITY_FILE]
    expected_md5 = entry["checksum"].removeprefix("md5:")

    destination = priority_path(root_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and _md5(destination) == expected_md5:
        logger.info("%s already present and verified, skipping.", destination)
        return destination

    temp_path = destination.with_name(f"{destination.name}.tmp-{os.getpid()}")
    try:
        content = requests.get(entry["links"]["self"], timeout=600)
        content.raise_for_status()
        temp_path.write_bytes(content.content)
        actual = _md5(temp_path)
        if actual != expected_md5:
            raise RuntimeError(f"{PRIORITY_FILE}: md5 {actual} != Zenodo's {expected_md5}")
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)
    (destination.parent / RECORD_FILENAME).write_text(json.dumps(record, indent=1, ensure_ascii=False))
    logger.info("Wrote the Priority List to %s", destination)
    return destination
