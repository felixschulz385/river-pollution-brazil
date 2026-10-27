import logging
import os
from pathlib import Path

import requests

from .constants import MAPBIOMAS_COVERAGE_URL_TEMPLATE, MAPBIOMAS_FILENAME_TEMPLATE, MAPBIOMAS_YEARS


logger = logging.getLogger(__name__)


def _download(url, destination):
    """Stream `url` to `destination` through a temporary file, so a partial download is never kept."""
    temp_path = destination.with_name(f"{destination.name}.tmp-{os.getpid()}")
    try:
        with requests.get(url, stream=True, timeout=600) as response:
            response.raise_for_status()
            expected = response.headers.get("Content-Length")
            with open(temp_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 22):
                    handle.write(chunk)
        if expected is not None and temp_path.stat().st_size != int(expected):
            raise RuntimeError(
                f"{destination.name}: got {temp_path.stat().st_size} of {expected} bytes"
            )
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)


def fetch_mapbiomas_coverage(datadir, years=MAPBIOMAS_YEARS):
    """Download the MapBiomas Collection 10 annual coverage GeoTIFFs into `datadir`.

    Years whose file is already present are skipped, so the step can be rerun
    after an interrupted download.
    """
    datadir = Path(datadir)
    datadir.mkdir(parents=True, exist_ok=True)
    for year in years:
        destination = datadir / MAPBIOMAS_FILENAME_TEMPLATE.format(year=year)
        if destination.exists():
            logger.info("%s already present, skipping.", destination.name)
            continue
        url = MAPBIOMAS_COVERAGE_URL_TEMPLATE.format(year=year)
        logger.info("Downloading %s", url)
        _download(url, destination)
    return datadir
