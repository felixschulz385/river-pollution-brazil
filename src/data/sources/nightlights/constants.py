from __future__ import annotations

import re
from pathlib import Path

from src.data.shared.paths import processed_dir, raw_dir as _shared_raw_dir, source_root


# Earth Observation Group (EOG, Colorado School of Mines) annual VIIRS
# night-light composites (VNL), downloaded by hand behind EOG's login into
# folders under `data/nightlights/raw/`, kept as served (`.dat.tif.gz`):
# - `eog_vnl_v21/`: v2.1, 2012 (April-December) to 2021, Suomi NPP; a link to
#   the copies of the growth-and-temperature project;
# - `eog_vnl_<YYYY-MM-DD>/`: v2.2, 2022 onward (2022 combines Suomi NPP with
#   three months of NOAA-20), with PROVENANCE.md, SCHEMA.md and SHA256SUMS.
# Every file covers the globe on one grid: 86401 x 33601 pixels of 15 arc
# seconds, EPSG:4326, pixel centres from (-180, 75), no nodata value, one strip
# per row. Radiance is in nW/cm2/sr; `cf_cvg` counts cloud-free observations.
FILE_GLOB = "*/*.dat.tif.gz"
AVERAGE = "average_masked"  # background, fires and aurora masked to 0
COVERAGE = "cf_cvg"
VARIANTS = (AVERAGE, COVERAGE)
# `VNL_v21_npp_2013_global_vcmcfg_c202205302300.average_masked.dat.tif`,
# `VNL_v21_npp_201204-201212_...` (2012 starts in April),
# `VNL_v22_npp-j01_2022_...`, `VNL_npp_2023_global_vcmslcfg_v2_c202402081600...`.
NAME_PATTERN = re.compile(
    r"VNL_(?:v(?P<version>2\d)_)?(?P<satellite>[a-z0-9-]+)_(?P<year>\d{4})(?:\d\d-\d{6})?_global_\w+?"
    r"(?P<v2>_v2)?_c\d+\.(?P<variant>\w+?)\.dat\.tif"
)
# Some copies were saved under another file's name (in v2.1, the 2012
# median_masked file holds the 2012 cf_cvg, and the 2013 cf_cvg file the 2012
# median_masked). Each file is identified by the original name EOG wrote into
# its TIFF document-name tag; a file whose tag disagrees with its name is skipped.
DOCUMENT_NAME_TAG = "TIFFTAG_DOCUMENTNAME"
FIRST_YEAR = 2012

# Masked radiance has a few negative cells (-1.5 marks a missing median in v2);
# they are set to 0. A pixel is lit when its masked radiance is above 0.
LIT_THRESHOLD = 0.0
# Zones within each municipality: the whole municipality, areas that bar mining
# and were in force by 2012 (`protected_areas`, lead 1's IGD areas), within
# `GOLD_BUFFER_M` of an SGB gold occurrence (`geology`), and both. Pixels belong
# to the polygon holding their centre.
GOLD_BUFFER_M = 5_000
ZONES = {"": "municipality", "_protected": "protected", "_gold5km": "gold", "_gold5km_protected": "gold_protected"}

MUN_ID_COLUMN = "mun_id"
YEAR_COLUMN = "year"
VERSION_COLUMN = "viirs_version"
SATELLITE_COLUMN = "viirs_satellite"
COVERAGE_COLUMN = "viirs_cf_cvg_mean"


def zone_columns(suffix):
    return (f"viirs_pixels{suffix}", f"viirs_lit_pixels{suffix}", f"viirs_radiance{suffix}")


MEASURE_COLUMNS = (*(column for suffix in ZONES for column in zone_columns(suffix)), COVERAGE_COLUMN)
ADM2_COLUMNS = (MUN_ID_COLUMN, YEAR_COLUMN, VERSION_COLUMN, SATELLITE_COLUMN, *MEASURE_COLUMNS)

DEFAULT_ADM2_OUTPUT_PATH = str(processed_dir(".", "nightlights") / "nightlights_adm2.parquet")


def nightlights_dir(root_dir: str | Path = ".") -> Path:
    return source_root(root_dir, "nightlights")


def raw_dir(root_dir: str | Path = ".") -> Path:
    return _shared_raw_dir(root_dir, "nightlights")


def parse_name(name):
    """(year, variant, version, satellite) of an EOG file name, or None."""
    match = NAME_PATTERN.search(name)
    if match is None:
        return None
    version = f"v2.{match['version'][1]}" if match["version"] else ("v2.2" if match["v2"] else None)
    return int(match["year"]), match["variant"], version, match["satellite"]


def raw_files(raw):
    """(path, year, variant, version, satellite) of each `average_masked` and `cf_cvg` file under `raw`."""
    files = []
    for path in sorted(Path(raw).glob(FILE_GLOB)):
        parsed = parse_name(path.name)
        if parsed is not None and parsed[1] in VARIANTS and parsed[0] >= FIRST_YEAR:
            files.append((path, *parsed))
    return files
