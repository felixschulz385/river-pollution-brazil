from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from src.data.shared.paths import processed_dir, raw_dir as _shared_raw_dir, source_root


# INPE's DETER cloud masks and alerts, placed by hand in `data/cloud_cover/raw/`.
#
# `132281-V1.zip` is the openICPSR replication package of Assunção, Gandour &
# Rocha (2023, AEJ: Applied, "DETERring Deforestation in the Amazon"; data CC
# BY 4.0, code BSD; INPE's files CC BY-SA 3.0). It holds INPE's monthly cloud
# shapefiles, May 2005 - December 2016, as the authors downloaded them from
# www1.dpi.inpe.br/obt/deter/dados/ (2016 and 2018), in a zip inside the zip;
# TerraBrasilis does not publish them. Only that inner zip is read.
REPLICATION_FILE = "132281-V1.zip"
CLOUD_MEMBER = "data/raw2clean/land_cover/deter_inpe/input.zip"
# The authors' municipality x month panel (7-digit IBGE codes, their 2007
# municipal map), read only to compare against.
AUTHORS_PANEL_MEMBER = "data/built/land_cover/deter/policy_la_deter_built_clouds_coverage_panel_df.Rdata"

# One folder per month with data, `input/<year>/Nuvem_<YYYYMMDD>_shp/` or
# `Nuvens_<YYYYMM>_shp/`, holding one or two polygon layers. The month comes
# from the date in the folder name.
CLOUD_LAYER_PATTERN = re.compile(r"^input/\d{4}/Nuve(?:m|ns)_(?P<year>\d{4})(?P<month>\d{2})\d*_shp/(?P<layer>[^/]+)\.shp$")
FIRST_YEAR = 2005
LAST_YEAR = 2016

# Coordinate systems of layers without a .prj, from INPE's documentation as the
# authors record it: SAD69 to 2010, SAD69 (pre-1996, Brazil's shift) from 2011
# to July 2015, SIRGAS 2000 from August 2015. The authors give January 2016 the
# second; it has no .prj, and every later layer's .prj is SIRGAS 2000.
SAD69 = "EPSG:4618"
SAD69_PRE96_BR = "+proj=longlat +ellps=aust_SA +towgs84=-66.87,4.37,-38.52,0,0,0,0 +no_defs"
SIRGAS2000 = "EPSG:4674"
CRS_PERIODS = ((date(2011, 1, 1), SAD69), (date(2015, 8, 1), SAD69_PRE96_BR), (date(9999, 1, 1), SIRGAS2000))

# Areas in SIRGAS 2000 / Brazil Polyconic, as `spatial_tabular` (the authors
# used SAD69 / Brazil Polyconic).
AREA_CRS = "EPSG:5880"

# The Legal Amazon: the eight full states and Maranhão's municipalities
# reaching west of 44°W (the rule of `moreno_louzada.legal_amazon`).
LEGAL_AMAZON_STATES = ("11", "12", "13", "14", "15", "16", "17", "51")
MARANHAO = "21"
LEGAL_AMAZON_MERIDIAN = -44.0

MUN_ID_COLUMN = "mun_id"
YEAR_COLUMN = "year"
MONTH_COLUMN = "month"

# Clouds. A month with two layers (most of 2006-2010: two image passes) counts
# an area as clouded only where both are. A month without a folder counts the
# whole Legal Amazon as clouded (INPE's guidance; 13 months in 2005-2009).
CLOUD_SHARE_COLUMN = "deter_cloud_share"
CLOUD_IMPUTED_COLUMN = "deter_cloud_imputed"
CLOUD_PRODES_COLUMN = "deter_cloud_share_prodes"
CLOUD_MONTH_COLUMNS = (MUN_ID_COLUMN, YEAR_COLUMN, MONTH_COLUMN, CLOUD_SHARE_COLUMN, CLOUD_IMPUTED_COLUMN)
# The year table: the mean over the calendar year and over the PRODES year
# ending in July of `year` (August of the year before to July), as the authors
# define it; PRODES 2005 lacks August-December 2004 and is missing.
CLOUD_ADM2_COLUMNS = (MUN_ID_COLUMN, YEAR_COLUMN, CLOUD_SHARE_COLUMN, CLOUD_PRODES_COLUMN)
PRODES_FIRST_MONTH = 8

# DETER-B alerts from TerraBrasilis (terrabrasilis.dpi.inpe.br, "deter-amz",
# downloaded as a shapefile zip named for the download time). One polygon per
# alert and municipality, from August 2016, with the class, the image date and
# the municipality's 7-digit code and area. Alerts of the current month
# (`_curr` IDs) are not yet audited.
ALERTS_GLOB = "deter-amz-*.zip"
ALERTS_FIRST_MONTH = (2016, 8)
ALERTS_DATE_PATTERN = re.compile(r"deter-amz-(?P<day>\d{2})-(?P<month>\d{2})-(?P<year>\d{4})")
ALERT_FIELDS = ("CLASSNAME", "VIEW_DATE", "AREAMUNKM", "GEOCODIBGE")
# Alert groups: every class, clear-cut deforestation, and mining.
ALERT_GROUPS = {
    "all": None,
    "deforestation": ("DESMATAMENTO_CR", "DESMATAMENTO_VEG"),
    "mining": ("MINERACAO",),
}


def alert_columns(group):
    return (f"deter_alerts_{group}", f"deter_alert_area_km2_{group}")


ALERT_MEASURE_COLUMNS = tuple(column for group in ALERT_GROUPS for column in alert_columns(group))
ALERT_MONTH_COLUMNS = (MUN_ID_COLUMN, YEAR_COLUMN, MONTH_COLUMN, *ALERT_MEASURE_COLUMNS)
ALERT_ADM2_COLUMNS = (MUN_ID_COLUMN, YEAR_COLUMN, *ALERT_MEASURE_COLUMNS)

CLOUDS_PRODUCT = "clouds"
ALERTS_PRODUCT = "deter_alerts"
PRODUCTS = (CLOUDS_PRODUCT, ALERTS_PRODUCT)

DEFAULT_CLOUD_MONTH_OUTPUT_PATH = str(processed_dir(".", "cloud_cover") / "deter_clouds_adm2_month.parquet")
DEFAULT_CLOUD_ADM2_OUTPUT_PATH = str(processed_dir(".", "cloud_cover") / "deter_clouds_adm2.parquet")
DEFAULT_ALERT_MONTH_OUTPUT_PATH = str(processed_dir(".", "cloud_cover") / "deter_alerts_adm2_month.parquet")
DEFAULT_ALERT_ADM2_OUTPUT_PATH = str(processed_dir(".", "cloud_cover") / "deter_alerts_adm2.parquet")


def cloud_cover_dir(root_dir: str | Path = ".") -> Path:
    return source_root(root_dir, "cloud_cover")


def raw_dir(root_dir: str | Path = ".") -> Path:
    return _shared_raw_dir(root_dir, "cloud_cover")


def replication_path(root_dir: str | Path = ".") -> Path:
    return raw_dir(root_dir) / REPLICATION_FILE


def alerts_date(path):
    """The download date in a DETER-B file name (`deter-amz-<DD>-<MM>-<YYYY>-...`), or None."""
    match = ALERTS_DATE_PATTERN.search(Path(path).name)
    return None if match is None else date(int(match["year"]), int(match["month"]), int(match["day"]))


def alerts_path(root_dir: str | Path = ".") -> Path | None:
    """The newest DETER-B download by the date in its name, or None."""
    files = [path for path in raw_dir(root_dir).glob(ALERTS_GLOB) if alerts_date(path)]
    return max(files, key=alerts_date) if files else None


def layer_crs(year, month):
    """The coordinate system of a cloud layer without a .prj."""
    first_day = date(year, month, 1)
    return next(crs for end, crs in CRS_PERIODS if first_day < end)
