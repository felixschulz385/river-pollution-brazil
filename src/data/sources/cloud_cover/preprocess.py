import logging
import multiprocessing
import os
import shutil
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely

from src.data.sources.biomes.preprocess import _load_adm2_boundaries
from src.data.sources.land_cover.constants import derive_mun_id_from_adm2_id

from .constants import (
    ALERT_ADM2_COLUMNS,
    ALERT_FIELDS,
    ALERT_GROUPS,
    ALERTS_FIRST_MONTH,
    ALERT_MONTH_COLUMNS,
    AREA_CRS,
    CLOUD_ADM2_COLUMNS,
    CLOUD_IMPUTED_COLUMN,
    CLOUD_LAYER_PATTERN,
    CLOUD_MEMBER,
    CLOUD_MONTH_COLUMNS,
    CLOUD_PRODES_COLUMN,
    CLOUD_SHARE_COLUMN,
    DEFAULT_ALERT_ADM2_OUTPUT_PATH,
    DEFAULT_ALERT_MONTH_OUTPUT_PATH,
    DEFAULT_CLOUD_ADM2_OUTPUT_PATH,
    DEFAULT_CLOUD_MONTH_OUTPUT_PATH,
    FIRST_YEAR,
    LAST_YEAR,
    LEGAL_AMAZON_MERIDIAN,
    LEGAL_AMAZON_STATES,
    MARANHAO,
    MONTH_COLUMN,
    MUN_ID_COLUMN,
    PRODES_FIRST_MONTH,
    YEAR_COLUMN,
    alert_columns,
    alerts_date,
    alerts_path,
    layer_crs,
    replication_path,
)


logger = logging.getLogger(__name__)

POLYGON = shapely.GeometryType.POLYGON
# Municipalities of the running build, set before the month workers fork.
_MUNICIPALITIES = []


def legal_amazon_municipalities(root_dir=".", gadm_path=None, layer=None, adm2_id_column=None):
    """The Legal Amazon's municipalities (`mun_id`, geometry) in `AREA_CRS`, one row each."""
    boundaries = _load_adm2_boundaries(root_dir, gadm_path=gadm_path, layer=layer, adm2_id_column=adm2_id_column)
    boundaries[MUN_ID_COLUMN] = boundaries["adm2_id"].map(derive_mun_id_from_adm2_id)
    state = boundaries[MUN_ID_COLUMN].str[:2]
    west = boundaries.to_crs(4326).geometry.bounds["minx"] < LEGAL_AMAZON_MERIDIAN
    amazon = boundaries[state.isin(LEGAL_AMAZON_STATES) | (state.eq(MARANHAO) & west)]
    amazon = amazon[[MUN_ID_COLUMN, "geometry"]].dissolve(MUN_ID_COLUMN).reset_index()
    return amazon.to_crs(AREA_CRS).sort_values(MUN_ID_COLUMN, ignore_index=True)


# --------------------------------------------------------------------------
# clouds
# --------------------------------------------------------------------------

def cloud_layers(names):
    """{(year, month): [shapefile paths]} of the cloud layers among a zip's member names, layers sorted by name."""
    layers = {}
    for name in names:
        match = CLOUD_LAYER_PATTERN.match(name)
        if match:
            layers.setdefault((int(match["year"]), int(match["month"])), []).append(name)
    return {key: sorted(paths) for key, paths in sorted(layers.items())}


def polygon_parts(geometries):
    """The polygons of `geometries` after repair; lines and points left by the repair are dropped."""
    parts = shapely.get_parts(shapely.make_valid(np.asarray(geometries, dtype=object)))
    parts = shapely.get_parts(parts)  # collections from the repair hold multipolygons
    return parts[(shapely.get_type_id(parts) == POLYGON) & ~shapely.is_empty(parts)]


def month_clouds(layers):
    """The clouded area of a month: the union of each layer's polygons, intersected across layers."""
    unions = [shapely.union_all(polygon_parts(layer)) for layer in layers]
    clouds = unions[0]
    for union in unions[1:]:
        clouds = shapely.intersection(clouds, union)
    return clouds


def covered_shares(municipalities, clouds):
    """Share of each municipality's area under `clouds` (a union, so its parts do not overlap)."""
    geometries = np.asarray(municipalities, dtype=object)
    parts = polygon_parts([clouds])
    covered = np.zeros(len(geometries))
    if len(parts):
        cut, part = shapely.STRtree(parts).query(geometries, predicate="intersects")
        areas = shapely.area(shapely.intersection(geometries[cut], parts[part]))
        covered = np.bincount(cut, weights=areas, minlength=len(geometries))
    return np.clip(covered / shapely.area(geometries), 0.0, 1.0)


def read_layer(source, name, year, month):
    """A cloud layer's polygons in `AREA_CRS`; a layer without a .prj gets INPE's system for its month."""
    frame = pyogrio.read_dataframe(f"/vsizip/{source}/{name}", columns=[])
    if frame.crs is None:
        frame = frame.set_crs(layer_crs(year, month))
    return frame.to_crs(AREA_CRS).geometry.to_numpy()


def month_shares(source, year, month, names):
    """Each municipality's clouded share in one month (`_MUNICIPALITIES`)."""
    clouds = month_clouds([read_layer(source, name, year, month) for name in names])
    shares = covered_shares(_MUNICIPALITIES, clouds)
    logger.info("%d-%02d: %d layer(s), mean share %.3f", year, month, len(names), shares.mean())
    return shares


def cloud_month_table(mun_ids, measured):
    """The municipality x month table from {(year, month): shares or None}; None is a month without data (all clouded)."""
    frames = []
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        for month in range(1, 13):
            shares = measured.get((year, month))
            frames.append(pd.DataFrame({
                MUN_ID_COLUMN: mun_ids, YEAR_COLUMN: year, MONTH_COLUMN: month,
                CLOUD_SHARE_COLUMN: 1.0 if shares is None else shares, CLOUD_IMPUTED_COLUMN: shares is None,
            }))
    return pd.concat(frames, ignore_index=True)[list(CLOUD_MONTH_COLUMNS)]


def cloud_year_table(monthly):
    """Per municipality and year, the mean share over the calendar year and over the PRODES year ending in July.

    A PRODES year needs all twelve months, so the first one is missing.
    """
    keys = [MUN_ID_COLUMN, YEAR_COLUMN]
    calendar = monthly.groupby(keys)[CLOUD_SHARE_COLUMN].mean()
    prodes_year = monthly[YEAR_COLUMN] + (monthly[MONTH_COLUMN] >= PRODES_FIRST_MONTH)
    grouped = monthly.groupby([monthly[MUN_ID_COLUMN], prodes_year.rename(YEAR_COLUMN)])[CLOUD_SHARE_COLUMN]
    prodes = grouped.mean().where(grouped.size() == 12)
    result = calendar.to_frame().join(prodes.rename(CLOUD_PRODES_COLUMN), how="left")
    return result.reset_index()[list(CLOUD_ADM2_COLUMNS)]


def _extract_member(path, member, destination):
    with zipfile.ZipFile(path) as bundle, bundle.open(member) as source, open(destination, "wb") as target:
        shutil.copyfileobj(source, target, 1 << 24)


def _write(frame, root_dir, path):
    resolved = Path(root_dir) / path
    resolved.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(resolved, index=False)
    return resolved


def build_clouds(root_dir=".", path=None, gadm_path=None, layer=None, adm2_id_column=None, month_output_path=None,
                 adm2_output_path=None, workers=None):
    """Measure DETER's monthly cloud cover per Legal Amazon municipality, 2005-2016, and its year means."""
    path = Path(path) if path else replication_path(root_dir)
    if not path.exists():
        raise FileNotFoundError(f"No replication package at {path}; openICPSR 132281-V1 is placed by hand.")
    municipalities = legal_amazon_municipalities(root_dir, gadm_path, layer, adm2_id_column)
    _MUNICIPALITIES[:] = municipalities.geometry.to_numpy()
    logger.info("%d Legal Amazon municipalities", len(municipalities))

    workers = workers or int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    # The cloud layers sit in a zip inside the package; GDAL reads a copy of it far faster than the nested zip.
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as scratch:
        source = Path(scratch) / "input.zip"
        _extract_member(path, CLOUD_MEMBER, source)
        with zipfile.ZipFile(source) as bundle:
            layers = cloud_layers(bundle.namelist())
        layers = {key: names for key, names in layers.items() if FIRST_YEAR <= key[0] <= LAST_YEAR}
        missing = [f"{year}-{month:02d}" for year in range(FIRST_YEAR, LAST_YEAR + 1) for month in range(1, 13)
                   if (year, month) not in layers]
        logger.info("%d months with cloud layers; %d without, counted as clouded: %s", len(layers), len(missing),
                    missing)
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork")) as pool:
            futures = {key: pool.submit(month_shares, source, *key, names) for key, names in layers.items()}
            measured = {key: future.result() for key, future in futures.items()}

    monthly = cloud_month_table(municipalities[MUN_ID_COLUMN].to_numpy(), measured)
    yearly = cloud_year_table(monthly)
    month_path = _write(monthly, root_dir, month_output_path or DEFAULT_CLOUD_MONTH_OUTPUT_PATH)
    year_path = _write(yearly, root_dir, adm2_output_path or DEFAULT_CLOUD_ADM2_OUTPUT_PATH)
    logger.info("Wrote cloud shares %s to %s and year means %s to %s", monthly.shape, month_path, yearly.shape,
                year_path)
    return monthly, yearly


# --------------------------------------------------------------------------
# DETER-B alerts
# --------------------------------------------------------------------------

def last_full_month(download):
    """(year, month) of the month before the download's."""
    return (download.year, download.month - 1) if download.month > 1 else (download.year - 1, 12)


def alert_month_table(alerts, mun_ids, first_month, last_month):
    """Alert counts and areas (km²) per municipality and month and group, zero where there are none.

    `alerts` has `ALERT_FIELDS`; rows outside `mun_ids` or the months are dropped.
    """
    dates = pd.to_datetime(alerts["VIEW_DATE"])
    frame = pd.DataFrame({
        MUN_ID_COLUMN: alerts["GEOCODIBGE"].astype(str).str.strip().str.replace(r"\.0+$", "", regex=True).str[:6],
        YEAR_COLUMN: dates.dt.year, MONTH_COLUMN: dates.dt.month, "class": alerts["CLASSNAME"],
        "area": pd.to_numeric(alerts["AREAMUNKM"], errors="coerce").fillna(0.0),
    })
    outside = ~frame[MUN_ID_COLUMN].isin(set(mun_ids))
    if outside.any():
        logger.warning("%d alerts in municipalities outside the Legal Amazon map dropped: %s", int(outside.sum()),
                       sorted(alerts.loc[outside, "GEOCODIBGE"].astype(str).unique()))
    months = pd.period_range(pd.Period(year=first_month[0], month=first_month[1], freq="M"),
                             pd.Period(year=last_month[0], month=last_month[1], freq="M"), freq="M")
    period = pd.PeriodIndex.from_fields(year=frame[YEAR_COLUMN], month=frame[MONTH_COLUMN], freq="M")
    frame = frame[~outside & period.isin(months)]

    keys = [MUN_ID_COLUMN, YEAR_COLUMN, MONTH_COLUMN]
    index = pd.MultiIndex.from_tuples(
        [(mun_id, month.year, month.month) for mun_id in sorted(mun_ids) for month in months], names=keys)
    result = pd.DataFrame(index=index)
    for group, classes in ALERT_GROUPS.items():
        members = frame if classes is None else frame[frame["class"].isin(classes)]
        grouped = members.groupby(keys)
        count, area = alert_columns(group)
        result[count] = grouped.size().reindex(index, fill_value=0).astype("int64")
        result[area] = grouped["area"].sum().reindex(index, fill_value=0.0).astype(float)
    return result.reset_index()[list(ALERT_MONTH_COLUMNS)]


def alert_year_table(monthly):
    """Year sums of the monthly alerts, for years with all twelve months."""
    keys = [MUN_ID_COLUMN, YEAR_COLUMN]
    months = monthly.groupby(YEAR_COLUMN)[MONTH_COLUMN].nunique()
    full = monthly[monthly[YEAR_COLUMN].isin(months.index[months == 12])]
    measures = [column for column in ALERT_ADM2_COLUMNS if column not in keys]
    return full.groupby(keys, as_index=False)[measures].sum()[list(ALERT_ADM2_COLUMNS)]


def build_alerts(root_dir=".", path=None, gadm_path=None, layer=None, adm2_id_column=None, month_output_path=None,
                 adm2_output_path=None):
    """Count DETER-B alerts per Legal Amazon municipality and month, from August 2016 to the month before the download."""
    path = Path(path) if path else alerts_path(root_dir)
    if path is None or not path.exists():
        raise FileNotFoundError("No DETER-B download (deter-amz-<DD>-<MM>-<YYYY>-*.zip) under data/cloud_cover/raw; "
                                "it is downloaded by hand from TerraBrasilis.")
    download = alerts_date(path)
    alerts = pyogrio.read_dataframe(f"/vsizip/{path}", columns=list(ALERT_FIELDS), read_geometry=False)
    mun_ids = legal_amazon_municipalities(root_dir, gadm_path, layer, adm2_id_column)[MUN_ID_COLUMN].to_numpy()
    monthly = alert_month_table(alerts, mun_ids, ALERTS_FIRST_MONTH, last_full_month(download))
    yearly = alert_year_table(monthly)
    logger.info("DETER-B %s: %d alerts, %d in the table; %d mining", download, len(alerts),
                int(monthly[alert_columns("all")[0]].sum()), int(monthly[alert_columns("mining")[0]].sum()))
    month_path = _write(monthly, root_dir, month_output_path or DEFAULT_ALERT_MONTH_OUTPUT_PATH)
    year_path = _write(yearly, root_dir, adm2_output_path or DEFAULT_ALERT_ADM2_OUTPUT_PATH)
    logger.info("Wrote alerts %s to %s and year sums %s to %s", monthly.shape, month_path, yearly.shape, year_path)
    return monthly, yearly
