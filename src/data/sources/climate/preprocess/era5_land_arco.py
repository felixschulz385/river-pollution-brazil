from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..constants import BOUNDARY_HOURS
from ..fetch.common import ERA5_AREA, _timestamp, climate_file_lock
from ..fetch.era5_land_arco import ARCO_GROUPS, open_arco_group_dataset
from .era5_land import (
    ERA5_DAILY_VARIABLE_NAME_MAP,
    ERA5_OUTPUT_END,
    ERA5_OUTPUT_START,
    ERA5L_VAR_CONFIG,
    _drop_incomplete_boundary_day,
    _ensure_store_shape,
    _era5_store_path,
    _normalize_time_index,
    _rename_dataset_dims,
    bootstrap_era5_store,
    load_or_create_geobox_state,
    resample_era5l_hourly_to_daily,
    write_dataset_region,
)

logger = logging.getLogger(__name__)

ARCO_PROGRESS_FILENAME = "era5_land_arco_progress.json"


def _arco_progress_path(root_dir=".") -> Path:
    return _era5_store_path(root_dir).parent / ARCO_PROGRESS_FILENAME


def _load_arco_progress(root_dir=".") -> dict:
    path = _arco_progress_path(root_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_arco_progress(root_dir, progress: dict) -> None:
    path = _arco_progress_path(root_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(progress, indent=2, sort_keys=True), encoding="utf-8")


def _is_month_processed(progress: dict, group: str, year_month: str) -> bool:
    return progress.get(group, {}).get(year_month, {}).get("status") == "processed"


def _mark_month_processed(progress: dict, group: str, year_month: str) -> dict:
    progress.setdefault(group, {})[year_month] = {
        "status": "processed",
        "processed_at": _timestamp(),
    }
    return progress


def slice_arco_to_area(ds: xr.Dataset, area=ERA5_AREA) -> xr.Dataset:
    """Slice an ARCO dataset to `area` ([N, W, S, E]).

    Returns latitude in descending (N->S) order and longitude in ascending
    (W->E) order to match the orientation of the existing local zarr store's
    geobox - ARCO's native latitude order and the store's are not the same,
    and this must never be assumed either way.
    """
    north, west, south, east = area
    lat = np.asarray(ds["latitude"].values)
    lon = np.asarray(ds["longitude"].values)

    lat_ascending = lat[0] < lat[-1]
    lat_select = slice(south, north) if lat_ascending else slice(north, south)
    ds = ds.sel(latitude=lat_select)
    if lat_ascending:
        ds = ds.isel(latitude=slice(None, None, -1))

    west_query, east_query = west, east
    if lon.min() >= 0 and (west < 0 or east < 0):
        west_query, east_query = west % 360, east % 360

    lon_ascending = lon[0] < lon[-1]
    lon_select = slice(west_query, east_query) if lon_ascending else slice(east_query, west_query)
    ds = ds.sel(longitude=lon_select)
    if not lon_ascending:
        ds = ds.isel(longitude=slice(None, None, -1))

    return ds


def _assert_matches_geobox(ds: xr.Dataset, geobox_state: dict, *, atol: float = 1e-6) -> None:
    lat = np.asarray(ds["latitude"].values)
    lon = np.asarray(ds["longitude"].values)
    expected_lat = np.asarray(geobox_state["latitude"])
    expected_lon = np.asarray(geobox_state["longitude"])

    if lat.shape != expected_lat.shape or not np.allclose(lat, expected_lat, atol=atol):
        raise ValueError(
            "ARCO latitude grid does not match the existing store geobox: "
            f"got n={lat.size} ({lat[:2]}...{lat[-2:]}), expected n={expected_lat.size} "
            f"({expected_lat[:2]}...{expected_lat[-2:]})."
        )
    if lon.shape != expected_lon.shape or not np.allclose(lon, expected_lon, atol=atol):
        raise ValueError(
            "ARCO longitude grid does not match the existing store geobox: "
            f"got n={lon.size} ({lon[:2]}...{lon[-2:]}), expected n={expected_lon.size} "
            f"({expected_lon[:2]}...{expected_lon[-2:]})."
        )


def _pending_year_months(start: str, end: str, progress: dict, group: str) -> list[pd.Period]:
    return [
        period
        for period in pd.period_range(start=start, end=end, freq="M")
        if not _is_month_processed(progress, group, str(period))
    ]


def _month_time_bounds(period: pd.Period) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Return `(month_start, month_end, boundary_end)`.

    `boundary_end` extends `BOUNDARY_HOURS` past the month's own last hour,
    into the following month's first `BOUNDARY_HOURS` UTC hours --
    `resample_era5l_hourly_to_daily` buckets by Brazil-local calendar day
    (shifting timestamps back before flooring to a date), so this month's
    own last local day needs those hours to be complete.
    """
    month_start = period.start_time
    month_end = period.end_time
    boundary_end = (period + 1).start_time + pd.Timedelta(hours=BOUNDARY_HOURS - 1)
    return month_start, month_end, boundary_end


def preprocess_era5_land_arco(
    root_dir=".",
    *,
    start: str = ERA5_OUTPUT_START,
    end: str = ERA5_OUTPUT_END,
    chunks: str = "geo",
    area=ERA5_AREA,
) -> Path:
    store_path = bootstrap_era5_store(root_dir=root_dir)
    geobox_state = load_or_create_geobox_state(root_dir=root_dir)
    progress = _load_arco_progress(root_dir)

    for group, group_cfg in ARCO_GROUPS.items():
        # group_cfg["vars"] lists ARCO's native short names (e.g. t2m, d2m),
        # which aren't always our internal short names (2t, 2d) - resolve
        # through the same rename map applied to the data itself below.
        internal_names = [ERA5_DAILY_VARIABLE_NAME_MAP.get(v, v) for v in group_cfg["vars"]]
        var_config = {v: ERA5L_VAR_CONFIG[v] for v in internal_names if v in ERA5L_VAR_CONFIG}
        if not var_config:
            continue

        pending_months = _pending_year_months(start, end, progress, group)
        if not pending_months:
            logger.info("ARCO group %s already fully processed for %s..%s.", group, start, end)
            continue

        source_ds = open_arco_group_dataset(group, root_dir=root_dir, chunks=chunks)
        try:
            source_ds = slice_arco_to_area(source_ds, area=area)
            source_ds = source_ds.rename(
                {k: v for k, v in ERA5_DAILY_VARIABLE_NAME_MAP.items() if k in source_ds.data_vars}
            )
            source_ds = _rename_dataset_dims(source_ds)

            for period in pending_months:
                month_start, month_end, boundary_end = _month_time_bounds(period)
                month_ds = source_ds.sel(time=slice(month_start, month_end))
                if month_ds.sizes.get("time", 0) == 0:
                    logger.warning(
                        "ARCO group %s has no data for %s yet; stopping early.", group, period
                    )
                    break

                # This month's own last Brazil-local day needs `BOUNDARY_HOURS`
                # of the *next* month's first UTC hours (see `_month_time_bounds`);
                # if the ARCO store doesn't have them yet, defer this month
                # (and, since later months need this same not-yet-arrived data
                # too, everything after it) rather than writing an incomplete day.
                boundary_ds = source_ds.sel(time=slice(month_end, boundary_end))
                if boundary_ds.sizes.get("time", 0) < BOUNDARY_HOURS:
                    logger.warning(
                        "ARCO group %s doesn't yet have the %d boundary hour(s) "
                        "needed to complete %s's own last Brazil-local day; "
                        "stopping early.",
                        group,
                        BOUNDARY_HOURS,
                        period,
                    )
                    break

                daily = resample_era5l_hourly_to_daily(
                    xr.concat([month_ds, boundary_ds], dim="time"), var_config
                )
                daily = _drop_incomplete_boundary_day(daily, month_start)
                _assert_matches_geobox(daily, geobox_state)
                daily = _ensure_store_shape(daily, geobox_state)
                daily = daily.assign_coords(
                    time=_normalize_time_index(daily.indexes["time"])
                ).sortby("time")
                # ARCO's remote dask chunking rarely lines up with the local
                # store's fixed zarr chunk scheme; a lazy write can fail (or
                # silently misalign) chunk validation. One month for our small
                # bbox is tiny, so just materialize it before writing.
                daily = daily.load()

                with climate_file_lock(store_path, owner="climate_preprocess_arco"):
                    write_dataset_region(daily, store_path)

                progress = _mark_month_processed(progress, group, str(period))
                _save_arco_progress(root_dir, progress)
                logger.info("Processed ARCO group=%s month=%s.", group, period)
        finally:
            close = getattr(source_ds, "close", None)
            if callable(close):
                close()

    return store_path
