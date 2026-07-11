from __future__ import annotations

import os
import json
import logging
from pathlib import Path
import pickle
import re
import gc
import shutil
import tempfile

import dask
import dask.array as da
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rioxarray  # noqa: F401
import xarray as xr
from joblib import Parallel, delayed

from ..constants import (
    DATE_COLUMN,
    DEFAULT_ERA5_LAND_STORE_PATH,
    DEFAULT_ERA5_LAND_TRENCH_DAY_PATH,
    DEFAULT_RIVER_NETWORK_PATH,
    ERA5_LAND_PREPROCESS_STAGES,
    ERA5_LAND_PREPROCESS_SUBTYPES,
    MONTH_COLUMN,
    TRENCH_ID_COLUMN,
    YEAR_COLUMN,
)
from ..fetch.common import (
    _timestamp,
    _worker_wait,
    climate_file_lock,
    load_download_manifest,
    lock_path_for,
    manifest_path_for,
    _wait_for_lock_release,
    write_download_manifest,
)
from land_cover.preprocess import deduplicate_drainage_polygons
from river_network import RiverNetwork


logger = logging.getLogger(__name__)


ERA5_OUTPUT_START = "1985-01-01"
ERA5_OUTPUT_END = "2024-12-31"
ERA5_OUTPUT_FREQ = "1D"
ERA5_OUTPUT_TIME_INDEX = pd.date_range(ERA5_OUTPUT_START, ERA5_OUTPUT_END, freq=ERA5_OUTPUT_FREQ)
ERA5_OUTPUT_DIMS = ("time", "latitude", "longitude")
ERA5_OUTPUT_CHUNKS = (365, -1, -1)
SUPPORTED_ERA5_PREPROCESS_SUBTYPES = ERA5_LAND_PREPROCESS_SUBTYPES
GEObOX_FILENAME = "geobox.pickle"
ERA5_FILENAME_PATTERN = re.compile(r"era5_land_(hourly|daily)_(?P<year>\d{4})_(?P<month>\d{2})\.grib$")
EARTHKIT_TEMP_DIR_PREFIX = "tmp"
EARTHKIT_TEMP_RETENTION_SECONDS = 6 * 3600
ERA5_TABULAR_TIME_CHUNK_DAYS = 7
ERA5_PARQUET_TIME_SLICE_DAYS = 7
ERA5_TRENCH_SUBBATCH_SIZE = 1000
ERA5_POINT_SELECTION_TOLERANCE_DEGREES = 0.1
TRENCH_CENTROID_AREA_CRS = "EPSG:5880"
DEFAULT_ERA5_PARQUET_N_JOBS = 2
ERA5_DAILY_VARIABLE_NAME_MAP = {
    "2m_dewpoint_temperature": "2d",
    "2m_temperature": "2t",
    "d2m": "2d",
    "t2m": "2t",
}

ERA5L_VAR_CONFIG = {
    "tp": {
        "name": "Total precipitation",
        "units_in": "m",
        "units_out": "mm/day",
        "aggregation": {
            "kind": "sum",
            "resample_freq": "1D",
            "skipna": False,
            "scale_factor": 1000.0,
            "offset": 0.0,
        },
        "preferred_spatial_role": "upstream_hydrology",
    },
    "sro": {
        "name": "Surface runoff",
        "units_in": "m",
        "units_out": "mm/day",
        "aggregation": {
            "kind": "sum",
            "resample_freq": "1D",
            "skipna": False,
            "scale_factor": 1000.0,
            "offset": 0.0,
        },
        "preferred_spatial_role": "upstream_hydrology",
    },
    "ssro": {
        "name": "Sub-surface runoff",
        "units_in": "m",
        "units_out": "mm/day",
        "aggregation": {
            "kind": "sum",
            "resample_freq": "1D",
            "skipna": False,
            "scale_factor": 1000.0,
            "offset": 0.0,
        },
        "preferred_spatial_role": "upstream_hydrology",
    },
    "pev": {
        "name": "Potential evaporation",
        "units_in": "m",
        "units_out": "mm/day",
        "aggregation": {
            "kind": "sum",
            "resample_freq": "1D",
            "skipna": False,
            "scale_factor": 1000.0,
            "offset": 0.0,
        },
        "preferred_spatial_role": "upstream_hydrology_or_water_balance",
    },
    "2t": {
        "name": "2 metre temperature",
        "units_in": "K",
        "units_out": "degC",
        "aggregation": {
            "kind": "mean",
            "resample_freq": "1D",
            "skipna": True,
            "scale_factor": 1.0,
            "offset": -273.15,
            "extras": {
                "daily_min": "min",
                "daily_max": "max",
            },
        },
        "preferred_spatial_role": "station_control",
    },
    "2d": {
        "name": "2 metre dewpoint temperature",
        "units_in": "K",
        "units_out": "degC",
        "aggregation": {
            "kind": "mean",
            "resample_freq": "1D",
            "skipna": True,
            "scale_factor": 1.0,
            "offset": -273.15,
        },
        "preferred_spatial_role": "station_control",
    },
    "swvl1": {
        "name": "Volumetric soil water layer 1",
        "units_in": "m3 m-3",
        "units_out": "m3 m-3",
        "aggregation": {
            "kind": "mean",
            "resample_freq": "1D",
            "skipna": True,
            "scale_factor": 1.0,
            "offset": 0.0,
        },
        "preferred_spatial_role": "upstream_wetness_state",
    },
    "swvl2": {
        "name": "Volumetric soil water layer 2",
        "units_in": "m3 m-3",
        "units_out": "m3 m-3",
        "aggregation": {
            "kind": "mean",
            "resample_freq": "1D",
            "skipna": True,
            "scale_factor": 1.0,
            "offset": 0.0,
        },
        "preferred_spatial_role": "upstream_wetness_state",
    },
}

def _root(root_dir=".") -> Path:
    return Path(root_dir)


def _era5_raw_dir(root_dir=".", subtype="era5_land_hourly") -> Path:
    return _root(root_dir) / "data" / "climate" / "raw" / subtype


def _era5_processed_dir(root_dir=".") -> Path:
    return _root(root_dir) / "data" / "climate" / "processed" / "era5_land"


def _era5_cache_dir(root_dir=".") -> Path:
    return _root(root_dir) / "data" / "climate" / "processed" / "cache_nobackup"


def _era5_store_path(root_dir=".") -> Path:
    return _root(root_dir) / DEFAULT_ERA5_LAND_STORE_PATH


def _era5_trench_day_path(root_dir=".") -> Path:
    return _root(root_dir) / DEFAULT_ERA5_LAND_TRENCH_DAY_PATH


def _era5_geobox_path(root_dir=".") -> Path:
    return _era5_raw_dir(root_dir, "era5_land_hourly") / GEObOX_FILENAME


def _river_network_path(root_dir=".") -> Path:
    return _root(root_dir) / DEFAULT_RIVER_NETWORK_PATH


def _dataset_name_for_subtype(subtype: str) -> str:
    if subtype == "era5_land_hourly":
        return "reanalysis-era5-land"
    if subtype == "era5_land_daily":
        return "derived-era5-land-daily-statistics"
    raise ValueError(f"Unsupported ERA5 preprocess subtype: {subtype}")


def _expected_output_var_attrs() -> dict[str, dict]:
    expected = {}
    for var_name, cfg in ERA5L_VAR_CONFIG.items():
        agg = cfg["aggregation"]
        expected[var_name] = {
            "long_name": cfg.get("name", var_name),
            "source_short_name": var_name,
            "units_original": cfg.get("units_in"),
            "units": cfg.get("units_out"),
            "preferred_spatial_role": cfg.get("preferred_spatial_role", ""),
            "aggregation_kind": agg.get("kind"),
            "aggregation_frequency": agg.get("resample_freq"),
            "aggregation_skipna": agg.get("skipna", True),
            "scale_factor_applied": agg.get("scale_factor", 1.0),
            "offset_applied": agg.get("offset", 0.0),
        }
        for extra_suffix, extra_kind in agg.get("extras", {}).items():
            expected[f"{var_name}_{extra_suffix}"] = {
                "long_name": f"{cfg.get('name', var_name)} {extra_suffix.replace('_', ' ')}",
                "source_short_name": var_name,
                "units_original": cfg.get("units_in"),
                "units": cfg.get("units_out"),
                "preferred_spatial_role": cfg.get("preferred_spatial_role", ""),
                "aggregation_kind": extra_kind,
                "aggregation_frequency": agg.get("resample_freq"),
                "aggregation_skipna": agg.get("skipna", True),
                "scale_factor_applied": agg.get("scale_factor", 1.0),
                "offset_applied": agg.get("offset", 0.0),
            }
    return expected


def _available_era5_output_variables(dataset: xr.Dataset) -> list[str]:
    expected = set(_expected_output_var_attrs())
    return [var_name for var_name in dataset.data_vars if var_name in expected]


def _load_drainage_polygons(root_dir=".", geobox_state: dict | None = None):
    network_path = _river_network_path(root_dir)
    logger.info("Loading river network drainage areas from %s", network_path)
    network = RiverNetwork()
    network.load(str(network_path))
    if network.drainage_areas is None:
        raise ValueError("River network must include drainage_areas for climate tabularization.")

    drainage_polygons = network.drainage_areas.to_crs(4326)
    if "within_brazil" not in drainage_polygons.columns:
        raise ValueError(
            "Drainage areas missing 'within_brazil' column. "
            "Run river-network generate with --gadm-path to annotate this column."
        )

    if geobox_state is None:
        geobox_state = load_or_create_geobox_state(root_dir=root_dir)

    centroid_points = (
        drainage_polygons
        .to_crs(TRENCH_CENTROID_AREA_CRS)
        .geometry
        .centroid
        .to_crs(4326)
    )
    within_geobox = centroid_points.within(
        geobox_state["geobox"].extent.geom.buffer(0.01)
    )
    drainage_polygons = drainage_polygons[
        drainage_polygons["within_brazil"] & within_geobox
    ]
    drainage_polygons = deduplicate_drainage_polygons(drainage_polygons).reset_index(drop=True)
    return drainage_polygons[[TRENCH_ID_COLUMN, "geometry"]].copy().reset_index(drop=True)


def _empty_trench_day_frame(climate_columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=[TRENCH_ID_COLUMN, DATE_COLUMN, *climate_columns])


def _trench_centroid_coordinates(drainage_polygons: pd.DataFrame) -> pd.DataFrame:
    centroid_points = (
        drainage_polygons
        .to_crs(TRENCH_CENTROID_AREA_CRS)
        .geometry
        .centroid
        .to_crs(4326)
    )
    coordinates = centroid_points.get_coordinates().reset_index(drop=True)
    trench_coordinates = pd.DataFrame(
        {
            TRENCH_ID_COLUMN: drainage_polygons[TRENCH_ID_COLUMN].to_numpy(dtype=np.int64),
            "latitude": coordinates["y"].to_numpy(dtype=np.float64),
            "longitude": coordinates["x"].to_numpy(dtype=np.float64),
        }
    )
    return (
        trench_coordinates.dropna(subset=["latitude", "longitude"])
        .drop_duplicates(subset=[TRENCH_ID_COLUMN], keep="first")
        .sort_values(TRENCH_ID_COLUMN)
        .reset_index(drop=True)
    )


def _time_windows(total_size: int, window_size: int):
    for start in range(0, total_size, window_size):
        yield start, min(start + window_size, total_size)


def _chunk_layout(dataset: xr.Dataset) -> dict[str, tuple[int, ...]]:
    chunk_map = getattr(dataset, "chunks", None) or getattr(dataset, "chunksizes", None)
    if chunk_map is None:
        raise ValueError("ERA5-Land tabularization requires chunked xarray data.")
    return {name: tuple(int(size) for size in sizes) for name, sizes in chunk_map.items()}


def _effective_era5_parquet_n_jobs(n_jobs: int | None) -> int:
    if n_jobs is None:
        return max(1, min(os.cpu_count() or 1, DEFAULT_ERA5_PARQUET_N_JOBS))
    return max(1, int(n_jobs))


def _time_chunk_windows(dataset: xr.Dataset) -> list[tuple[int, int]]:
    chunk_layout = _chunk_layout(dataset)
    total_time = dataset.sizes["time"]
    source_time_chunk_size = chunk_layout["time"][0]
    return list(_time_windows(total_time, source_time_chunk_size))


def _extract_trench_subbatch(
    climate_chunk: xr.Dataset,
    *,
    trench_batch: pd.DataFrame,
    climate_columns: list[str],
) -> pd.DataFrame:
    trench_ids = trench_batch[TRENCH_ID_COLUMN].to_numpy(dtype=np.int64)
    latitudes = xr.DataArray(
        trench_batch["latitude"].to_numpy(dtype=np.float64),
        dims=("trench",),
        coords={TRENCH_ID_COLUMN: ("trench", trench_ids)},
    )
    longitudes = xr.DataArray(
        trench_batch["longitude"].to_numpy(dtype=np.float64),
        dims=("trench",),
        coords={TRENCH_ID_COLUMN: ("trench", trench_ids)},
    )
    selected = climate_chunk[climate_columns].sel(
        latitude=latitudes,
        longitude=longitudes,
        method="nearest",
        tolerance=ERA5_POINT_SELECTION_TOLERANCE_DEGREES,
    )
    frame = selected.to_dataframe().reset_index()
    if frame.empty:
        return _empty_trench_day_frame(climate_columns)

    if "time" not in frame.columns:
        raise ValueError("Expected `time` column after ERA5-Land trench extraction.")

    frame = frame.rename(columns={"time": DATE_COLUMN})
    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], errors="coerce").dt.normalize()
    frame[TRENCH_ID_COLUMN] = frame[TRENCH_ID_COLUMN].astype(np.int64, copy=False)
    frame = frame.drop(columns=["trench", "latitude", "longitude"], errors="ignore")
    return frame[[TRENCH_ID_COLUMN, DATE_COLUMN, *climate_columns]]


def _extract_time_slice(
    climate_chunk: xr.Dataset,
    *,
    time_start: int,
    time_stop: int,
    trench_coordinates: pd.DataFrame,
    climate_columns: list[str],
) -> pd.DataFrame:
    logger.debug(
        "Selecting ERA5-Land time slice %s:%s within loaded chunk of %s day(s)",
        time_start,
        time_stop,
        climate_chunk.sizes["time"],
    )
    chunk_frames = []
    n_trenches = len(trench_coordinates)
    climate_slice = climate_chunk.isel(time=slice(time_start, time_stop))
    for batch_start, batch_stop in _time_windows(n_trenches, ERA5_TRENCH_SUBBATCH_SIZE):
        logger.debug(
            "Selecting ERA5-Land trench subbatch %s:%s for days %s:%s",
            batch_start,
            batch_stop,
            time_start,
            time_stop,
        )
        trench_batch = trench_coordinates.iloc[batch_start:batch_stop].reset_index(drop=True)
        chunk_frames.append(
            _extract_trench_subbatch(
                climate_slice,
                trench_batch=trench_batch,
                climate_columns=climate_columns,
            )
        )

    if not chunk_frames:
        return _empty_trench_day_frame(climate_columns)

    return (
        pd.concat(chunk_frames, ignore_index=True)
        .sort_values([TRENCH_ID_COLUMN, DATE_COLUMN], kind="mergesort")
        .reset_index(drop=True)
    )


def _write_time_chunk_parts(
    dataset: xr.Dataset,
    *,
    time_start: int,
    time_stop: int,
    trench_coordinates: pd.DataFrame,
    climate_columns: list[str],
    output_dir: Path,
) -> int:
    logger.info(
        "Selecting ERA5-Land time chunk %s:%s of %s",
        time_start,
        time_stop,
        dataset.sizes["time"],
    )
    with dask.config.set(scheduler="single-threaded"):
        climate_chunk = dataset.isel(time=slice(time_start, time_stop)).load()

    parts_written = 0
    try:
        for local_start, local_stop in _time_windows(
            climate_chunk.sizes["time"],
            ERA5_PARQUET_TIME_SLICE_DAYS,
        ):
            global_start = time_start + local_start
            global_stop = time_start + local_stop
            frame = _extract_time_slice(
                climate_chunk,
                time_start=local_start,
                time_stop=local_stop,
                trench_coordinates=trench_coordinates,
                climate_columns=climate_columns,
            )
            if frame.empty:
                del frame
                continue

            _write_partitioned_time_slice_frame(
                frame,
                output_dir=output_dir,
                basename=f"part-{global_start:05d}-{global_stop:05d}.parquet",
            )
            parts_written += 1
            del frame
            gc.collect()
    finally:
        close = getattr(climate_chunk, "close", None)
        if callable(close):
            close()
        del climate_chunk
        gc.collect()

    return parts_written


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _partition_path(output_dir: Path, *, year: int, month: int) -> Path:
    return output_dir / f"{YEAR_COLUMN}={year:04d}" / f"{MONTH_COLUMN}={month:02d}"


def _write_partitioned_time_slice_frame(
    frame: pd.DataFrame,
    *,
    output_dir: Path,
    basename: str,
) -> None:
    partition_index = pd.DataFrame(
        {
            YEAR_COLUMN: frame[DATE_COLUMN].dt.year.to_numpy(dtype=np.int16, copy=False),
            MONTH_COLUMN: frame[DATE_COLUMN].dt.month.to_numpy(dtype=np.int8, copy=False),
        }
    )
    group_indices = partition_index.groupby([YEAR_COLUMN, MONTH_COLUMN], sort=False).indices
    for (year, month), indices in group_indices.items():
        partition_dir = _partition_path(output_dir, year=int(year), month=int(month))
        partition_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pandas(frame.iloc[indices].copy(), preserve_index=False),
            partition_dir / basename,
        )


def _write_chunked_trench_day_table(
    dataset: xr.Dataset,
    *,
    climate_columns: list[str],
    trench_coordinates: pd.DataFrame,
    output_path: Path,
    n_jobs: int | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    _remove_path(temp_path)
    temp_path.mkdir(parents=True, exist_ok=True)

    effective_n_jobs = _effective_era5_parquet_n_jobs(n_jobs)
    chunk_windows = _time_chunk_windows(dataset)
    logger.info(
        "Extracting %d ERA5-Land time chunk(s) with %d worker(s).",
        len(chunk_windows),
        effective_n_jobs,
    )

    if effective_n_jobs == 1:
        parts_written = sum(
            _write_time_chunk_parts(
                dataset,
                time_start=chunk_start,
                time_stop=chunk_stop,
                trench_coordinates=trench_coordinates,
                climate_columns=climate_columns,
                output_dir=temp_path,
            )
            for chunk_start, chunk_stop in chunk_windows
        )
    else:
        parts_written = sum(
            Parallel(
                n_jobs=effective_n_jobs,
                backend="threading",
                pre_dispatch=effective_n_jobs,
                batch_size=1,
            )(
                delayed(_write_time_chunk_parts)(
                    dataset,
                    time_start=chunk_start,
                    time_stop=chunk_stop,
                    trench_coordinates=trench_coordinates,
                    climate_columns=climate_columns,
                    output_dir=temp_path,
                )
                for chunk_start, chunk_stop in chunk_windows
            )
        )

    if parts_written == 0:
        empty = _empty_trench_day_frame(climate_columns)
        pq.write_table(
            pa.Table.from_pandas(empty, preserve_index=False),
            temp_path / "part-00000.parquet",
        )

    _remove_path(output_path)
    temp_path.replace(output_path)


def tabularize_era5_land_by_trench(root_dir=".", n_jobs: int | None = None) -> Path:
    store_path = _era5_store_path(root_dir)
    if not store_path.exists():
        raise FileNotFoundError(f"Processed ERA5-Land store not found at {store_path}.")

    geobox_state = load_or_create_geobox_state(root_dir=root_dir)
    drainage_polygons = _load_drainage_polygons(root_dir, geobox_state=geobox_state)
    logger.info("Opening processed ERA5-Land store for trench/day tabularization: %s", store_path)
    dataset = xr.open_zarr(
        store_path,
        consolidated=False,
    )
    try:
        climate_columns = _available_era5_output_variables(dataset)
        if not climate_columns:
            raise ValueError("Processed ERA5-Land store does not contain any supported variables.")

        trench_coordinates = _trench_centroid_coordinates(drainage_polygons)
        omitted_trenches = len(drainage_polygons) - len(trench_coordinates)
        logger.info(
            "Prepared centroid coordinates for %s/%s drainage polygons; omitting %s trench(es) without valid coordinates",
            len(trench_coordinates),
            len(drainage_polygons),
            omitted_trenches,
        )
        output_path = _era5_trench_day_path(root_dir)
        _write_chunked_trench_day_table(
            dataset[climate_columns],
            climate_columns=climate_columns,
            trench_coordinates=trench_coordinates,
            output_path=output_path,
            n_jobs=n_jobs,
        )
    finally:
        close = getattr(dataset, "close", None)
        if callable(close):
            close()

    logger.info("Saved ERA5-Land trench/day table to %s", output_path)
    return output_path


def _time_index() -> pd.DatetimeIndex:
    return ERA5_OUTPUT_TIME_INDEX


def _normalize_time_index(values) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(values)).tz_localize(None).normalize()


def _file_sort_key(path: Path) -> tuple[int, int]:
    match = ERA5_FILENAME_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unrecognized ERA5 filename: {path.name}")
    return int(match.group("year")), int(match.group("month"))


def discover_era5_input_files(root_dir=".", subtype="era5_land_hourly") -> list[Path]:
    if subtype not in SUPPORTED_ERA5_PREPROCESS_SUBTYPES:
        raise ValueError(f"Unsupported ERA5 preprocess subtype: {subtype}")

    raw_dir = _era5_raw_dir(root_dir=root_dir, subtype=subtype)
    return sorted(
        [
            path
            for path in raw_dir.glob("*.grib")
            if ERA5_FILENAME_PATTERN.fullmatch(path.name) is not None
        ],
        key=_file_sort_key,
    )


def _load_manifest_or_placeholder(path: Path, subtype: str) -> dict:
    manifest = load_download_manifest(path)
    if manifest is not None:
        return manifest
    return {
        "target_path": str(path),
        "dataset": _dataset_name_for_subtype(subtype),
        "request": None,
        "download_status": "downloaded",
        "status": "downloaded",
        "error": None,
    }


def _write_preprocess_manifest(
    target_path: Path,
    *,
    subtype: str,
    base_manifest: dict,
    preprocess_status: str,
    store_path: Path | None = None,
    raw_deleted: bool | None = None,
    error: str | None = None,
    **extra_fields,
) -> Path:
    payload = dict(base_manifest)
    payload["download_status"] = payload.get("download_status", payload.get("status"))
    payload["preprocess_subtype"] = subtype
    payload["preprocess_status"] = preprocess_status
    payload["preprocess_updated_at"] = _timestamp()
    payload["status"] = {
        "processing": "preprocessing",
        "processed": "processed",
        "failed": "preprocess_failed",
    }[preprocess_status]
    payload["error"] = error
    if store_path is not None:
        payload["processed_store_path"] = str(store_path)
    if preprocess_status == "processed":
        payload["preprocessed_at"] = payload["preprocess_updated_at"]
    if raw_deleted is not None:
        payload["raw_deleted"] = raw_deleted
        if raw_deleted:
            payload["raw_deleted_at"] = payload["preprocess_updated_at"]
    payload.update(extra_fields)
    manifest_path = manifest_path_for(target_path)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest_path


def _extract_geobox_from_dataset(dataset: xr.Dataset):
    return dataset.odc.geobox


def _extract_spatial_ref_from_dataset(dataset: xr.Dataset):
    if "spatial_ref" not in dataset.coords:
        return None
    return dataset.coords["spatial_ref"].values


def _coord_values(values) -> np.ndarray:
    if hasattr(values, "values"):
        return np.asarray(values.values)
    return np.asarray(values)


def _build_geobox_state(geobox, spatial_ref=None) -> dict:
    latitude = _coord_values(geobox.coords["latitude"])
    longitude = _coord_values(geobox.coords["longitude"])
    return {
        "geobox": geobox,
        "latitude": latitude,
        "longitude": longitude,
        "spatial_ref": spatial_ref,
        "shape": (len(latitude), len(longitude)),
    }


def _load_geobox_state(path: Path) -> dict:
    with path.open("rb") as handle:
        payload = pickle.load(handle)

    if isinstance(payload, dict):
        return payload
    return _build_geobox_state(payload)


def _save_geobox_state(path: Path, geobox_state: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(geobox_state, handle)
    return path


def _metadata_column(metadata: pd.DataFrame, *candidates: str) -> pd.Series | None:
    for name in candidates:
        if name in metadata.columns:
            return metadata[name]
    return None


def _metadata_time_offsets(metadata: pd.DataFrame) -> pd.Series:
    data_time = _metadata_column(metadata, "dataTime", "validityTime")
    hour_offsets = pd.Series(0, index=metadata.index, dtype="int64")

    if data_time is not None:
        hour_offsets = data_time.fillna(0).astype(int).floordiv(100)

    step_range = _metadata_column(metadata, "stepRange")
    if step_range is not None:
        step_offsets = (
            step_range.astype(str).str.split("-").str[-1].replace({"nan": "0"}).astype(int)
        )
        hour_offsets = hour_offsets + step_offsets

    return hour_offsets


def _field_valid_datetimes(metadata: pd.DataFrame) -> pd.Series:
    valid_datetime = _metadata_column(metadata, "valid_datetime")
    if valid_datetime is not None:
        return pd.to_datetime(valid_datetime)

    base_date = _metadata_column(metadata, "validityDate", "dataDate", "date")
    if base_date is None:
        raise KeyError(
            "Could not derive datetimes from ERA5 metadata because no date column was found."
        )

    hour_offsets = _metadata_time_offsets(metadata)
    base_dates = pd.to_datetime(base_date.astype(str), format="%Y%m%d")
    return base_dates + pd.to_timedelta(hour_offsets, unit="h")


def _field_band_names(metadata: pd.DataFrame) -> pd.Series:
    band_names = _metadata_column(metadata, "shortName", "variable")
    if band_names is None:
        raise KeyError(
            "Could not derive variable names from ERA5 metadata because no variable column was found."
        )
    return band_names.astype(str).replace(ERA5_DAILY_VARIABLE_NAME_MAP)


def _looks_like_earthkit_temp_dir(path: Path) -> bool:
    if not path.is_dir() or not path.name.startswith(EARTHKIT_TEMP_DIR_PREFIX):
        return False
    try:
        for child in path.iterdir():
            if child.is_dir() and child.name.startswith("file-") and child.name.endswith(".d"):
                return True
    except PermissionError:
        return False
    return False


def _cleanup_stale_earthkit_temp_dirs(*, min_age_seconds=EARTHKIT_TEMP_RETENTION_SECONDS) -> int:
    temp_root = Path(tempfile.gettempdir())
    now = pd.Timestamp.utcnow().timestamp()
    removed = 0
    for candidate in temp_root.iterdir():
        if not _looks_like_earthkit_temp_dir(candidate):
            continue
        try:
            age_seconds = now - candidate.stat().st_mtime
        except (FileNotFoundError, PermissionError):
            continue
        if age_seconds < min_age_seconds:
            continue
        try:
            shutil.rmtree(candidate)
            removed += 1
        except (PermissionError, FileNotFoundError, OSError):
            continue
    if removed:
        logger.info(
            "Removed %s stale Earthkit temp director%s.",
            removed,
            "y" if removed == 1 else "ies",
        )
    return removed


def _open_era5_dataset(path: Path) -> xr.Dataset:
    import earthkit.data as ekd

    _cleanup_stale_earthkit_temp_dirs()
    field_list = ekd.from_source("file", str(path))
    try:
        metadata = field_list.ls().copy()
        metadata["datetime"] = _field_valid_datetimes(metadata)
        metadata["band_name"] = _field_band_names(metadata)

        bands = pd.Index(metadata["band_name"]).drop_duplicates().tolist()
        times = pd.Index(metadata["datetime"]).drop_duplicates().tolist()
        latitude = np.asarray(field_list.data(keys="lat")[0, :, 0])
        longitude = np.asarray(field_list.data(keys="lon")[0, 0, :])
        values = np.asarray(field_list.to_numpy())

        expected_fields = len(bands) * len(times)
        if values.shape[0] != expected_fields:
            raise ValueError(
                "Unexpected ERA5 field count while reshaping notebook-style GRIB data: "
                f"{values.shape[0]} != {len(bands)} x {len(times)}."
            )

        dataset = xr.DataArray(
            values.reshape(len(bands), len(times), len(latitude), len(longitude)),
            dims=["band", "time", "latitude", "longitude"],
            coords={
                "band": bands,
                "time": pd.DatetimeIndex(times),
                "latitude": latitude,
                "longitude": longitude,
            },
            attrs={"source_path": str(path)},
        ).to_dataset(dim="band")
    finally:
        close = getattr(field_list, "close", None)
        if callable(close):
            close()
        del field_list
        gc.collect()
        _cleanup_stale_earthkit_temp_dirs(min_age_seconds=0)

    return dataset


def _close_dataset(dataset: xr.Dataset) -> None:
    close = getattr(dataset, "close", None)
    if callable(close):
        close()


def load_or_create_geobox_state(root_dir=".", sample_path: Path | None = None) -> dict:
    geobox_path = _era5_geobox_path(root_dir)
    if geobox_path.exists():
        return _load_geobox_state(geobox_path)

    store_path = _era5_store_path(root_dir)
    if sample_path is None and store_path.exists():
        dataset = xr.open_zarr(store_path, consolidated=False)
        try:
            geobox_state = _build_geobox_state(
                _extract_geobox_from_dataset(dataset),
                spatial_ref=_extract_spatial_ref_from_dataset(dataset),
            )
        finally:
            _close_dataset(dataset)
        _save_geobox_state(geobox_path, geobox_state)
        return geobox_state

    if sample_path is None:
        hourly_files = discover_era5_input_files(root_dir=root_dir, subtype="era5_land_hourly")
        daily_files = discover_era5_input_files(root_dir=root_dir, subtype="era5_land_daily")
        candidate_files = hourly_files or daily_files
        if not candidate_files:
            raise FileNotFoundError("No ERA5 GRIB files found to derive the geobox.")
        sample_path = candidate_files[0]

    dataset = _open_era5_dataset(sample_path)
    try:
        geobox_state = _build_geobox_state(
            _extract_geobox_from_dataset(dataset),
            spatial_ref=_extract_spatial_ref_from_dataset(dataset),
        )
    finally:
        _close_dataset(dataset)
    _save_geobox_state(geobox_path, geobox_state)
    return geobox_state


def _base_store_dataset(geobox_state: dict) -> xr.Dataset:
    base = xr.Dataset(
        coords={
            "time": _time_index(),
            "latitude": geobox_state["latitude"],
            "longitude": geobox_state["longitude"],
        }
    )
    if geobox_state.get("spatial_ref") is not None:
        base = base.assign_coords(spatial_ref=geobox_state["spatial_ref"])
    return base


def _store_region_for_dataset(dataset: xr.Dataset) -> dict[str, slice]:
    time_values = _normalize_time_index(dataset.indexes["time"])
    full_time_index = _time_index()
    start = full_time_index.get_loc(time_values[0])
    end = full_time_index.get_loc(time_values[-1]) + 1
    return {
        "time": slice(start, end),
        "latitude": slice(0, dataset.sizes["latitude"]),
        "longitude": slice(0, dataset.sizes["longitude"]),
    }


def _missing_store_variables(store_path: Path) -> set[str]:
    if not store_path.exists():
        return set(_expected_output_var_attrs())

    opened = xr.open_zarr(store_path, consolidated=False)
    try:
        existing = set(opened.data_vars)
    finally:
        close = getattr(opened, "close", None)
        if callable(close):
            close()
    return set(_expected_output_var_attrs()) - existing


def bootstrap_era5_store(root_dir=".", sample_path: Path | None = None) -> Path:
    store_path = _era5_store_path(root_dir)
    geobox_state = load_or_create_geobox_state(root_dir=root_dir, sample_path=sample_path)
    base = _base_store_dataset(geobox_state)

    if not store_path.exists():
        store_path.parent.mkdir(parents=True, exist_ok=True)
        base.to_zarr(
            store_path,
            mode="w",
            compute=False,
            zarr_format=3,
            consolidated=False,
        )

    missing_vars = _missing_store_variables(store_path)
    if not missing_vars:
        return store_path

    shape = (len(base.time), len(base.latitude), len(base.longitude))
    chunks = (
        ERA5_OUTPUT_CHUNKS[0],
        len(base.latitude) if ERA5_OUTPUT_CHUNKS[1] == -1 else ERA5_OUTPUT_CHUNKS[1],
        len(base.longitude) if ERA5_OUTPUT_CHUNKS[2] == -1 else ERA5_OUTPUT_CHUNKS[2],
    )
    expected_attrs = _expected_output_var_attrs()

    for var_name in sorted(missing_vars):
        var_da = xr.DataArray(
            da.empty(shape, dtype=np.float32, chunks=chunks),
            dims=ERA5_OUTPUT_DIMS,
            coords=base.coords,
            attrs=expected_attrs[var_name],
        )
        xr.Dataset({var_name: var_da}).to_zarr(
            store_path,
            mode="a",
            compute=False,
            zarr_format=3,
            consolidated=False,
        )

    return store_path


def _reduce_resample(da: xr.DataArray, freq: str, kind: str, skipna: bool) -> xr.DataArray:
    rs = da.resample(time=freq)

    if kind == "sum":
        return rs.sum(skipna=skipna, keep_attrs=True)
    if kind == "mean":
        return rs.mean(skipna=skipna, keep_attrs=True)
    if kind == "min":
        return rs.min(skipna=skipna, keep_attrs=True)
    if kind == "max":
        return rs.max(skipna=skipna, keep_attrs=True)

    raise ValueError(f"Unsupported aggregation kind: {kind}")


def _rename_dataset_dims(dataset: xr.Dataset) -> xr.Dataset:
    rename_map = {}
    for old_name, new_name in (
        ("lat", "latitude"),
        ("lon", "longitude"),
        ("x", "longitude"),
        ("y", "latitude"),
        ("valid_time", "time"),
    ):
        if old_name in dataset.dims or old_name in dataset.coords:
            rename_map[old_name] = new_name
    return dataset.rename(rename_map) if rename_map else dataset


def _ensure_store_shape(dataset: xr.Dataset, geobox_state: dict) -> xr.Dataset:
    dataset = dataset.assign_coords(
        latitude=geobox_state["latitude"],
        longitude=geobox_state["longitude"],
    )
    return dataset.transpose("time", "latitude", "longitude")


def _configured_dataset(dataset: xr.Dataset) -> xr.Dataset:
    return dataset[[var for var in ERA5L_VAR_CONFIG if var in dataset.data_vars]]


def resample_era5l_hourly_to_daily(ds: xr.Dataset, var_config: dict) -> xr.Dataset:
    ds = _rename_dataset_dims(ds)
    ds = _configured_dataset(ds)
    out = xr.Dataset(attrs=dict(ds.attrs))
    out.attrs.update(
        {
            "temporal_aggregation": "daily",
            "aggregation_engine": "xarray.resample",
            "aggregation_note": (
                "Accumulation variables aggregated by daily sum; "
                "instantaneous/state variables aggregated by daily mean unless configured otherwise."
            ),
            "time_label": "left",
            "time_closed": "left",
        }
    )

    for var_name, cfg in var_config.items():
        if var_name not in ds.data_vars:
            continue

        da_in = ds[var_name]
        agg = cfg["aggregation"]
        kind = agg["kind"]
        freq = agg["resample_freq"]
        skipna = agg.get("skipna", True)
        scale = agg.get("scale_factor", 1.0)
        offset = agg.get("offset", 0.0)

        daily = _reduce_resample(da_in, freq=freq, kind=kind, skipna=skipna)
        if scale != 1.0 or offset != 0.0:
            daily = daily * scale + offset

        daily.attrs = _expected_output_var_attrs()[var_name]
        out[var_name] = daily.astype(np.float32)

        for extra_suffix, extra_kind in agg.get("extras", {}).items():
            extra_name = f"{var_name}_{extra_suffix}"
            extra = _reduce_resample(da_in, freq=freq, kind=extra_kind, skipna=skipna)
            if scale != 1.0 or offset != 0.0:
                extra = extra * scale + offset
            extra.attrs = _expected_output_var_attrs()[extra_name]
            out[extra_name] = extra.astype(np.float32)

    return out


def prepare_daily_era5_dataset(ds: xr.Dataset) -> xr.Dataset:
    ds = _rename_dataset_dims(ds)
    ds = _configured_dataset(ds)
    out = xr.Dataset(attrs=dict(ds.attrs))
    expected_attrs = _expected_output_var_attrs()

    for var_name, cfg in ERA5L_VAR_CONFIG.items():
        if var_name not in ds.data_vars:
            continue

        da_in = ds[var_name]
        agg = cfg["aggregation"]
        scale = agg.get("scale_factor", 1.0)
        offset = agg.get("offset", 0.0)
        prepared = da_in
        if scale != 1.0 or offset != 0.0:
            prepared = prepared * scale + offset
        prepared.attrs = expected_attrs[var_name]
        out[var_name] = prepared.astype(np.float32)

    return out


def _prepare_dataset_for_store(
    dataset: xr.Dataset,
    subtype: str,
    geobox_state: dict,
) -> xr.Dataset:
    if subtype == "era5_land_hourly":
        prepared = resample_era5l_hourly_to_daily(dataset, ERA5L_VAR_CONFIG)
    elif subtype == "era5_land_daily":
        prepared = prepare_daily_era5_dataset(dataset)
    else:
        raise ValueError(f"Unsupported ERA5 preprocess subtype: {subtype}")

    prepared = _ensure_store_shape(prepared, geobox_state)
    prepared = prepared.assign_coords(time=_normalize_time_index(prepared.indexes["time"]))
    return prepared.sortby("time")


def write_dataset_region(dataset: xr.Dataset, store_path: Path) -> Path:
    if not dataset.data_vars:
        return store_path

    dataset.to_zarr(
        store_path,
        mode="r+",
        region=_store_region_for_dataset(dataset),
        zarr_format=3,
        consolidated=False,
    )
    return store_path


def _delete_raw_input_file(path: Path) -> None:
    if path.exists():
        path.unlink()


def _manifest_download_is_active(manifest: dict | None) -> bool:
    if manifest is None:
        return False
    download_status = manifest.get("download_status", manifest.get("status"))
    return download_status in {"submitted", "downloading"}


def _manifest_ready_for_preprocess(manifest: dict | None) -> bool:
    if manifest is None:
        return True
    download_status = manifest.get("download_status", manifest.get("status"))
    return download_status == "downloaded" and manifest.get("preprocess_status") != "processed"


def _candidate_manifest_paths(root_dir=".", subtype="era5_land_hourly") -> list[Path]:
    raw_dir = _era5_raw_dir(root_dir=root_dir, subtype=subtype)
    return sorted(raw_dir.glob("*.grib.manifest.json"))


def _active_download_requests_exist(root_dir=".", subtype="era5_land_hourly") -> bool:
    for manifest_path in _candidate_manifest_paths(root_dir=root_dir, subtype=subtype):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _manifest_download_is_active(manifest):
            return True
    return False


def process_era5_input_file(path: Path, *, root_dir=".", subtype="era5_land_hourly") -> Path:
    if subtype not in SUPPORTED_ERA5_PREPROCESS_SUBTYPES:
        raise ValueError(f"Unsupported ERA5 preprocess subtype: {subtype}")

    logger.info("Processing ERA5 GRIB %s for subtype %s.", path.name, subtype)
    with climate_file_lock(path, owner="climate_preprocess_worker"):
        base_manifest = _load_manifest_or_placeholder(path, subtype)
        store_path = bootstrap_era5_store(root_dir=root_dir, sample_path=path)
        geobox_state = load_or_create_geobox_state(root_dir=root_dir, sample_path=path)

        _write_preprocess_manifest(
            path,
            subtype=subtype,
            base_manifest=base_manifest,
            preprocess_status="processing",
            store_path=store_path,
            raw_deleted=False,
            file_lock=str(lock_path_for(path)),
        )

        dataset = _open_era5_dataset(path)
        try:
            try:
                prepared = _prepare_dataset_for_store(dataset, subtype=subtype, geobox_state=geobox_state)
                write_dataset_region(prepared, store_path)
            finally:
                _close_dataset(dataset)
        except Exception as exc:
            _write_preprocess_manifest(
                path,
                subtype=subtype,
                base_manifest=base_manifest,
                preprocess_status="failed",
                store_path=store_path,
                raw_deleted=False,
                error=str(exc),
                file_lock=str(lock_path_for(path)),
            )
            raise

        _delete_raw_input_file(path)
        _write_preprocess_manifest(
            path,
            subtype=subtype,
            base_manifest=base_manifest,
            preprocess_status="processed",
            store_path=store_path,
            raw_deleted=True,
            error=None,
        )
        return store_path


def preprocess_era5_land(root_dir=".", subtype="era5_land_hourly", stage="all", n_jobs: int | None = None) -> Path:
    if subtype not in SUPPORTED_ERA5_PREPROCESS_SUBTYPES:
        raise ValueError(f"Unsupported ERA5 preprocess subtype: {subtype}")
    if stage not in ERA5_LAND_PREPROCESS_STAGES:
        raise ValueError(
            f"Unsupported ERA5 preprocess stage: {stage}. "
            f"Available stages: {sorted(ERA5_LAND_PREPROCESS_STAGES)}"
        )

    if stage in {"all", "zarr"}:
        input_files = discover_era5_input_files(root_dir=root_dir, subtype=subtype)
        if not input_files:
            raise FileNotFoundError(f"No ERA5 GRIB files found for subtype {subtype!r}.")

        store_path = bootstrap_era5_store(root_dir=root_dir, sample_path=input_files[0])
        for input_file in input_files:
            store_path = process_era5_input_file(input_file, root_dir=root_dir, subtype=subtype)
    else:
        store_path = _era5_store_path(root_dir)

    if stage == "zarr":
        return store_path

    if not store_path.exists():
        raise FileNotFoundError(f"Processed ERA5-Land store not found at {store_path}.")
    return tabularize_era5_land_by_trench(root_dir=root_dir, n_jobs=n_jobs)


def preprocess_era5_land_worker(
    root_dir=".",
    subtype="era5_land_hourly",
    poll_seconds=120,
    stage="all",
    n_jobs: int | None = None,
) -> Path:
    if subtype not in SUPPORTED_ERA5_PREPROCESS_SUBTYPES:
        raise ValueError(f"Unsupported ERA5 preprocess subtype: {subtype}")
    if stage not in ERA5_LAND_PREPROCESS_STAGES:
        raise ValueError(
            f"Unsupported ERA5 preprocess stage: {stage}. "
            f"Available stages: {sorted(ERA5_LAND_PREPROCESS_STAGES)}"
        )

    if stage == "parquet":
        return tabularize_era5_land_by_trench(root_dir=root_dir, n_jobs=n_jobs)

    last_store_path = _era5_store_path(root_dir)
    while True:
        ready_files = []
        for path in discover_era5_input_files(root_dir=root_dir, subtype=subtype):
            _wait_for_lock_release(path)
            manifest = load_download_manifest(path)
            if _manifest_ready_for_preprocess(manifest):
                ready_files.append(path)

        if ready_files:
            bootstrap_era5_store(root_dir=root_dir, sample_path=ready_files[0])
            for path in ready_files:
                last_store_path = process_era5_input_file(path, root_dir=root_dir, subtype=subtype)
            continue

        if not _active_download_requests_exist(root_dir=root_dir, subtype=subtype):
            if last_store_path.exists():
                if stage == "zarr":
                    return last_store_path
                return tabularize_era5_land_by_trench(root_dir=root_dir, n_jobs=n_jobs)
            raise FileNotFoundError(
                f"No downloaded or active ERA5 files found for subtype {subtype!r}."
            )

        _worker_wait(poll_seconds)
