from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_ROOT = ROOT / "code" / "data"
if str(DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ROOT))

from climate.preprocess.era5_land import (
    ERA5L_VAR_CONFIG,
    ERA5_OUTPUT_TIME_INDEX,
    bootstrap_era5_store,
    load_or_create_geobox_state,
    prepare_daily_era5_dataset,
    preprocess_era5_land_worker,
    preprocess_era5_land,
    resample_era5l_hourly_to_daily,
    write_dataset_region,
)
from climate.fetch.common import load_download_manifest, write_download_manifest


class _FakeCoord:
    def __init__(self, values):
        self.values = np.asarray(values)


class FakeGeoBox:
    def __init__(self, latitude, longitude):
        self.coords = {
            "latitude": _FakeCoord(latitude),
            "longitude": _FakeCoord(longitude),
        }


def _hourly_dataset() -> xr.Dataset:
    time = pd.date_range("1985-01-01", periods=48, freq="1h")
    lat = np.array([-10.0, -11.0], dtype=np.float64)
    lon = np.array([-50.0, -49.0], dtype=np.float64)
    base = np.arange(48 * 2 * 2, dtype=np.float32).reshape(48, 2, 2)

    return xr.Dataset(
        data_vars={
            "tp": (("time", "latitude", "longitude"), np.ones((48, 2, 2), dtype=np.float32) * 0.001),
            "2t": (("time", "latitude", "longitude"), 273.15 + base),
            "2d": (("time", "latitude", "longitude"), 270.15 + base),
            "swvl1": (("time", "latitude", "longitude"), np.ones((48, 2, 2), dtype=np.float32) * 0.2),
            "swvl2": (("time", "latitude", "longitude"), np.ones((48, 2, 2), dtype=np.float32) * 0.4),
        },
        coords={"time": time, "latitude": lat, "longitude": lon},
    )


def _daily_dataset() -> xr.Dataset:
    time = pd.date_range("1985-02-01", periods=3, freq="1D")
    lat = np.array([-10.0, -11.0], dtype=np.float64)
    lon = np.array([-50.0, -49.0], dtype=np.float64)
    return xr.Dataset(
        data_vars={
            "2t": (("time", "latitude", "longitude"), np.ones((3, 2, 2), dtype=np.float32) * 300.0),
            "2d": (("time", "latitude", "longitude"), np.ones((3, 2, 2), dtype=np.float32) * 295.0),
            "swvl1": (("time", "latitude", "longitude"), np.ones((3, 2, 2), dtype=np.float32) * 0.3),
            "swvl2": (("time", "latitude", "longitude"), np.ones((3, 2, 2), dtype=np.float32) * 0.6),
        },
        coords={"time": time, "latitude": lat, "longitude": lon},
    )


def test_load_or_create_geobox_state_persists_first_dataset_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample_path = tmp_path / "data" / "climate" / "raw" / "era5_land_hourly" / "era5_land_hourly_1985_01.grib"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text("grib", encoding="utf-8")
    dataset = xr.Dataset(coords={"spatial_ref": xr.DataArray(4326)})
    fake_geobox = FakeGeoBox(latitude=[-10.0, -11.0], longitude=[-50.0, -49.0])

    monkeypatch.setattr(
        "climate.preprocess.era5_land._open_era5_dataset",
        lambda path: dataset,
    )
    monkeypatch.setattr(
        "climate.preprocess.era5_land._extract_geobox_from_dataset",
        lambda ds: fake_geobox,
    )
    monkeypatch.setattr(
        "climate.preprocess.era5_land._extract_spatial_ref_from_dataset",
        lambda ds: 4326,
    )

    state = load_or_create_geobox_state(root_dir=tmp_path, sample_path=sample_path)

    assert state["latitude"].tolist() == [-10.0, -11.0]
    assert state["longitude"].tolist() == [-50.0, -49.0]
    assert state["spatial_ref"] == 4326
    assert (tmp_path / "data" / "climate" / "raw" / "era5_land_hourly" / "geobox.pickle").exists()


def test_bootstrap_era5_store_creates_coords_only_store_and_missing_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    geobox_state = {
        "geobox": FakeGeoBox(latitude=[-10.0, -11.0], longitude=[-50.0, -49.0]),
        "latitude": np.array([-10.0, -11.0]),
        "longitude": np.array([-50.0, -49.0]),
        "spatial_ref": 4326,
    }

    monkeypatch.setattr(
        "climate.preprocess.era5_land.load_or_create_geobox_state",
        lambda root_dir=".", sample_path=None: geobox_state,
    )

    store_path = bootstrap_era5_store(root_dir=tmp_path)
    opened = xr.open_zarr(store_path, consolidated=False)
    try:
        assert {"time", "latitude", "longitude", "spatial_ref"} == set(opened.coords)
        assert len(opened.time) == len(ERA5_OUTPUT_TIME_INDEX)
        assert {"tp", "2t", "2t_daily_min", "2t_daily_max", "swvl2"} <= set(opened.data_vars)
    finally:
        close = getattr(opened, "close", None)
        if callable(close):
            close()


def test_bootstrap_era5_store_appends_only_missing_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    geobox_state = {
        "geobox": FakeGeoBox(latitude=[-10.0, -11.0], longitude=[-50.0, -49.0]),
        "latitude": np.array([-10.0, -11.0]),
        "longitude": np.array([-50.0, -49.0]),
        "spatial_ref": None,
    }
    monkeypatch.setattr(
        "climate.preprocess.era5_land.load_or_create_geobox_state",
        lambda root_dir=".", sample_path=None: geobox_state,
    )

    store_path = bootstrap_era5_store(root_dir=tmp_path)
    opened = xr.open_zarr(store_path, consolidated=False)
    try:
        drop_vars = [var for var in opened.data_vars if var != "tp"]
        rebuilt = opened.drop_vars(drop_vars)
    finally:
        close = getattr(opened, "close", None)
        if callable(close):
            close()

    rebuilt.to_zarr(store_path, mode="w", zarr_format=3, consolidated=False)

    store_path = bootstrap_era5_store(root_dir=tmp_path)
    reopened = xr.open_zarr(store_path, consolidated=False)
    try:
        assert "tp" in reopened.data_vars
        assert "2t" in reopened.data_vars
        assert "2t_daily_min" in reopened.data_vars
    finally:
        close = getattr(reopened, "close", None)
        if callable(close):
            close()


def test_resample_hourly_to_daily_applies_expected_aggregations() -> None:
    daily = resample_era5l_hourly_to_daily(_hourly_dataset(), ERA5L_VAR_CONFIG)

    assert daily.time.size == 2
    assert daily["tp"].isel(time=0, latitude=0, longitude=0).item() == pytest.approx(24.0, abs=1e-4)
    assert daily["2t"].isel(time=0, latitude=0, longitude=0).item() == pytest.approx(46.0, abs=1e-4)
    assert daily["2t_daily_min"].isel(time=0, latitude=0, longitude=0).item() == pytest.approx(0.0)
    assert daily["2t_daily_max"].isel(time=1, latitude=0, longitude=0).item() == pytest.approx(188.0, abs=1e-4)
    assert daily["2t"].attrs["units"] == "degC"


def test_prepare_daily_era5_dataset_writes_daily_values_without_resampling() -> None:
    prepared = prepare_daily_era5_dataset(_daily_dataset())

    assert prepared.time.size == 3
    assert prepared["2t"].isel(time=0, latitude=0, longitude=0).item() == pytest.approx(26.85)
    assert prepared["swvl1"].isel(time=2, latitude=1, longitude=1).item() == pytest.approx(0.3)
    assert "2t_daily_min" not in prepared.data_vars


def test_write_dataset_region_writes_to_matching_month_slice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    geobox_state = {
        "geobox": FakeGeoBox(latitude=[-10.0, -11.0], longitude=[-50.0, -49.0]),
        "latitude": np.array([-10.0, -11.0]),
        "longitude": np.array([-50.0, -49.0]),
        "spatial_ref": None,
    }
    monkeypatch.setattr(
        "climate.preprocess.era5_land.load_or_create_geobox_state",
        lambda root_dir=".", sample_path=None: geobox_state,
    )
    store_path = bootstrap_era5_store(root_dir=tmp_path)

    daily = prepare_daily_era5_dataset(_daily_dataset())
    daily = daily.assign_coords(latitude=geobox_state["latitude"], longitude=geobox_state["longitude"])
    daily = daily.transpose("time", "latitude", "longitude")

    write_dataset_region(daily, store_path)

    opened = xr.open_zarr(store_path, consolidated=False)
    try:
        time_index = pd.DatetimeIndex(opened.time.values)
        target_idx = time_index.get_loc(pd.Timestamp("1985-02-01"))
        assert opened["2t"].isel(time=target_idx, latitude=0, longitude=0).load().item() == pytest.approx(26.85)
        assert np.isnan(opened["2t"].isel(time=0, latitude=0, longitude=0).load().item())
    finally:
        close = getattr(opened, "close", None)
        if callable(close):
            close()


def test_preprocess_era5_land_processes_files_and_leaves_store_reusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = tmp_path / "data" / "climate" / "raw" / "era5_land_hourly"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for month in ("01", "02"):
        (raw_dir / f"era5_land_hourly_1985_{month}.grib").write_text("grib", encoding="utf-8")

    geobox_state = {
        "geobox": FakeGeoBox(latitude=[-10.0, -11.0], longitude=[-50.0, -49.0]),
        "latitude": np.array([-10.0, -11.0]),
        "longitude": np.array([-50.0, -49.0]),
        "spatial_ref": None,
    }
    monkeypatch.setattr(
        "climate.preprocess.era5_land.load_or_create_geobox_state",
        lambda root_dir=".", sample_path=None: geobox_state,
    )
    monkeypatch.setattr(
        "climate.preprocess.era5_land._open_era5_dataset",
        lambda path: _hourly_dataset(),
    )
    for month in ("01", "02"):
        write_download_manifest(
            raw_dir / f"era5_land_hourly_1985_{month}.grib",
            dataset="reanalysis-era5-land",
            request={"year": ["1985"], "month": [month]},
            status="downloaded",
        )

    first_store = preprocess_era5_land(root_dir=tmp_path, subtype="era5_land_hourly")

    opened = xr.open_zarr(first_store, consolidated=False)
    try:
        assert "2t_daily_min" in opened.data_vars
        assert opened["tp"].isel(time=0, latitude=0, longitude=0).load().item() == pytest.approx(24.0, abs=1e-4)
    finally:
        close = getattr(opened, "close", None)
        if callable(close):
            close()
    assert not any(raw_dir.glob("*.grib"))


def test_preprocess_era5_land_deletes_raw_file_and_updates_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = tmp_path / "data" / "climate" / "raw" / "era5_land_hourly"
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / "era5_land_hourly_1985_01.grib"
    target.write_text("grib", encoding="utf-8")
    write_download_manifest(
        target,
        dataset="reanalysis-era5-land",
        request={"year": ["1985"], "month": ["01"]},
        status="downloaded",
        request_id="req-1",
    )

    geobox_state = {
        "geobox": FakeGeoBox(latitude=[-10.0, -11.0], longitude=[-50.0, -49.0]),
        "latitude": np.array([-10.0, -11.0]),
        "longitude": np.array([-50.0, -49.0]),
        "spatial_ref": None,
    }
    monkeypatch.setattr(
        "climate.preprocess.era5_land.load_or_create_geobox_state",
        lambda root_dir=".", sample_path=None: geobox_state,
    )
    monkeypatch.setattr(
        "climate.preprocess.era5_land._open_era5_dataset",
        lambda path: _hourly_dataset(),
    )

    store_path = preprocess_era5_land(root_dir=tmp_path, subtype="era5_land_hourly")
    manifest = load_download_manifest(target)

    assert store_path.exists()
    assert not target.exists()
    assert manifest is not None
    assert manifest["status"] == "downloaded"
    assert manifest["preprocess_status"] == "processed"
    assert manifest["raw_deleted"] is True
    assert manifest["request_id"] == "req-1"
    assert manifest["processed_store_path"].endswith("era5_land.zarr")


def test_preprocess_worker_waits_for_new_downloaded_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = tmp_path / "data" / "climate" / "raw" / "era5_land_hourly"
    raw_dir.mkdir(parents=True, exist_ok=True)

    geobox_state = {
        "geobox": FakeGeoBox(latitude=[-10.0, -11.0], longitude=[-50.0, -49.0]),
        "latitude": np.array([-10.0, -11.0]),
        "longitude": np.array([-50.0, -49.0]),
        "spatial_ref": None,
    }
    monkeypatch.setattr(
        "climate.preprocess.era5_land.load_or_create_geobox_state",
        lambda root_dir=".", sample_path=None: geobox_state,
    )
    monkeypatch.setattr(
        "climate.preprocess.era5_land._open_era5_dataset",
        lambda path: _hourly_dataset(),
    )

    waits = {"count": 0}

    def fake_wait(seconds):
        waits["count"] += 1
        if waits["count"] == 1:
            target = raw_dir / "era5_land_hourly_1985_01.grib"
            target.write_text("grib", encoding="utf-8")
            write_download_manifest(
                target,
                dataset="reanalysis-era5-land",
                request={"year": ["1985"], "month": ["01"]},
                status="downloaded",
            )

    monkeypatch.setattr("climate.preprocess.era5_land._worker_wait", fake_wait)
    monkeypatch.setattr(
        "climate.preprocess.era5_land._active_download_requests_exist",
        lambda root_dir=".", subtype="era5_land_hourly": waits["count"] == 0,
    )

    store_path = preprocess_era5_land_worker(root_dir=tmp_path, subtype="era5_land_hourly", poll_seconds=0)
    manifest = load_download_manifest(raw_dir / "era5_land_hourly_1985_01.grib")

    assert waits["count"] >= 1
    assert store_path.exists()
    assert manifest is not None
    assert manifest["preprocess_status"] == "processed"
