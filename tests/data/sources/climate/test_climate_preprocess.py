from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.data.sources.climate.preprocess.era5_land import (
    ERA5L_VAR_CONFIG,
    ERA5_OUTPUT_TIME_INDEX,
    _open_era5_dataset,
    bootstrap_era5_store,
    load_or_create_geobox_state,
    prepare_daily_era5_dataset,
    preprocess_era5_land_worker,
    preprocess_era5_land,
    resample_era5l_hourly_to_daily,
    write_dataset_region,
)
from src.data.sources.climate.fetch.common import load_download_manifest, write_download_manifest


class _FakeEra5FieldList:
    """Mimics the tiny slice of earthkit.data.FieldList's interface that
    _open_era5_dataset relies on."""

    def __init__(self, metadata: pd.DataFrame, values: np.ndarray, latitude: np.ndarray, longitude: np.ndarray):
        self._metadata = metadata
        self._values = values
        self._latitude = latitude
        self._longitude = longitude

    def ls(self):
        return self._metadata

    def data(self, keys):
        if keys == "lat":
            plane = np.repeat(self._latitude[:, None], len(self._longitude), axis=1)
        elif keys == "lon":
            plane = np.repeat(self._longitude[None, :], len(self._latitude), axis=0)
        else:
            raise ValueError(f"Unsupported key: {keys!r}")
        return np.broadcast_to(plane, (len(self._metadata), *plane.shape))

    def to_numpy(self):
        return self._values

    def close(self):
        pass


def test_open_era5_dataset_maps_fields_by_metadata_not_stream_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for the reshape/transpose bug: the flat GRIB field
    array must be mapped into (band, time) using the metadata's band/datetime
    columns, not by assuming a band-major message order. Real GRIB streams
    are time-major (all bands for hour 0, then hour 1, ...) - this test uses
    exactly that ordering and would fail against a blind
    `.reshape(len(bands), len(times), ...)`."""
    times = pd.date_range("1985-01-01", periods=3, freq="1h")
    bands = ["2t", "tp"]
    latitude = np.array([1.0, 0.0])
    longitude = np.array([10.0, 11.0])

    rows = []
    planes = []
    for time_idx, valid_time in enumerate(times):
        for band_idx, band in enumerate(bands):
            rows.append({"shortName": band, "valid_datetime": valid_time})
            sentinel = (band_idx + 1) * 1000 + time_idx
            planes.append(np.full((len(latitude), len(longitude)), sentinel, dtype=np.float64))

    metadata = pd.DataFrame(rows)
    values = np.stack(planes, axis=0)
    fake_field_list = _FakeEra5FieldList(metadata, values, latitude, longitude)

    monkeypatch.setattr("earthkit.data.from_source", lambda *args, **kwargs: fake_field_list)

    dataset = _open_era5_dataset(Path("dummy.grib"))

    for time_idx in range(len(times)):
        assert dataset["2t"].isel(time=time_idx, latitude=0, longitude=0).item() == 1000 + time_idx
        assert dataset["tp"].isel(time=time_idx, latitude=0, longitude=0).item() == 2000 + time_idx


def test_open_era5_dataset_rejects_duplicate_band_time_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = pd.date_range("1985-01-01", periods=2, freq="1h")
    metadata = pd.DataFrame(
        [
            {"shortName": "2t", "valid_datetime": times[0]},
            {"shortName": "2t", "valid_datetime": times[0]},
        ]
    )
    latitude = np.array([1.0, 0.0])
    longitude = np.array([10.0, 11.0])
    values = np.zeros((2, len(latitude), len(longitude)), dtype=np.float64)
    fake_field_list = _FakeEra5FieldList(metadata, values, latitude, longitude)

    monkeypatch.setattr("earthkit.data.from_source", lambda *args, **kwargs: fake_field_list)

    with pytest.raises(ValueError, match="Duplicate ERA5 fields"):
        _open_era5_dataset(Path("dummy.grib"))


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
        "src.data.sources.climate.preprocess.era5_land._open_era5_dataset",
        lambda path: dataset,
    )
    monkeypatch.setattr(
        "src.data.sources.climate.preprocess.era5_land._extract_geobox_from_dataset",
        lambda ds: fake_geobox,
    )
    monkeypatch.setattr(
        "src.data.sources.climate.preprocess.era5_land._extract_spatial_ref_from_dataset",
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
        "src.data.sources.climate.preprocess.era5_land.load_or_create_geobox_state",
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
        "src.data.sources.climate.preprocess.era5_land.load_or_create_geobox_state",
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
        "src.data.sources.climate.preprocess.era5_land.load_or_create_geobox_state",
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
        "src.data.sources.climate.preprocess.era5_land.load_or_create_geobox_state",
        lambda root_dir=".", sample_path=None: geobox_state,
    )
    monkeypatch.setattr(
        "src.data.sources.climate.preprocess.era5_land._open_era5_dataset",
        lambda path: _hourly_dataset(),
    )
    for month in ("01", "02"):
        write_download_manifest(
            raw_dir / f"era5_land_hourly_1985_{month}.grib",
            dataset="reanalysis-era5-land",
            request={"year": ["1985"], "month": [month]},
            status="downloaded",
        )

    first_store = preprocess_era5_land(root_dir=tmp_path, stage="zarr")

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
        "src.data.sources.climate.preprocess.era5_land.load_or_create_geobox_state",
        lambda root_dir=".", sample_path=None: geobox_state,
    )
    monkeypatch.setattr(
        "src.data.sources.climate.preprocess.era5_land._open_era5_dataset",
        lambda path: _hourly_dataset(),
    )

    store_path = preprocess_era5_land(root_dir=tmp_path, stage="zarr")
    manifest = load_download_manifest(target)

    assert store_path.exists()
    assert not target.exists()
    assert manifest is not None
    # `_write_preprocess_manifest` advances the top-level "status" through the
    # preprocess lifecycle too (-> "processed"); `download_status` is what
    # retains the original download outcome.
    assert manifest["status"] == "processed"
    assert manifest["download_status"] == "downloaded"
    assert manifest["preprocess_status"] == "processed"
    assert manifest["raw_deleted"] is True
    assert manifest["request_id"] == "req-1"
    assert manifest["processed_store_path"].endswith("era5_land.zarr_nobackup")


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
        "src.data.sources.climate.preprocess.era5_land.load_or_create_geobox_state",
        lambda root_dir=".", sample_path=None: geobox_state,
    )
    monkeypatch.setattr(
        "src.data.sources.climate.preprocess.era5_land._open_era5_dataset",
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

    monkeypatch.setattr("src.data.sources.climate.preprocess.era5_land._worker_wait", fake_wait)
    monkeypatch.setattr(
        "src.data.sources.climate.preprocess.era5_land._active_download_requests_exist",
        lambda root_dir=".", subtype="era5_land_hourly": waits["count"] == 0,
    )

    store_path = preprocess_era5_land_worker(root_dir=tmp_path, poll_seconds=0, stage="zarr")
    manifest = load_download_manifest(raw_dir / "era5_land_hourly_1985_01.grib")

    assert waits["count"] >= 1
    assert store_path.exists()
    assert manifest is not None
    assert manifest["preprocess_status"] == "processed"
