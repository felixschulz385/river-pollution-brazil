from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.data.climate.fetch.era5_land_arco import (
    arco_store_url,
    open_arco_group_dataset,
)
from src.data.climate.preprocess.era5_land_arco import (
    _assert_matches_geobox,
    _is_month_processed,
    _load_arco_progress,
    _mark_month_processed,
    _save_arco_progress,
    preprocess_era5_land_arco,
    slice_arco_to_area,
)


class _FakeCoord:
    def __init__(self, values):
        self.values = np.asarray(values)


class FakeGeoBox:
    def __init__(self, latitude, longitude):
        self.coords = {
            "latitude": _FakeCoord(latitude),
            "longitude": _FakeCoord(longitude),
        }


def _write_cdsapi(root: Path, contents: str) -> None:
    credentials_path = root / "setup" / "secrets" / ".cdsapi"
    credentials_path.parent.mkdir(parents=True, exist_ok=True)
    credentials_path.write_text(contents, encoding="utf-8")


def test_arco_store_url_builds_expected_pattern() -> None:
    assert arco_store_url("sfc-2m-temperature", chunks="geo") == (
        "https://arco.datastores.ecmwf.int/cadl-arco-geo-007/"
        "arco/reanalysis_era5_land/sfc-2m-temperature/geoChunked.zarr"
    )
    assert arco_store_url("sfc-soil-water", chunks="time") == (
        "https://arco.datastores.ecmwf.int/cadl-arco-time-005/"
        "arco/reanalysis_era5_land/sfc-soil-water/timeChunked.zarr"
    )
    with pytest.raises(ValueError, match="Unsupported ARCO group"):
        arco_store_url("not-a-group")


def test_open_arco_group_dataset_uses_bearer_auth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_cdsapi(tmp_path, "url: https://example/api\nkey: my-secret-key\n")
    captured = {}

    def fake_open_zarr(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return xr.Dataset()

    monkeypatch.setattr("xarray.open_zarr", fake_open_zarr)

    open_arco_group_dataset("sfc-2m-temperature", root_dir=tmp_path)

    assert captured["url"] == arco_store_url("sfc-2m-temperature")
    assert captured["kwargs"]["consolidated"] is True
    assert captured["kwargs"]["storage_options"] == {
        "headers": {"Authorization": "Bearer my-secret-key"}
    }


def _make_dataset(latitude, longitude, *, ascending_lat: bool, ascending_lon: bool) -> xr.Dataset:
    lat = np.array(latitude, dtype=float)
    lon = np.array(longitude, dtype=float)
    if not ascending_lat:
        lat = lat[::-1]
    if not ascending_lon:
        lon = lon[::-1]
    data = np.zeros((1, len(lat), len(lon)), dtype=np.float32)
    return xr.Dataset(
        {"t2m": (("time", "latitude", "longitude"), data)},
        coords={
            "time": pd.date_range("2000-01-01", periods=1),
            "latitude": lat,
            "longitude": lon,
        },
    )


def test_slice_arco_to_area_ascending_lat_neg180_lon() -> None:
    ds = _make_dataset(
        [-12.0, -11.0, -10.0, -9.0, -8.0],
        [-52.0, -51.0, -50.0, -49.0, -48.0],
        ascending_lat=True,
        ascending_lon=True,
    )
    sliced = slice_arco_to_area(ds, area=[-9.0, -51.0, -11.0, -49.0])

    assert list(sliced["latitude"].values) == [-9.0, -10.0, -11.0]
    assert list(sliced["longitude"].values) == [-51.0, -50.0, -49.0]


def test_slice_arco_to_area_descending_lat_0_360_lon() -> None:
    ds = _make_dataset(
        [-8.0, -9.0, -10.0, -11.0, -12.0],
        [308.0, 309.0, 310.0, 311.0, 312.0],
        ascending_lat=False,
        ascending_lon=True,
    )
    sliced = slice_arco_to_area(ds, area=[-9.0, -51.0, -11.0, -49.0])

    assert list(sliced["latitude"].values) == [-9.0, -10.0, -11.0]
    assert list(sliced["longitude"].values) == [309.0, 310.0, 311.0]


def test_assert_matches_geobox_passes_and_raises() -> None:
    geobox_state = {
        "latitude": np.array([-10.0, -11.0]),
        "longitude": np.array([-50.0, -49.0]),
    }
    ok = xr.Dataset(coords={"latitude": np.array([-10.0, -11.0]), "longitude": np.array([-50.0, -49.0])})
    _assert_matches_geobox(ok, geobox_state)

    bad_shape = xr.Dataset(
        coords={"latitude": np.array([-10.0, -11.0, -12.0]), "longitude": np.array([-50.0, -49.0])}
    )
    with pytest.raises(ValueError, match="latitude grid"):
        _assert_matches_geobox(bad_shape, geobox_state)

    bad_values = xr.Dataset(
        coords={"latitude": np.array([-10.0, -11.0]), "longitude": np.array([-50.0, -48.0])}
    )
    with pytest.raises(ValueError, match="longitude grid"):
        _assert_matches_geobox(bad_values, geobox_state)


def test_arco_progress_round_trip(tmp_path: Path) -> None:
    progress = _load_arco_progress(tmp_path)
    assert progress == {}

    progress = _mark_month_processed(progress, "sfc-2m-temperature", "1985-01")
    _save_arco_progress(tmp_path, progress)

    reloaded = _load_arco_progress(tmp_path)
    assert _is_month_processed(reloaded, "sfc-2m-temperature", "1985-01")
    assert not _is_month_processed(reloaded, "sfc-2m-temperature", "1985-02")


def test_preprocess_era5_land_arco_writes_store_and_marks_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    geobox_state = {
        "geobox": FakeGeoBox(latitude=[-10.0, -11.0], longitude=[-50.0, -49.0]),
        "latitude": np.array([-10.0, -11.0]),
        "longitude": np.array([-50.0, -49.0]),
        "spatial_ref": None,
    }
    monkeypatch.setattr(
        "src.data.climate.preprocess.era5_land.load_or_create_geobox_state",
        lambda root_dir=".", sample_path=None: geobox_state,
    )
    monkeypatch.setattr(
        "src.data.climate.preprocess.era5_land_arco.load_or_create_geobox_state",
        lambda root_dir=".": geobox_state,
    )
    monkeypatch.setattr(
        "src.data.climate.preprocess.era5_land_arco.ARCO_GROUPS",
        {"sfc-2m-temperature": {"id": "007", "vars": ["t2m", "d2m"]}},
    )

    hours = pd.date_range("1985-01-01", periods=48, freq="1h")
    fixture = xr.Dataset(
        {
            "t2m": (("time", "latitude", "longitude"), np.full((48, 2, 2), 290.0, dtype=np.float32)),
            "d2m": (("time", "latitude", "longitude"), np.full((48, 2, 2), 285.0, dtype=np.float32)),
        },
        coords={
            "time": hours,
            "latitude": np.array([-11.0, -10.0]),  # ascending; store geobox is descending
            "longitude": np.array([-50.0, -49.0]),
        },
    )

    monkeypatch.setattr(
        "src.data.climate.preprocess.era5_land_arco.open_arco_group_dataset",
        lambda group, root_dir=".", chunks="geo": fixture,
    )

    store_path = preprocess_era5_land_arco(
        root_dir=tmp_path, start="1985-01-01", end="1985-01-31"
    )

    opened = xr.open_zarr(store_path, consolidated=False)
    try:
        day = opened.sel(time="1985-01-01")
        assert float(day["2t"].isel(latitude=0, longitude=0).values) == pytest.approx(290.0 - 273.15)
        assert float(day["2d"].isel(latitude=0, longitude=0).values) == pytest.approx(285.0 - 273.15)
        assert float(day["2t_daily_min"].isel(latitude=0, longitude=0).values) == pytest.approx(290.0 - 273.15)
        assert float(day["2t_daily_max"].isel(latitude=0, longitude=0).values) == pytest.approx(290.0 - 273.15)
    finally:
        close = getattr(opened, "close", None)
        if callable(close):
            close()

    progress = _load_arco_progress(tmp_path)
    assert _is_month_processed(progress, "sfc-2m-temperature", "1985-01")

    # Re-running should skip the already-processed month rather than re-opening the group.
    calls = []
    monkeypatch.setattr(
        "src.data.climate.preprocess.era5_land_arco.open_arco_group_dataset",
        lambda group, root_dir=".", chunks="geo": calls.append(group) or fixture,
    )
    preprocess_era5_land_arco(root_dir=tmp_path, start="1985-01-01", end="1985-01-31")
    assert calls == []
