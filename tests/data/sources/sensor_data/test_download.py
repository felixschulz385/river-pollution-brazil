from __future__ import annotations

from selenium.common.exceptions import InvalidSessionIdException

from src.data.sources.sensor_data.fetch.data import download as download_module
from src.data.sources.sensor_data.fetch.data.download import (
    _station_code_matches,
    _wait_for_download_completion,
    download_by_id,
)


def test_station_code_matches_requires_digit_boundaries() -> None:
    assert _station_code_matches("12_conventional_mdb_20200101.zip", "12")
    assert not _station_code_matches("112_conventional_mdb_20200101.zip", "12")
    assert not _station_code_matches("1234_conventional_mdb_20200101.zip", "12")
    assert _station_code_matches("prefix_12.zip", "12")


def test_wait_for_download_completion_ignores_substring_matching_station(tmp_path) -> None:
    # A file for an unrelated station ("112") that contains this station's code
    # ("12") as a substring must not be picked up as station 12's download.
    (tmp_path / "112_conventional_mdb_20200101.zip").write_bytes(b"data")

    try:
        _wait_for_download_completion(tmp_path, "12", timeout_seconds=1)
    except TimeoutError:
        pass
    else:
        raise AssertionError("Expected a TimeoutError since no matching file for station 12 exists.")


def test_download_by_id_returns_restarted_driver_after_session_loss(monkeypatch, tmp_path) -> None:
    """After a session-loss restart, `download_by_id` must hand the caller its new
    driver -- otherwise the caller's stale reference gets reused for every
    subsequent station until the periodic restart happens to catch up."""
    old_driver = object()
    new_driver = object()

    monkeypatch.setattr(
        download_module,
        "_load_station_category_history_from_db",
        lambda root_dir, station_code: {},
    )
    monkeypatch.setattr(download_module, "_refresh_session", lambda driver: None)
    monkeypatch.setattr(download_module, "_decency_wait", lambda *args, **kwargs: None)
    monkeypatch.setattr(download_module, "_dump_debug_html", lambda *args, **kwargs: None)

    calls = {"n": 0}

    def fake_download_conventional_archives(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise InvalidSessionIdException("invalid session id")
        return [
            download_module._download_record(
                "run-1",
                "2024-01-01T00:00:00Z",
                "default",
                "12",
                result_tab="conventional",
                station_type="type-a",
                status="downloaded",
                success=1,
                attempts=1,
            )
        ]

    monkeypatch.setattr(
        download_module, "_download_conventional_archives", fake_download_conventional_archives
    )

    class _FakeBrowserManager:
        def restart(self):
            return new_driver

    records, returned_driver = download_by_id(
        "12",
        "Station Twelve",
        old_driver,
        tmp_path / "downloads",
        tmp_path / "raw",
        root_dir=str(tmp_path),
        browser_manager=_FakeBrowserManager(),
    )

    assert returned_driver is new_driver
    assert returned_driver is not old_driver
    assert len(records) == 1


def test_current_raw_archives_frame_excludes_truncated_zip(tmp_path) -> None:
    import zipfile

    good_path = tmp_path / "12_conventional_mdb_20200101.zip"
    with zipfile.ZipFile(good_path, "w") as archive:
        archive.writestr("STATION12.mdb", b"fake mdb bytes")

    # Simulate a download interrupted mid-write (e.g. `shutil.move` killed
    # partway, or process killed before the file finished writing): a
    # truncated file that starts with a real ZIP signature but has no valid
    # central directory.
    truncated_path = tmp_path / "34_conventional_mdb_20200102.zip"
    truncated_path.write_bytes(b"PK\x03\x04" + b"\x00" * 40)

    frame = download_module._current_raw_archives_frame(tmp_path)

    assert good_path.name in frame["filename"].tolist()
    assert truncated_path.name not in frame["filename"].tolist()


def test_current_raw_archives_frame_caches_zip_verification_across_runs(tmp_path, monkeypatch) -> None:
    import zipfile

    good_path = tmp_path / "12_conventional_mdb_20200101.zip"
    with zipfile.ZipFile(good_path, "w") as archive:
        archive.writestr("STATION12.mdb", b"fake mdb bytes")

    calls = {"n": 0}
    real_is_parseable_zip = download_module._is_parseable_zip

    def counting_is_parseable_zip(path):
        calls["n"] += 1
        return real_is_parseable_zip(path)

    monkeypatch.setattr(download_module, "_is_parseable_zip", counting_is_parseable_zip)

    first = download_module._current_raw_archives_frame(tmp_path)
    assert good_path.name in first["filename"].tolist()
    assert calls["n"] == 1

    # Unchanged file on a second scan: the cached verdict is reused, not
    # re-verified via a full CRC pass.
    second = download_module._current_raw_archives_frame(tmp_path)
    assert good_path.name in second["filename"].tolist()
    assert calls["n"] == 1

    # File rewritten (e.g. redownloaded): the cache entry is stale and the
    # file is re-verified.
    with zipfile.ZipFile(good_path, "w") as archive:
        archive.writestr("STATION12.mdb", b"different fake mdb bytes")
    import os as _os

    _os.utime(good_path, ns=(_os.stat(good_path).st_atime_ns, _os.stat(good_path).st_mtime_ns + 1))

    third = download_module._current_raw_archives_frame(tmp_path)
    assert good_path.name in third["filename"].tolist()
    assert calls["n"] == 2
