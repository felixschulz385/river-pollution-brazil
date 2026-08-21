from __future__ import annotations

import io
import zipfile

import pytest

from src.data.sources.biomes import fetch as fetch_module
from src.data.sources.biomes.constants import archive_path


def _zip_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("bioma.txt", b"placeholder")
    return buffer.getvalue()


class _FakeResponse:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        return None


def test_fetch_biomes_downloads_atomically_and_leaves_no_temp_file(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        fetch_module.requests, "get", lambda url, timeout: _FakeResponse(_zip_bytes())
    )

    fetch_module.fetch_biomes(root_dir=tmp_path)

    destination = archive_path(tmp_path)
    assert destination.exists()
    leftover_temp_files = list(destination.parent.glob(f"{destination.name}.tmp-*"))
    assert leftover_temp_files == []


def test_fetch_biomes_skips_download_when_destination_already_exists(tmp_path, monkeypatch: pytest.MonkeyPatch):
    destination = archive_path(tmp_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_zip_bytes())

    calls = []
    monkeypatch.setattr(
        fetch_module.requests,
        "get",
        lambda url, timeout: calls.append(url) or _FakeResponse(_zip_bytes()),
    )

    fetch_module.fetch_biomes(root_dir=tmp_path)

    assert calls == []
