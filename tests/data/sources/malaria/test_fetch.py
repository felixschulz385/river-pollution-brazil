from __future__ import annotations

import hashlib

import pytest

from src.data.sources.malaria import fetch as fetch_module
from src.data.sources.malaria.constants import METADATA_FILENAMES, YEAR_FILENAME_TEMPLATE, raw_dir

CONTENT = b"sivep-bytes" * 200


class _FakeResponse:
    def __init__(self, payload=None, content=b""):
        self.payload = payload
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload

    def iter_content(self, chunk_size):
        yield self.content

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_get(md5, calls):
    names = [YEAR_FILENAME_TEMPLATE.format(year=2020), *METADATA_FILENAMES]
    record = {
        "files": [
            {"key": name, "size": len(CONTENT), "checksum": f"md5:{md5}", "links": {"self": f"https://z/{name}"}}
            for name in names
        ]
    }

    def get(url, timeout, stream=False):
        calls.append(url)
        return _FakeResponse(payload=record) if not stream else _FakeResponse(content=CONTENT)

    return get


def test_fetch_downloads_verified_files_and_skips_them_on_rerun(tmp_path, monkeypatch: pytest.MonkeyPatch):
    calls = []
    monkeypatch.setattr(fetch_module.requests, "get", _fake_get(hashlib.md5(CONTENT).hexdigest(), calls))

    fetch_module.fetch_sivep(root_dir=tmp_path, years=(2020,))
    assert (raw_dir(tmp_path) / YEAR_FILENAME_TEMPLATE.format(year=2020)).read_bytes() == CONTENT
    downloads = len(calls) - 1

    calls.clear()
    fetch_module.fetch_sivep(root_dir=tmp_path, years=(2020,))
    assert downloads == 1 + len(METADATA_FILENAMES)
    assert len(calls) == 1  # only the record lookup


def test_fetch_rejects_a_checksum_mismatch(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(fetch_module.requests, "get", _fake_get("0" * 32, []))

    with pytest.raises(RuntimeError, match="md5"):
        fetch_module.fetch_sivep(root_dir=tmp_path, years=(2020,))
    assert list(raw_dir(tmp_path).iterdir()) == []
