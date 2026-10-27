from __future__ import annotations

from pathlib import Path

import pytest

from src.data.sources.land_cover import fetch as land_cover_fetch
from src.data.sources.land_cover.core import LandCover, product_paths
from src.data.sources.land_cover.mining import (
    MINING_ADM2_UPSTREAM_PATH,
    MINING_DATADIR,
    MINING_EXTRACT_PATH,
    MINING_LEGEND_PATH,
)


class _Response:
    def __init__(self, body, length=None):
        self.body = body
        self.headers = {"Content-Length": str(len(body) if length is None else length)}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.body


def test_fetch_downloads_missing_years_and_skips_present_ones(tmp_path, monkeypatch):
    (tmp_path / "brazil_coverage_2000.tif").write_bytes(b"kept")
    urls = []

    def fake_get(url, **kwargs):
        urls.append(url)
        return _Response(b"tif")

    monkeypatch.setattr(land_cover_fetch.requests, "get", fake_get)

    land_cover_fetch.fetch_mapbiomas_coverage(tmp_path, years=(2000, 2001))

    assert urls == [
        "https://storage.googleapis.com/mapbiomas-public/initiatives/brasil/"
        "collection_10/lulc/coverage/brazil_coverage_2001.tif"
    ]
    assert (tmp_path / "brazil_coverage_2000.tif").read_bytes() == b"kept"
    assert (tmp_path / "brazil_coverage_2001.tif").read_bytes() == b"tif"


def test_fetch_rejects_truncated_download_without_leaving_a_file(tmp_path, monkeypatch):
    monkeypatch.setattr(land_cover_fetch.requests, "get", lambda url, **kwargs: _Response(b"ti", length=3))

    with pytest.raises(RuntimeError, match="got 2 of 3 bytes"):
        land_cover_fetch.fetch_mapbiomas_coverage(tmp_path, years=(2001,))

    assert list(tmp_path.iterdir()) == []


def test_mining_product_resolves_c11_paths(tmp_path):
    paths = product_paths("mining", tmp_path)

    assert paths["datadir"] == Path(tmp_path) / MINING_DATADIR
    assert paths["legend_path"] == Path(tmp_path) / MINING_LEGEND_PATH
    assert paths["extract_path"] == MINING_EXTRACT_PATH
    assert paths["adm2_upstream_path"] == MINING_ADM2_UPSTREAM_PATH


def test_unknown_product_is_rejected():
    with pytest.raises(ValueError, match="Unknown land-cover product"):
        product_paths("soy")


def test_mining_product_defaults_extract_output_and_rejects_the_sensor_variant(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        "src.data.sources.land_cover.core._preprocess_land_cover",
        lambda agent, **kwargs: calls.setdefault("preprocess", kwargs),
    )
    monkeypatch.setattr(
        "src.data.sources.land_cover.core._assemble_land_cover",
        lambda agent, **kwargs: calls.setdefault("assemble", kwargs),
    )
    agent = LandCover(product="mining", output_columns=[])

    agent.preprocess()
    agent.assemble(variant="adm2")

    assert calls["preprocess"]["output_path"] == MINING_EXTRACT_PATH
    assert calls["assemble"]["land_cover_path"] == MINING_EXTRACT_PATH
    assert calls["assemble"]["output_path"] == MINING_ADM2_UPSTREAM_PATH
    with pytest.raises(ValueError, match="only the adm2 variant"):
        agent.assemble(variant="sensor")
