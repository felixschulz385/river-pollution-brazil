from __future__ import annotations

import geopandas as gpd
import pytest

from src.data.sources.geology import fetch as fetch_module
from src.data.sources.geology.constants import SGB_LAYERS, gpkg_path


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def _feature(x, y):
    return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [x, y]}, "properties": {"SUBSTANCIAS": "Ouro"}}


def _fake_get(count):
    def get(url, params, timeout):
        if params.get("returnCountOnly"):
            return _FakeResponse({"count": count})
        return _FakeResponse({"type": "FeatureCollection", "features": [_feature(-55, -5), _feature(-54, -5)]})

    return get


def test_fetch_writes_every_layer_and_no_temp_file(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(fetch_module.requests, "get", _fake_get(count=2))

    fetch_module.fetch_sgb_gold(root_dir=tmp_path)

    destination = gpkg_path(tmp_path)
    for layer in SGB_LAYERS:
        assert len(gpd.read_file(destination, layer=layer)) == 2
    assert list(destination.parent.glob("*.tmp-*")) == []


def test_fetch_rejects_a_truncated_layer(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(fetch_module.requests, "get", _fake_get(count=3))

    with pytest.raises(RuntimeError, match="got 2 of 3"):
        fetch_module.fetch_sgb_gold(root_dir=tmp_path)
    destination = gpkg_path(tmp_path)
    assert not destination.exists()
    assert list(destination.parent.glob("*.tmp-*")) == []
