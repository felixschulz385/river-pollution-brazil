from __future__ import annotations

from src.data.health.core import Health


def test_health_initializes_constructor_options():
    agent = Health(root_dir="/tmp/example", headless=True, download_dir="/tmp/downloads")

    assert agent.root_dir == "/tmp/example"
    assert agent.headless is True
    assert agent.download_dir == "/tmp/downloads"


def test_health_fetch_routes_subtype(monkeypatch):
    seen = {}

    def fake_fetch_health_data(root_dir, subtype, headless, download_dir):
        seen["root_dir"] = root_dir
        seen["subtype"] = subtype
        return []

    monkeypatch.setattr(
        "src.data.health.fetch.fetch_health_data",
        fake_fetch_health_data,
    )

    agent = Health(root_dir="/tmp/example")
    agent.fetch(subtype="mortality")

    assert seen == {"root_dir": "/tmp/example", "subtype": "mortality"}
