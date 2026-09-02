from __future__ import annotations

import argparse

import pytest

from src.data.sources.land_cover.__main__ import configure_parser, run


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    configure_parser(parser, include_action=False)
    args = parser.parse_args(argv)
    args.action = "assemble"
    return args


class _RecordingAgent:
    def __init__(self, *args, **kwargs):
        self.calls: list[str] = []

    def assemble(self, *, variant, **kwargs):
        self.calls.append(variant)


def _patch_agent(monkeypatch) -> _RecordingAgent:
    agent = _RecordingAgent()
    monkeypatch.setattr(
        "src.data.sources.land_cover.__main__.LandCover",
        lambda *a, **k: agent,
    )
    return agent


def test_assemble_without_variant_runs_both_in_order(monkeypatch):
    agent = _patch_agent(monkeypatch)

    run(_parse_args([]))

    assert agent.calls == ["sensor", "adm2"]


def test_assemble_with_explicit_variant_runs_only_that_one(monkeypatch):
    agent = _patch_agent(monkeypatch)

    run(_parse_args(["--variant", "adm2"]))

    assert agent.calls == ["adm2"]


def test_assemble_all_rejects_output_override(monkeypatch):
    _patch_agent(monkeypatch)

    with pytest.raises(ValueError, match="--output is not supported with --variant all"):
        run(_parse_args(["--output", "somewhere.parquet"]))
