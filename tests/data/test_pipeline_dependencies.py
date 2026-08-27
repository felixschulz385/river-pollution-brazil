from __future__ import annotations

import pytest

from src.data.pipeline_dependencies import (
    ASSEMBLY_NODE,
    StagePrerequisites,
    assembly_prerequisites,
    build_chain,
    unmet_prerequisites,
)
from src.data.verification.core import SourceReport


def _stub_verification(monkeypatch, states: dict[str, dict]):
    """Replace `Verification` with a stub reporting caller-supplied
    fetch_status/preprocess_complete per source, so these tests exercise
    pipeline_dependencies' own graph/ordering logic in isolation from
    verification's actual artifact-checking (covered elsewhere)."""

    class _StubVerification:
        def __init__(self, root_dir="."):
            self.root_dir = root_dir

        def verify(self, source=None, force=False):
            state = states.get(source, {"fetch_status": "not_present_locally", "preprocess_complete": False})
            report = SourceReport(
                source=source,
                status="verified" if state["preprocess_complete"] else "outstanding",
                fingerprint="",
                verified_at="",
                fetch_status=state["fetch_status"],
                preprocess_complete=state["preprocess_complete"],
            )
            return {source: report}

    monkeypatch.setattr("src.data.verification.core.Verification", _StubVerification)


# --------------------------------------------------------------------------
# unmet_prerequisites
# --------------------------------------------------------------------------

def test_unmet_prerequisites_land_cover_preprocess_reports_everything_missing(monkeypatch):
    _stub_verification(monkeypatch, {})  # everything defaults to not_present_locally / incomplete

    problems = unmet_prerequisites("land_cover", "preprocess")

    assert any("'land_cover' has no raw data fetched yet" in p for p in problems)
    assert any("'river_network' has not finished preprocessing yet" in p for p in problems)
    assert any("'sensor_data' has not finished preprocessing yet" in p for p in problems)
    assert len(problems) == 3


def test_unmet_prerequisites_land_cover_preprocess_empty_when_satisfied(monkeypatch):
    _stub_verification(
        monkeypatch,
        {
            "land_cover": {"fetch_status": "verified", "preprocess_complete": False},
            "river_network": {"fetch_status": "verified", "preprocess_complete": True},
            "sensor_data": {"fetch_status": "verified", "preprocess_complete": True},
        },
    )

    assert unmet_prerequisites("land_cover", "preprocess") == []


def test_unmet_prerequisites_sensor_data_fetch_requires_river_network_preprocessed(monkeypatch):
    _stub_verification(monkeypatch, {"river_network": {"fetch_status": "verified", "preprocess_complete": False}})

    problems = unmet_prerequisites("sensor_data", "fetch")

    assert len(problems) == 1
    assert "'river_network' has not finished preprocessing yet" in problems[0]


def test_unmet_prerequisites_sensor_data_fetch_has_no_self_check(monkeypatch):
    """Unlike "preprocess", the "fetch" verb has no universal self-check --
    a source obviously doesn't need to already be fetched to fetch it."""
    _stub_verification(monkeypatch, {"river_network": {"fetch_status": "verified", "preprocess_complete": True}})

    assert unmet_prerequisites("sensor_data", "fetch") == []


def test_unmet_prerequisites_biomes_requires_sensor_data_fetched(monkeypatch):
    _stub_verification(
        monkeypatch,
        {
            "biomes": {"fetch_status": "verified", "preprocess_complete": False},
            "sensor_data": {"fetch_status": "not_present_locally", "preprocess_complete": False},
        },
    )

    problems = unmet_prerequisites("biomes", "preprocess")

    assert len(problems) == 1
    assert "'sensor_data' has not been fetched yet" in problems[0]


def test_unmet_prerequisites_river_network_manual_placement_message(monkeypatch):
    """river_network has fetch=False in SOURCE_REGISTRY -- the self-check
    message for it must point at manual placement, not a `data fetch` command."""
    _stub_verification(monkeypatch, {})

    problems = unmet_prerequisites("river_network", "preprocess")

    assert len(problems) == 1
    assert "must be placed manually" in problems[0]
    assert "data fetch" not in problems[0]


# --------------------------------------------------------------------------
# assembly_prerequisites
# --------------------------------------------------------------------------

_FIXTURE_ASSEMBLY_YAML = """
datasets:
  - id: sensor_panel
    mode: sensor
    index: [station_code, datetime]
    output_path: data/assembly/sensor_panel.parquet
    sources:
      - name: water_quality
        path: data/sensor_data/processed/aggregate/water_quality_streamflow.parquet
        join_keys: [station_code, datetime]
        variables: [ph]
      - name: river_system
        path: data/river_network/processed/river_trenches.parquet
        join_keys: [station_code]
        variables: []
"""


def test_assembly_prerequisites_derived_from_fixture_config(tmp_path, monkeypatch):
    config_path = tmp_path / "assembly_datasets.yaml"
    config_path.write_text(_FIXTURE_ASSEMBLY_YAML)
    monkeypatch.setattr(
        "src.data.assembly.constants.DEFAULT_CONFIG_PATH", str(config_path.relative_to(tmp_path))
    )

    prereqs = assembly_prerequisites(root_dir=tmp_path)

    assert set(prereqs.requires_preprocessed) == {"sensor_data", "river_network"}
    assert prereqs.requires_fetched == ()


def test_assembly_prerequisites_missing_config_yields_no_prerequisites(tmp_path):
    assert assembly_prerequisites(root_dir=tmp_path) == StagePrerequisites()


def test_assembly_prerequisites_against_real_config_excludes_health():
    """Regression-locks today's actual setup/assembly_datasets.yaml: health
    isn't wired into any dataset's source paths yet (per its own "Deferred"
    comment), so it must not show up as an assembly prerequisite."""
    prereqs = assembly_prerequisites(root_dir=".")

    assert "health" not in prereqs.requires_preprocessed
    assert "river_network" in prereqs.requires_preprocessed
    assert "sensor_data" in prereqs.requires_preprocessed


# --------------------------------------------------------------------------
# build_chain
# --------------------------------------------------------------------------

def test_build_chain_raises_on_unresolvable_manual_source(monkeypatch):
    """river_network can't be auto-fetched -- if its raw data isn't present,
    the chain can't be built automatically."""
    _stub_verification(monkeypatch, {})

    with pytest.raises(ValueError, match="must be placed manually"):
        build_chain("land_cover", "preprocess")


def test_build_chain_orders_multihop_dependencies(monkeypatch):
    _stub_verification(monkeypatch, {"river_network": {"fetch_status": "verified", "preprocess_complete": True}})

    chain = build_chain("land_cover", "preprocess")

    assert chain == [
        ("land_cover", "fetch"),
        ("sensor_data", "fetch"),
        ("sensor_data", "preprocess"),
        ("land_cover", "preprocess"),
    ]


def test_build_chain_skips_already_satisfied_steps(monkeypatch):
    _stub_verification(
        monkeypatch,
        {
            "land_cover": {"fetch_status": "verified", "preprocess_complete": False},
            "river_network": {"fetch_status": "verified", "preprocess_complete": True},
            "sensor_data": {"fetch_status": "verified", "preprocess_complete": True},
        },
    )

    assert build_chain("land_cover", "preprocess") == [("land_cover", "preprocess")]


def test_build_chain_assembly_pulls_in_preprocess_steps_for_every_prerequisite(tmp_path, monkeypatch):
    config_path = tmp_path / "assembly_datasets.yaml"
    config_path.write_text(_FIXTURE_ASSEMBLY_YAML)
    monkeypatch.setattr(
        "src.data.assembly.constants.DEFAULT_CONFIG_PATH", str(config_path.relative_to(tmp_path))
    )
    _stub_verification(monkeypatch, {"river_network": {"fetch_status": "verified", "preprocess_complete": True}})

    chain = build_chain(ASSEMBLY_NODE, "assemble", root_dir=tmp_path)

    assert chain[-1] == (ASSEMBLY_NODE, "assemble")
    assert ("sensor_data", "preprocess") in chain
    assert ("river_network", "preprocess") not in chain  # already satisfied
    assert chain.index(("sensor_data", "preprocess")) < chain.index((ASSEMBLY_NODE, "assemble"))
