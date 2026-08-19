from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_cli_summary_on_empty_root_reports_not_present_locally(tmp_path):
    """The key regression test: an empty root must not crash and every
    source must gracefully degrade to `not_present_locally`."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli",
            "data",
            "summary",
            "--root-dir",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    assert "Data Verification Summary" in result.stdout
    for source in (
        "river_network",
        "land_cover",
        "sensor_data",
        "climate",
        "biomes",
        "population",
        "health",
        "assembly",
    ):
        assert source in result.stdout

    # Every sidecar written against an empty root should be not_present_locally.
    for source in (
        "river_network",
        "land_cover",
        "sensor_data",
        "climate",
        "biomes",
        "population",
        "health",
        "assembly",
    ):
        sidecar = tmp_path / "data" / source / ".verification.json"
        assert sidecar.exists()
        assert '"status": "not_present_locally"' in sidecar.read_text()


def test_cli_verify_single_source_on_empty_root(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli",
            "data",
            "verify",
            "--root-dir",
            str(tmp_path),
            "--source",
            "assembly",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr

    sidecar = tmp_path / "data" / "assembly" / ".verification.json"
    assert sidecar.exists()


def test_standalone_module_invocation(tmp_path):
    """The module must also work outside the top-level `src.cli` dispatcher."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.data.verification",
            "verify",
            "--root-dir",
            str(tmp_path),
            "--source",
            "assembly",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    sidecar = tmp_path / "data" / "assembly" / ".verification.json"
    assert sidecar.exists()
