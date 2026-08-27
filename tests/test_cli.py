from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run_cli(args: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "src.cli", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_preprocess_hard_blocks_on_unmet_prerequisites(tmp_path):
    """land_cover's preprocess requires river_network to be preprocessed
    first -- on an empty root, that's unmet, so the CLI must refuse to run
    and exit non-zero with a clear, actionable message instead of crashing
    deep inside land_cover's own preprocessing code."""
    result = _run_cli(
        ["data", "preprocess", "--source", "land_cover", "--phase", "extract", "--root-dir", str(tmp_path)]
    )

    assert result.returncode == 1
    assert "Cannot preprocess 'land_cover': prerequisites not satisfied." in result.stderr
    assert "river_network" in result.stderr
    assert "--chain" in result.stderr
    assert "--skip-dependency-check" in result.stderr


def test_preprocess_skip_dependency_check_bypasses_the_gate(tmp_path):
    result = _run_cli(
        [
            "data",
            "preprocess",
            "--source",
            "land_cover",
            "--phase",
            "extract",
            "--skip-dependency-check",
            "--root-dir",
            str(tmp_path),
        ]
    )

    # The gate must not have fired -- whatever happens next (it will still
    # fail, since there's no real land_cover/river_network data on an empty
    # root) is a separate, expected failure, not the dependency check.
    assert "Cannot preprocess" not in result.stderr
    assert "prerequisites not satisfied" not in result.stderr


def test_preprocess_chain_stops_at_unresolvable_manual_source(tmp_path):
    """river_network has no automated fetch -- --chain can run every
    auto-resolvable prerequisite, but must fail clearly (not crash) once it
    hits a source whose raw data can only be placed manually."""
    result = _run_cli(
        [
            "data",
            "preprocess",
            "--source",
            "land_cover",
            "--phase",
            "extract",
            "--chain",
            "--root-dir",
            str(tmp_path),
        ]
    )

    assert result.returncode == 1
    assert "must be placed manually" in result.stderr
    assert "river_network" in result.stderr


def test_preprocess_passes_through_silently_once_prerequisites_are_satisfied(tmp_path):
    """biomes.preprocess requires sensor_data to be fetched (its raw
    'stations' table), gadm to be preprocessed (its simplified boundary
    output), and biomes' own raw archive -- once all three exist, the gate
    must let the run proceed to biomes' own preprocessing code instead of
    blocking."""
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point

    from src.data.sources.biomes.constants import archive_path
    from src.data.sources.gadm.constants import DEFAULT_ADM0_LAYER, DEFAULT_ADM2_LAYER, DEFAULT_SIMPLIFIED_GADM_PATH
    from src.data.sources.sensor_data.fetch.database import STATIONS_TABLE, write_geodataframe_table

    stations = pd.DataFrame({"station_code": ["11111111"]})
    stations_geo = gpd.GeoDataFrame(
        stations, geometry=gpd.points_from_xy([-45.0], [-10.0]), crs=4326
    )
    write_geodataframe_table(str(tmp_path), STATIONS_TABLE, stations_geo)

    biomes_archive = archive_path(tmp_path)
    biomes_archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(biomes_archive, "w") as archive:
        archive.writestr("biomes.shp", "fake-shapefile-bytes" * 100)

    gadm_path = tmp_path / DEFAULT_SIMPLIFIED_GADM_PATH
    gadm_path.parent.mkdir(parents=True, exist_ok=True)
    frame = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs=4326)
    frame.to_file(gadm_path, layer=DEFAULT_ADM0_LAYER, driver="GPKG")
    frame.to_file(gadm_path, layer=DEFAULT_ADM2_LAYER, driver="GPKG")

    result = _run_cli(["data", "preprocess", "--source", "biomes", "--root-dir", str(tmp_path)])

    assert "Cannot preprocess" not in result.stderr
    assert "prerequisites not satisfied" not in result.stderr
