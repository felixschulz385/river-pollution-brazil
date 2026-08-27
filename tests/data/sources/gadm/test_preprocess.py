from __future__ import annotations

import geopandas as gpd
import pyogrio
from shapely.geometry import Point

from src.data.sources.gadm.constants import DEFAULT_ADM0_LAYER, DEFAULT_ADM2_LAYER, DEFAULT_SIMPLIFY_TOLERANCE
from src.data.sources.gadm.preprocess import simplify_gadm


def _write_raw_fixture(path, *, adm0_layer=DEFAULT_ADM0_LAYER, adm2_layer=DEFAULT_ADM2_LAYER):
    # A high-resolution circle has many vertices -- simplification at any
    # meaningful tolerance measurably reduces that count, unlike a simple
    # square/triangle where a coarse tolerance might not remove any point.
    circle = Point(0, 0).buffer(1.0, quad_segs=64)
    adm0 = gpd.GeoDataFrame({"geometry": [circle]}, crs=4326)
    adm0.to_file(path, layer=adm0_layer, driver="GPKG")
    adm2 = gpd.GeoDataFrame({"geometry": [circle], "CC_2": ["1234567"]}, crs=4326)
    adm2.to_file(path, layer=adm2_layer, driver="GPKG")
    return len(list(circle.exterior.coords))


def test_simplify_gadm_writes_both_layers(tmp_path):
    raw_path = tmp_path / "raw.gpkg"
    _write_raw_fixture(raw_path)
    output_path = tmp_path / "simplified.gpkg"

    result_path = simplify_gadm(gadm_path=str(raw_path), output_path=str(output_path))

    assert result_path == output_path
    layers = [name for name, _geom_type in pyogrio.list_layers(output_path)]
    assert set(layers) == {DEFAULT_ADM0_LAYER, DEFAULT_ADM2_LAYER}


def test_simplify_gadm_reduces_vertex_count(tmp_path):
    raw_path = tmp_path / "raw.gpkg"
    raw_vertex_count = _write_raw_fixture(raw_path)
    output_path = tmp_path / "simplified.gpkg"

    simplify_gadm(gadm_path=str(raw_path), output_path=str(output_path), tolerance=0.1)

    simplified = gpd.read_file(output_path, layer=DEFAULT_ADM0_LAYER)
    simplified_vertex_count = len(list(simplified.geometry.iloc[0].exterior.coords))
    assert simplified_vertex_count < raw_vertex_count


def test_simplify_gadm_default_tolerance_matches_constant(tmp_path):
    raw_path = tmp_path / "raw.gpkg"
    _write_raw_fixture(raw_path)

    default_output = tmp_path / "default.gpkg"
    explicit_output = tmp_path / "explicit.gpkg"
    simplify_gadm(gadm_path=str(raw_path), output_path=str(default_output))
    simplify_gadm(gadm_path=str(raw_path), output_path=str(explicit_output), tolerance=DEFAULT_SIMPLIFY_TOLERANCE)

    default_geom = gpd.read_file(default_output, layer=DEFAULT_ADM0_LAYER).geometry.iloc[0]
    explicit_geom = gpd.read_file(explicit_output, layer=DEFAULT_ADM0_LAYER).geometry.iloc[0]
    assert default_geom.equals(explicit_geom)


def test_simplify_gadm_rerun_rebuilds_cleanly(tmp_path):
    """A second run (e.g. after the raw file changes) must not leave a stale
    or duplicated layer behind from the first run's output."""
    raw_path = tmp_path / "raw.gpkg"
    _write_raw_fixture(raw_path)
    output_path = tmp_path / "simplified.gpkg"

    simplify_gadm(gadm_path=str(raw_path), output_path=str(output_path), tolerance=0.01)
    simplify_gadm(gadm_path=str(raw_path), output_path=str(output_path), tolerance=0.2)

    layers = [name for name, _geom_type in pyogrio.list_layers(output_path)]
    assert sorted(layers) == sorted({DEFAULT_ADM0_LAYER, DEFAULT_ADM2_LAYER})
