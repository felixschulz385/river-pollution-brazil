from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, Polygon, box

from src.data.sources.protected_areas import preprocess
from src.data.sources.protected_areas.constants import AREA_COLUMNS, ADM2_COLUMNS
from src.data.sources.protected_areas.preprocess import build_protected_areas, protected_mask
from src.data.verification.sources import _protected_area_checks, _protected_areas_adm2_checks


LAYERS = {
    "indigenous_land": ("ws:indigenous_area_legal_amazon", "ws:indigenous_area_amazon_biome"),
    "conservation_unit": ("ws:conservation_units_legal_amazon", "ws:conservation_units_amazon_biome"),
}


def _write_inputs(tmp_path):
    """Two 1°x1° municipalities. A regularised land split across two layers over the west half of the first;
    a land under study over the second; a park (2005), a RESEX (2015), a FLONA and an APA."""
    gadm_path = tmp_path / "gadm.gpkg"
    gpd.GeoDataFrame(
        {"CC_2": ["1500100", "1500200"], "geometry": [box(-56, -6, -55, -5), box(-55, -6, -54, -5)]}, crs=4326
    ).to_file(gadm_path, layer="ADM_ADM_2", driver="GPKG")

    gpkg = tmp_path / "areas.gpkg"
    stray = GeometryCollection([box(-56, -5.5, -55.5, -5), LineString([(-56, -5), (-55, -5)])])
    gpd.GeoDataFrame({
        "terrai_cod": [101, 102], "terrai_nom": ["A", "B"], "etnia_nome": ["x", "y"],
        "fase_ti": ["Regularizada", "Em Estudo"], "geometry": [box(-56, -6, -55.5, -5.5), box(-55, -6, -54, -5)],
    }, crs=4326).to_file(gpkg, layer="indigenous_area_legal_amazon", driver="GPKG")
    gpd.GeoDataFrame({
        "terrai_cod": [101], "terrai_nom": ["A"], "etnia_nome": ["x"], "fase_ti": ["Regularizada"],
        "geometry": [stray],  # the land's other piece, with a stray line
    }, crs=4326).to_file(gpkg, layer="indigenous_area_amazon_biome", driver="GPKG")
    degenerate = MultiPolygon([box(-55.5, -6, -55.25, -5), Polygon([(-55.3, -5.5)] * 4)])
    gpd.GeoDataFrame({
        "nome": ["Parque P", "Resex R", "Flona F"], "categoria": ["Parque", "Reserva Extrativista", "Floresta"],
        "grupo": ["PI", "US", "US"], "esfera": ["federal", "federal", "federal"],
        "ano_cria": ["2005", "e 2015 de 01/01/2015", "2001"],
        "geometry": [degenerate, box(-55.25, -6, -55, -5), box(-55, -6, -54.5, -5)],
    }, crs=4326).to_file(gpkg, layer="conservation_units_legal_amazon", driver="GPKG")
    gpd.GeoDataFrame({
        "nome": ["APA Q"], "categoria": ["Área de Proteção Ambiental"], "grupo": ["US"], "esfera": ["estadual"],
        "ano_cria": ["2000"], "geometry": [box(-54.5, -6, -54, -5)],
    }, crs=4326).to_file(gpkg, layer="conservation_units_amazon_biome", driver="GPKG")
    return gadm_path, gpkg


@pytest.fixture
def built(tmp_path, monkeypatch):
    monkeypatch.setattr(preprocess, "LAYERS", LAYERS)
    gadm_path, gpkg = _write_inputs(tmp_path)
    areas, shares = build_protected_areas(root_dir=str(tmp_path), gpkg=gpkg, gadm_path=str(gadm_path),
                                          layer="ADM_ADM_2")
    return areas.set_index("area_id"), shares.set_index("mun_id")


def test_areas_merge_pieces_and_flag_barred_and_2012(built):
    areas, _ = built

    assert sorted(areas.index) == ["ti:101", "ti:102", "uc:0000", "uc:0001", "uc:0002", "uc:0003"]
    land = areas.loc["ti:101"]
    assert (land["mining_barred"], land["in_force_2012"]) == (True, True)
    assert land.geometry.area == pytest.approx(0.5)  # both pieces, without the stray line
    assert not areas.loc["ti:102", "mining_barred"]  # under study
    units = areas[areas["kind"] == "conservation_unit"].set_index("name")
    assert units["mining_barred"].to_dict() == {"APA Q": False, "Flona F": False, "Parque P": True, "Resex R": True}
    assert units["in_force_2012"].to_dict() == {"APA Q": True, "Flona F": True, "Parque P": True, "Resex R": False}
    assert units.loc["Resex R", "year_created"] == 2015
    assert list(areas.reset_index().columns) == list(AREA_COLUMNS)


def test_shares_cover_barred_areas_and_those_in_force_by_2012(built):
    _, shares = built

    first = shares.loc["150010"]
    assert first["indigenous_land_share"] == pytest.approx(0.5, abs=0.01)
    assert first["conservation_barred_share"] == pytest.approx(0.5, abs=0.01)  # park and RESEX
    assert first["protected_share"] == pytest.approx(1.0, abs=0.01)
    assert first["protected_2012_share"] == pytest.approx(0.75, abs=0.01)  # not the 2015 RESEX
    assert shares.loc["150020"].sum() == pytest.approx(0.0)  # land under study, FLONA and APA
    assert list(shares.reset_index().columns) == list(ADM2_COLUMNS)


def test_protected_mask_and_verification(built):
    areas, shares = built
    points = gpd.GeoDataFrame(geometry=[Point(-55.8, -5.8), Point(-55.1, -5.5), Point(-54.2, -5.5)], crs=4326)

    assert protected_mask(points, areas.reset_index()).tolist() == [True, False, False]
    assert all(check.ok for check in _protected_area_checks(areas.reset_index()))
    assert all(check.ok for check in _protected_areas_adm2_checks(shares.reset_index()))
