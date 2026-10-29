from __future__ import annotations

import zipfile

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, Polygon, box

from src.data.sources.protected_areas import preprocess
from src.data.sources.protected_areas.constants import AREA_COLUMNS, ADM2_COLUMNS
from src.data.sources.protected_areas.preprocess import build_protected_areas, protected_mask
from src.data.verification.sources import _protected_area_checks, _protected_areas_adm2_checks


LAYERS = {"conservation_unit": ("ws:conservation_units_legal_amazon", "ws:conservation_units_amazon_biome")}


def _write_funai(tmp_path, lands):
    """FUNAI's layer as served: a zipped shapefile in EPSG:4674 in a dated snapshot folder."""
    folder = tmp_path / "shapefile"
    folder.mkdir()
    dates = [column for column in lands.columns if column.startswith("data_")]
    lands = lands.assign(**{column: lands[column].map(lambda value: None if pd.isna(value) else value.date())
                            for column in dates})  # shapefile Date fields
    lands.to_crs(4674).to_file(folder / "tis_poligonais_portarias.shp", driver="ESRI Shapefile")
    snapshot = tmp_path / "data" / "protected_areas" / "raw" / "funai_2026-01-01"
    snapshot.mkdir(parents=True)
    with zipfile.ZipFile(snapshot / "tis_poligonais_portarias.zip", "w") as archive:
        for path in folder.iterdir():
            archive.write(path, path.name)


def _write_inputs(tmp_path):
    """Two 1°x1° municipalities. Over the west half of the first: a land homologated in 1991 and, inside it,
    a land declared in 2016, an undated reserve still being acquired, an undated regularised reserve and a land
    still marked Delimitada with a 2010 declaration; a land under study over the second. A park (2005) in two
    pieces, a RESEX (2015), a FLONA and an APA."""
    gadm_path = tmp_path / "gadm.gpkg"
    gpd.GeoDataFrame(
        {"CC_2": ["1500100", "1500200"], "geometry": [box(-56, -6, -55, -5), box(-55, -6, -54, -5)]}, crs=4326
    ).to_file(gadm_path, layer="ADM_ADM_2", driver="GPKG")

    date = pd.Timestamp
    _write_funai(tmp_path, gpd.GeoDataFrame({
        "terrai_cod": [101, 102, 103, 104, 105, 106],
        "terrai_nom": ["A", "B", "C", "D", "E", "F"],
        "fase_ti": ["Regularizada", "Em Estudo", "Homologada", "Encaminhada RI", "Regularizada", "Delimitada"],
        "modalidade": ["Tradicionalmente ocupada", "Tradicionalmente ocupada", "Tradicionalmente ocupada",
                       "Reserva Indígena", "Reserva Indígena", "Tradicionalmente ocupada"],
        "data_decla": [pd.NaT, pd.NaT, date("2016-04-19"), pd.NaT, pd.NaT, date("2010-05-03")],
        "data_homol": [date("1991-10-29"), pd.NaT, date("2025-11-17"), pd.NaT, pd.NaT, pd.NaT],
        "data_regul": [date("1993-01-05"), pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT],
        "geometry": [
            MultiPolygon([box(-56, -6, -55.5, -5.5), box(-56, -5.5, -55.5, -5)]), box(-55, -6, -54, -5),
            box(-56, -6, -55.75, -5), box(-55.6, -5.6, -55.5, -5.5), box(-55.7, -5.6, -55.6, -5.5),
            box(-55.8, -5.6, -55.7, -5.5),
        ],
    }, crs=4326))

    gpkg = tmp_path / "areas.gpkg"
    degenerate = MultiPolygon([box(-55.5, -6, -55.25, -5.5), Polygon([(-55.3, -5.8)] * 4)])
    stray = GeometryCollection([box(-55.5, -5.5, -55.25, -5), LineString([(-55.5, -5), (-55, -5)])])
    gpd.GeoDataFrame({
        "nome": ["Parque P", "Resex R", "Flona F"], "categoria": ["Parque", "Reserva Extrativista", "Floresta"],
        "grupo": ["PI", "US", "US"], "esfera": ["federal", "federal", "federal"],
        "ano_cria": ["2005", "e 2015 de 01/01/2015", "2001"],
        "geometry": [degenerate, box(-55.25, -6, -55, -5), box(-55, -6, -54.5, -5)],
    }, crs=4326).to_file(gpkg, layer="conservation_units_legal_amazon", driver="GPKG")
    gpd.GeoDataFrame({
        "nome": ["APA Q", "Parque P"], "categoria": ["Área de Proteção Ambiental", "Parque"], "grupo": ["US", "PI"],
        "esfera": ["estadual", "federal"], "ano_cria": ["2000", "2005"],
        "geometry": [box(-54.5, -6, -54, -5), stray],  # the park's other piece, with a stray line
    }, crs=4326).to_file(gpkg, layer="conservation_units_amazon_biome", driver="GPKG")
    return gadm_path, gpkg


@pytest.fixture
def built(tmp_path, monkeypatch):
    monkeypatch.setattr(preprocess, "LAYERS", LAYERS)  # FUNAI's layer is found in the snapshot folder
    gadm_path, gpkg = _write_inputs(tmp_path)
    areas, shares = build_protected_areas(root_dir=str(tmp_path), gpkg=gpkg, gadm_path=str(gadm_path),
                                          layer="ADM_ADM_2")
    return areas.set_index("area_id"), shares.set_index("mun_id")


def test_areas_merge_pieces_and_flag_barred_and_2012(built):
    areas, _ = built

    assert sorted(areas.index) == [
        "ti:101", "ti:102", "ti:103", "ti:104", "ti:105", "ti:106", "uc:0000", "uc:0001", "uc:0002", "uc:0003",
    ]
    lands = areas[areas["kind"] == "indigenous_land"]
    assert lands["mining_barred"].to_dict() == {
        "ti:101": True, "ti:102": False, "ti:103": True, "ti:104": True, "ti:105": True,
        "ti:106": True,  # declared, though still marked Delimitada
    }
    assert lands["year_barred"].tolist() == [1991, pd.NA, 2016, pd.NA, pd.NA, 2010]
    assert lands["in_force_2012"].to_dict() == {
        "ti:101": True, "ti:102": False, "ti:103": False, "ti:104": False,
        "ti:105": True,  # undated but regularised
        "ti:106": True,
    }
    assert lands.loc["ti:103", "date_homologated"] == pd.Timestamp("2025-11-17")
    assert lands.loc["ti:104", "category"] == "Reserva Indígena"
    assert lands.loc["ti:101"].geometry.area == pytest.approx(0.5)
    units = areas[areas["kind"] == "conservation_unit"].set_index("name")
    assert units.loc["Parque P"].geometry.area == pytest.approx(0.25)  # both pieces, without the stray line
    assert units["mining_barred"].to_dict() == {"APA Q": False, "Flona F": False, "Parque P": True, "Resex R": True}
    assert units["in_force_2012"].to_dict() == {"APA Q": True, "Flona F": True, "Parque P": True, "Resex R": False}
    assert units["year_barred"].tolist() == [pd.NA, pd.NA, 2005, 2015]
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
