from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from src.data.sources.geology.preprocess import build_adm2_geology


def _write_inputs(tmp_path: Path):
    """Two 1°x1° municipalities side by side; a district over the west half of the first."""
    gadm_path = tmp_path / "gadm.gpkg"
    gpd.GeoDataFrame(
        {"CC_2": ["1500100", "1500200"], "geometry": [box(-56, -6, -55, -5), box(-55, -6, -54, -5)]},
        crs=4326,
    ).to_file(gadm_path, layer="ADM_ADM_2", driver="GPKG")

    gpkg = tmp_path / "sgb_gold.gpkg"
    gpd.GeoDataFrame(
        {
            "SUBSTANCIAS": ["Ouro", "Ouro, Prata", "Ouro", "Estanho", "Ouro", "Ouro"],
            "STATUS_ECONOMICO": ["Garimpo", "Não explotado", "Mina", "Garimpo", "Não explotado", "Garimpo"],
            "ROCHAS_HOSPEDEIRAS": ["Alúvio", None, "Veio de quartzo", "Cascalho", "Cascalho", None],
            "geometry": [
                Point(-55.5, -5.5),
                Point(-55.2, -5.5),
                Point(-54.5, -5.5),
                Point(-54.5, -5.6),  # tin: counted separately
                Point(-54.0 + 0.01, -5.5),  # ~1 km east of the second municipality
                Point(-50.0, -5.5),  # far from both: dropped
            ],
        },
        crs=4326,
    ).to_file(gpkg, layer="occurrences", driver="GPKG")
    gpd.GeoDataFrame({"geometry": [box(-56, -6, -55.5, -5)]}, crs=4326).to_file(
        gpkg, layer="gold_districts", driver="GPKG"
    )
    # Overlapping provinces must not be double-counted.
    gpd.GeoDataFrame({"geometry": [box(-56, -6, -55, -5), box(-56, -6, -55, -5)]}, crs=4326).to_file(
        gpkg, layer="gold_provinces", driver="GPKG"
    )
    return gadm_path, gpkg


def test_build_adm2_geology_counts_and_shares(tmp_path: Path):
    gadm_path, gpkg = _write_inputs(tmp_path)

    result = build_adm2_geology(
        root_dir=str(tmp_path),
        gpkg=gpkg,
        gadm_path=str(gadm_path),
        layer="ADM_ADM_2",
        output_path=str(tmp_path / "geology_adm2.parquet"),
    ).set_index("mun_id")

    assert result.index.tolist() == ["150010", "150020"]
    assert result["gold_occurrences"].tolist() == [2, 2]
    assert result["gold_occurrences_garimpo"].tolist() == [1, 0]
    assert result["gold_occurrences_unexploited"].tolist() == [1, 1]
    assert result["gold_occurrences_mine"].tolist() == [0, 1]
    assert result["gold_occurrences_alluvial"].tolist() == [1, 1]
    assert result["tin_occurrences"].tolist() == [0, 1]
    assert result["tin_occurrences_unexploited"].tolist() == [0, 0]
    assert result.loc["150010", "gold_district_share"] == pytest.approx(0.5, abs=0.01)
    assert result.loc["150020", "gold_district_share"] == 0.0
    assert result.loc["150010", "gold_province_share"] == pytest.approx(1.0)
    assert result.loc["150020", "gold_province_share"] == pytest.approx(0.0, abs=1e-6)
    assert result["area_km2"].between(12_000, 12_500).all()
