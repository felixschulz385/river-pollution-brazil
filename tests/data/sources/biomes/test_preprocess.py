from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point

from src.data.sources.biomes.preprocess import build_adm2_biomes


def test_build_adm2_biomes_excludes_non_municipality_gadm_lagoons(tmp_path: Path):
    """GADM's Brazil ADM2 layer includes Lagoa dos Patos/Lagoa Mirim as
    fake-CC_2 "municipality" polygons (4300001/4300002). Overlaying them
    alongside a real municipality and truncating both to mun_id would
    collapse them onto the same bogus mun_id="430000", duplicating this
    table's key; they must be excluded before the overlay is even built."""
    base = Point(-47.9, -15.8)

    gadm_path = tmp_path / "gadm.gpkg"
    adm2_boundary = gpd.GeoDataFrame(
        {
            "CC_2": ["4314902", "4300001"],
            "geometry": [base.buffer(1.0), base.buffer(1.0)],
        },
        crs=4326,
    )
    adm2_boundary.to_file(gadm_path, layer="ADM_ADM_2", driver="GPKG")

    biome_path = tmp_path / "biomes.shp"
    biome_polygons = gpd.GeoDataFrame(
        {"NM_BIOMA": ["Cerrado"], "geometry": [base.buffer(2.0)]},
        crs=4326,
    )
    biome_polygons.to_file(biome_path)

    result = build_adm2_biomes(
        root_dir=str(tmp_path),
        shapefile_path=biome_path,
        gadm_path=str(gadm_path),
        layer="ADM_ADM_2",
        output_path=str(tmp_path / "biomes_adm2.parquet"),
    )

    assert result["mun_id"].tolist() == ["431490"]
