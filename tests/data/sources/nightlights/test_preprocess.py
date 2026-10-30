from __future__ import annotations

import gzip
import shutil

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box

from src.data.sources.nightlights.constants import ADM2_COLUMNS, parse_name
from src.data.sources.nightlights.preprocess import build_nightlights
from src.data.verification.sources import _nightlights_adm2_checks


PIXEL = 0.01
TRANSFORM = from_origin(-56.5, -4.5, PIXEL, PIXEL)
SHAPE = (200, 300)  # -56.5 to -53.5, -6.5 to -4.5


def _write(folder, name, values, document=None):
    """A gzipped single-band GeoTIFF as EOG serves it, with the original name in its document-name tag."""
    folder.mkdir(parents=True, exist_ok=True)
    plain = folder.parent / name.removesuffix(".gz")
    with rasterio.open(plain, "w", driver="GTiff", width=SHAPE[1], height=SHAPE[0], count=1, dtype=values.dtype,
                       crs="EPSG:4326", transform=TRANSFORM) as target:
        target.write(values, 1)
        target.update_tags(TIFFTAG_DOCUMENTNAME=f"./{document or name.removesuffix('.gz')}")
    with open(plain, "rb") as source, gzip.open(folder / name, "wb") as target:
        shutil.copyfileobj(source, target)
    plain.unlink()


def _cell(lon, lat):
    return int((-4.5 - lat) / PIXEL), int((lon + 56.5) / PIXEL)


def _radiance(level):
    """`level` over the west half of the first municipality and 2x `level` over the second; one negative cell."""
    values = np.zeros(SHAPE, dtype="float32")
    (r0, c0), (r1, c1) = _cell(-56, -5), _cell(-55.5, -6)
    values[r0:r1, c0:c1] = level
    (r0, c0), (r1, c1) = _cell(-55, -5), _cell(-54, -6)
    values[r0:r1, c0:c1] = 2 * level
    values[_cell(-54.105, -5.905)] = -1.5
    return values


@pytest.fixture
def built(tmp_path):
    gadm_path = tmp_path / "gadm.gpkg"
    gpd.GeoDataFrame(
        {"CC_2": ["1500100", "1500200"], "geometry": [box(-56, -6, -55, -5), box(-55, -6, -54, -5)]}, crs=4326
    ).to_file(gadm_path, layer="ADM_ADM_2", driver="GPKG")
    areas_path = tmp_path / "protected_areas.parquet"
    gpd.GeoDataFrame({
        "mining_barred": [True, True, False], "in_force_2012": [True, False, True],
        "geometry": [box(-56, -6, -55.5, -5), box(-55, -6, -54.5, -5), box(-54.5, -6, -54, -5)],
    }, crs=4326).to_parquet(areas_path)
    geology_gpkg = tmp_path / "sgb.gpkg"
    gpd.GeoDataFrame({
        "SUBSTANCIAS": ["Ouro", "Estanho"], "ROCHAS_HOSPEDEIRAS": [None, None],
        "STATUS_ECONOMICO": ["Não explotado", "Não explotado"],
        "geometry": [Point(-55.5, -5.5), Point(-54.5, -5.5)],
    }, crs=4326).to_file(geology_gpkg, layer="occurrences", driver="GPKG")

    raw = tmp_path / "data" / "nightlights" / "raw"
    v21, v22 = raw / "eog_vnl_v21", raw / "eog_vnl_2026-01-01"
    stem21 = "VNL_v21_npp_2021_global_vcmslcfg_c202205302300"
    stem22 = "VNL_v22_npp-j01_2022_global_vcmslcfg_c202303062300"
    _write(v21, f"{stem21}.average_masked.dat.tif.gz", _radiance(1.0))
    _write(v21, f"{stem21}.cf_cvg.dat.tif.gz", np.full(SHAPE, 10, dtype="uint16"))
    _write(v22, f"{stem22}.average_masked.dat.tif.gz", _radiance(3.0))
    # Saved under the wrong name: the file holds the 2021 median.
    _write(v22, f"{stem22}.cf_cvg.dat.tif.gz", _radiance(1.0), document=f"{stem21}.median_masked.dat.tif")
    result = build_nightlights(root_dir=str(tmp_path), gadm_path=str(gadm_path), protected_areas_path=str(areas_path),
                               geology_gpkg=str(geology_gpkg), workers=1)
    return result.set_index(["mun_id", "year"])


def test_names_parse_across_versions():
    assert parse_name("VNL_v21_npp_201204-201212_global_vcmcfg_c202205302300.average_masked.dat.tif.gz") == (
        2012, "average_masked", "v2.1", "npp")
    assert parse_name("VNL_v22_npp-j01_2022_global_vcmslcfg_c202303062300.cf_cvg.dat.tif.gz") == (
        2022, "cf_cvg", "v2.2", "npp-j01")
    assert parse_name("VNL_npp_2023_global_vcmslcfg_v2_c202402081600.average_masked.dat.tif.gz") == (
        2023, "average_masked", "v2.2", "npp")


def test_radiance_and_lit_pixels_per_municipality_with_negatives_set_to_zero(built):
    assert list(built.reset_index().columns) == list(ADM2_COLUMNS)
    first, second = built.loc[("150010", 2021)], built.loc[("150020", 2021)]
    assert first["viirs_pixels"] == second["viirs_pixels"] == 10_000
    assert first["viirs_lit_pixels"] == 5_000 and first["viirs_radiance"] == pytest.approx(5_000)
    assert second["viirs_lit_pixels"] == 9_999 and second["viirs_radiance"] == pytest.approx(2 * 9_999)
    assert built.loc[("150010", 2022), "viirs_radiance"] == pytest.approx(15_000)
    assert built.loc[("150010", 2021), "viirs_version"] == "v2.1"
    assert built.loc[("150010", 2022), "viirs_version"] == "v2.2"


def test_zones_are_barred_areas_in_force_by_2012_and_5km_around_gold(built):
    first, second = built.loc[("150010", 2021)], built.loc[("150020", 2021)]
    # The west half of the first municipality; the second's barred area came after 2012.
    assert first["viirs_pixels_protected"] == 5_000 and first["viirs_radiance_protected"] == pytest.approx(5_000)
    assert second["viirs_pixels_protected"] == 0
    # A 5 km circle (~64 pixels of 0.01 degrees) on the edge of the protected half; tin does not count.
    assert 50 < first["viirs_pixels_gold5km"] < 80 and second["viirs_pixels_gold5km"] == 0
    assert 0 < first["viirs_pixels_gold5km_protected"] < first["viirs_pixels_gold5km"]
    assert first["viirs_radiance_gold5km"] == pytest.approx(first["viirs_pixels_gold5km_protected"])


def test_misnamed_coverage_file_is_skipped_and_the_table_verifies(built):
    assert built.loc[("150010", 2021), "viirs_cf_cvg_mean"] == pytest.approx(10.0)
    assert built.xs(2022, level="year")["viirs_cf_cvg_mean"].isna().all()
    checks = _nightlights_adm2_checks(built.reset_index())
    assert all(check.ok for check in checks), [check.message for check in checks if not check.ok]
