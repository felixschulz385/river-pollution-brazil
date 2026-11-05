from __future__ import annotations

import zipfile
from datetime import date

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon, box

from src.data.sources.cloud_cover.constants import (
    ALERTS_FIRST_MONTH,
    CLOUD_MEMBER,
    CLOUD_PRODES_COLUMN,
    CLOUD_SHARE_COLUMN,
    FIRST_YEAR,
    LAST_YEAR,
    SAD69,
    SAD69_PRE96_BR,
    SIRGAS2000,
    alerts_date,
    layer_crs,
)
from src.data.sources.cloud_cover.preprocess import (
    alert_month_table,
    alert_year_table,
    build_alerts,
    build_clouds,
    cloud_layers,
    cloud_month_table,
    cloud_year_table,
    covered_shares,
    last_full_month,
    month_clouds,
)
from src.data.verification.sources import (
    _alert_month_checks,
    _alert_year_checks,
    _cloud_month_checks,
    _cloud_year_checks,
)


def test_cloud_layers_take_the_month_from_the_folder_and_keep_both_layers():
    names = [
        "input/2006/Nuvem_20060531_shp/Nuvem_20060531_B_pol.shp",
        "input/2006/Nuvem_20060531_shp/Nuvem_20060531_A_pol.shp",
        "input/2006/Nuvem_20060531_shp/Nuvem_20060531_A_pol.dbf",
        "input/2011/Nuvens_201103_shp/DETER2011_MARCO_NUVEM_pol.shp",
        "input/2011/Nuvens_201103_shp/DETER2011_MARCO_NUVEM_props.txt",
        "input/2011/Deter_201103_shp/DETER2011_MARCO_pol.shp",
        "input/2008/Nuvens_20081130_shp/",
    ]
    assert cloud_layers(names) == {
        (2006, 5): ["input/2006/Nuvem_20060531_shp/Nuvem_20060531_A_pol.shp",
                    "input/2006/Nuvem_20060531_shp/Nuvem_20060531_B_pol.shp"],
        (2011, 3): ["input/2011/Nuvens_201103_shp/DETER2011_MARCO_NUVEM_pol.shp"],
    }


def test_layer_crs_follows_inpe_periods():
    assert layer_crs(2010, 12) == SAD69
    assert layer_crs(2011, 1) == SAD69_PRE96_BR
    assert layer_crs(2015, 7) == SAD69_PRE96_BR
    assert layer_crs(2015, 8) == SIRGAS2000
    assert layer_crs(2016, 1) == SIRGAS2000


def test_month_clouds_unions_within_a_layer_and_intersects_across_layers():
    overlapping = [box(0, 0, 2, 2), box(1, 0, 3, 2)]
    assert month_clouds([overlapping]).area == pytest.approx(6.0)
    assert month_clouds([overlapping, [box(2, 0, 4, 2)]]).area == pytest.approx(2.0)
    assert month_clouds([[box(0, 0, 1, 1)], [box(5, 5, 6, 6)]]).is_empty


def test_month_clouds_repairs_invalid_polygons():
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])  # self-intersecting: two triangles of area 1
    assert month_clouds([[bowtie]]).area == pytest.approx(2.0)


def test_covered_shares():
    municipalities = [box(0, 0, 2, 1), box(2, 0, 4, 1), box(10, 10, 11, 11)]
    clouds = month_clouds([[box(1, 0, 3, 1)]])
    np.testing.assert_allclose(covered_shares(municipalities, clouds), [0.5, 0.5, 0.0])
    np.testing.assert_allclose(covered_shares(municipalities, Polygon()), [0.0, 0.0, 0.0])


def test_cloud_tables_impute_missing_months_and_average_prodes_years():
    measured = {(year, month): np.array([0.0, 0.5]) for year in range(FIRST_YEAR, LAST_YEAR + 1)
                for month in range(1, 13) if (year, month) != (2006, 2)}
    monthly = cloud_month_table(np.array(["150010", "150020"]), measured)
    assert len(monthly) == 2 * 12 * (LAST_YEAR - FIRST_YEAR + 1)
    february = monthly[(monthly["year"] == 2006) & (monthly["month"] == 2)]
    assert february["deter_cloud_imputed"].all() and (february[CLOUD_SHARE_COLUMN] == 1.0).all()

    yearly = cloud_year_table(monthly).set_index(["mun_id", "year"])
    assert yearly.loc[("150010", 2006), CLOUD_SHARE_COLUMN] == pytest.approx(1 / 12)
    # PRODES 2006 runs from August 2005 to July 2006 and holds the imputed February.
    assert yearly.loc[("150010", 2006), CLOUD_PRODES_COLUMN] == pytest.approx(1 / 12)
    assert yearly.loc[("150010", 2007), CLOUD_PRODES_COLUMN] == pytest.approx(0.0)
    assert yearly.loc[("150020", 2008), CLOUD_PRODES_COLUMN] == pytest.approx(0.5)
    assert np.isnan(yearly.loc[("150010", FIRST_YEAR), CLOUD_PRODES_COLUMN])
    assert sorted(yearly.index.get_level_values("year").unique()) == list(range(FIRST_YEAR, LAST_YEAR + 1))
    assert all(check.ok for check in _cloud_month_checks(monthly)), _cloud_month_checks(monthly)
    assert all(check.ok for check in _cloud_year_checks(yearly.reset_index()))


def _gadm(tmp_path):
    """Two Pará municipalities, one in Maranhão west of 44°W, one east of it and one in São Paulo."""
    path = tmp_path / "gadm.gpkg"
    gpd.GeoDataFrame(
        {"CC_2": ["1500100", "1500200", "2100100", "2100200", "3500100"],
         "geometry": [box(-56, -6, -55, -5), box(-55, -6, -54, -5), box(-45, -3, -44.5, -2),
                      box(-43, -3, -42, -2), box(-48, -22, -47, -21)]},
        crs=4326,
    ).to_file(path, layer="ADM_ADM_2", driver="GPKG")
    return path


def _replication_package(tmp_path):
    """A package whose inner zip has May 2005 (one layer, no .prj) and June 2006 (two layers, SIRGAS 2000)."""
    layers = tmp_path / "layers"
    files = {
        "input/2005/Nuvem_20050531_shp/nuvem_20050531_pol": (None, [box(-56, -6, -55.5, -5)]),
        "input/2006/Nuvem_20060630_shp/Nuvem_A_pol": (SIRGAS2000, [box(-55, -6, -54, -5)]),
        "input/2006/Nuvem_20060630_shp/Nuvem_B_pol": (SIRGAS2000, [box(-54.5, -6, -53, -5)]),
    }
    inner = tmp_path / "input.zip"
    with zipfile.ZipFile(inner, "w") as bundle:
        for name, (crs, geometries) in files.items():
            target = layers / f"{name}.shp"
            target.parent.mkdir(parents=True, exist_ok=True)
            frame = gpd.GeoDataFrame({"class_name": ["NUVEM"] * len(geometries)}, geometry=geometries, crs=crs)
            frame.to_file(target, driver="ESRI Shapefile")
            for part in target.parent.glob(f"{target.stem}.*"):
                bundle.write(part, f"{name}{part.suffix}")
    package = tmp_path / "132281-V1.zip"
    with zipfile.ZipFile(package, "w") as bundle:
        bundle.write(inner, CLOUD_MEMBER)
    return package


@pytest.mark.filterwarnings("ignore:'crs' was not provided:UserWarning")
def test_build_clouds(tmp_path):
    monthly, yearly = build_clouds(
        root_dir=tmp_path, path=_replication_package(tmp_path), gadm_path=_gadm(tmp_path), workers=1,
        month_output_path="clouds_month.parquet", adm2_output_path="clouds.parquet",
    )
    assert sorted(monthly["mun_id"].unique()) == ["150010", "150020", "210010"]
    shares = monthly.set_index(["mun_id", "year", "month"])[CLOUD_SHARE_COLUMN]
    assert shares[("150010", 2005, 5)] == pytest.approx(0.5, abs=1e-3)
    assert shares[("150020", 2005, 5)] == pytest.approx(0.0)
    assert shares[("150020", 2006, 6)] == pytest.approx(0.5, abs=1e-3)  # clouded in both layers
    assert shares[("150010", 2006, 6)] == pytest.approx(0.0)
    assert shares[("150010", 2005, 4)] == 1.0  # no data
    assert monthly["deter_cloud_imputed"].sum() == 3 * (12 * (LAST_YEAR - FIRST_YEAR + 1) - 2)
    assert (tmp_path / "clouds_month.parquet").exists() and (tmp_path / "clouds.parquet").exists()
    assert all(check.ok for check in _cloud_month_checks(monthly))
    assert all(check.ok for check in _cloud_year_checks(yearly))


def _alerts(rows):
    return pd.DataFrame(rows, columns=["CLASSNAME", "VIEW_DATE", "AREAMUNKM", "GEOCODIBGE"])


def test_alert_month_table_groups_and_fills_the_grid():
    alerts = _alerts([
        ("MINERACAO", "2016-08-03", 0.2, 1500107),
        ("MINERACAO", "2016-08-20", 0.3, 1500107),
        ("DESMATAMENTO_CR", "2016-09-01", 1.0, 1500107),
        ("CICATRIZ_DE_QUEIMADA", "2016-09-02", 2.0, 1500206),
        ("MINERACAO", "2016-11-01", 9.0, 1500107),  # after the last month
        ("MINERACAO", "2016-09-01", 9.0, 5100102),  # outside the map
    ])
    monthly = alert_month_table(alerts, np.array(["150010", "150020"]), (2016, 8), (2016, 10))
    assert len(monthly) == 2 * 3
    table = monthly.set_index(["mun_id", "year", "month"])
    assert table.loc[("150010", 2016, 8), "deter_alerts_mining"] == 2
    assert table.loc[("150010", 2016, 8), "deter_alert_area_km2_mining"] == pytest.approx(0.5)
    assert table.loc[("150010", 2016, 9), "deter_alerts_deforestation"] == 1
    assert table.loc[("150020", 2016, 9), "deter_alerts_all"] == 1
    assert table.loc[("150020", 2016, 9), "deter_alerts_deforestation"] == 0
    assert table["deter_alerts_all"].sum() == 4
    assert all(check.ok for check in _alert_month_checks(monthly))


def test_alert_year_table_keeps_full_years():
    alerts = _alerts([("MINERACAO", "2017-03-01", 0.2, 1500107), ("MINERACAO", "2016-12-01", 0.1, 1500107)])
    yearly = alert_year_table(alert_month_table(alerts, np.array(["150010"]), (2016, 8), (2018, 5)))
    assert yearly["year"].tolist() == [2017]
    assert yearly["deter_alerts_mining"].tolist() == [1]
    assert all(check.ok for check in _alert_year_checks(yearly))


def test_last_full_month_and_download_date():
    assert last_full_month(date(2026, 11, 4)) == (2026, 10)
    assert last_full_month(date(2027, 1, 2)) == (2026, 12)
    assert alerts_date("deter-amz-04-11-2026-16_04_29.zip") == date(2026, 11, 4)


def test_build_alerts(tmp_path):
    shapefile = tmp_path / "deter" / "deter-amz-deter-public.shp"
    shapefile.parent.mkdir()
    gpd.GeoDataFrame(
        {"CLASSNAME": ["MINERACAO", "DESMATAMENTO_CR"], "VIEW_DATE": ["2019-05-02", "2017-01-10"],
         "AREAMUNKM": [0.4, 1.5], "GEOCODIBGE": [1500107, 1500206], "UF": ["PA", "PA"]},
        geometry=[box(-55.9, -5.9, -55.8, -5.8), box(-54.9, -5.9, -54.8, -5.8)], crs=4674,
    ).to_file(shapefile, driver="ESRI Shapefile")
    download = tmp_path / "deter-amz-04-02-2020-10_00_00.zip"
    with zipfile.ZipFile(download, "w") as bundle:
        for part in shapefile.parent.iterdir():
            bundle.write(part, part.name)

    monthly, yearly = build_alerts(root_dir=tmp_path, path=download, gadm_path=_gadm(tmp_path),
                                   month_output_path="alerts_month.parquet", adm2_output_path="alerts.parquet")
    months = (2020 - ALERTS_FIRST_MONTH[0]) * 12 + 1 - ALERTS_FIRST_MONTH[1] + 1  # August 2016 to January 2020
    assert len(monthly) == 3 * months
    assert yearly["year"].unique().tolist() == [2017, 2018, 2019]
    totals = yearly.groupby("mun_id")[["deter_alerts_mining", "deter_alerts_deforestation"]].sum()
    assert totals.loc["150010", "deter_alerts_mining"] == 1
    assert totals.loc["150020", "deter_alerts_deforestation"] == 1
    assert all(check.ok for check in _alert_month_checks(monthly))
