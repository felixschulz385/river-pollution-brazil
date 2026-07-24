from pathlib import Path
import pickle
import re

import geopandas as gpd
import numpy as np
import odc.geo.xr  # noqa: F401
import pandas as pd
import rioxarray  # noqa: F401
from tqdm import tqdm
import xarray as xr
from odc.geo.geobox import GeoBox

from shared.spatial_tabular import mapping_to_long_frame, rasterize_feature_values


MONTH_DICT = {
    "JANEIRO": "01",
    "JANEIR0": "01",
    "FEVEREIRO": "02",
    "MARCO": "03",
    "MARÃ‡O": "03",
    "ABRIL": "04",
    "MAIO": "05",
    "JUNHO": "06",
    "JULHO": "07",
    "AGOSTO": "08",
    "SETEMBRO": "09",
    "OUTUBRO": "10",
    "NOVEMBRO": "11",
    "DEZEMBRO": "12",
    "jan": "01",
    "fev": "02",
    "mar": "03",
    "abr": "04",
    "mai": "05",
    "jun": "06",
    "jul": "07",
    "ago": "08",
    "set": "09",
    "out": "10",
    "nov": "11",
    "dez": "12",
    "Jan": "01",
    "Fev": "02",
    "Mar": "03",
    "Abr": "04",
    "Mai": "05",
    "Jun": "06",
    "Jul": "07",
    "Ago": "08",
    "Set": "09",
    "Out": "10",
    "Nov": "11",
    "Dez": "12",
    "JAN": "01",
    "FEV": "02",
    "MAR": "03",
    "ABR": "04",
    "MAI": "05",
    "JUN": "06",
    "JUL": "07",
    "AGO": "08",
    "SET": "09",
    "OUT": "10",
    "NOV": "11",
    "DEZ": "12",
}


def _root(root_dir="."):
    return Path(root_dir)


def _climate_raw_dir(root_dir="."):
    return _root(root_dir) / "data" / "climate" / "raw"


def _load_boundaries(root_dir="."):
    gpkg_path = _root(root_dir) / "data" / "gadm" / "gadm41_BRA.gpkg"
    json_path = _root(root_dir) / "data" / "misc" / "raw" / "gadm" / "gadm41_BRA_2.json"

    if gpkg_path.exists():
        boundaries = gpd.read_file(
            gpkg_path,
            layer="ADM_ADM_2",
            engine="pyogrio",
        )
    elif json_path.exists():
        boundaries = gpd.read_file(
            json_path,
            engine="pyogrio",
        )
    else:
        raise FileNotFoundError(
            "Could not find GADM ADM2 boundaries. Checked "
            f"{gpkg_path} and {json_path}."
        )

    boundaries["CC_2r"] = boundaries.CC_2.str.slice(0, 6).astype(int)
    return boundaries


def _load_weather_data(root_dir="."):
    raw_dir = _climate_raw_dir(root_dir)
    precipitation = xr.open_mfdataset(
        str(raw_dir / "precip.*.nc"),
        chunks="auto",
        decode_times=True,
        decode_cf=True,
    )
    temperature = xr.open_mfdataset(
        str(raw_dir / "tmax.*.nc"),
        chunks="auto",
        decode_times=True,
        decode_cf=True,
    )

    weather_data = xr.merge([temperature, precipitation])
    weather_data = weather_data.assign_coords(
        {
            "lon": np.vectorize(
                lambda lon: lon - 360 if lon > 180 else lon
            )(precipitation.lon)
        }
    )
    weather_data = weather_data.sortby("lon")
    return weather_data.rio.write_crs("epsg:4326").rio.set_spatial_dims(
        x_dim="lon",
        y_dim="lat",
    ).rio.write_coordinate_system(inplace=True)


def _extract_small_boundary_timeseries(boundaries, weather_data):
    def worker(row):
        tmp = weather_data.sel(lon=row.x, lat=row.y, method="nearest")
        return pd.DataFrame(
            {
                "time": tmp.time.values,
                "cloud_cover": tmp.cloud_cover.values,
                "tmax": tmp.tmax.values,
                "precip": tmp.precip.values,
            }
        )

    missing = boundaries[~boundaries.CC_2r.isin(weather_data.CC_2r.to_series().dropna().unique())]
    if missing.empty:
        return pd.DataFrame(columns=["CC_2r", "time", "cloud_cover", "tmax", "precip"])

    extracted = pd.concat(
        missing.set_index("CC_2r").centroid.apply(worker).to_dict()
    )
    return extracted.reset_index(names=["CC_2r", "t"]).drop(columns="t")


def _list_deter_shapefiles(root_dir="."):
    raw_dir = _climate_raw_dir(root_dir)
    year_dirs = [path for path in raw_dir.iterdir() if path.is_dir() and path.name.isdigit()]
    files = []
    for year_dir in year_dirs:
        for shapefile in year_dir.glob("*.shp"):
            files.append({"file": shapefile.relative_to(raw_dir).as_posix(), "year": year_dir.name})
    if not files:
        return pd.DataFrame(columns=["file", "year", "month", "type"])

    result = pd.DataFrame(files)
    result["month"] = result.file.str.extract(
        r"(" + "|".join(map(re.escape, MONTH_DICT.keys())) + ")",
        expand=False,
    ).replace(MONTH_DICT)
    result.loc[result.month.isna(), "month"] = result.loc[
        result.month.isna(), "file"
    ].str.extract(r".*_\d{4}(\d{2})\d{2}_.*\.shp", expand=False)
    result.loc[result.month.isna(), "month"] = result.loc[
        result.month.isna(), "file"
    ].str.extract(r".*_\d{4}(\d{2})_.*\.shp", expand=False)
    result.loc[result.month.isna(), "month"] = result.loc[
        result.month.isna(), "file"
    ].str.extract(r".*_\d{4}_(\d{2}).*\.shp", expand=False)
    result.loc[result.month.isna(), "month"] = result.loc[
        result.month.isna(), "file"
    ].str.extract(r".*\d{4}(\d{2})\d{2}.*\.shp", expand=False)
    result.loc[result.month.isna(), "month"] = result.loc[
        result.month.isna(), "file"
    ].str.extract(r".*\d{4}(\d{2})\.shp", expand=False)
    result["type"] = result.file.str.contains(r"uvem|uvens", flags=re.IGNORECASE).map(
        {True: "cloud_cover", False: "DETER"}
    )
    return result


def _build_deter_cloud_cover(root_dir="."):
    files = _list_deter_shapefiles(root_dir=root_dir)
    if files.empty:
        return pd.DataFrame(columns=["CC_2r", "year", "cloud_cover_DETER"])

    boundaries = _load_boundaries(root_dir=root_dir)
    raw_dir = _climate_raw_dir(root_dir)
    out_dict = {}

    for file in tqdm(
        files.query("type == 'cloud_cover'").file,
        total=files.query("type == 'cloud_cover'").file.size,
    ):
        cloud_cover = gpd.read_file(raw_dir / file, engine="pyogrio")
        if cloud_cover.crs is not None:
            cloud_cover = cloud_cover.to_crs(4326)
        else:
            cloud_cover = cloud_cover.set_crs(4326)

        cloud_cover["cloud_cover"] = 1
        cloud_cover = cloud_cover[["cloud_cover", "geometry"]]

        geobox = GeoBox.from_bbox(
            tuple(cloud_cover.total_bounds),
            crs="EPSG:4326",
            resolution=0.01,
        )
        cloud_cover_grid = rasterize_feature_values(
            cloud_cover,
            geobox,
            value_column="cloud_cover",
            dtype=np.uint8,
            fill_value=0,
        ).to_dataset(name="cloud_cover")
        boundaries_grid = rasterize_feature_values(
            boundaries[["CC_2r", "geometry"]],
            geobox,
            value_column="CC_2r",
            dtype=float,
            fill_value=np.nan,
        )
        cloud_cover_grid["CC_2r"] = boundaries_grid
        out_dict[file] = (
            cloud_cover_grid.set_coords("CC_2r").groupby("CC_2r").mean().cloud_cover.to_pandas()
        )

    with (_root(root_dir) / "data" / "climate" / "DETER_cc2r.pkl").open("wb") as handle:
        pickle.dump(out_dict, handle)

    out_df = mapping_to_long_frame(
        out_dict,
        index_name="file",
        column_name="CC_2r",
        value_name="cloud_cover",
    )
    out_df = pd.merge(files[["file", "year", "month"]], out_df, on="file")
    out_df = out_df.groupby("CC_2r").filter(
        lambda frame: not ((frame["cloud_cover"] == 0) | frame["cloud_cover"].isna()).all()
    )
    return (
        out_df.groupby(["CC_2r", "year"])
        .agg({"cloud_cover": "mean"})
        .reset_index()
        .astype({"year": int})
        .rename(columns={"cloud_cover": "cloud_cover_DETER"})
    )


def preprocess_cloud_cover(root_dir="."):
    boundaries = _load_boundaries(root_dir=root_dir)
    weather_data = _load_weather_data(root_dir=root_dir)
    weather_data = weather_data.rio.clip_box(*boundaries.total_bounds).persist()

    cloud_cover = xr.open_mfdataset(
        str(_climate_raw_dir(root_dir) / "*.nc"),
        chunks="auto",
        decode_times=True,
        decode_cf=True,
    )
    weather_data["cloud_cover"] = cloud_cover.cfc.reindex_like(
        weather_data, method="nearest"
    ).persist()
    weather_data = weather_data.resample(time="1Y").mean().load()

    geobox = weather_data[list(weather_data.data_vars)[0]].odc.geobox
    boundaries_grid = rasterize_feature_values(
        boundaries[["CC_2r", "geometry"]],
        geobox,
        value_column="CC_2r",
        dtype=float,
        fill_value=np.nan,
    )
    boundaries_grid = boundaries_grid.rename({"latitude": "lat", "longitude": "lon"})
    weather_data["CC_2r"] = boundaries_grid

    weather_data_df = weather_data.groupby("CC_2r").mean().to_dataframe().reset_index()
    if "spatial_ref" in weather_data_df.columns:
        weather_data_df = weather_data_df.drop(columns="spatial_ref")

    weather_data_df_merge = _extract_small_boundary_timeseries(boundaries, weather_data)
    deter_cloud_cover = _build_deter_cloud_cover(root_dir=root_dir)

    weather_data_df = pd.concat([weather_data_df, weather_data_df_merge], ignore_index=True)
    weather_data_df["year"] = pd.to_datetime(weather_data_df.time).dt.year
    weather_data_df = weather_data_df.drop(columns="time")
    weather_data_df = weather_data_df.rename(
        columns={"tmax": "temperature", "precip": "precipitation"}
    )
    weather_data_df = pd.merge(
        weather_data_df,
        deter_cloud_cover,
        on=["CC_2r", "year"],
        how="outer",
    )

    output_path = _root(root_dir) / "data" / "climate" / "climate_data.parquet"
    weather_data_df[
        [
            "CC_2r",
            "year",
            "cloud_cover",
            "cloud_cover_DETER",
            "temperature",
            "precipitation",
        ]
    ].to_parquet(output_path, index=False)
    return output_path
