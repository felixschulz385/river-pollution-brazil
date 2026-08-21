import numpy as np
import pandas as pd
import xarray as xr
from odc.geo import xr as odc_xr
from odc.geo.geom import Geometry


def geometry_with_crs(geometry, crs=4326):
    """Wrap a geometry with a CRS for ODC raster crop calls."""
    return Geometry(geometry, crs)


def deduplicate_drainage_polygons(drainage_polygons, *, trench_id_column="trench_id"):
    """Keep the first drainage polygon for each trench id."""
    if trench_id_column not in drainage_polygons.columns:
        raise ValueError(
            f"Drainage polygons must include `{trench_id_column}` as an explicit column."
        )
    drainage_polygons = drainage_polygons.drop_duplicates(
        subset=[trench_id_column],
        keep="first",
    )
    return drainage_polygons.reset_index(drop=True)


def crop_unique_counts(raster, geometry, *, crs=4326, dtype=np.int64):
    """Crop a raster to one geometry and return unique finite values with counts.

    Non-finite (nodata) pixels are excluded before counting so callers summing
    ``counts`` for a total-valid-area figure don't silently include nodata area.
    """
    cropped = raster.odc.crop(geometry_with_crs(geometry, crs=crs))
    arr = np.asarray(cropped).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        empty = np.array([], dtype=dtype)
        return empty, empty
    return np.unique(arr.astype(dtype, copy=False), return_counts=True)


def is_extent_mismatch_error(exc):
    """Return True when a raster crop failed because extents do not overlap."""
    message = str(exc).lower()
    return "overlap" in message or "extent" in message


def mapping_to_long_frame(mapping, *, index_name, column_name, value_name):
    """Convert a mapping of aligned series-like values into a long dataframe."""
    if not mapping:
        return pd.DataFrame(columns=[index_name, column_name, value_name])

    return (
        pd.DataFrame(mapping)
        .transpose()
        .reset_index(names=index_name)
        .melt([index_name], var_name=column_name, value_name=value_name)
    )


def order_features_by_area(
    frame,
    *,
    geometry_column="geometry",
    id_column=None,
    area_crs="EPSG:5880",
):
    """Sort larger polygons first so they win ownership in overlap resolution."""
    projected = frame.to_crs(area_crs)
    areas = projected[geometry_column].area
    sort_columns = ["_feature_area"]
    ascending = [False]
    if id_column is not None:
        sort_columns.append(id_column)
        ascending.append(True)
    ordered = frame.assign(_feature_area=areas.to_numpy(copy=False)).sort_values(
        sort_columns,
        ascending=ascending,
        kind="mergesort",
    )
    return ordered.drop(columns="_feature_area").reset_index(drop=True)


def rasterize_feature_labels(
    frame,
    geobox,
    *,
    label_column,
    geometry_column="geometry",
    crs=4326,
    dtype="int64",
    fill_value=0,
    all_touched=False,
):
    """Rasterize vector features into a single-owner label grid."""
    grid = odc_xr.xr_zeros(geobox, dtype=dtype, name=label_column)
    for row in frame.itertuples(index=False):
        geometry = getattr(row, geometry_column, None)
        if geometry is None or geometry.is_empty:
            continue
        label = getattr(row, label_column)
        hits = odc_xr.rasterize(
            geometry_with_crs(geometry, crs=crs),
            geobox,
            all_touched=all_touched,
        )
        grid = xr.where((grid == fill_value) & hits, label, grid)
    return grid.astype(dtype)


def build_feature_label_grid(
    frame,
    geobox,
    *,
    label_column,
    geometry_column="geometry",
    crs=4326,
    area_crs="EPSG:5880",
    fill_value=0,
):
    """Rasterize features into a direct label grid."""
    ordered = order_features_by_area(
        frame,
        geometry_column=geometry_column,
        id_column=label_column,
        area_crs=area_crs,
    )
    return rasterize_feature_labels(
        ordered,
        geobox,
        label_column=label_column,
        geometry_column=geometry_column,
        crs=crs,
        fill_value=fill_value,
        all_touched=True,
    )

