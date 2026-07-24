import numpy as np
import pandas as pd
import xarray as xr
from odc.geo import xr as odc_xr
from odc.geo.geom import Geometry


def geometry_with_crs(geometry, crs=4326):
    """Wrap a geometry with a CRS for ODC raster crop calls."""
    return Geometry(geometry, crs)


def crop_unique_counts(raster, geometry, *, crs=4326):
    """Crop a raster to one geometry and return unique values with counts."""
    cropped = raster.odc.crop(geometry_with_crs(geometry, crs=crs))
    return np.unique(cropped, return_counts=True)


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


def rasterize_feature_values(
    frame,
    geobox,
    *,
    value_column,
    geometry_column="geometry",
    crs=4326,
    dtype=None,
    fill_value=np.nan,
):
    """Rasterize one vector row at a time into a labeled xarray grid."""
    grid = None
    for row in frame.itertuples(index=False):
        geometry = getattr(row, geometry_column, None)
        if geometry is None or geometry.is_empty:
            continue
        mask = odc_xr.rasterize(
            geometry_with_crs(geometry, crs=crs),
            geobox,
            all_touched=False,
        )
        value = getattr(row, value_column)
        layer = mask.astype(dtype or type(value)) * value
        if grid is None:
            grid = layer
        else:
            grid = grid.where(~mask, layer)

    if grid is None:
        template_geometry = frame.iloc[0][geometry_column]
        grid = odc_xr.rasterize(geometry_with_crs(template_geometry, crs=crs), geobox)
        grid = grid.astype(dtype or float)
        grid[:] = fill_value
        return grid

    if np.isnan(fill_value):
        return grid.where(grid != 0)
    return grid.where(grid != 0, fill_value)


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


masked_unique_counts = crop_unique_counts
is_no_overlap_error = is_extent_mismatch_error
rasterize_value_grid = rasterize_feature_values
