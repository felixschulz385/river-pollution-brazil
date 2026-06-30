import numpy as np
import pandas as pd
from odc.geo.geom import Geometry
from odc.geo import xr as odc_xr


def geometry_with_crs(geometry, crs=4326):
    """Wrap a geometry with a CRS for ODC raster crop calls."""
    return Geometry(geometry, crs)


def masked_unique_counts(raster, geometry, *, crs=4326):
    """Crop a raster to one geometry and return unique values with counts."""
    cropped = raster.odc.crop(geometry_with_crs(geometry, crs=crs))
    return np.unique(cropped, return_counts=True)


def is_no_overlap_error(exc):
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


def rasterize_value_grid(frame, geobox, *, value_column, crs=4326, dtype=None, fill_value=np.nan):
    """Rasterize one vector row at a time into a labeled xarray grid."""
    grid = None
    for row in frame.itertuples(index=False):
        geometry = getattr(row, "geometry", None)
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
        grid = odc_xr.rasterize(geometry_with_crs(frame.iloc[0].geometry, crs=crs), geobox)
        grid = grid.astype(dtype or float)
        grid[:] = fill_value
        return grid

    if np.isnan(fill_value):
        return grid.where(grid != 0)
    return grid.where(grid != 0, fill_value)
