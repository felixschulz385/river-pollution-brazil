import numpy as np
import odc.geo.xr  # noqa: F401 -- registers the .odc xarray accessor
import xarray as xr
from shapely.geometry import box

from src.data.shared.spatial_tabular import crop_unique_counts


def _fixture_raster():
    data = np.array(
        [
            [1, 1, 3, 3],
            [1, 1, 3, 3],
            [np.nan, np.nan, 5, 5],
            [np.nan, np.nan, 5, 5],
        ],
        dtype="float64",
    )
    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"y": [10.0, 9.0, 8.0, 7.0], "x": [0.0, 1.0, 2.0, 3.0]},
    )
    return da.odc.assign_crs("EPSG:4326")


def test_crop_unique_counts_excludes_nodata_pixels_from_total():
    raster = _fixture_raster()
    geometry = box(0.5, 7.5, 3.5, 9.5)

    values, counts = crop_unique_counts(raster, geometry)

    assert values.tolist() == [1, 3, 5]
    assert counts.tolist() == [1, 2, 2]
    assert int(counts.sum()) == 5
    assert values.dtype == np.int64


def test_crop_unique_counts_returns_empty_arrays_when_crop_is_all_nodata():
    raster = _fixture_raster()
    geometry = box(-0.5, 6.5, 1.5, 8.5)

    values, counts = crop_unique_counts(raster, geometry)

    assert values.tolist() == []
    assert counts.tolist() == []
    assert values.dtype == np.int64
