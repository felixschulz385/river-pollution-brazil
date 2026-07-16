import os
import numpy as np
import xarray as xr
import rioxarray as rxr
from itertools import product

def expand_bounds(bounds, factor = 1.5):
    """
    Expands a bounding box by a specified factor outward from its center.

    Parameters:
    - bounds (list[float]): A list of four floats representing the coordinates of the bounding box [xmin, ymin, xmax, ymax].
    - factor (float, optional): The factor by which the bounds should be expanded. Default is 1.5.

    Returns:
    - list[float]: A list of four floats representing the new expanded bounding box.
    """
    return [bounds[0] - (bounds[2] - bounds[0]) * (factor - 1) / 2, bounds[1] - (bounds[3] - bounds[1]) * (factor - 1) / 2, bounds[2] + (bounds[2] - bounds[0]) * (factor - 1) / 2, bounds[3] + (bounds[3] - bounds[1]) * (factor - 1) / 2]


def load_height_profile(bbox, polygon = None, dem_path = "/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/misc/raw/DEM_GLO-90/"):
    """
    Load and return the height profile for a given bounding box.

    Args:
    bbox (tuple): A tuple representing the bounding box with four values (min latitude, min longitude, max latitude, max longitude).

    Returns:
    xarray.DataArray: An array containing the height profile within the specified bounding box.
    """

    # Generate file path suffixes for latitude and longitude using grid conventions
    # Latitudes and longitudes are formatted based on their hemisphere and rounded to the nearest degree
    lat_lon = list(product(
        [lat for lat in np.arange(np.floor(bbox[0]), np.ceil(bbox[2]), 1)],
        [lon for lon in np.arange(np.floor(bbox[1]), np.ceil(bbox[3]), 1)]
    ))
    
    # Filter out files that do not intersect the polygon
    if polygon is not None:
        lat_lon = filter(lambda x: shapely.box(x[0], x[1], x[0] + 1, x[1] + 1).intersects(polygon), lat_lon)
    
    lat_lon = [
        (f"E{int(lat):03}" if lat >= 0 else f"W{-int(lat):03}", 
         f"N{int(lon):02}" if lon >= 0 else f"S{-int(lon):02}") for lat, lon in lat_lon
    ]
    
    # Construct file paths for DEM (Digital Elevation Model) tiles within the bounding box
    files_dem_cop = [
        f"{dem_path}Copernicus_DSM_30_{lon}_00_{lat}_00/DEM/Copernicus_DSM_30_{lon}_00_{lat}_00_DEM.tif"
        for lat, lon in lat_lon
    ]
    
    # Filter out files that do not exist
    files_dem_cop = [file for file in files_dem_cop if os.path.exists(file)]

    # Load the DEM files and combine them into a single DataArray using xarray, keeping the first channel and excluding the last pixel on each edge
    height_profile = xr.combine_by_coords([
        rxr.open_rasterio(file)[0, :-1, :-1] for file in files_dem_cop
    ])

    # # Clip the combined height profile to exactly match the bounding box using rioxarray
    # height_profile = height_profile.rio.clip_box(*bbox)

    return height_profile