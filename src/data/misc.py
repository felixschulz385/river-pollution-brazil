import os
import pandas as pd
import geopandas as gpd
import sqlite3
import xarray as xr
import rioxarray as rxr
import shapely



def preprocess_grid(root_dir):
    """
    This function preprocesses a grid system for geographical data processing.
    
    It reads the anchor grid and boundaries, calculates the grid cells, and
    saves the processed grid data into a SQLite database and a Feather file.

    Parameters:
    root_dir (str): The root directory where the data files are located.
    """
    # Load anchor grid example as a raster file
    ex_empty = rxr.open_rasterio(root_dir + "data/imagery/grid_anchor.tif")

    # Load geographical boundaries as GeoDataFrame and convert CRS
    boundaries = gpd.read_file(root_dir + "data/boundaries/gadm_410-BRA.geojson",
                               engine="pyogrio").to_crs(5641)

    # Calculate the minimum and maximum X coordinates, and the step size for grid cells in X direction
    min_x = int(np.min(ex_empty.x.to_numpy()))
    max_x = int(boundaries.total_bounds[2])
    step_x = 30 * len(ex_empty.x)
    grid_cells_x = int((max_x - min_x) // step_x) + 2

    # Calculate the minimum and maximum Y coordinates, and the step size for grid cells in Y direction
    min_y = int(boundaries.total_bounds[1])
    max_y = int(np.max(ex_empty.y.to_numpy())) + 30
    step_y = 30 * len(ex_empty.y)
    grid_cells_y = int((max_y - min_y) // step_y) + 2

    # Generate a meshgrid for X and Y coordinates
    xv, yv = np.meshgrid(np.array([min_x + i * step_x for i in range(grid_cells_x)]),
                         np.array([max_y - i * step_y for i in range(grid_cells_y)]))

    # Create bounding boxes for each grid cell
    boxes = gpd.GeoDataFrame(dict(Row=np.repeat(range(len(xv) - 1), len(xv[0]) - 1),
                                  Column=np.tile(range(len(xv[0]) - 1), len(xv) - 1)),
                             geometry=[shapely.box(xv[i, j], yv[i, j] - 30, xv[i, j + 1] - 30, yv[i + 1, j])
                                       for i in range(len(xv) - 1) for j in range(len(xv[0]) - 1)],
                             crs=5641)

    # Reset index and assign a name to it
    boxes.reset_index(names="CellID", inplace=True)

    # Extract X and Y coordinates from the box geometry
    boxes[["X", "Y"]] = pd.DataFrame(boxes.exterior.map(lambda x: [min(y) for y in x.xy]).to_list(), columns=["x", "y"])

    # Determine if boxes are internal to the boundary
    boxes["Internal"] = boxes.intersects(boundaries.geometry.iloc[0])

    # Connect to SQLite database
    conn = sqlite3.connect(root_dir + 'data/imagery/imagery.db')

    # Insert data into the GridCells table, replacing if it already exists
    boxes.drop(columns="geometry").to_sql("GridCells", conn, if_exists="replace", index=False)

    # Store the geometry data externally in a Feather file
    boxes[["CellID", "geometry"]].to_feather(root_dir + "data/imagery/grid_cells.feather")

    # Close the connection
    conn.close()