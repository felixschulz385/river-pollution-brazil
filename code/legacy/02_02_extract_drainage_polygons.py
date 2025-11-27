# Standard library imports
import os
import pickle
import sqlite3
import sys
from collections import defaultdict
from itertools import chain
import warnings

# Suppress CRS warnings (buffer(0))
warnings.filterwarnings("ignore", category=UserWarning)

# Third-party imports for data handling and computation
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
from tqdm import tqdm

# Geospatial and image processing libraries
import cv2
import shapely
from pysheds.grid import Grid
from rasterio.io import MemoryFile

# Multiprocessing imports
from multiprocessing import Pool

# Local module imports for specific functionality
sys.path.append("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/code")
from code.data.river_network import river_network, calculate_distance_from_estuary
from data.preprocess.drainage_polygons.extract import extract_polygons_grid_cell

# A function to calculate a drainage area
def worker(grid_cell_index, drainage_polygons_dissolved, rivers_brazil_shapefile, grid_data_projected):
    print(f"--- Processing grid cell {grid_cell_index} ---")
    tmp = drainage_polygons_dissolved[drainage_polygons_dissolved.centroid.within(grid_data_projected.geometry.iloc[grid_cell_index])]
    if tmp.empty:
        return
    results = extract_polygons_grid_cell(tmp, rivers_brazil_shapefile)
    pickle.dump(results, open(f"/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/drainage/temp_extracted_grid_cells/{grid_cell_index}.pkl", "wb"))

def process_chunk(grid_cell_index):
    worker(grid_cell_index, drainage_polygons_dissolved, rivers_brazil_shapefile, grid_data_projected)
    
if __name__ == '__main__':
    # Database connection to fetch grid data
    conn = sqlite3.connect('/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/imagery/imagery.db')
    grid_data = pd.read_sql_query("SELECT * FROM GridCells WHERE Internal=1", conn)
    grid_geoms = gpd.read_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/imagery/grid_cells.feather")
    grid_data = gpd.GeoDataFrame(grid_data.merge(grid_geoms, on="CellID"))
    grid_data_projected = grid_data.to_crs(4326)
    conn.close()
    
    # Check which grid cells have already been processed
    os.makedirs("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/drainage/temp_extracted_grid_cells", exist_ok=True)
    processed_grid_cells = os.listdir("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/drainage/temp_extracted_grid_cells")
    processed_grid_cells = [int(i.split(".")[0]) for i in processed_grid_cells]

    # Loading river network data and computing distances from the estuary
    rivers_brazil_shapefile = gpd.read_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/river_network/shapefile.feather")
    rivers_brazil_topology = gpd.read_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/river_network/topology.feather")
    rivers_brazil_shapefile = calculate_distance_from_estuary(rivers_brazil_shapefile, rivers_brazil_topology)

    # Get the dissolved drainage polygons
    drainage_polygons_dissolved = gpd.read_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/drainage/drainage_polygons_dissolved_filtered.feather")
    
    with Pool(processes = int(os.environ["SLURM_CPUS_PER_TASK"])) as pool:
        # Apply the function to each chunk using multiprocessing
        pool.map(process_chunk, [x for x in grid_data_projected.index if x not in processed_grid_cells])