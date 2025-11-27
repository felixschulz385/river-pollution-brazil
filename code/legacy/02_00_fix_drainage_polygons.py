# Standard library imports
import os
import pickle
import sqlite3
import sys
from collections import defaultdict
from itertools import chain
import warnings

warnings.filterwarnings("ignore")

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
from data.preprocess.drainage_polygons.fix_ana_drainage_polygons import fix_rivers_in_grid

# Database connection to fetch grid data
conn = sqlite3.connect('/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/imagery/imagery.db')
grid_data = pd.read_sql_query("SELECT * FROM GridCells WHERE Internal=1", conn)
grid_geoms = gpd.read_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/imagery/grid_cells.feather")
grid_data = gpd.GeoDataFrame(grid_data.merge(grid_geoms, on="CellID"))
conn.close()

# Reading and processing river data
rivers = gpd.read_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/river_network/msc_rivers.feather").to_crs(5641)
rivers_subset = gpd.clip(rivers, grid_data.geometry.iloc[525])

# Loading administrative boundaries and clipping to the grid area
boundaries = gpd.read_file("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/misc/raw/gadm/gadm41_BRA_2.json", engine="pyogrio").to_crs(5641)
boundaries_subset = gpd.clip(boundaries, grid_data.geometry.iloc[525])

# Loading river network data and computing distances from the estuary
rivers_brazil = pickle.load(open("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/river_network/network.pkl", "rb"))
rivers_brazil_shapefile = gpd.read_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/river_network/shapefile.feather")
rivers_brazil_topology = gpd.read_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/river_network/topology.feather")
rivers_brazil_shapefile = calculate_distance_from_estuary(rivers_brazil_shapefile, rivers_brazil_topology)

# Reading and processing drainage areas
drainage_polygons = gpd.read_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/drainage_area.feather")
drainage_polygons_projected = drainage_polygons.to_crs(5641)
drainage_polygons_gridded = grid_data.sjoin(gpd.GeoDataFrame(geometry=drainage_polygons_projected.centroid, index=drainage_polygons_projected.index), how="right").dropna(subset=["index_left"])

# Checking for previously saved work and updating the grid calculations

processed_grid_cells = os.listdir("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/drainage/temp_fixed_grid_cells")
processed_grid_cells = [int(i.split(".")[0]) for i in processed_grid_cells]
    
def wrapper(i):
    try:
        update_set = fix_rivers_in_grid(i, rivers_brazil_shapefile, rivers_brazil_topology, drainage_polygons, drainage_polygons_gridded)
        with open(f"/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/drainage/temp_fixed_grid_cells/{i}.pkl", "wb") as f:
            pickle.dump(update_set, f)
    except:
        pass  
    
with Pool(int(os.environ["SLURM_CPUS_PER_TASK"])) as p:
    p.map(wrapper, [i for i in grid_data.index if i not in processed_grid_cells])