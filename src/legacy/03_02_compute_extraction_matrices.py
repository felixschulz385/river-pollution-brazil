import os
import pickle
from multiprocessing import Pool
import numpy as np
import pandas as pd
import geopandas as gpd
import sparse
import xarray as xr
import rioxarray as rxr
from rasterio.features import rasterize

import sys
sys.path.append("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/code")
from data.preprocess.drainage_polygons.aux_functions import expand_bounds, load_height_profile

def rasterize_polygon(payload, shape, transform):
    t_rasterized = rasterize(payload, out_shape=shape, transform=transform, all_touched=True)
    rasterized_sparse = sparse.COO.from_numpy(t_rasterized)
    return rasterized_sparse

def worker(payload):
    t_template = lc_t.rio.clip_box(*expand_bounds(payload[1].total_bounds))
    polygons_raster = [rasterize_polygon([x], payload[2], payload[3]) for x in payload[1].geometry]
    pickle.dump(polygons_raster, open(f"/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/drainage/temp_extraction_masks/{payload[0]}.pkl", "wb"))
    
if __name__ == "__main__":
    # Load extracted polygons
    extracted_drainage_polygons = gpd.read_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/drainage/extracted_drainage_polygons.feather")
    extracted_drainage_polygons = extracted_drainage_polygons.set_crs(4326)[~extracted_drainage_polygons.is_empty].dropna(subset=["geometry"])
    
    # Load land cover data as template
    lc_t = rxr.open_rasterio(f"/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/land_cover/raw/lc_mapbiomas8_30/mapbiomas_brasil_coverage_{1990}.tif", chunks=True).squeeze()

    # Get grid ids to work on
    grid_ids = extracted_drainage_polygons.index.get_level_values(0).unique()
    # Remove already processed grid cells
    files = os.listdir("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/drainage/temp_extraction_masks")
    files = pd.Series(files).str.extract(r"(\d*)").squeeze().astype(int).unique()
    grid_ids = [x for x in grid_ids if not x in files]
    
    # Prepare payloads
    payloads = [
        (idx, 
         extracted_drainage_polygons.loc[idx], 
         lc_t.rio.clip_box(*expand_bounds(extracted_drainage_polygons.loc[idx].total_bounds)).shape,
         lc_t.rio.clip_box(*expand_bounds(extracted_drainage_polygons.loc[idx].total_bounds)).rio.transform()) 
        for idx in grid_ids]

    with Pool(8) as p:
        p.map(worker, payloads)
