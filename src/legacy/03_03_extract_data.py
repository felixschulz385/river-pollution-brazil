import os
import pickle
from datetime import datetime
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rxr

from tqdm import tqdm
import multiprocessing as mp

import sys
sys.path.append("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/code")
from data.preprocess.drainage_polygons.aux_functions import expand_bounds

def extract(x, x_1):
    #
    land_cover = np.unique(x, return_counts=True)
    #
    df_g = np.sum(np.isin(x_1, class_labels["forest"]) & (~ np.isin(x, class_labels["forest"])))
    df_f = np.sum(np.isin(x_1, class_labels["forest"]) & (np.isin(x, class_labels["farming"])))
    df_p = np.sum(np.isin(x_1, class_labels["forest"]) & (np.isin(x, class_labels["pasture"])))
    df_a = np.sum(np.isin(x_1, class_labels["forest"]) & (np.isin(x, class_labels["agriculture"])))
    df_u = np.sum(np.isin(x_1, class_labels["forest"]) & (np.isin(x, class_labels["urban"])))
    df_m = np.sum(np.isin(x_1, class_labels["forest"]) & (np.isin(x, class_labels["mining"])))
    deforestation = np.array([df_g, df_f, df_p, df_a, df_u, df_m])
    #
    return land_cover, deforestation
 
    
def worker(payload):
    print(f"*** Processing grid_id: {payload[0]} at {datetime.now().strftime('%H:%M:%S')} ***")

    # load extraction masks
    polygons_raster = pickle.load(open(f"/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/drainage/temp_extraction_masks/{payload[0]}.pkl", "rb"))

    # extract raw pixel values
    c_extracted = {i: payload[2][:,polygons_raster[i].astype(bool).todense()] for i in range(len(polygons_raster))}
    
    out_df = pd.DataFrame()  
    for year in range(1986, 2023):
        # calculate land cover and deforestation data
        c_extracted_processed = [extract(c_extracted[i][year-1985], c_extracted[i][year-1985-1]) for i in range(len(polygons_raster))]
        
        ## post-process land cover data
        # prepare dataframe
        lc_df = pd.DataFrame(columns = legend.ID, index = payload[1])
        # fill dataframe
        for i in range(len(c_extracted_processed)):
            lc_df.loc[payload[1][i], c_extracted_processed[i][0][0]] = c_extracted_processed[i][0][1]
        # fill missing values with 0 and calculate total area 
        lc_df = lc_df.astype(float).fillna(0)
        lc_df["total"] = lc_df.sum(axis=1)

        # sum columns based on IDs and create new columns with names from the dictionary
        for name, id_ in class_labels.items():
            lc_df[name] = lc_df[id_].sum(axis=1)
            
        ## post-process deforestation data
        df_df = pd.DataFrame().from_dict({payload[1][i]: c_extracted_processed[i][1] for i in range(len(c_extracted_processed))}, 
                                        orient = "index",
                                        columns = ["deforestation", "deforestation_f", "deforestation_p", "deforestation_a", "deforestation_u", "deforestation_m"])
        
        # merge dataframes
        merged_df = pd.concat([lc_df, df_df], axis=1).astype(np.uint32)
        # add year and grid_id
        merged_df["year"] = year; merged_df["grid_id"] = payload[0]
        # set index
        out_df = pd.concat([out_df, merged_df.reset_index().set_index(["grid_id", "index"])])
    
    out_df.to_parquet(f"/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/land_cover/temp_extracted_land_cover/{payload[0]}.parquet")

def main():
    # import drainage polygons
    extracted_drainage_polygons = gpd.read_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/drainage/extracted_drainage_polygons.feather")
    extracted_drainage_polygons = extracted_drainage_polygons.set_crs(4326)[~extracted_drainage_polygons.is_empty].dropna(subset=["geometry"])
    
    # sort by size for efficient chunking
    sizes = extracted_drainage_polygons.groupby(level=0).size()
    sorted_first_index = sizes.sort_values(ascending=False).index
    extracted_drainage_polygons = extracted_drainage_polygons.loc[sorted_first_index]
    
    # 
    files = os.listdir("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/land_cover/temp_extracted_land_cover")
    files = pd.Series(files)[pd.Series(files).str.contains(r"^\d+")].str.extract(r"(^\d+)").squeeze().astype(int).unique()
    extract_indices = extracted_drainage_polygons.index.get_level_values(0).unique()
    extract_indices = np.array([extract_indices for extract_indices in extract_indices if extract_indices not in files])
    
    def load_data(grid_id):
        data = [grid_id, extracted_drainage_polygons.loc[grid_id].index, {}]
        
        for year in range(1985, 2023):
            # load land cover data    
            with rxr.open_rasterio(f"/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/land_cover/raw/lc_mapbiomas8_30/mapbiomas_brasil_coverage_{year}.tif", chunks=True).squeeze() as lc_t:       
                data[2][year] = lc_t.rio.clip_box(*expand_bounds(extracted_drainage_polygons.loc[grid_id].total_bounds))
        
        data[2] = xr.concat(data[2].values(), pd.Index(data[2].keys(), name = "year")).values
        return data
            
    def producer_task(task_queue, data_queue, load_semaphore):
        for grid_id in extract_indices:
            load_semaphore.acquire()  # Ensure only one load operation at a time
            data = load_data(grid_id)
            data_queue.put(data)
            #load_semaphore.release()

    def consumer_task(data_queue, load_semaphore):
        while True:
            data = data_queue.get()
            if data is None:
                break
            worker(data)
            load_semaphore.release()  # Signal that a dataset slot is free
            
    # Queues
    max_workers = int(os.getenv("SLURM_CPUS_PER_TASK", 8)) - 1
    data_queue = mp.Queue(maxsize=max_workers)

    # Semaphore to control the number of datasets in memory
    load_semaphore = mp.Semaphore(max_workers)

    # Create and start the producer process
    producer = mp.Process(target=producer_task, args=(data_queue, data_queue, load_semaphore))
    producer.start()

    # Create and start the consumer processes
    consumers = []
    for _ in range(max_workers):
        consumer = mp.Process(target=consumer_task, args=(data_queue, load_semaphore))
        consumer.start()
        consumers.append(consumer)

    # Wait for the producer to finish
    producer.join()

    # Signal consumers to stop
    for _ in range(num_processors):
        data_queue.put(None)

    # Wait for the consumers to finish
    for consumer in consumers:
        consumer.join()

if __name__ == "__main__":
    
    # import the legend
    legend = pd.read_excel("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/land_cover/legend/mapbiomas_legend.xlsx")

    # create class labels to summarize the land cover data
    class_labels = {
        "forest": legend.loc[legend.Class.str.match(r"(^1\.)"), "ID"].values,
        "non-forest/natural": legend.loc[legend.Class.str.match(r"(^2\.)"), "ID"].values,
        "farming": legend.loc[legend.Class.str.match(r"(^3\.)"), "ID"].values,
        "pasture": legend.loc[legend.Class.str.match(r"(^3\.1)"), "ID"].values,
        "agriculture": legend.loc[legend.Class.str.match(r"(^3\.2)"), "ID"].values,
        "non-vegetated": legend.loc[legend.Class.str.match(r"(^4\.)"), "ID"].values,
        "urban": legend.loc[legend.Class.str.match(r"(^4\.2)"), "ID"].values,
        "mining": legend.loc[legend.Class.str.match(r"(^4\.3)"), "ID"].values,
        "water": legend.loc[legend.Class.str.match(r"(^5\.)"), "ID"].values,
    }
    
    main()