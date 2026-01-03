import os
import pickle
from datetime import datetime
from collections import defaultdict
import json
import numpy as np
import pandas as pd
import dask.dataframe as dd
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
from itertools import chain
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp

import sys
sys.path.append("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/code")
from data.river_network import calculate_distance_from_estuary
from code.data.drainage_polygons import expand_bounds


def extract_worker(x, x_1):
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
        c_extracted_processed = [extract_worker(c_extracted[i][year-1985], c_extracted[i][year-1985-1]) for i in range(len(polygons_raster))]
        
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

def extract():
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


def aggregate_stations(root_dir = "/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/"):
    """
    Aggregate land- and cloud cover data to match it with water quality stations.

    Parameters:
    root_dir (str): The root directory where the data files are located. Default is 
                    "/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/".

    Returns:
    None

    Outputs:
    - A parquet file containing processed land cover data for water quality stations, 
      saved to the specified root directory.
    """
    # import data
    rivers = gpd.read_feather(f"{root_dir}data/river_network/shapefile.feather")
    topology = gpd.read_feather(f"{root_dir}data/river_network/topology.feather")
    stations_reachability = pd.read_pickle(f"{root_dir}data/water_quality/stations_reachability.pkl")
    water_quality_panel = pd.read_parquet(f"{root_dir}data/water_quality/quality_indicators_panel.parquet")
    drainage_polygons = gpd.read_feather(f"{root_dir}data/drainage/extracted_drainage_polygons.feather")
    land_cover = dd.read_parquet(f"{root_dir}data/land_cover/temp_extracted_land_cover/", columns=["year", "deforestation", "deforestation_p", "deforestation_a", "deforestation_u", "deforestation_m", "forest", "pasture", "agriculture", "urban", "mining", "total"]).astype(np.uint32)

    # calculate distance from estuary
    rivers = calculate_distance_from_estuary(rivers, topology)
    del topology

    # compute lookup from node to list of polygons
    drainage_polygons_tmp = drainage_polygons[((~ drainage_polygons.is_empty) & (~ drainage_polygons.geometry.isna()))].drop(columns="geometry").reset_index(names = ["grid", "index"])
    drainage_polygons_tmp = pd.merge(drainage_polygons_tmp, rivers.drop(columns = ["NORIOCOMP", "CORIO", "geometry"]), on = ["estuary", "river", "segment", "subsegment"])
    drainage_polygons_tmp["identifier"] = drainage_polygons_tmp.apply(lambda x: [x["grid"], x["index"]], axis = 1)
    node_polygon_lookup = drainage_polygons_tmp.set_index("upstream_node_id").groupby(level=0).apply(lambda x: x["identifier"].tolist()).to_dict()
    
    # compute lookup from station to list of upstream nodes
    reachability_lookup = stations_reachability.set_index("Codigo", drop = True).reachability.apply(lambda x: list(x) if x else None)
    reachability_lookup = reachability_lookup[~reachability_lookup.index.duplicated(keep='first')]
    
    # compute lookup from node to distance from estuary
    distance_lookup = drainage_polygons[["distance_from_estuary"]].reset_index(names = ["grid_id", "index"])
    #distance_lookup = rivers[["upstream_node_id", "distance_from_estuary"]].rename(columns={"upstream_node_id": "node"}).reset_index(drop = True)
    del rivers, drainage_polygons, drainage_polygons_tmp
    
    # compute lookup from station to distance from estuary
    station_distance_lookup = stations_reachability.set_index("Codigo", drop=True).distance_from_estuary
    station_distance_lookup = station_distance_lookup[~station_distance_lookup.index.duplicated(keep='first')]
    del stations_reachability
    
    # get all station codes
    station_codes = water_quality_panel.station.unique()
    del water_quality_panel
    
    # get upstream nodes for all stations
    t_node_ids = {x: reachability_lookup.loc[x] for x in station_codes}
    del reachability_lookup, station_codes
    
    # prepare deforestation data in dask dataframe
    t_deforestation = land_cover.groupby(["grid_id", "index", "year"]).sum().persist()
    del land_cover
    
    ## chunk all stations into chunks if 1M nodes
    # assign chunks of 1M nodes
    t_chunks = np.cumsum([len(x) if x is not None else 0 for i, x in t_node_ids.items()]) // 1e6
    # get indices of chunks
    t_chunks = [(int(np.argmax(t_chunks == i)), int(len(t_chunks) - np.argmax(t_chunks[::-1] == i) - 1)) for i in np.unique(t_chunks)]
    # get nodes split into chunks
    c_node_ids = [{y: t_node_ids[y] for y in list(t_node_ids.keys())[x[0]:x[1]]} for x in t_chunks]
    #c_polygon_ids = [{station_id: {node_id: node_polygon_lookup.get(node_id, [None, None]) for node_id in node_ids if node_id is not None} for station_id, node_ids in chunk.items() if node_ids is not None} for chunk in c_node_ids]
    del t_node_ids
    
    # iterate over chunks
    out_df = [None] * len(c_node_ids)
    for i in range(len(c_node_ids)):
        ## prepare data frame for final estimation
        # get polygon ids
        t_index_prep = {(key, int(value)): node_polygon_lookup.get(value, [None, None]) for key, values in c_node_ids[i].items() if values is not None for value in values}
        # combine in tuple for index
        t_index_prep = [(key[0], key[1], int(value[0]), int(value[1])) for key, values in t_index_prep.items() for value in values if value is not None]         
        # create dataframe with indices
        c_final_df = dd.from_pandas(pd.DataFrame().from_records(t_index_prep, columns = ["station", "node", "grid_id", "index"]))
        # merge with deforestation data
        c_final_df = dd.merge(c_final_df, t_deforestation.reset_index(), on = ["grid_id", "index"], how = "left")
        # merge with distance data
        c_final_df = dd.merge(c_final_df, dd.from_pandas(distance_lookup).astype(np.int64), on=["grid_id", "index"], how = "left")
        # calculate distance from station
        c_final_df["distance_from_station"] = (c_final_df.distance_from_estuary - c_final_df.station.map(station_distance_lookup, meta=('station', 'uint32')))
        
        # Define the windows
        max_ = 1000 * 1e3; step_ = 50 * 1e3
        windows = np.arange(0, max_, step_)
        # Create bin labels
        bin_labels = [f"[{start},{start + step_})" for start in windows]
        # Bin the sorting column
        c_final_df['bins'] = c_final_df["distance_from_station"].map_partitions(pd.cut, bins=np.concatenate([windows, [max_ + step_]]), labels=bin_labels, right=False, include_lowest=True)
        
        #
        #c_final_df.to_parquet(f"{root_dir}data/land_cover/land_cover_stations_detailed_{i}/")
        
        # Group by the bins and sum the value column
        out_df[i] = c_final_df.drop(columns=["distance_from_station", "distance_from_estuary", "grid_id", "index", "node"]).groupby(["station", "year", "bins"]).sum().compute()
        out_df[i] = out_df[i].groupby(["station", "year", "bins"]).sum()
    
    # combine all chunks
    out_df = pd.concat(out_df)
    
    # save to parquet
    out_df.reset_index().astype({"year": np.int16}).to_parquet(f"{root_dir}data/land_cover/land_cover_stations.parquet")

def aggregate_municipalities(root_dir = "/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/"):
    # import data
    municipalities = gpd.read_file(f"{root_dir}data/misc/raw/gadm/gadm41_BRA_2.json", engine="pyogrio")
    rivers = gpd.read_feather(f"{root_dir}data/river_network/shapefile.feather")
    drainage_polygons = gpd.read_feather(f"{root_dir}data/drainage/extracted_drainage_polygons.feather")
    land_cover = dd.read_parquet(f"{root_dir}data/land_cover/temp_extracted_land_cover/", columns=["year", "deforestation", "deforestation_p", "deforestation_a", "deforestation_u", "deforestation_m", "forest", "pasture", "agriculture", "urban", "mining", "total"]).astype(np.uint32)
    #cloud_cover = dd.read_parquet(f"{root_dir}data/cloud_cover/cloud_cover.parquet").astype({"grid_id": "uint32", "index": "uint32", "year": "uint32", "cloud_cover": "float32"})
    reachability_municipalities = json.load(open(f"{root_dir}data/river_network/reachability_municipalities.json", "r"))

    # create mapping from GID_2 to integer ID
    gid_id_lookup = municipalities["GID_2"].reset_index().set_index("GID_2")["index"].to_dict()
    id_cc_lookup = municipalities["CC_2"].str.slice(0, 6).to_dict()
    # convert reachability_municipalities keys to integer ID
    reachability_municipalities = {gid_id_lookup.get(k, k): v for k, v in reachability_municipalities.items()}
    del municipalities
    
    # prepare deforestation data in dask dataframe
    t_deforestation = land_cover.groupby(["grid_id", "index", "year"]).sum().reset_index().astype(np.uint32).persist()
    del land_cover
    
    ## aggregation within adm2 regions
    
    # prepare a table from adm2 to grid_id
    adm2_table = pd.merge(
        rivers[["adm2", "estuary", "river", "segment", "subsegment"]], 
        drainage_polygons[["estuary", "river", "segment", "subsegment"]].reset_index(names = ["grid_id", "index"]), 
        on=["estuary", "river", "segment", "subsegment"], how="right",
        ).dropna()[["grid_id", "index", "adm2"]].astype(np.uint32)
    
    # merge deforestation data with adm2
    t_deforestation_adm = dd.merge(t_deforestation, dd.from_pandas(adm2_table, npartitions=1), on=["grid_id", "index"], how="left")
    # calculate deforestation data for each adm2
    c_final_df_deforestation = t_deforestation_adm.drop(columns=["grid_id", "index"]).groupby(["adm2", "year"]).sum().compute()
    # merge cloud cover data with adm2
    #t_cloud_cover_adm = dd.merge(cloud_cover, dd.from_pandas(adm2_table, npartitions=1), on=["grid_id", "index"], how="left")
    # calculate cloud cover data for each adm2
    c_final_df_cloud_cover = t_cloud_cover_adm.drop(columns=["grid_id", "index"]).groupby(["adm2", "year"]).mean().compute()
    # merge deforestation and cloud cover data
    out_df = pd.merge(c_final_df_deforestation, c_final_df_cloud_cover, on=["adm2", "year"], how="left")
    # write GID
    out_df = out_df.reset_index().astype({"year": np.int16})
    out_df["municipality"] = out_df.adm2.map(id_cc_lookup)
    out_df.drop(columns=["adm2"], inplace=True)

    # save to parquet
    out_df.to_parquet(f"{root_dir}data/land_cover/land_cover_municipalities.parquet", index=False)
    
    # ## aggregation of upstream nodes
    
    # # compute lookup from node to list of polygons
    # drainage_polygons_tmp = drainage_polygons[((~ drainage_polygons.is_empty) & (~ drainage_polygons.geometry.isna()))].drop(columns="geometry").reset_index(names = ["grid", "index"])
    # drainage_polygons_tmp = pd.merge(drainage_polygons_tmp, rivers.drop(columns = ["NORIOCOMP", "CORIO", "geometry"]), on = ["estuary", "river", "segment", "subsegment"])
    # drainage_polygons_tmp["identifier"] = drainage_polygons_tmp.apply(lambda x: [x["grid"], x["index"]], axis = 1)
    # node_polygon_lookup = drainage_polygons_tmp.set_index("upstream_node_id").groupby(level=0).apply(lambda x: x["identifier"].tolist()).to_dict()
    # del drainage_polygons, rivers
    
    # # assign chunks of 1M nodes
    # t_chunks = np.cumsum([len(x) if x is not None else 0 for i, x in reachability_municipalities.items()]) // 1e6
    # # get indices of chunks
    # t_chunks = [(int(np.argmax(t_chunks == i)), int(len(t_chunks) - np.argmax(t_chunks[::-1] == i) - 1)) for i in np.unique(t_chunks)]
    # # get nodes split into chunks
    # c_node_ids = [{y: reachability_municipalities[y] for y in list(reachability_municipalities.keys())[x[0]:x[1]]} for x in t_chunks]
    
    # # iterate over chunks
    # out_df = [None] * len(c_node_ids)
    # for i in range(len(c_node_ids)):
    #     ## prepare data frame for final estimation
    #     # get polygon ids
    #     t_index_prep = {(key, int(value)): node_polygon_lookup.get(value, [None, None]) for key, values in c_node_ids[i].items() if values is not None for value in values}
    #     # combine in tuple for index
    #     t_index_prep = [(key[0], key[1], int(value[0]), int(value[1])) for key, values in t_index_prep.items() for value in values if value is not None]         
    #     # create dataframe with indices
    #     c_final_df = dd.from_pandas(pd.DataFrame().from_records(t_index_prep, columns = ["municipality", "node", "grid_id", "index"])).astype(np.uint32)
    #     # merge with deforestation data and summarize
    #     c_final_df_deforestation = dd.merge(c_final_df, t_deforestation, on = ["grid_id", "index"], how = "left")
    #     c_final_df_deforestation = c_final_df_deforestation.drop(columns=["grid_id", "index", "node"]).groupby(["municipality", "year"]).sum().compute().astype(np.float32)
    #     # merge with cloud cover data and summarize
    #     c_final_df_cloud_cover = dd.merge(c_final_df, cloud_cover, on = ["grid_id", "index"], how = "left")
    #     c_final_df_cloud_cover = c_final_df_cloud_cover.groupby(["municipality", "year"]).agg({"cloud_cover": "mean"}).compute().astype(np.float32)
    #     # Group by the bins and sum the value column
    #     #agg_dict = {"deforestation": "sum", "deforestation_p": "sum", "deforestation_a": "sum", "deforestation_u": "sum", "deforestation_m": "sum", "forest": "sum", "pasture": "sum", "agriculture": "sum", "urban": "sum", "mining": "sum", "total": "sum"}
    #     out_df[i] = pd.merge(c_final_df_deforestation, c_final_df_cloud_cover, on = ["municipality", "year"], how = "outer")
    # # combine all chunks
    # out_df = pd.concat(out_df).reset_index().astype({"year": np.int16})
    # # get GID_2
    # out_df["municipality"] = out_df.municipality.map(id_cc_lookup)

    # # save to parquet
    # out_df.to_parquet(f"{root_dir}data/land_cover/deforestation_municipalities_upstream.parquet")
      
if __name__ == "__main__":
    prepare_control_variables()