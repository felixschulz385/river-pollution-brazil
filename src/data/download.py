# Standard library imports
import os
import time
from time import sleep
import re
import io
import requests
import pickle
import tarfile
import zipfile
import urllib.request
from functools import partial
from itertools import compress

# Third-party library imports
import pandas as pd
import geopandas as gpd
import shapely
from tqdm import tqdm
from multiprocessing import Pool
#import ee
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import cdsapi

class DataAgent:
    """
    DataAgent class for handling data preparation and fetching.
    """
# -- this function needs to be defined on __main__ so that multiprocessing is able to pickle it
def imagery_downloader(link, root_dir):
    tmp = root_dir + "data/imagery/raw/" + re.compile("(?<=\=).*(?=[\&]requestSignature\=)").search(link).group(0) + ".tar"
    urllib.request.urlretrieve(link, tmp)

class download_agent:
    
    def __init__(self, root_dir, area = "BRA", year = 2010):
        """
        Constructor for DataAgent.

        Parameters:
        root_dir (str): Root directory where data will be stored.
        area (str): Area identifier.
        """
        self.root_dir = root_dir
        self.area = area
        self.year = year
            
    def fetch(self, dataset):
        """
        Fetch data from the provided dataset.

        Parameters:
        dataset (dict): Dataset containing setup details.
        """
        def prepare_filesystem(dataset):
            """
            Prepare filesystem directories for storing data.
            """
            os.makedirs(os.path.dirname(self.root_dir + dataset["path"]), exist_ok = True)

        def dl(dataset):
            """
            Download function to fetch data from a URL.
            """
            if type(dataset["url"]) == dict:
                for file in dataset["url"]["files"]:
                    urllib.request.urlretrieve(dataset["url"]["base"] + file, self.root_dir + dataset["path"] + "/" + file)
            if not type(dataset["url"]) == dict:
                urllib.request.urlretrieve(dataset["url"], self.root_dir + dataset["path"])
            
        def ex(dataset):
            """
            Extract function to extract data from a zip file.
            """
            if type(dataset["url"]) == dict:
                for file in dataset["url"]["files"]:
                    with zipfile.ZipFile(self.root_dir + dataset["path"] + "/" + file, 'r') as zip_ref:
                        zip_ref.extractall(self.root_dir + dataset["path"])
            if not type(dataset["url"]) == dict:
                with zipfile.ZipFile(self.root_dir + dataset["path"], 'r') as zip_ref:
                    zip_ref.extractall((self.root_dir + dataset["path"]).split("raw")[0] + "raw")
                    
        def fe_de_co(dataset):
            """
            Fetch Copernicus DEM data.
            """
            # create shapely polygons on a grid, spanning the entire planet with a resolution of 1 degrees
            grid = gpd.GeoDataFrame(geometry = [shapely.geometry.box(i, j, i+1, j+1) for i in range(-180, 180, 1) for j in range(-90, 90, 1)])
            # filter for the area of interest
            boundaries_limits = gpd.read_file("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/misc/gadm_410-BRA.geojson", engine="pyogrio").to_crs(5641).buffer(200 * 1e3).to_crs(4326)
            grid["in_bbox"] = grid.intersects(shapely.geometry.box(*boundaries_limits.total_bounds))
            grid["in_boundaries"] = grid.intersects(boundaries_limits.geometry.iloc[0])
            grid_filtered = grid.copy()[grid.in_boundaries]
            grid_filtered = grid_filtered[grid_filtered.centroid.map(lambda x: x.coords[0][0]) < -32.5]
            # format the coordinates
            grid_filtered["formatted_lon"] = grid_filtered.bounds.minx.map(lambda x: "W" + str(int(x*-1)).zfill(3) if x < 0 else "E" + str(int(x)).zfill(3))
            grid_filtered["formatted_lat"] = grid_filtered.bounds.miny.map(lambda x: "S" + str(int(x*-1)).zfill(2) if x < 0 else "N" + str(int(x)).zfill(2))
            # get the links
            grid_filtered["filename"] = grid_filtered.apply(lambda x: f"Copernicus_DSM_30_{x.formatted_lat}_00_{x.formatted_lon}_00.tar", axis = 1)
            grid_filtered["dl_link"] = grid_filtered.apply(lambda x: f"https://prism-dem-open.copernicus.eu/pd-desk-open-access/prismDownload/COP-DEM_GLO-90-DGED__2021_1/{x.filename}", axis = 1)
            # download and extract the files
            for idx, row in tqdm(grid_filtered.iterrows(), total = grid_filtered.shape[0]):
                if os.path.exists(self.root_dir + dataset["path"] + "/" + row.filename):
                    continue
                try:
                    urllib.request.urlretrieve(row.dl_link, self.root_dir + dataset["path"] + "/" + row.filename)
                    tarfile.open(self.root_dir + dataset["path"] + "/" + row.filename, "r").extractall(self.root_dir + dataset["path"])
                except:
                    pass
