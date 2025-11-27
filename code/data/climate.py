import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
import zipfile
import cdsapi
from geocube.api.core import make_geocube
from tqdm import tqdm
import multiprocessing as mp
import pickle

class climate:
    """
    A class to download climate data.
    """
    
    def __init__(self):
        pass
    
    def fetch(self):
        
        def fe_cc_co(dataset):
            """
            Fetch Copernicus Cloud Cover data
            
            Note: Requires a Copernicus Climate Data Store account and API key installed on the device.
            """
            
            # Create a CDS API client
            c = cdsapi.Client()
            # Retrieve the data
            c.retrieve(
                'satellite-cloud-properties',
                {
                    'format': 'zip',
                    'product_family': 'clara_a3',
                    'origin': 'eumetsat',
                    'variable': 'cloud_fraction',
                    'climate_data_record_type': 'thematic_climate_data_record',
                    'time_aggregation': 'monthly_mean',
                    'year': [
                        '1979', '1980', '1981',
                        '1982', '1983', '1984',
                        '1985', '1986', '1987',
                        '1988', '1989', '1990',
                        '1991', '1992', '1993',
                        '1994', '1995', '1996',
                        '1997', '1998', '1999',
                        '2000', '2001', '2002',
                        '2003', '2004', '2005',
                        '2006', '2007', '2008',
                        '2009', '2010', '2011',
                        '2012', '2013', '2014',
                        '2015', '2016', '2017',
                        '2018', '2019', '2020',
                    ],
                    'month': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                    ],
                },
                self.root_dir + dataset["path"])
            
            # extract the files
            with zipfile.ZipFile(self.root_dir + dataset["path"], 'r') as zip_ref:
                    zip_ref.extractall((self.root_dir + dataset["path"]).split("raw")[0] + "raw")
        
        def fe_cc_deter(dataset):
            t_website_index = requests.get("https://www.dpi.inpe.br/prodesdigital/dados/deter_modis_2004_2017")
            # parse index
            t_website_index_parsed = BeautifulSoup(t_website_index.text, "html.parser")
            # get years
            t_index_years = pd.Series([x.get("href") for x in t_website_index_parsed.find_all("a")])
            t_index_years = t_index_years[t_index_years.str.match(r"\d{4}\/")]
            for c_year in t_index_years:
                t_website_index = requests.get(f"https://www.dpi.inpe.br/prodesdigital/dados/deter_modis_2004_2017/{c_year}")
                # parse index
                t_website_index_parsed = BeautifulSoup(t_website_index.text, "html.parser")
                #
                t_index_files = pd.Series([x.get("href") for x in t_website_index_parsed.find_all("a")])
                #
                t_download_files = t_index_files[t_index_files.str.match(r"([Nn]uvem|[Nn]uvens|[Dd]eter)")]
                # download files with urllib
                for c_file in t_download_files:
                    if not os.path.exists(f"{(self.root_dir + dataset['path']).split('raw')[0] + 'raw'}/{c_year}"):
                        os.makedirs(f"{(self.root_dir + dataset['path']).split('raw')[0] + 'raw'}/{c_year}")
                    if not os.path.exists(f"{(self.root_dir + dataset['path']).split('raw')[0] + 'raw'}/{c_year}{c_file}"):
                        urlretrieve(f"https://www.dpi.inpe.br/prodesdigital/dados/deter_modis_2004_2017/{c_year}{c_file}", f"{(self.root_dir + dataset['path']).split('raw')[0] + 'raw'}/{c_year}{c_file}")
                            
            for c_year in t_index_years:
                t_extract_files = os.listdir(f"{(self.root_dir + dataset['path']).split('raw')[0] + 'raw'}/{c_year}")
                # extract files
                for c_file in t_extract_files:
                    if c_file.endswith(".zip"):
                        ZipFile(f"{(self.root_dir + dataset['path']).split('raw')[0] + 'raw'}/{c_year}/{c_file}").extractall(f"{(self.root_dir + dataset['path']).split('raw')[0] + 'raw'}/{c_year}")
        
        agent.fetch(get_fetch_instructions("cl_noaa_temp"))
        agent.fetch(get_fetch_instructions("cl_noaa_precip"))
        fe_cc_co()
        fe_cc_deter()
                  
    def preprocess(self):
        # read in the boundaries
        boundaries = gpd.read_file("data/misc/raw/gadm/gadm41_BRA_2.json", engine="pyogrio")
        boundaries["CC_2r"] = boundaries.CC_2.str.slice(0, 6).astype(int)
        # read in the climate data
        precipitation = xr.open_mfdataset("data/climate/raw/precip.*.nc", chunks = "auto", decode_times=True, decode_cf = True)
        temperature = xr.open_mfdataset("data/climate/raw/tmax.*.nc", chunks = "auto", decode_times=True, decode_cf = True)
        
        # merge
        weather_data = xr.merge([temperature, precipitation])
        
        # fix longitudes
        weather_data = weather_data.assign_coords({"lon": (np.vectorize(lambda lon: lon - 360 if lon > 180 else lon)(precipitation.lon))})
        # sort by lon
        weather_data = weather_data.sortby("lon")
        # set CRS
        weather_data = weather_data.rio.write_crs("epsg:4326").rio.set_spatial_dims(x_dim="lon", y_dim="lat").rio.write_coordinate_system(inplace=True)
        
        # subset
        weather_data = weather_data.rio.clip_box(*boundaries.total_bounds).persist()
        
        # read in cloud cover
        cloud_cover = xr.open_mfdataset(
            "data/climate/raw/*.nc", 
            chunks = "auto", decode_times=True, decode_cf = True
            )

        # merge
        cloud_cover = cloud_cover.cfc.reindex_like(weather_data, method="nearest").persist()
        weather_data["cloud_cover"] = cloud_cover
        
        # resample by year
        weather_data = weather_data.resample(time="1Y").mean().load()
        
        # turn boundaries into a grid
        boundaries_grid = make_geocube(
            vector_data=boundaries[["CC_2r", "geometry"]],
            like=weather_data
        ).CC_2r.rename({"x": "lon", "y": "lat"})

        # merge
        weather_data["CC_2r"] = boundaries_grid
        
        # get mean by CC_2r
        weather_data_df = weather_data.groupby("CC_2r").mean().to_dataframe()
        weather_data_df = weather_data_df.reset_index().drop(columns="spatial_ref")

        # an extraction worker
        def worker(x):
            tmp = weather_data.sel(lon=x.x, lat=x.y, method="nearest")
            return pd.DataFrame({"time": tmp.time.values, "cloud_cover": tmp.cloud_cover.values, "tmax": tmp.tmax.values, "precip": tmp.precip.values})
        
        # extract closest for those that are too small for rasterization
        weather_data_df_merge = pd.concat(boundaries[~boundaries.CC_2r.isin(weather_data_df.CC_2r.unique())].set_index("CC_2r").centroid.apply(worker).to_dict())
        weather_data_df_merge = weather_data_df_merge.reset_index(names = ["CC_2r", "t"]).drop(columns="t")

        
        # get all downloaded DETER cloud cover data files
        def worker(c_year):
            tmp = pd.Series(os.listdir(f"{root_dir}data/climate/raw/{c_year}"))
            return c_year + tmp[tmp.str.contains(r"\.shp$")]
        files = pd.DataFrame({"file": pd.concat([worker(file) for file in t_index_years])}).reset_index(drop = True)
        files["year"] = files.file.str.extract(r"(\d{4})")
        
        # a dictionary of ALL CAPS month names in brazilian portuguese with their respective number
        month_dict = {
            "JANEIRO": "01",
            "JANEIR0": "01",
            "FEVEREIRO": "02",
            "MARCO": "03",
            "MARÇO": "03",
            "ABRIL": "04",
            "MAIO": "05",
            "JUNHO": "06",
            "JULHO": "07",
            "AGOSTO": "08",
            "SETEMBRO": "09",
            "OUTUBRO": "10",
            "NOVEMBRO": "11",
            "DEZEMBRO": "12",
            "jan": "01",
            "fev": "02",
            "mar": "03",
            "abr": "04",
            "mai": "05",
            "jun": "06",
            "jul": "07",
            "ago": "08",
            "set": "09",
            "out": "10",
            "nov": "11",
            "dez": "12",
            "Jan": "01",
            "Fev": "02",
            "Mar": "03",
            "Abr": "04",
            "Mai": "05",
            "Jun": "06",
            "Jul": "07",
            "Ago": "08",
            "Set": "09",
            "Out": "10",
            "Nov": "11",
            "Dez": "12",
            "JAN": "01",
            "FEV": "02",
            "MAR": "03",
            "ABR": "04",
            "MAI": "05",
            "JUN": "06",
            "JUL": "07",
            "AGO": "08",
            "SET": "09",
            "OUT": "10",
            "NOV": "11",
            "DEZ": "12"
        }
    
        # detect and extract month from file name based on month_dict
        files["month"] = files.file.str.extract(r"(" + "|".join(month_dict.keys()) + ")").replace(month_dict)
        # if month is not found, get from numeric in file name
        files.loc[files.month.isna(), "month"] = files.loc[files.month.isna(), "file"].str.extract(".*_\d{4}(\d{2})\d{2}_.*\.shp", expand=False)
        files.loc[files.month.isna(), "month"] = files.loc[files.month.isna(), "file"].str.extract(".*_\d{4}(\d{2})_.*\.shp", expand=False)
        files.loc[files.month.isna(), "month"] = files.loc[files.month.isna(), "file"].str.extract(".*_\d{4}_(\d{2}).*\.shp", expand=False)
        files.loc[files.month.isna(), "month"] = files.loc[files.month.isna(), "file"].str.extract(".*\d{4}(\d{2})\d{2}.*\.shp", expand=False)
        files.loc[files.month.isna(), "month"] = files.loc[files.month.isna(), "file"].str.extract(".*\d{4}(\d{2})\.shp", expand=False)
        #files = files.astype({"year": int, "month": int})
        files["type"] = files.file.str.contains(r"uvem|uvens", flags=re.IGNORECASE).map({True: "cloud_cover", False: "DETER"})
        boundaries = gpd.read_file(f"{root_dir}data/misc/raw/gadm/gadm41_BRA_2.json", engine="pyogrio")
        boundaries["CC_2r"] = boundaries.CC_2.str.slice(0, 6).astype(int)

        # iterate over all cloud cover files and extract
        out_dict = {}
        for file in tqdm(files.query("type == 'cloud_cover'").file, total = files.query("type == 'cloud_cover'").file.size):
            
            # load the file
            c_clouds = gpd.read_file(f"{root_dir}/data/climate/raw/{file}", engine="pyogrio")
            
            # set CRS if not set
            if c_clouds.crs is not None:
                c_clouds = c_clouds.to_crs(4326)
            else:
                c_clouds = c_clouds.set_crs(4326)
            
            # prepare to be turned into mask
            c_clouds["cloud_cover"] = 1
            c_clouds = c_clouds[["cloud_cover", "geometry"]]
            
            # turn into grid
            cloud_cover_grid = make_geocube(
                vector_data=c_clouds,
                measurements=["cloud_cover"],
                fill = 0,
                output_crs="epsg:4326",
                resolution=(-.01, .01)
            )
            
            # match with boundaries
            boundaries_grid = make_geocube(
                vector_data=boundaries[["CC_2r", "geometry"]],
                like=cloud_cover_grid
            ).CC_2r
            
            # merge
            cloud_cover_grid["CC_2r"] = boundaries_grid
            # get mean by CC_2r
            out_dict[file] = cloud_cover_grid.set_coords("CC_2r").groupby("CC_2r").mean().cloud_cover.to_pandas()
        # save to pickle
        pickle.dump(out_dict, open(f"{root_dir}data/climate/DETER_cc2r.pkl", "wb"))
        
        # turn into long dataframe
        out_df = pd.DataFrame(out_dict).transpose().reset_index(names="file").melt(["file"], value_name = "cloud_cover")
        # merge with month and year
        out_df = pd.merge(files[["file", "year", "month"]], out_df, on="file")
        # filter out all CC_2r for which all are NaN or 0
        out_df = out_df.groupby('CC_2r').filter(lambda x: not ((x['cloud_cover'] == 0) | (x['cloud_cover'].isna())).all())
        # aggregate to yearly mean
        out_df_agg = out_df.groupby(["CC_2r", "year"]).agg({"cloud_cover": "mean"}).reset_index().astype({"year": int}).rename(columns = {"cloud_cover": "cloud_cover_DETER"})

        # combine
        weather_data_df = pd.concat([weather_data_df, weather_data_df_merge])
        weather_data_df["year"] = pd.to_datetime(weather_data_df.time).dt.year
        weather_data_df = weather_data_df.drop(columns="time")
        weather_data_df = weather_data_df.rename(columns={"tmax": "temperature", "precip": "precipitation"})
        
        weather_data_df = pd.merge(weather_data_df, out_df_agg, on=["CC_2r", "year"], how="outer")
        
        weather_data_df[["CC_2r", "year", "cloud_cover", "cloud_cover_DETER", "temperature", "precipitation"]].to_parquet(f"{root_dir}data/climate/climate_data.parquet", index = False)
                