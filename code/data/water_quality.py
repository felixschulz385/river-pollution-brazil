import os
import zipfile
import requests
import shelve
import numpy as np
import pandas as pd
import geopandas as gpd


class water_quality:
    """
    A class to process health data.
    """
    
    def __init__(self):
        pass
    
    def fetch(self):
        
        def get_stations_list(root_dir="/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/"):
            """
            Fetches the list of hydrological stations, filters outliers, checks their 
            geographical boundaries within Brazil, and saves the result to a Feather file.

            Parameters:
            root_dir (str): The root directory path where the output file and additional 
                            resources (e.g., boundary data) are located.

            Returns:
            None
            """
            # URL to fetch the list of hydrological stations
            url = "http://telemetriaws1.ana.gov.br/ServiceANA.asmx/HidroInventario?codEstDE=&codEstATE=&tpEst=&nmEst=&nmRio=&codSubBacia=&codBacia=&nmMunicipio=&nmEstado=&sgResp=&sgOper=&telemetrica="
            
            # Send a GET request to the URL and get the response content
            response = requests.get(url=url)
            
            # Parse the XML response to extract the station data into a DataFrame
            stations = pd.read_xml(response.content, xpath=".//Table")
            
            # Convert the DataFrame to a GeoDataFrame with geometry based on Longitude and Latitude
            stations_geo = gpd.GeoDataFrame(stations, geometry=gpd.points_from_xy(stations.Longitude, stations.Latitude))
            
            # Filter out stations with Longitude less than -100 to remove outliers
            stations_geo = stations_geo[stations_geo.Longitude > -100]
            
            # Load Brazilian boundaries from a geojson file
            brazil = gpd.read_file(f"{root_dir}data/misc/gadm_410-BRA.geojson", engine="pyogrio")
            
            # Check if stations are within the boundaries of Brazil
            stations_geo["in_bounds"] = stations_geo.intersects(brazil.iloc[0].geometry)
            
            # Save the filtered GeoDataFrame to a Feather file
            stations_geo.to_feather(f"{root_dir}data/water_quality/stations.feather")
            
            # Perform spatial join to find the nearest river for each station
            stations_rivers = gpd.sjoin_nearest(stations_geo.set_crs(4326).to_crs(5641), rivers)
            
            # Save the resulting GeoDataFrame to a Feather file
            stations_rivers[["Codigo", "geometry", "estuary", "river", "segment", "subsegment", "adm2", "distance_from_estuary"]].to_feather(f"{root_dir}data/water_quality/stations_rivers.feather")

            # Open the reachability information shelve database 
            with shelve.open(f"{root_dir}data/river_network/reachability.db") as reachability:
                # Define a worker function to retrieve reachability information for each station
                def worker(x):
                    # If upstream_node_id is NaN, return None
                    if np.isnan(x.upstream_node_id):
                        return None
                    # If the upstream_node_id is not in the reachability database, return None
                    if not str(int(x.upstream_node_id)) in reachability:
                        return None
                    # Return the reachability information from the database
                    return reachability[str(int(x.upstream_node_id))]
                
                # Apply the worker function to each row in the stations_rivers DataFrame to get reachability
                stations_rivers["reachability"] = stations_rivers.apply(worker, axis=1)

            # Save the resulting DataFrame to a pickle file
            stations_rivers.to_pickle(f"{root_dir}data/water_quality/stations_reachability.pkl")
        
        def fe_wq_ana(dataset):
            """
            Fetch (scrape) water quality data from the SNIRH website.
            """           
            
            # Check if the queries.json file already exists in the specified path
            if not os.path.exists(f"{self.root_dir}data/water_quality/queries.json"):
                # Load geographic data of stations from a Feather file
                stations_geo = gpd.read_feather(f"{self.root_dir}data/water_quality/stations.feather")
                
                # Filter stations where 'TipoEstacaoQualAgua' equals 1 and select only the 'Codigo' column
                stations_to_query = stations_geo.loc[stations_geo["TipoEstacaoQualAgua"] == 1, ["Codigo"]]
                
                # Initialize a new column 'success' with None values to track query success
                stations_to_query.loc[:, "success"] = None
                
                # Reset the index for a clean DataFrame and set 'Codigo' as the new index
                stations_to_query = stations_to_query.reset_index(drop=True).set_index("Codigo")
                
                # List all files in the raw data directory
                existing_files = os.listdir(f"{self.root_dir}data/water_quality/raw")
                
                # Extract numeric IDs from the file names and filter unique IDs
                existing_file_ids = pd.Series(existing_files).str.extract(r"(^\d*)").squeeze().astype(int).unique()
                
                # Update the 'success' status for stations where corresponding raw data files exist
                stations_to_query.loc[existing_file_ids, "success"] = True

                # Save the DataFrame to a JSON file for later use
                stations_to_query.to_json(f"{self.root_dir}data/water_quality/queries.json")
                
            stations_to_query = pd.read_json(f"{self.root_dir}data/water_quality/queries.json")
            
            downloadPath = f"{self.root_dir}data/water_quality/raw"
            os.makedirs(downloadPath, exist_ok=True)

            # docker run -d -p 4444:4444 -p 7900:7900 -v "/home/ubuntu/ext_drive/scraping/Masterthesis/data/water_quality/raw":"/home/seluser/downloads" --shm-size="2g" selenium/standalone-chrome:latest
            options = webdriver.ChromeOptions()
            options.add_argument('--ignore-ssl-errors=yes')
            options.add_argument('--ignore-certificate-errors')
            #options.add_argument('--headless')
            options.add_argument("--disable-extensions") 
            options.add_argument("--disable-gpu") 

            prefs = {}
            prefs["profile.default_content_settings.popups"]=0
            prefs["download.default_directory"]="/home/seluser/downloads"
            options.add_experimental_option("prefs", prefs)
            # Connect to the WebDriver
            driver = webdriver.Remote(command_executor='http://localhost:4444/wd/hub', options=options)
            
            def download_by_ID(id, driver):
                n_tries = 0
                while n_tries < 5:
                    n_tries += 1
                    
                    try:
                        # clear field
                        driver.find_element(By.XPATH, '//button[@type="reset"]').click()
                        sleep(.5)
                        # enter ID
                        driver.find_element(By.XPATH, '//*[@name="codigoEstacao"]').send_keys(str(id))
                        sleep(.5)
                        # click to search
                        driver.find_element(By.XPATH, '//button[@color="primary"]').click()
                        # wait for results to load
                        element = WebDriverWait(driver, 20).until(
                            EC.presence_of_element_located((By.XPATH, '//td[contains(@class, "mat-column-csv")]/button'))
                        )
                        # download csv
                        element.click()
                        sleep(1)
                        return 1
                    except:
                        sleep(1)
                        if ((n_tries == 2) | (n_tries == 4)):
                            disconnected = True
                            while disconnected:
                                try:
                                    driver.get("https://www.snirh.gov.br/hidroweb/serieshistoricas")
                                    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, '//button[@type="reset"]')))
                                    disconnected = False
                                except:
                                    pass
                return 0
            
            driver.get("https://www.snirh.gov.br/hidroweb/serieshistoricas")
            WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, '//button[@type="reset"]')))

            indices_to_fetch = stations_to_query[stations_to_query.success.isna()].index
            for idx in tqdm(range(len(indices_to_fetch))):
                stations_to_query.loc[indices_to_fetch[idx], "success"] = download_by_ID(indices_to_fetch[idx], driver)
                if idx % 10 == 0:
                    stations_to_query.to_json(f"{self.root_dir}data/water_quality/queries.json")
            
            driver.close()
            
        get_stations_list()
        fe_wq_ana()
        
    def preprocess(self):

            """
            Preprocesses water quality data files by unzipping, extracting relevant CSV data, 
            cleaning, and combining them into a single DataFrame, which is then saved as a Feather file.

            Parameters:
            root_path (str): The root directory path where the zipped water quality data files are located.

            Returns:
            None
            """
            # Create a DataFrame containing filenames in the specified directory
            files = pd.DataFrame({"filename": os.listdir(root_path)})
            
            # Get the filesize for each file
            files["filesize"] = files.filename.apply(lambda x: os.path.getsize(f"{root_path}{x}"))
            
            # Extract file IDs from filenames
            files["file_id"] = files.filename.str.extract(r"(^\d*)").iloc[:, 0]
            
            # Filter out files with invalid sizes, IDs, or not containing '_csv.zip'
            files = files.loc[((files.filesize > 22) & ~(files.file_id == "") & files.filename.str.contains(r"_csv.zip")), :]
            
            # Convert file IDs to integers
            files["file_id"] = files.file_id.astype(int)
            
            # Initialize a list to store DataFrames
            dfs = [None] * files.shape[0]
            
            # Loop through each file
            for i in range(files.shape[0]):
                # Open the zip file
                with zipfile.ZipFile(f"{root_path}{files.filename.iloc[i]}", "r") as z:
                    # Prepare the expected CSV file name inside the zip file
                    t_csvname = f"{files.file_id.iloc[i]}_QualAgua.csv"
                    
                    # Check if the expected CSV file exists in the zip file
                    if t_csvname in [x.filename for x in z.filelist]:
                        # Get the number of header lines to skip
                        with z.open(t_csvname) as f:
                            t_it = ""
                            while not "EstacaoCodigo" in t_it:
                                t_it = f.readline().decode("latin-1")
                            # Read the CSV file into a DataFrame
                            dfs[i] = pd.read_csv(f, names=t_it.split(";"), delimiter=";", encoding="latin-1")
                    else:
                        # If the expected CSV file does not exist, create an empty DataFrame
                        dfs[i] = pd.DataFrame()
            
            # Filter out None and empty DataFrames from the list
            dfs = [y for y in [x for x in dfs if x is not None] if not y.empty]
            
            # Concatenate all DataFrames into a single DataFrame
            dfs = pd.concat(dfs).copy()
            
            # Replace commas with dots and convert columns to float
            dfs[["pH", "OD", "TempAmostra", "Turbidez", "DBO", "SolTotais", "NitrogenioAmoniacal", "Nitratos"]] = dfs[["pH", "OD", "TempAmostra", "Turbidez", "DBO", "SolTotais", "NitrogenioAmoniacal", "Nitratos"]].apply(lambda x: x.str.replace(",", ".").astype(float), axis=0)
            
            # Correct the date format
            dfs["Data"] = dfs["Data"].str.replace("/3", "/2")
            
            # Convert the date column to datetime format
            dfs["date"] = pd.to_datetime(dfs["Data"], format="mixed", dayfirst=True)
            
            # Save the processed DataFrame to a Feather file
            dfs.to_feather(f"{root_path}quality_indicators.feather")
            
            # Group the data by station code and resample annually, calculating the mean for each year
            water_quality_panel = dfs.groupby("EstacaoCodigo")[["pH", "Turbidez", "DBO", "OD", "SolTotais", "NitrogenioAmoniacal", "Nitratos", "date"]].resample("1YE", on="date").mean()

            # Data cleaning: Apply value limits for each parameter, setting values outside these ranges to NaN
            water_quality_panel["pH"] = water_quality_panel["pH"].apply(lambda x: x if 0 <= x <= 12 else np.nan)
            water_quality_panel["Turbidez"] = water_quality_panel["Turbidez"].apply(lambda x: x if 0 <= x <= 100 else np.nan)
            water_quality_panel["DBO"] = water_quality_panel["DBO"].apply(lambda x: x if 0 <= x <= 50 else np.nan)
            water_quality_panel["OD"] = water_quality_panel["OD"].apply(lambda x: x if 0 <= x <= 200 else np.nan)
            water_quality_panel["SolTotais"] = water_quality_panel["SolTotais"].apply(lambda x: x if 0 <= x <= 100 else np.nan)
            water_quality_panel["NitrogenioAmoniacal"] = water_quality_panel["NitrogenioAmoniacal"].apply(lambda x: x if 0 <= x <= 100 else np.nan)
            water_quality_panel["Nitratos"] = water_quality_panel["Nitratos"].apply(lambda x: x if 0 <= x <= 10 else np.nan)

            # Reset the index to make 'date' a regular column
            water_quality_panel.reset_index(inplace=True)

            # Extract the year from the 'date' column and create a new 'year' column
            water_quality_panel["year"] = water_quality_panel["date"].dt.year

            # Drop the original 'date' column as it's no longer needed
            water_quality_panel.drop(columns=["date"], inplace=True)

            # Rename columns to more descriptive names
            water_quality_panel.rename(
                columns={
                    "EstacaoCodigo": "station",
                    "Turbidez": "turbidity", 
                    "DBO": "biochem_oxygen_demand",
                    "OD": "dissolved_oxygen",
                    "SolTotais": "total_residue",
                    "NitrogenioAmoniacal": "total_nitrogen",
                    "Nitratos": "nitrates"
                }, 
                inplace=True
            )

            # Set the index to be a multi-index of station and year
            water_quality_panel.set_index(["station", "year"], inplace=True)

            # Convert specific columns to float32 for more efficient storage and reset the index
            water_quality_panel.astype({x: np.float32 for x in ["turbidity", "biochem_oxygen_demand", "dissolved_oxygen", "total_residue", "total_nitrogen", "nitrates"]}).reset_index()
            
            # Save the final DataFrame to a Parquet file
            water_quality_panel.to_parquet(f"{root_dir}data/water_quality/quality_indicators_panel.parquet")