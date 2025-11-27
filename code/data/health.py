import time
import io
import os
import pickle
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

class health:
    """
    A class to preprocess health data.
    """
    
    def __init__(self, headless=False, download_dir=None):
        """
        Initialize the health data scraper.
        
        Parameters:
        headless (bool): Whether to run Chrome in headless mode (default: False)
        download_dir (str): Directory for downloads (default: current directory)
        """
        self.headless = headless
        self.download_dir = download_dir or os.getcwd()
    
    def _get_chrome_driver(self):
        """
        Create and configure a Chrome WebDriver for local Mac use.
        
        Returns:
        webdriver.Chrome: Configured Chrome WebDriver instance
        """
        options = Options()
        options.add_argument('--ignore-ssl-errors=yes')
        options.add_argument('--ignore-certificate-errors')
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-gpu")
        
        if self.headless:
            options.add_argument('--headless')
        
        # Configure download preferences
        prefs = {
            "profile.default_content_settings.popups": 0,
            "download.default_directory": self.download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        }
        options.add_experimental_option("prefs", prefs)
        
        # Create driver using local Chrome
        # Note: This assumes chromedriver is in PATH or you can specify the path
        # service = Service('/path/to/chromedriver')  # Uncomment and set path if needed
        driver = webdriver.Chrome(options=options)
        
        return driver
    
    def fetch(self, subtype="all"):
        """
        Scrapes health data from the DATASUS TABNET website.
        Requires up to 3 hours to run
        
        Parameters:
        subtype (str): Type of data to fetch. Options: 'all', 'mortality', 'hospitalization', 'birth'
        """

        def fe_he_mo():
            """
            Fetch (scrape) mortality data from the DATASUS TABNET website.
            """
            
            def worker(mode):
                # Create local Chrome WebDriver
                driver = self._get_chrome_driver()
                
                # Years to query
                if mode == "pre":
                    years = list(range(79, 95))
                elif mode == "post":
                    years = list(range(96, 100)) + list(range(0, 22))
                years = [str(x).zfill(2) for x in years]

                # Dictionary to store the data
                out_df = {year: None for year in years}
                
                try:
                    for year in years:
                        # Open the URL
                        if mode == "pre":
                            driver.get("http://tabnet.datasus.gov.br/cgi/deftohtm.exe?sim/cnv/obt09br.def")
                        elif mode == "post":
                            driver.get("http://tabnet.datasus.gov.br/cgi/deftohtm.exe?sim/cnv/obt10br.def")
                            
                        # Wait for the page to load
                        time.sleep(3)
                        
                        # Select 'Faixa Etária' from the 'Coluna' dropdown
                        driver.find_element(By.XPATH, "//select[@name='Coluna']/option[@value='Faixa_Etária']").click()
                        
                        # If the year is not "22", select the corresponding year option
                        if ((not year == "22") and (mode == "pre")) or ((not year == "95") and (mode == "post")):
                            driver.find_element(By.XPATH, f"//option[@value='obtbr{year}.dbf']").click()
                        
                        # Select the 'prn' format
                        driver.find_element(By.XPATH, "//input[@name='formato' and @value='prn']").click()
                        
                        # Click the submit button
                        driver.find_element(By.XPATH, "//input[@class='mostra']").click()
                        
                        # Switch to the new window
                        driver.switch_to.window(driver.window_handles[-1])
                        
                        # Wait for the data to be displayed
                        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//pre")))
                        
                        # Extract the data from the 'pre' tag
                        data = driver.find_element(By.XPATH, "//pre").text
                        
                        # Read the data as a CSV from the string and store it in the dictionary
                        out_df[year] = pd.read_csv(io.StringIO(data), sep=';', encoding='latin1')
                        
                        # Close the current window
                        driver.close()
                        
                        # Switch back to the original window
                        driver.switch_to.window(driver.window_handles[0])
                        
                        # Optional: wait a bit before the next iteration
                        time.sleep(2)
                        
                        # Click the reset button
                        driver.find_element(By.XPATH, "//input[@class='limpa']").click()

                finally:
                    # Quit the WebDriver
                    driver.quit()
                    
                ## Data Postprocessing

                # Concatenate all dataframes in the dictionary into a single dataframe
                out_df = pd.concat(out_df)

                # Reset index and set 'year' as a column
                out_df = out_df.reset_index(level=0, names=["year"])

                # Adjust the 'year' column values (assuming years > 22 are in the 1900s and the rest are in the 2000s)
                out_df["year"] = out_df.year.astype(int).apply(lambda x: x + 1900 if x > 22 else x + 2000)

                # List of columns that need fixing (converting '-' to '0' and then to float)
                fix_cols = [
                    'Menor 1 ano', '1 a 4 anos', '5 a 9 anos',
                    '10 a 14 anos', '15 a 19 anos', '20 a 29 anos', '30 a 39 anos',
                    '40 a 49 anos', '50 a 59 anos', '60 a 69 anos', '70 a 79 anos',
                    '80 anos e mais', 'Idade ignorada'
                ]
                # Replace '-' with '0' and convert columns to float32
                out_df[fix_cols] = out_df[fix_cols].apply(lambda x: x.str.replace("-", "0"), axis=0).astype("float32")

                # Extract municipality ID and name from the 'Município' column
                out_df["mun_id"] = out_df.Município.str.extract(r"(\d{6})")[0].str.zfill(6)
                out_df["mun_name"] = out_df.Município.str.extract(r"\d{6}(.*)")[0].str.strip()

                # Drop the original 'Município' column as it's no longer needed
                out_df.drop(columns=["Município"], inplace=True)

                # Reorder columns to make 'mun_id', 'mun_name', and 'year' the first columns
                out_df = out_df[["mun_id", "mun_name", "year"] + [col for col in out_df.columns if col not in ["mun_id", "mun_name", "year"]]]

                # Rename columns to more parsable English names
                out_df.columns = [
                    'mun_id', 'mun_name', 'year', 'under_1', '1_to_4', '5_to_9', '10_to_14', '15_to_19',
                    '20_to_29', '30_to_39', '40_to_49', '50_to_59', '60_to_69', '70_to_79',
                    '80_and_more', 'age_unknown', 'total'
                ]

                # Drop rows with any missing values and save the cleaned dataframe to a CSV file
                out_df.dropna().to_csv(f"data/health/scraping_{mode}_1996.csv", index=False)

            worker("pre")
            worker("post")

        def fe_he_ho():
            """
            Fetch (scrape) hospital data from the DATASUS TABNET website.
            """
            
            def worker(mode="default"):
                ### --- OPTION "waterborne" NOT YET IMPLEMENTED ---
                
                # Create local Chrome WebDriver
                driver = self._get_chrome_driver()
                
                years = list(range(8, 22 + 1))
                years = [str(x).zfill(2) for x in years]

                # Dictionary to store the data
                out_df = {year: None for year in years}
                
                try:
                    for year in years:
                        # Open the URL
                        driver.get("http://tabnet.datasus.gov.br/cgi/deftohtm.exe?sih/cnv/qibr.def")
                        
                        # Wait for the page to load
                        time.sleep(3)  # Adjust the sleep time as needed
                        
                        # Select 'Faixa Etária' from the 'Coluna' dropdown
                        #driver.find_element(By.XPATH, "//select[@name='Coluna']/option[@value='Faixa_Etária']").click()
                        
                        # Select 'Valor aprovado' from the 'Incremento' dropdown
                        driver.find_element(By.XPATH, "//select[@name='Incremento']/option[@value='AIH_aprovadas']").click()
                        driver.find_element(By.XPATH, "//select[@name='Incremento']/option[@value='Internações']").click()
                        driver.find_element(By.XPATH, "//select[@name='Incremento']/option[@value='Valor_total']").click()
                        
                        # choose time period to query
                        driver.find_element(By.XPATH, "//option[@value='qibr2404.dbf']").click()
                        months = driver.find_elements(By.XPATH, f"//option[contains(@value, 'qibr{year}')]")
                        for month in months:
                            time.sleep(.2)
                            month.click()
                            
                        if mode == "waterborne":
                            # List of IDs corresponding to the queried medical procedures
                            procedure_ids = [
                                "0202040119", "0202040127", "0202040178",  # Stool Examination
                                "0213010240", "0213010275", "0213010216", "0213010453", "0202030750", "0202030873", "0202030776", "0213010020",  # Blood Tests
                                "0202080153", "0202020037", "0202020029", "0202020118", "0202010651", "0202010643",  # Blood Tests continued
                                "0214010120", "0214010139", "0214010180", "0214010058", "0214010104", "0214010090",  # Rapid Diagnostic Tests (RDTs)
                                "0213010208", "0213010194", "0213010186", "0213010011",  # PCR (Polymerase Chain Reaction)
                                "0301100209",  # Hydration Therapy
                                "0301100241", "0303010045", "0303010061",  # Antibiotic Treatment
                                "0303010100", "0303010150",  # Antiparasitic Treatment
                                "0303010118",  # Antiviral and Supportive Care
                                "0213010216", "0213010267",  # Antimalarial Treatment
                                "0303010142", "0303020032", "0303060301", "0303070129"  # Symptomatic Treatment
                            ]
                            
                            driver.find_element(By.XPATH, "//img[@id='fig15']").click()
                            time.sleep(1)
                            
                            driver.find_element(By.XPATH, f"//option[contains(text(), 'Todas as categorias')]").click()
                            for option_str in procedure_ids:
                                # select the procedure by its name
                                driver.find_element(By.XPATH, f"//option[contains(text(), '0101010010')]").click()
                                
                                driver.find_element(By.XPATH, f"//option[contains(text(), '{option_str}')]").click()
                        
                        # Select the 'prn' format
                        driver.find_element(By.XPATH, "//input[@name='formato' and @value='prn']").click()
                        
                        # Click the submit button
                        driver.find_element(By.XPATH, "//input[@class='mostra']").click()
                        
                        # Switch to the new window
                        driver.switch_to.window(driver.window_handles[-1])
                        
                        # Wait for the data to be displayed
                        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//pre")))
                        
                        # Extract the data from the 'pre' tag
                        data = driver.find_element(By.XPATH, "//pre").text
                        
                        # Read the data as a CSV from the string and store it in the dictionary
                        out_df[year] = pd.read_csv(io.StringIO(data), sep=';', encoding='latin1')
                        
                        # Close the current window
                        driver.close()
                        
                        # Switch back to the original window
                        driver.switch_to.window(driver.window_handles[0])
                        
                        # Optional: wait a bit before the next iteration
                        time.sleep(2)  # Adjust the sleep time as needed
                        
                        # Click the reset button
                        driver.find_element(By.XPATH, "//input[@class='limpa']").click()

                        pickle.dump(out_df, open(f"/home/ubuntu/ext_drive/scraping/Masterthesis/data/hospital/tmp_scraping.pkl", "wb"))
                        
                    # Concatenate out_df
                    out_df = pd.concat(out_df)
                    
                    # Reset index and set 'year' as a column
                    out_df = out_df.reset_index(level=0, names=["year"])

                    # Adjust the 'year' column values (assuming years > 22 are in the 1900s and the rest are in the 2000s)
                    out_df["year"] = out_df.year.astype(int).apply(lambda x: x + 1900 if x > 22 else x + 2000)

                    # Extract municipality ID and name from the 'Município' column
                    out_df["CC_2r"] = out_df.Município.str.extract(r"(\d{6})")[0].str.zfill(6)

                    # Drop the original 'Município' column as it's no longer needed
                    out_df.drop(columns=["Município"], inplace=True)

                    # Reorder columns to make 'CC_2r', 'mun_name', and 'year' the first columns
                    out_df = out_df[["CC_2r", "year"] + [col for col in out_df.columns if col not in ["CC_2r", "mun_name", "year"]]]

                    # Rename columns to more parsable English names
                    out_df.columns = [
                        'CC_2r', 'year', 'n_approved', 'hospitalizations', 'total_value'
                    ]

                    if not mode == "waterborne":
                        out_df.dropna().to_parquet("data/health/hospitalizations.parquet", index=False)
                    if mode == "waterborne":
                        out_df.dropna().to_parquet("data/health/hospitalizations_waterborne.parquet", index=False)
                
                finally:
                    # Quit the WebDriver
                    driver.quit()
            
            worker()
        
        def fe_he_bi():
            """
            Fetch (scrape) birth data (SINASC) from the DATASUS TABNET website.
            Fetches gestational duration and birth weight data for years 1994-2023.
            """
            
            def worker(column_type):
                """
                Worker function to scrape birth data for a specific column type.
                
                Parameters:
                column_type (str): Either 'gestational_duration' or 'birth_weight'
                """
                # Create local Chrome WebDriver
                driver = self._get_chrome_driver()
                
                # Years to query (1994 to 2023)
                years = list(range(1994, 2024))
                latest_year = 2023  # Latest available year in the data
                
                # Dictionary to store the data
                out_df = {year: None for year in years}
                
                # Map column types to XPATH values
                column_mapping = {
                    'gestational_duration': 'Duração_gestação',
                    'birth_weight': 'Peso_ao_nascer'
                }
                
                try:
                    for year in years:
                        print(f"Fetching {column_type} data for year {year}...")
                        
                        # Open the URL
                        driver.get("http://tabnet.datasus.gov.br/cgi/tabcgi.exe?sinasc/cnv/nvbr.def")
                        
                        # Wait for the page to load
                        time.sleep(3)
                        
                        # Select the appropriate column from the 'Coluna' dropdown
                        driver.find_element(By.XPATH, f"//select[@name='Coluna']/option[@value='{column_mapping[column_type]}']").click()
                        
                        # Convert full year to two-digit format for file selection
                        year_2digit = str(year % 100).zfill(2)
                        latest_year_2digit = str(latest_year % 100).zfill(2)
                        
                        # Select the corresponding year option
                        # Always deselect the latest year first, then select the target year
                        if year != latest_year:
                            driver.find_element(By.XPATH, f"//option[@value='nvbr{year_2digit}.dbf']").click()
                            driver.find_element(By.XPATH, f"//option[@value='nvbr{latest_year_2digit}.dbf']").click()
                            
                        
                        # Select the 'prn' format
                        driver.find_element(By.XPATH, "//input[@name='formato' and @value='prn']").click()
                        
                        # Click the submit button
                        driver.find_element(By.XPATH, "//input[@class='mostra']").click()
                        
                        # Switch to the new window
                        driver.switch_to.window(driver.window_handles[-1])
                        
                        # Wait for the data to be displayed
                        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//pre")))
                        
                        # Extract the data from the 'pre' tag
                        data = driver.find_element(By.XPATH, "//pre").text
                        
                        # Read the data as a CSV from the string and store it in the dictionary
                        out_df[year] = pd.read_csv(io.StringIO(data), sep=';', encoding='latin1')
                        
                        # Close the current window
                        driver.close()
                        
                        # Switch back to the original window
                        driver.switch_to.window(driver.window_handles[0])
                        
                        # Optional: wait a bit before the next iteration
                        time.sleep(2)
                        
                        # Click the reset button
                        driver.find_element(By.XPATH, "//input[@class='limpa']").click()
                
                finally:
                    # Quit the WebDriver
                    driver.quit()
                
                ## Data Postprocessing
                
                # Concatenate all dataframes in the dictionary into a single dataframe
                out_df = pd.concat(out_df)
                
                # Reset index and set 'year' as a column
                out_df = out_df.reset_index(level=0, names=["year"])
                
                # Year column is already in full format (1994-2023) from the dictionary keys
                
                # Extract municipality ID and name from the 'Município' column
                out_df["mun_id"] = out_df.Município.str.extract(r"(\d{6})")[0].str.zfill(6)
                out_df["mun_name"] = out_df.Município.str.extract(r"\d{6}(.*)")[0].str.strip()
                
                # Drop the original 'Município' column as it's no longer needed
                out_df.drop(columns=["Município"], inplace=True)
                
                # Get the column names (excluding mun_id, mun_name, year, and Total)
                value_cols = [col for col in out_df.columns if col not in ["mun_id", "mun_name", "year", "Total"]]
                
                # Replace '-' with '0' and convert columns to float32
                out_df[value_cols] = out_df[value_cols].apply(lambda x: x.str.replace("-", "0"), axis=0).astype("float32")
                
                # Reorder columns to make 'mun_id', 'mun_name', and 'year' the first columns
                out_df = out_df[["mun_id", "mun_name", "year"] + value_cols + ["Total"]]
                
                # Create output directory if it doesn't exist
                os.makedirs("data/health", exist_ok=True)
                
                # Drop rows with any missing values and save the cleaned dataframe to a CSV file
                out_df.dropna(subset=["mun_id"]).to_parquet(f"data/health/{column_type}.parquet", index=False)
                
                print(f"Saved {column_type} data to data/health/{column_type}.csv")
            
            # Fetch both types of birth data
            print("Fetching gestational duration data...")
            worker('gestational_duration')
            
            print("Fetching birth weight data...")
            worker('birth_weight')
        
        # ===========================================================
        # Execute scraping
        # ===========================================================
        
        if subtype in ["all", "mortality"]:
            print("Fetching mortality data...")
            fe_he_mo()
        
        if subtype in ["all", "hospitalization"]:
            print("Fetching hospitalization data...")
            fe_he_ho()
        
        if subtype in ["all", "birth"]:
            print("Fetching birth data...")
            fe_he_bi()
        
        if subtype not in ["all", "mortality", "hospitalization", "birth"]:
            raise ValueError(f"Invalid subtype: {subtype}. Choose from: 'all', 'mortality', 'hospitalization', 'birth'")
    
    def preprocess(self):
        pass