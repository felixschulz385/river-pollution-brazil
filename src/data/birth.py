import time
import io
import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options


class birth:
    """
    A class to fetch and preprocess birth data from DATASUS SINASC.
    """
    
    def __init__(self, headless=False, download_dir=None):
        """
        Initialize the birth data scraper.
        
        Parameters:
        headless (bool): Whether to run Chrome in headless mode (default: False)
        download_dir (str): Directory for downloads (default: current directory/data/birth)
        """
        self.headless = headless
        self.download_dir = download_dir or os.path.join(os.getcwd(), "data", "birth")
        os.makedirs(self.download_dir, exist_ok=True)
    
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
        driver = webdriver.Chrome(options=options)
        
        return driver
    
    def fetch(self, subtype="all"):
        """
        Scrapes birth data from the DATASUS SINASC website.
        
        Parameters:
        subtype (str): Type of data to fetch. Options: 'all', 'gestation', 'weight'
        """
        
        def fe_bi_gestation():
            """
            Fetch birth data by gestation duration from DATASUS SINASC.
            """
            driver = self._get_chrome_driver()
            
            # Years to query (1994 to 2023)
            years = list(range(1994, 2024))
            
            # Dictionary to store the data
            out_df = {year: None for year in years}
            
            try:
                for year in years:
                    print(f"Fetching gestation duration data for year {year}...")
                    
                    # Open the URL
                    driver.get("http://tabnet.datasus.gov.br/cgi/tabcgi.exe?sinasc/cnv/nvbr.def")
                    
                    # Wait for the page to load
                    time.sleep(3)
                    
                    # Select 'Duração gestação' from the 'Coluna' dropdown
                    driver.find_element(By.XPATH, "//select[@name='Coluna']/option[@value='Duração_da_gestação']").click()
                    
                    # Select the year
                    driver.find_element(By.XPATH, f"//option[@value='dnv{year}.dbf']").click()
                    
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
            
            # Extract municipality ID and name from the 'Município' column
            out_df["mun_id"] = out_df.Município.str.extract(r"(\d{6})")[0].str.zfill(6)
            out_df["mun_name"] = out_df.Município.str.extract(r"\d{6}(.*)")[0].str.strip()
            
            # Drop the original 'Município' column
            out_df.drop(columns=["Município"], inplace=True)
            
            # Reorder columns
            out_df = out_df[["mun_id", "mun_name", "year"] + [col for col in out_df.columns if col not in ["mun_id", "mun_name", "year"]]]
            
            # Save the cleaned dataframe to a CSV file
            out_df.dropna().to_csv(f"{self.download_dir}/birth_gestation_duration.csv", index=False)
            print(f"Saved gestation duration data to {self.download_dir}/birth_gestation_duration.csv")
        
        def fe_bi_weight():
            """
            Fetch birth data by birth weight from DATASUS SINASC.
            """
            driver = self._get_chrome_driver()
            
            # Years to query (1994 to 2023)
            years = list(range(1994, 2024))
            
            # Dictionary to store the data
            out_df = {year: None for year in years}
            
            try:
                for year in years:
                    print(f"Fetching birth weight data for year {year}...")
                    
                    # Open the URL
                    driver.get("http://tabnet.datasus.gov.br/cgi/tabcgi.exe?sinasc/cnv/nvbr.def")
                    
                    # Wait for the page to load
                    time.sleep(3)
                    
                    # Select 'Peso ao nascer' from the 'Coluna' dropdown
                    driver.find_element(By.XPATH, "//select[@name='Coluna']/option[@value='Peso_ao_nascer']").click()
                    
                    # Select the year
                    driver.find_element(By.XPATH, f"//option[@value='dnv{year}.dbf']").click()
                    
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
            
            # Extract municipality ID and name from the 'Município' column
            out_df["mun_id"] = out_df.Município.str.extract(r"(\d{6})")[0].str.zfill(6)
            out_df["mun_name"] = out_df.Município.str.extract(r"\d{6}(.*)")[0].str.strip()
            
            # Drop the original 'Município' column
            out_df.drop(columns=["Município"], inplace=True)
            
            # Reorder columns
            out_df = out_df[["mun_id", "mun_name", "year"] + [col for col in out_df.columns if col not in ["mun_id", "mun_name", "year"]]]
            
            # Save the cleaned dataframe to a CSV file
            out_df.dropna().to_csv(f"{self.download_dir}/birth_weight.csv", index=False)
            print(f"Saved birth weight data to {self.download_dir}/birth_weight.csv")
        
        # ===========================================================
        # Execute scraping
        # ===========================================================
        
        if subtype in ["all", "gestation"]:
            print("Fetching gestation duration data...")
            fe_bi_gestation()
        
        if subtype in ["all", "weight"]:
            print("Fetching birth weight data...")
            fe_bi_weight()
        
        if subtype not in ["all", "gestation", "weight"]:
            raise ValueError(f"Invalid subtype: {subtype}. Choose from: 'all', 'gestation', 'weight'")
    
    def preprocess(self):
        """
        Preprocess the birth data files.
        """
        print("Birth data preprocessing not yet implemented.")
        pass
