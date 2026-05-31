import os

from selenium import webdriver
from selenium.webdriver.chrome.options import Options


def create_chrome_driver(headless=False, download_dir=None):
    """Create a Chrome driver configured for local DATASUS scraping."""
    options = Options()
    options.add_argument("--ignore-ssl-errors=yes")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-gpu")

    if headless:
        options.add_argument("--headless")

    prefs = {
        "profile.default_content_settings.popups": 0,
        "download.default_directory": download_dir or os.getcwd(),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
    }
    options.add_experimental_option("prefs", prefs)
    return webdriver.Chrome(options=options)
