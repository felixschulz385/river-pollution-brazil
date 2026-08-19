from src.data.shared.webdriver import create_chrome_driver as _create_chrome_driver


def create_chrome_driver(headless=False, download_dir=None):
    return _create_chrome_driver(
        headless=headless,
        download_dir=download_dir,
        page_load_strategy="none",
    )

__all__ = ["create_chrome_driver"]
