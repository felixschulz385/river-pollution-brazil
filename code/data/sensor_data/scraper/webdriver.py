"""Chrome WebDriver lifecycle management."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from time import sleep
from typing import Generator

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

DEFAULT_WINDOW_WIDTH = 1600
DEFAULT_WINDOW_HEIGHT = 1000
DRIVER_CREATE_RETRIES = 3
DRIVER_RESTART_RETRIES = 2

logger = logging.getLogger(__name__)


class ManagedBrowser:
    """Context manager that owns a Chrome WebDriver instance."""

    def __init__(
        self,
        headless: bool = False,
        download_dir: str | None = None,
        extra_options: list[str] | None = None,
        keep_open_on_error: bool = False,
    ) -> None:
        self.headless = headless
        self.download_dir = download_dir
        self.extra_options = extra_options or []
        self.keep_open_on_error = keep_open_on_error
        self._driver: webdriver.Chrome | None = None
        self._driver_binary_path: str | None = None

    def __enter__(self) -> webdriver.Chrome:
        self._driver = self._create_driver()
        return self._driver

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None and self.keep_open_on_error:
            logger.warning(
                "Preserving Chrome window for debugging because an exception occurred: %s",
                exc_type.__name__,
            )
            return False

        self.quit()
        return False

    def quit(self) -> None:
        if self._driver is not None:
            try:
                self._driver.quit()
                logger.debug("Chrome driver quit successfully.")
            except Exception as exc:  # pragma: no cover
                logger.warning("Error while quitting driver: %s", exc)
            finally:
                self._driver = None

    @property
    def current_driver(self) -> webdriver.Chrome | None:
        return self._driver

    def restart(self) -> webdriver.Chrome:
        logger.warning("Restarting Chrome driver.")
        self.quit()
        last_error: Exception | None = None
        for attempt in range(1, DRIVER_RESTART_RETRIES + 1):
            try:
                self._driver = self._create_driver()
                return self._driver
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Chrome driver restart failed on attempt %s/%s: %s",
                    attempt,
                    DRIVER_RESTART_RETRIES,
                    exc,
                )
                sleep(min(5, attempt * 2))
        raise RuntimeError("Unable to restart Chrome driver after repeated failures.") from last_error

    def _create_driver(self) -> webdriver.Chrome:
        """Build a Chrome session with download preferences enabled."""
        options = webdriver.ChromeOptions()
        if self.headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(f"--window-size={DEFAULT_WINDOW_WIDTH},{DEFAULT_WINDOW_HEIGHT}")

        if self.download_dir:
            resolved_download_dir = str(Path(self.download_dir).expanduser().resolve())
            prefs = {
                "download.default_directory": resolved_download_dir,
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "safebrowsing.enabled": True,
            }
            options.add_experimental_option("prefs", prefs)

        for flag in self.extra_options:
            options.add_argument(flag)
        if self._driver_binary_path is None:
            self._driver_binary_path = ChromeDriverManager().install()

        last_error: Exception | None = None
        for attempt in range(1, DRIVER_CREATE_RETRIES + 1):
            try:
                service = Service(self._driver_binary_path)
                driver = webdriver.Chrome(service=service, options=options)
                driver.set_window_size(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
                logger.debug("Chrome driver created.")
                return driver
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Chrome driver creation failed on attempt %s/%s: %s",
                    attempt,
                    DRIVER_CREATE_RETRIES,
                    exc,
                )
                sleep(min(5, attempt * 2))
        raise RuntimeError("Unable to create Chrome driver after repeated failures.") from last_error


@contextmanager
def open_browser(
    headless: bool = False,
    download_dir: str | None = None,
    extra_options: list[str] | None = None,
    keep_open_on_error: bool = False,
) -> Generator[webdriver.Chrome, None, None]:
    with ManagedBrowser(
        headless=headless,
        download_dir=download_dir,
        extra_options=extra_options,
        keep_open_on_error=keep_open_on_error,
    ) as driver:
        yield driver
