"""Shared Chrome WebDriver lifecycle management."""

from __future__ import annotations

import logging
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from time import sleep
from typing import Generator

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import WebDriverException

DEFAULT_WINDOW_WIDTH = 1600
DEFAULT_WINDOW_HEIGHT = 1000
DRIVER_CREATE_RETRIES = 3
DRIVER_RESTART_RETRIES = 2
PAGE_LOAD_TIMEOUT_SECONDS = 30
PAGE_LOAD_RETRIES = 3
PAGE_LOAD_RETRY_BACKOFF_SECONDS = 2

logger = logging.getLogger(__name__)
NOISY_WEBDRIVER_LOGGERS = (
    "urllib3.connectionpool",
    "selenium.webdriver.remote.remote_connection",
    "selenium.webdriver.common.service",
    "selenium.webdriver.common.driver_finder",
    "selenium.webdriver.common.selenium_manager",
)


def _mute_noisy_webdriver_loggers() -> None:
    for logger_name in NOISY_WEBDRIVER_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.INFO)


def _build_chrome_options(
    *,
    headless: bool = False,
    download_dir: str | None = None,
    extra_options: list[str] | None = None,
    page_load_strategy: str = "eager",
    profile_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> webdriver.ChromeOptions:
    _mute_noisy_webdriver_loggers()
    options = webdriver.ChromeOptions()
    options.page_load_strategy = page_load_strategy
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-background-networking")
    options.add_argument("--no-first-run")
    options.add_argument("--no-default-browser-check")
    options.add_argument("--ignore-ssl-errors=yes")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--disable-features=Translate,OptimizationGuideModelDownloading")
    options.add_argument(f"--window-size={DEFAULT_WINDOW_WIDTH},{DEFAULT_WINDOW_HEIGHT}")

    if profile_dir is not None:
        options.add_argument(f"--user-data-dir={profile_dir}")
    if cache_dir is not None:
        options.add_argument(f"--disk-cache-dir={cache_dir}")

    if download_dir:
        resolved_download_dir = str(Path(download_dir).expanduser().resolve())
        prefs = {
            "profile.default_content_settings.popups": 0,
            "download.default_directory": resolved_download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
        }
        options.add_experimental_option("prefs", prefs)

    for flag in extra_options or []:
        options.add_argument(flag)
    return options


class ManagedBrowser:
    """Context manager that owns a Chrome WebDriver instance.

    `keep_open_on_error=True` is an interactive-debugging escape hatch: on an
    unhandled exception, `quit()` is skipped so the window stays open for
    inspection. That deliberately leaks the Chrome process and its
    `tempfile.mkdtemp` profile/cache directories for the rest of that
    process's lifetime -- there is no way to keep the window open for a
    human to look at while also cleaning those up. Do not leave this on for
    unattended/looped/scheduled runs, where each crash would leak another
    process and profile directory with nobody watching to close it.
    """

    def __init__(
        self,
        headless: bool = False,
        download_dir: str | None = None,
        extra_options: list[str] | None = None,
        page_load_strategy: str = "eager",
        keep_open_on_error: bool = False,
    ) -> None:
        self.headless = headless
        self.download_dir = download_dir
        self.extra_options = extra_options or []
        self.page_load_strategy = page_load_strategy
        self.keep_open_on_error = keep_open_on_error
        self._driver: webdriver.Chrome | None = None
        self._profile_dir: Path | None = None
        self._cache_dir: Path | None = None
        self._raw_driver_get = None
        self._raw_driver_quit = None

    def __enter__(self) -> webdriver.Chrome:
        logger.debug(
            "Opening managed Chrome browser: headless=%s, download_dir=%s, page_load_strategy=%s",
            self.headless,
            self.download_dir,
            self.page_load_strategy,
        )
        self._driver = self._create_driver()
        return self._driver

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None and self.keep_open_on_error:
            logger.warning(
                "Preserving Chrome window for debugging because an exception occurred: %s. "
                "The Chrome process and its profile/cache directory (%s) will NOT be cleaned "
                "up by this run -- close the window manually, and do not leave "
                "keep_open_on_error/keep_browser_on_error enabled for unattended runs.",
                exc_type.__name__,
                self._profile_dir,
            )
            return False

        self.quit()
        return False

    def quit(self) -> None:
        if self._driver is not None:
            try:
                logger.debug("Closing Chrome driver.")
                if self._raw_driver_quit is not None:
                    self._raw_driver_quit()
                else:
                    self._driver.quit()
                logger.debug("Chrome driver quit successfully.")
            except Exception as exc:  # pragma: no cover
                logger.warning("Error while quitting driver: %s", exc)
            finally:
                self._driver = None
                self._raw_driver_get = None
                self._raw_driver_quit = None
        self._cleanup_browser_dirs()

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
        self._cleanup_browser_dirs()
        self._profile_dir = Path(tempfile.mkdtemp(prefix="shared_chrome_profile_"))
        self._cache_dir = Path(tempfile.mkdtemp(prefix="shared_chrome_cache_"))
        logger.debug(
            "Prepared Chrome profile directories: profile_dir=%s, cache_dir=%s",
            self._profile_dir,
            self._cache_dir,
        )
        options = _build_chrome_options(
            headless=self.headless,
            download_dir=self.download_dir,
            extra_options=self.extra_options,
            page_load_strategy=self.page_load_strategy,
            profile_dir=self._profile_dir,
            cache_dir=self._cache_dir,
        )

        last_error: Exception | None = None
        for attempt in range(1, DRIVER_CREATE_RETRIES + 1):
            try:
                logger.debug(
                    "Creating Chrome driver attempt %s/%s with page_load_strategy=%s",
                    attempt,
                    DRIVER_CREATE_RETRIES,
                    self.page_load_strategy,
                )
                # No explicit driver binary/Service: Selenium Manager (built
                # into Selenium 4.6+) resolves and caches a chromedriver
                # matching the installed Chrome's own version automatically,
                # instead of relying on a separately-pinned binary (e.g. a
                # stale system-wide `chromedriver` on PATH) that can drift
                # out of sync with Chrome after a browser auto-update.
                driver = webdriver.Chrome(options=options)
                try:
                    self._raw_driver_get = driver.get
                    self._raw_driver_quit = driver.quit
                    driver.get = self.get
                    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT_SECONDS)
                    if not self.headless:
                        driver.set_window_size(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
                    logger.debug(
                        "Chrome driver created successfully: browser=%s, page_load_timeout=%ss",
                        driver.capabilities.get("browserVersion"),
                        PAGE_LOAD_TIMEOUT_SECONDS,
                    )
                    return driver
                except Exception:
                    # The Chrome process is already spawned at this point; if any
                    # setup step below fails, quit it before retrying so it isn't
                    # orphaned (a fresh Chrome process would otherwise be spawned
                    # on the next attempt with nothing left tracking this one).
                    try:
                        (self._raw_driver_quit or driver.quit)()
                    except Exception:
                        logger.debug("Failed to quit orphaned Chrome driver.", exc_info=True)
                    raise
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

    def _cleanup_browser_dirs(self) -> None:
        for path_attr in ("_profile_dir", "_cache_dir"):
            path = getattr(self, path_attr)
            if path is None:
                continue
            logger.debug("Cleaning Chrome temporary directory: %s", path)
            shutil.rmtree(path, ignore_errors=True)
            setattr(self, path_attr, None)

    def _stop_page_load(self) -> None:
        if self._driver is None:
            return
        try:
            logger.debug("Stopping in-flight Chrome page load.")
            self._driver.execute_script("window.stop();")
        except WebDriverException:
            logger.debug("Unable to stop current page load.", exc_info=True)

    def get(self, url: str) -> None:
        if self._driver is None or self._raw_driver_get is None:
            raise RuntimeError("Chrome driver is not initialized.")

        last_error: Exception | None = None
        for attempt in range(1, PAGE_LOAD_RETRIES + 1):
            try:
                logger.debug("Loading page attempt %s/%s: %s", attempt, PAGE_LOAD_RETRIES, url)
                self._raw_driver_get(url)
                logger.debug("Page load returned control for %s", url)
                return
            except TimeoutException as exc:
                last_error = exc
                self._stop_page_load()
                logger.warning(
                    "Chrome page load timed out for %s on attempt %s/%s.",
                    url,
                    attempt,
                    PAGE_LOAD_RETRIES,
                )
            except WebDriverException as exc:
                last_error = exc
                self._stop_page_load()
                logger.warning(
                    "Chrome page load failed for %s on attempt %s/%s: %s",
                    url,
                    attempt,
                    PAGE_LOAD_RETRIES,
                    exc,
                )

            if attempt == PAGE_LOAD_RETRIES:
                break

            logger.debug(
                "Backing off %ss before retrying page load for %s",
                min(5, attempt * PAGE_LOAD_RETRY_BACKOFF_SECONDS),
                url,
            )
            sleep(min(5, attempt * PAGE_LOAD_RETRY_BACKOFF_SECONDS))

        raise RuntimeError(f"Unable to load page after {PAGE_LOAD_RETRIES} attempts: {url}") from last_error


def create_chrome_driver(
    headless: bool = False,
    download_dir: str | None = None,
    page_load_strategy: str = "eager",
) -> webdriver.Chrome:
    """Create a managed Chrome driver and bind cleanup to `quit()`."""
    manager = ManagedBrowser(
        headless=headless,
        download_dir=download_dir,
        page_load_strategy=page_load_strategy,
    )
    # Not a `with` block on purpose: the driver is meant to outlive this call,
    # with cleanup deferred to the caller via the patched `driver.quit()`
    # below. If anything raises between `__enter__` (which already owns a
    # live Chrome process at that point) and the patched `quit` being wired
    # up, fall back to `manager.quit()` directly so that process isn't leaked.
    driver = manager.__enter__()
    try:
        driver.quit = manager.quit
        setattr(driver, "_shared_browser_manager", manager)
    except Exception:
        manager.quit()
        raise
    return driver


@contextmanager
def open_browser(
    headless: bool = False,
    download_dir: str | None = None,
    extra_options: list[str] | None = None,
    page_load_strategy: str = "eager",
    keep_open_on_error: bool = False,
) -> Generator[webdriver.Chrome, None, None]:
    with ManagedBrowser(
        headless=headless,
        download_dir=download_dir,
        extra_options=extra_options,
        page_load_strategy=page_load_strategy,
        keep_open_on_error=keep_open_on_error,
    ) as driver:
        yield driver
