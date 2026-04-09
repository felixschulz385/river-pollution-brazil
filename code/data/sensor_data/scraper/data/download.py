import random
import logging
import re
import shutil
import zipfile
from pathlib import Path
from time import monotonic, sleep
from datetime import datetime
import unicodedata
from uuid import uuid4

import duckdb
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import InvalidSessionIdException, TimeoutException, WebDriverException
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

from ..database import (
    RAW_ARCHIVES_TABLE,
    SENSOR_DOWNLOADS_TABLE,
    append_dataframe_table,
    table_exists,
    write_dataframe_table,
)
from ..paths import ensure_water_quality_dirs, get_download_log_database_path
from ..webdriver import ManagedBrowser
from .station_selection import load_queryable_stations

logger = logging.getLogger(__name__)

SERIES_URL = "https://www.snirh.gov.br/hidroweb/serieshistoricas"
RESTART_BROWSER_EVERY_N_STATIONS = 25
STATUS_LOG_EVERY_SECONDS = 60
STATUS_LOG_EVERY_STATIONS = 25
DOWNLOAD_LOG_COLUMNS = [
    "run_id",
    "attempted_at",
    "fetch_mode",
    "station_code",
    "result_tab",
    "station_type",
    "download_format",
    "status",
    "success",
    "attempts",
    "archive_path",
    "last_error",
]
FETCH_MODES = {
    "default",
    "missing-only",
    "retry-failed",
    "redownload-all",
}


def _decency_wait(min_seconds=1.2, max_seconds=2.8):
    """Throttle requests a little so test runs still behave politely."""
    sleep(random.uniform(min_seconds, max_seconds))


def _debug_dir(root_dir=".") -> Path:
    """Store live page snapshots for selector debugging."""
    debug_dir = ensure_water_quality_dirs(root_dir)[0] / "debug_html"
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir


def _dump_debug_html(driver, root_dir=".", label="page", station_code=None):
    """Persist the current DOM so scraper failures are inspectable afterwards."""
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    code_prefix = f"{station_code}_" if station_code is not None else ""
    output_path = _debug_dir(root_dir) / f"{code_prefix}{label}_{timestamp}.html"
    try:
        page_source = driver.page_source
    except Exception as exc:
        page_source = f"<!-- Unable to capture page source: {exc} -->"
        logger.warning("Could not capture debug HTML snapshot from the active browser: %s", exc)
    output_path.write_text(page_source, encoding="utf-8", errors="ignore")
    logger.info("Saved debug HTML snapshot to %s.", output_path)
    return output_path


def _browser_session_lost(exc: Exception) -> bool:
    if isinstance(exc, InvalidSessionIdException):
        return True
    if not isinstance(exc, WebDriverException):
        return False
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "invalid session id",
            "session deleted",
            "disconnected",
            "not connected to devtools",
            "browser has closed the connection",
        )
    )


def _archive_storage_name(station_code: str) -> str:
    """Create a unique but readable archive filename for local storage."""
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid4().hex[:8]
    return f"{station_code}_mdb_{timestamp}_{suffix}.zip"


def _slugify_label(value: str | None) -> str:
    if pd.isna(value):
        value = None
    text = unicodedata.normalize("NFKD", str(value or "unknown"))
    text = text.encode("ascii", "ignore").decode("ascii").strip().lower()
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def _category_key(result_tab: str, station_type: str | None, download_format: str = "mdb") -> tuple[str, str, str]:
    return (str(result_tab), str(station_type or ""), str(download_format))


def _current_timestamp() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _format_duration(total_seconds: float) -> str:
    total_seconds = max(int(total_seconds), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _log_status_window(
    download_records: list[dict[str, object]],
    completed_stations: int,
    pending_stations: int,
    total_prepared_stations: int,
    run_started_at: float,
    last_station_id: str | None = None,
) -> None:
    elapsed_seconds = monotonic() - run_started_at
    remaining_pending_stations = max(pending_stations - completed_stations, 0)
    skipped_stations = max(total_prepared_stations - pending_stations, 0)
    done_total_stations = min(skipped_stations + completed_stations, total_prepared_stations)
    stations_per_minute = (completed_stations / elapsed_seconds * 60) if elapsed_seconds > 0 else 0.0
    eta_seconds = (
        (remaining_pending_stations / completed_stations * elapsed_seconds)
        if completed_stations > 0 else 0.0
    )

    downloaded_categories = sum(1 for record in download_records if record["status"] == "downloaded")
    failed_categories = sum(
        1 for record in download_records if str(record["status"]).endswith("failed") or record["status"] == "empty_archive"
    )
    skipped_categories = sum(1 for record in download_records if int(record["attempts"]) == 0)

    status_lines = [
        f"Prepared stations : {done_total_stations}/{total_prepared_stations}",
        f"Pending stations  : {completed_stations}/{pending_stations} processed, {remaining_pending_stations} remaining",
        f"Skipped stations  : {skipped_stations}",
        f"Categories        : {downloaded_categories} downloaded, {failed_categories} failed, {skipped_categories} skipped",
        f"Elapsed           : {_format_duration(elapsed_seconds)}",
        f"Rate              : {stations_per_minute:.2f} pending stations/min",
    ]
    if completed_stations > 0 and remaining_pending_stations > 0:
        status_lines.append(f"ETA               : {_format_duration(eta_seconds)}")
    if last_station_id is not None:
        status_lines.append(f"Last station      : {last_station_id}")

    content_width = max(len(line) for line in status_lines + ["Fetch Status"])
    top_border = "+" + "-" * (content_width + 2) + "+"
    boxed_lines = [top_border, f"| {'Fetch Status'.ljust(content_width)} |"]
    boxed_lines.append("|" + " " * (content_width + 2) + "|")
    boxed_lines.extend(f"| {line.ljust(content_width)} |" for line in status_lines)
    boxed_lines.append(top_border)
    logger.info("\n%s", "\n".join(boxed_lines))


def _download_log_connection(root_dir: str) -> duckdb.DuckDBPyConnection:
    ensure_water_quality_dirs(root_dir)
    return duckdb.connect(str(get_download_log_database_path(root_dir)))


def _current_raw_archives_frame(raw_dir: Path) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "filename": [path.name for path in raw_dir.iterdir() if path.is_file()],
            "file_size_bytes": [path.stat().st_size for path in raw_dir.iterdir() if path.is_file()],
        }
    )


def _compute_pending_station_ids(
    root_dir: str,
    stations_to_query: pd.DataFrame,
    fetch_mode: str,
) -> list[str]:
    stations_frame = stations_to_query.reset_index().copy()
    stations_frame["station_code"] = stations_frame["station_code"].astype(str)
    has_download_history = table_exists(root_dir, SENSOR_DOWNLOADS_TABLE)
    with _download_log_connection(root_dir) as connection:
        connection.register("_stations_to_query", stations_frame)
        query_by_mode = {
            "redownload-all": """
                SELECT station_code
                FROM _stations_to_query
                ORDER BY station_code
            """,
        }
        if has_download_history:
            query_by_mode.update(
                {
                    "default": """
                        WITH successful_stations AS (
                            SELECT DISTINCT CAST(station_code AS VARCHAR) AS station_code
                            FROM sensor_downloads
                            WHERE COALESCE(CAST(success AS INTEGER), 0) = 1
                        ),
                        archived_stations AS (
                            SELECT DISTINCT regexp_extract(filename, '^(\\d+)', 1) AS station_code
                            FROM raw_archives
                            WHERE regexp_matches(filename, '^(\\d+)_')
                        )
                        SELECT s.station_code
                        FROM _stations_to_query s
                        LEFT JOIN successful_stations ok USING (station_code)
                        LEFT JOIN archived_stations arc USING (station_code)
                        WHERE ok.station_code IS NULL
                          AND arc.station_code IS NULL
                        ORDER BY s.station_code
                    """,
                    "missing-only": """
                        WITH seen_stations AS (
                            SELECT DISTINCT CAST(station_code AS VARCHAR) AS station_code
                            FROM sensor_downloads
                        ),
                        archived_stations AS (
                            SELECT DISTINCT regexp_extract(filename, '^(\\d+)', 1) AS station_code
                            FROM raw_archives
                            WHERE regexp_matches(filename, '^(\\d+)_')
                        )
                        SELECT s.station_code
                        FROM _stations_to_query s
                        LEFT JOIN seen_stations seen USING (station_code)
                        LEFT JOIN archived_stations arc USING (station_code)
                        WHERE seen.station_code IS NULL
                          AND arc.station_code IS NULL
                        ORDER BY s.station_code
                    """,
                    "retry-failed": """
                        WITH successful_stations AS (
                            SELECT DISTINCT CAST(station_code AS VARCHAR) AS station_code
                            FROM sensor_downloads
                            WHERE COALESCE(CAST(success AS INTEGER), 0) = 1
                        ),
                        failed_stations AS (
                            SELECT DISTINCT CAST(station_code AS VARCHAR) AS station_code
                            FROM sensor_downloads
                            WHERE COALESCE(CAST(success AS INTEGER), 0) = 0
                        )
                        SELECT s.station_code
                        FROM _stations_to_query s
                        INNER JOIN failed_stations failed USING (station_code)
                        LEFT JOIN successful_stations ok USING (station_code)
                        WHERE ok.station_code IS NULL
                        ORDER BY s.station_code
                    """,
                }
            )
        else:
            query_by_mode.update(
                {
                    "default": """
                        WITH archived_stations AS (
                            SELECT DISTINCT regexp_extract(filename, '^(\\d+)', 1) AS station_code
                            FROM raw_archives
                            WHERE regexp_matches(filename, '^(\\d+)_')
                        )
                        SELECT s.station_code
                        FROM _stations_to_query s
                        LEFT JOIN archived_stations arc USING (station_code)
                        WHERE arc.station_code IS NULL
                        ORDER BY s.station_code
                    """,
                    "missing-only": """
                        WITH archived_stations AS (
                            SELECT DISTINCT regexp_extract(filename, '^(\\d+)', 1) AS station_code
                            FROM raw_archives
                            WHERE regexp_matches(filename, '^(\\d+)_')
                        )
                        SELECT s.station_code
                        FROM _stations_to_query s
                        LEFT JOIN archived_stations arc USING (station_code)
                        WHERE arc.station_code IS NULL
                        ORDER BY s.station_code
                    """,
                    "retry-failed": """
                        SELECT station_code
                        FROM _stations_to_query
                        WHERE 1 = 0
                    """,
                }
            )
        pending = connection.execute(query_by_mode[fetch_mode]).fetchdf()
        connection.unregister("_stations_to_query")
    return pending["station_code"].astype(str).tolist()


def _load_station_category_history_from_db(
    root_dir: str,
    station_code: str,
) -> dict[tuple[str, str, str], pd.DataFrame]:
    if not table_exists(root_dir, SENSOR_DOWNLOADS_TABLE):
        return {}
    with _download_log_connection(root_dir) as connection:
        station_history = connection.execute(
            """
            SELECT *
            FROM sensor_downloads
            WHERE CAST(station_code AS VARCHAR) = ?
            """,
            [str(station_code)],
        ).fetchdf()
    if station_history.empty:
        return {}
    for column in DOWNLOAD_LOG_COLUMNS:
        if column not in station_history.columns:
            station_history[column] = pd.NA
    return _station_category_history(station_history.loc[:, DOWNLOAD_LOG_COLUMNS], str(station_code))


def _station_category_history(
    history: pd.DataFrame,
    station_code: str,
) -> dict[tuple[str, str, str], pd.DataFrame]:
    if history.empty:
        return {}

    station_history = history.loc[history["station_code"].astype(str) == str(station_code)].copy()
    if station_history.empty:
        return {}

    grouped_history: dict[tuple[str, str, str], pd.DataFrame] = {}
    for key, group in station_history.groupby(["result_tab", "station_type", "download_format"], dropna=False):
        grouped_history[_category_key(str(key[0]), _slugify_label(key[1]), str(key[2]))] = group.reset_index(drop=True)
    return grouped_history


def _has_successful_history(category_history: pd.DataFrame) -> bool:
    if category_history.empty:
        return False
    return bool((pd.to_numeric(category_history["success"], errors="coerce").fillna(0) == 1).any())


def _has_failed_history(category_history: pd.DataFrame) -> bool:
    if category_history.empty:
        return False
    return bool((pd.to_numeric(category_history["success"], errors="coerce").fillna(0) == 0).any())


def _should_attempt_category(
    fetch_mode: str,
    existing_keys: set[tuple[str, str, str]],
    key: tuple[str, str, str],
    category_history: pd.DataFrame | None,
) -> tuple[bool, str]:
    has_file = key in existing_keys
    history = category_history if category_history is not None else pd.DataFrame()
    has_success = _has_successful_history(history)
    has_failure = _has_failed_history(history)

    if fetch_mode == "redownload-all":
        return True, "forced_redownload"
    if fetch_mode == "retry-failed":
        if has_success:
            return False, "already_succeeded"
        if has_failure:
            return True, "retry_failed"
        return False, "no_failed_history"
    if fetch_mode == "missing-only":
        if has_file or has_success or has_failure:
            return False, "already_seen"
        return True, "missing_history"
    if fetch_mode == "default":
        if has_file or has_success:
            return False, "already_succeeded"
        return True, "needs_download"
    raise ValueError(f"Unsupported fetch mode: {fetch_mode}")


def _station_has_any_archives(raw_dir: Path, station_code: str) -> bool:
    pattern = f"{station_code}_*.zip"
    return any(path.is_file() for path in raw_dir.glob(pattern))


def _existing_category_keys(raw_dir: Path, station_code: str) -> set[tuple[str, str, str]]:
    """Infer already-downloaded categories from category-aware archive names."""
    existing_keys: set[tuple[str, str, str]] = set()
    pattern = re.compile(
        rf"^{re.escape(str(station_code))}_(?P<tab>[a-z0-9_]+)_(?P<category>[a-z0-9_]+)_mdb_",
        re.IGNORECASE,
    )
    for path in raw_dir.iterdir():
        if not path.is_file() or path.suffix.lower() != ".zip":
            continue
        match = pattern.match(path.name)
        if not match:
            continue
        existing_keys.add(
            _category_key(
                match.group("tab"),
                match.group("category"),
            )
        )
    return existing_keys


def _wait_for_series_page(driver):
    """Open the historical series page and wait until the search form appears."""
    logger.debug("Opening historical series page to establish cookies.")
    driver.get(SERIES_URL)
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.XPATH, "//input[@formcontrolname='codigoEstacao']"))
    )
    _decency_wait()


def _refresh_session(driver):
    """Reload the site to refresh authentication cookies after failures."""
    logger.debug("Refreshing Selenium-backed session cookies.")
    try:
        driver.refresh()
        _decency_wait(1.0, 2.0)
    except Exception:
        # If the browser got into a bad state, a clean navigation usually
        # restores the session faster than surfacing a transient failure.
        pass
    _wait_for_series_page(driver)


def _wait_for_download_completion(download_dir: Path, station_code: str, timeout_seconds: int = 120) -> Path:
    """Wait until Chrome finishes writing the requested station archive."""
    candidate_pattern = f"*{station_code}*.zip"
    existing_files = {path.name for path in download_dir.glob(candidate_pattern)}

    for partial in download_dir.glob(f"*{station_code}*.crdownload"):
        partial.unlink(missing_ok=True)

    deadline = pd.Timestamp.utcnow() + pd.Timedelta(seconds=timeout_seconds)
    while pd.Timestamp.utcnow() < deadline:
        partial_files = list(download_dir.glob(f"*{station_code}*.crdownload"))
        candidate_files = [
            path
            for path in download_dir.glob(candidate_pattern)
            if path.name not in existing_files
        ]
        if candidate_files and not partial_files:
            newest = max(candidate_files, key=lambda path: path.stat().st_mtime)
            if newest.stat().st_size > 0:
                return newest
        sleep(1)
    raise TimeoutError(f"Timed out waiting for download of station {station_code}.")


def _safe_click(driver, element):
    """Click an element, falling back to JavaScript when needed."""
    try:
        element.click()
    except Exception:
        driver.execute_script("arguments[0].click();", element)


def _active_tab_xpath() -> str:
    return "//mat-tab-body[contains(@class,'mat-tab-body-active')]"


def _normalise_station_code_text(value: str) -> str:
    return "".join(str(value).split())


def _activate_tab(driver, label_fragment: str) -> None:
    """Switch to a named result tab when it exists."""
    matching_tabs = driver.find_elements(
        By.XPATH,
        f"//div[@role='tab'][contains(., '{label_fragment}')]",
    )
    if not matching_tabs:
        logger.debug("Tab containing '%s' not found.", label_fragment)
        return
    _safe_click(driver, matching_tabs[0])
    _decency_wait(0.8, 1.6)


def _search_station(
    driver,
    station_code: str,
    station_name: str,
    root_dir=".",
    match_name: bool = True,
):
    """Enter station code and station name into the search form."""
    station_input = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable(
            (
                By.XPATH,
                "//input[@formcontrolname='codigoEstacao']",
            )
        )
    )
    station_input.clear()
    station_input.send_keys(station_code)

    station_name_input = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable(
            (
                By.XPATH,
                "//input[@formcontrolname='NomeEstacao']",
            )
        )
    )
    station_name_input.clear()
    if match_name:
        station_name_input.send_keys(station_name)
    _decency_wait(0.5, 1.2)

    search_button_selectors = [
        "//button[.//span[contains(normalize-space(.), 'Pesquisar')]]",
        "//button[.//mat-icon[normalize-space(.)='search']]",
        "//button[@type='submit']",
    ]
    for selector in search_button_selectors:
        buttons = driver.find_elements(By.XPATH, selector)
        if buttons:
            _safe_click(driver, buttons[0])
            break
    else:
        station_input.send_keys(Keys.ENTER)

    try:
        WebDriverWait(driver, 30).until(
            lambda current_driver: len(
                current_driver.find_elements(
                    By.XPATH,
                    f"{_active_tab_xpath()}//tr[contains(@class,'mat-row')]",
                )
            )
            > 0
        )
    except TimeoutException:
        _dump_debug_html(driver, root_dir=root_dir, label="search_results_missing", station_code=station_code)
        raise
    _decency_wait(1.0, 2.0)


def _store_downloaded_file(downloaded_path: Path, raw_dir: Path, station_code: str, source_label: str) -> Path:
    """Move a browser download into the raw archive directory using a unique name."""
    suffix = downloaded_path.suffix or ".zip"
    archive_path = raw_dir / _archive_storage_name(f"{station_code}_{source_label}")
    archive_path = archive_path.with_suffix(suffix)
    shutil.move(str(downloaded_path), str(archive_path))
    return archive_path


def _row_station_type(row) -> str | None:
    station_type_cells = row.find_elements(By.XPATH, ".//td[contains(@class,'cdk-column-tipoEstacao')]")
    if not station_type_cells:
        return None
    text = station_type_cells[0].text.strip()
    return text or None


def _download_record(
    run_id: str,
    attempted_at: str,
    fetch_mode: str,
    station_code: str,
    result_tab: str,
    station_type: str | None,
    status: str,
    success: int,
    attempts: int,
    archive_path: str | None = None,
    last_error: str | None = None,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "attempted_at": attempted_at,
        "fetch_mode": fetch_mode,
        "station_code": str(station_code),
        "result_tab": result_tab,
        "station_type": station_type,
        "download_format": "mdb",
        "status": status,
        "success": int(success),
        "attempts": int(attempts),
        "archive_path": archive_path,
        "last_error": last_error,
    }


def _merge_attempt_records(
    aggregated_records: dict[tuple[str, str, str], dict[str, object]],
    attempt_records: list[dict[str, object]],
) -> None:
    for record in attempt_records:
        key = _category_key(record["result_tab"], record["station_type"], record["download_format"])
        existing = aggregated_records.get(key)
        if existing is None:
            aggregated_records[key] = record.copy()
            continue

        existing["attempts"] = int(existing["attempts"]) + int(record["attempts"])
        existing["status"] = record["status"]
        existing["success"] = record["success"]
        if record["archive_path"]:
            existing["archive_path"] = record["archive_path"]
        if record["last_error"] is not None:
            existing["last_error"] = record["last_error"]
        elif record["success"]:
            existing["last_error"] = None


def _download_matching_rows_in_active_tab(
    run_id: str,
    attempted_at: str,
    station_code: str,
    driver,
    download_dir: Path,
    raw_dir: Path,
    result_tab: str,
    fetch_mode: str,
    station_history: dict[tuple[str, str, str], pd.DataFrame],
) -> list[dict[str, object]]:
    """Download every MDB exposed for matching rows in the currently active tab."""
    records: list[dict[str, object]] = []
    normalized_station_code = _normalise_station_code_text(station_code)
    existing_keys = _existing_category_keys(raw_dir, station_code)
    saw_matching_row = False

    while True:
        rows = driver.find_elements(
            By.XPATH,
            f"{_active_tab_xpath()}//tr[contains(@class,'mat-row')]",
        )

        matching_indices = []
        for index, row in enumerate(rows):
            code_cells = row.find_elements(By.XPATH, ".//td[contains(@class,'cdk-column-id')]")
            if not code_cells:
                continue
            code_text = _normalise_station_code_text(code_cells[0].text)
            if code_text == normalized_station_code:
                matching_indices.append(index)

        for row_index in matching_indices:
            rows = driver.find_elements(
                By.XPATH,
                f"{_active_tab_xpath()}//tr[contains(@class,'mat-row')]",
            )
            if row_index >= len(rows):
                continue
            row = rows[row_index]
            saw_matching_row = True
            station_type = _row_station_type(row)
            station_type_slug = _slugify_label(station_type)
            key = _category_key(result_tab, station_type_slug)
            category_history = station_history.get(key)
            should_attempt, skip_reason = _should_attempt_category(
                fetch_mode,
                existing_keys,
                key,
                category_history,
            )
            if not should_attempt:
                skip_success = int(
                    key in existing_keys
                    or _has_successful_history(category_history if category_history is not None else pd.DataFrame())
                )
                records.append(
                    _download_record(
                        run_id,
                        attempted_at,
                        fetch_mode,
                        station_code,
                        result_tab=result_tab,
                        station_type=station_type,
                        status=skip_reason,
                        success=skip_success,
                        attempts=0,
                    )
                )
                continue

            mdb_buttons = row.find_elements(By.XPATH, ".//td[contains(@class,'cdk-column-mdb')]//button")
            if not mdb_buttons:
                records.append(
                    _download_record(
                        run_id,
                        attempted_at,
                        fetch_mode,
                        station_code,
                        result_tab=result_tab,
                        station_type=station_type,
                        status="missing_mdb_button",
                        success=0,
                        attempts=1,
                        last_error="MDB button missing for matching result row.",
                    )
                )
                continue

            try:
                _safe_click(driver, mdb_buttons[0])
                downloaded_path = _wait_for_download_completion(download_dir, station_code)
                source_label = f"{_slugify_label(result_tab)}_{station_type_slug}"
                archive_path = _store_downloaded_file(
                    downloaded_path,
                    raw_dir,
                    str(station_code),
                    source_label,
                )
                if _is_valid_archive(archive_path):
                    records.append(
                        _download_record(
                            run_id,
                            attempted_at,
                            fetch_mode,
                            station_code,
                            result_tab=result_tab,
                            station_type=station_type,
                            status="downloaded",
                            success=1,
                            attempts=1,
                            archive_path=str(archive_path),
                        )
                    )
                    existing_keys.add(key)
                else:
                    archive_path.unlink(missing_ok=True)
                    records.append(
                        _download_record(
                            run_id,
                            attempted_at,
                            fetch_mode,
                            station_code,
                            result_tab=result_tab,
                            station_type=station_type,
                            status="empty_archive",
                            success=0,
                            attempts=1,
                            last_error="Website returned an empty archive.",
                        )
                    )
                    existing_keys.add(key)
            except Exception as exc:
                records.append(
                    _download_record(
                        run_id,
                        attempted_at,
                        fetch_mode,
                        station_code,
                        result_tab=result_tab,
                        station_type=station_type,
                        status="download_failed",
                        success=0,
                        attempts=1,
                        last_error=str(exc),
                    )
                )

        next_buttons = driver.find_elements(
            By.XPATH,
            f"{_active_tab_xpath()}//button[contains(@class,'mat-paginator-navigation-next')]",
        )
        if not next_buttons:
            break
        next_button = next_buttons[-1]
        if next_button.get_attribute("disabled") is not None:
            break
        previous_rows = rows
        _safe_click(driver, next_button)
        if previous_rows:
            WebDriverWait(driver, 20).until(EC.staleness_of(previous_rows[0]))
        _decency_wait(0.8, 1.6)

    if not saw_matching_row:
        records.append(
            _download_record(
                run_id,
                attempted_at,
                fetch_mode,
                station_code,
                result_tab=result_tab,
                station_type=None,
                status="station_not_found",
                success=0,
                attempts=1,
                last_error=f"Station {station_code} not found in {result_tab} results.",
            )
        )
    return records


def _download_conventional_archives(
    run_id: str,
    attempted_at: str,
    station_code,
    station_name: str,
    driver,
    download_dir: Path,
    raw_dir: Path,
    root_dir=".",
    fetch_mode: str = "default",
    station_history: dict[tuple[str, str, str], pd.DataFrame] | None = None,
    match_name: bool = True,
):
    """Download every MDB exposed for the searched station in the conventional table."""
    logger.debug(
        "Searching conventional data and downloading all MDB rows for station %s (%s).",
        station_code,
        station_name,
    )
    _activate_tab(driver, "Convencion")
    _search_station(
        driver,
        station_code,
        station_name,
        root_dir=root_dir,
        match_name=match_name,
    )
    return _download_matching_rows_in_active_tab(
        run_id=run_id,
        attempted_at=attempted_at,
        station_code=str(station_code),
        driver=driver,
        download_dir=Path(download_dir),
        raw_dir=Path(raw_dir),
        result_tab="conventional",
        fetch_mode=fetch_mode,
        station_history=station_history or {},
    )


def _is_valid_archive(archive_path: Path) -> bool:
    """Check whether the downloaded file is a non-empty ZIP archive.

    Some station exports complete successfully from the website but contain no
    MDB members. Those should be treated as skippable empty results rather than
    successful downloads.
    """
    if not archive_path.exists() or archive_path.stat().st_size <= 22:
        return False
    with archive_path.open("rb") as handle:
        signature = handle.read(4)
    if signature != b"PK\x03\x04":
        return False
    try:
        with zipfile.ZipFile(archive_path, "r") as archive_file:
            mdb_members = [
                member_name
                for member_name in archive_file.namelist()
                if member_name.lower().endswith(".mdb")
            ]
            return len(mdb_members) > 0
    except zipfile.BadZipFile:
        return False


def download_by_id(
    station_code,
    station_name: str,
    driver,
    download_dir,
    raw_dir,
    root_dir=".",
    fetch_mode: str = "default",
    run_id: str | None = None,
    browser_manager: ManagedBrowser | None = None,
):
    """Download all MDB categories exposed for one station, with retries."""
    if fetch_mode not in FETCH_MODES:
        raise ValueError(f"Unknown fetch mode: {fetch_mode}")

    run_id = run_id or uuid4().hex
    attempted_at = _current_timestamp()
    station_history = _load_station_category_history_from_db(root_dir, str(station_code))
    aggregated_records: dict[tuple[str, str, str], dict[str, object]] = {}
    for attempt in range(1, 6):
        try:
            logger.info("Downloading MDB archives for station %s (attempt %s/5).", station_code, attempt)
            attempt_records = _download_conventional_archives(
                run_id,
                attempted_at,
                station_code,
                station_name,
                driver,
                Path(download_dir),
                Path(raw_dir),
                root_dir=root_dir,
                fetch_mode=fetch_mode,
                station_history=station_history,
                match_name=(attempt == 1),
            )
            _merge_attempt_records(aggregated_records, attempt_records)
            if any(record["station_type"] is not None for record in aggregated_records.values()):
                aggregated_records.pop(_category_key("conventional", None), None)

            retryable_statuses = {"download_failed", "station_not_found"}
            retryable_records = [
                record for record in aggregated_records.values() if record["status"] in retryable_statuses
            ]
            if not retryable_records:
                successful_records = [record for record in aggregated_records.values() if record["success"] == 1]
                logger.info(
                    "Finished station %s with %s successful MDB category download(s).",
                    station_code,
                    len(successful_records),
                )
                return list(aggregated_records.values())
        except Exception as exc:
            session_lost = _browser_session_lost(exc)
            if not session_lost:
                _dump_debug_html(
                    driver,
                    root_dir=root_dir,
                    label=f"attempt_{attempt}_failure",
                    station_code=station_code,
                )
            logger.warning(
                "Download failed for station %s on attempt %s/5: %s",
                station_code,
                attempt,
                exc,
            )
            _decency_wait(1.5, 3.5)
            if session_lost and browser_manager is not None:
                logger.warning(
                    "Browser session was lost while processing station %s; restarting Selenium before retry.",
                    station_code,
                )
                driver = browser_manager.restart()
                _refresh_session(driver)
                continue
            if attempt in {2, 4}:
                _refresh_session(driver)
            elif attempt == 3:
                driver.refresh()
                _decency_wait(2.0, 4.0)
    if not aggregated_records:
        aggregated_records[_category_key("conventional", None)] = _download_record(
            run_id,
            attempted_at,
            fetch_mode,
            station_code,
            result_tab="conventional",
            station_type=None,
            status="download_failed",
            success=0,
            attempts=5,
            last_error=f"No downloadable archive found for station {station_code}.",
        )
    return list(aggregated_records.values())


def fetch_station_data(
    root_dir=".",
    download_dir=None,
    headless=False,
    keep_browser_on_error=False,
    single_station=None,
    fetch_mode: str = "default",
):
    """Download station archives after priming website cookies with Selenium."""
    if fetch_mode not in FETCH_MODES:
        raise ValueError(f"Unknown fetch mode: {fetch_mode}. Expected one of: {sorted(FETCH_MODES)}")

    _, raw_dir = ensure_water_quality_dirs(root_dir)
    browser_download_dir = raw_dir if download_dir is None else download_dir
    stations_to_query = load_queryable_stations(root_dir)
    if single_station is not None:
        single_station = str(single_station).strip()
        stations_to_query = stations_to_query.loc[
            stations_to_query.index.astype(str) == single_station
        ]
        if len(stations_to_query.index) == 0:
            raise KeyError(f"Unknown station code in stations_rivers: {single_station}")
    logger.info("Prepared %s station(s) for archive download.", len(stations_to_query))
    current_raw_archives = _current_raw_archives_frame(raw_dir)
    write_dataframe_table(
        root_dir,
        RAW_ARCHIVES_TABLE,
        current_raw_archives,
    )
    run_id = uuid4().hex
    download_records: list[dict[str, object]] = []
    download_record_buffer: list[dict[str, object]] = []
    run_started_at = monotonic()
    last_status_log_at = run_started_at
    pending_ids = _compute_pending_station_ids(root_dir, stations_to_query, fetch_mode)

    browser = ManagedBrowser(
        headless=headless,
        download_dir=str(browser_download_dir),
        keep_open_on_error=keep_browser_on_error,
    )
    with browser as driver:
        _refresh_session(driver)
        _dump_debug_html(driver, root_dir=root_dir, label="initial_series_page")

        logger.info(
            "Starting download loop for %s pending station(s) after applying fetch mode '%s'.",
            len(pending_ids),
            fetch_mode,
        )
        if pending_ids:
            _log_status_window(
                download_records,
                completed_stations=0,
                pending_stations=len(pending_ids),
                total_prepared_stations=len(stations_to_query),
                run_started_at=run_started_at,
            )
        for idx, station_id in enumerate(tqdm(pending_ids)):
            station_name = stations_to_query.loc[station_id, "station_name"]
            if idx > 0 and idx % RESTART_BROWSER_EVERY_N_STATIONS == 0:
                logger.info(
                    "Restarting Selenium browser after %s station(s) to keep the session fresh.",
                    idx,
                )
                driver = browser.restart()
                _refresh_session(driver)
            station_records = download_by_id(
                station_id,
                station_name,
                driver,
                browser_download_dir,
                raw_dir,
                root_dir=root_dir,
                fetch_mode=fetch_mode,
                run_id=run_id,
                browser_manager=browser,
            )
            download_records.extend(station_records)
            download_record_buffer.extend(station_records)
            # The query table is checkpointed every few stations so interrupted
            # scraping sessions can resume without starting over.
            if idx % 10 == 0 and download_record_buffer:
                append_dataframe_table(
                    root_dir,
                    SENSOR_DOWNLOADS_TABLE,
                    pd.DataFrame(download_record_buffer, columns=DOWNLOAD_LOG_COLUMNS),
                )
                download_record_buffer = []
            completed_stations = idx + 1
            current_time = monotonic()
            if (
                completed_stations % STATUS_LOG_EVERY_STATIONS == 0
                or (current_time - last_status_log_at) >= STATUS_LOG_EVERY_SECONDS
                or completed_stations == len(pending_ids)
            ):
                _log_status_window(
                    download_records,
                    completed_stations=completed_stations,
                    pending_stations=len(pending_ids),
                    total_prepared_stations=len(stations_to_query),
                    run_started_at=run_started_at,
                    last_station_id=str(station_id),
                )
                last_status_log_at = current_time
            _decency_wait()

    if download_record_buffer:
        append_dataframe_table(
            root_dir,
            SENSOR_DOWNLOADS_TABLE,
            pd.DataFrame(download_record_buffer, columns=DOWNLOAD_LOG_COLUMNS),
        )
    write_dataframe_table(
        root_dir,
        RAW_ARCHIVES_TABLE,
        _current_raw_archives_frame(raw_dir),
    )
    return pd.DataFrame(download_records, columns=DOWNLOAD_LOG_COLUMNS)
