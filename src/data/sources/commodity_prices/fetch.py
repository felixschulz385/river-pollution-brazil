import logging
import re
import time

import pandas as pd
import requests

from src.data.shared.batches import atomic_write_bytes

from .constants import (
    BCB_ATTEMPTS,
    BCB_FIRST_DATE,
    BCB_RETRY_SECONDS,
    BCB_SGS_URL,
    BCB_WINDOW_YEARS,
    PINK_SHEET_FALLBACK_URL,
    PINK_SHEET_LINK_PATTERN,
    PINK_SHEET_PAGE_URL,
    bcb_path,
    pink_sheet_path,
)


logger = logging.getLogger(__name__)


def pink_sheet_url():
    """The current monthly Pink Sheet link from the World Bank page, or the fallback release."""
    try:
        response = requests.get(PINK_SHEET_PAGE_URL, timeout=120)
        response.raise_for_status()
        match = re.search(PINK_SHEET_LINK_PATTERN, response.text)
        if match:
            return match.group(0)
    except requests.RequestException as error:
        logger.warning("Could not read %s (%s); using the fallback release.", PINK_SHEET_PAGE_URL, error)
    return PINK_SHEET_FALLBACK_URL


def fetch_pink_sheet(root_dir="."):
    """Download the monthly Pink Sheet workbook (replaced on every run: it is revised monthly)."""
    url = pink_sheet_url()
    logger.info("Downloading %s", url)
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    if not response.content.startswith(b"PK"):
        raise RuntimeError(f"{url} did not return an xlsx file.")
    destination = pink_sheet_path(root_dir)
    atomic_write_bytes(str(destination), response.content)
    return destination


def bcb_windows(first_date, last_date, years=BCB_WINDOW_YEARS):
    """Consecutive (start, end) date windows of at most `years` years covering [first_date, last_date]."""
    start, last = pd.Timestamp(first_date), pd.Timestamp(last_date)
    windows = []
    while start <= last:
        end = min(start + pd.DateOffset(years=years) - pd.Timedelta(days=1), last)
        windows.append((start, end))
        start = end + pd.Timedelta(days=1)
    return windows


def _bcb_window(start, end, attempts=BCB_ATTEMPTS):
    """One window of SGS series 1; the API intermittently answers with an HTML error page, so retry."""
    for attempt in range(1, attempts + 1):
        response = requests.get(
            BCB_SGS_URL,
            params={"formato": "json", "dataInicial": f"{start:%d/%m/%Y}", "dataFinal": f"{end:%d/%m/%Y}"},
            timeout=120,
        )
        if response.ok and "json" in response.headers.get("Content-Type", ""):
            return response.json()
        logger.warning("BCB SGS gave no JSON for %s to %s (attempt %d of %d).", f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}",
                       attempt, attempts)
        time.sleep(BCB_RETRY_SECONDS * attempt)
    raise RuntimeError(f"BCB SGS returned no JSON for {start:%Y-%m-%d} to {end:%Y-%m-%d}.")


def fetch_bcb_exchange_rate(root_dir=".", last_date=None):
    """Download SGS series 1 (BRL per USD, daily) from `BCB_FIRST_DATE` to `last_date` (default yesterday).

    The API answers with an HTML error page, not JSON, when the window ends on
    a day without a published rate yet (e.g. today), hence yesterday.
    """
    last_date = last_date or pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    frames = [pd.DataFrame(_bcb_window(start, end)) for start, end in bcb_windows(BCB_FIRST_DATE, last_date)]
    rates = pd.concat(frames, ignore_index=True).drop_duplicates("data")
    rates = pd.DataFrame({
        "date": pd.to_datetime(rates["data"], format="%d/%m/%Y"),
        "brl_per_usd": pd.to_numeric(rates["valor"]),
    }).sort_values("date")
    destination = bcb_path(root_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rates.to_parquet(destination, index=False)
    logger.info("BCB SGS 1: %d daily rates, %s to %s", len(rates), rates.date.min().date(), rates.date.max().date())
    return destination


def fetch_commodity_prices(root_dir="."):
    return {"pink_sheet": fetch_pink_sheet(root_dir), "bcb": fetch_bcb_exchange_rate(root_dir)}
