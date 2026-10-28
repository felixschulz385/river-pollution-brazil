from __future__ import annotations

from pathlib import Path

from src.data.shared.paths import processed_dir, raw_dir as _shared_raw_dir, source_root


# World Bank Commodity Price Data (the Pink Sheet), monthly nominal USD. The
# file's URL changes with each monthly release, so it is read from the
# commodity-markets page; the fallback is the release current when this
# source was written.
PINK_SHEET_PAGE_URL = "https://www.worldbank.org/en/research/commodity-markets"
PINK_SHEET_LINK_PATTERN = r"https://[^\"']*CMO-Historical-Data-Monthly\.xlsx"
PINK_SHEET_FALLBACK_URL = (
    "https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/"
    "related/CMO-Historical-Data-Monthly.xlsx"
)
PINK_SHEET_FILENAME = "cmo_historical_data_monthly.xlsx"
PINK_SHEET_SHEET = "Monthly Prices"
PINK_SHEET_GOLD_COLUMN = "Gold"  # $/troy oz

# Banco Central do Brasil SGS series 1: BRL per USD, daily (PTAX sell rate).
# The API returns at most ten years per request, so the series is fetched in
# windows from the start of the Real.
BCB_SGS_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.1/dados"
BCB_FIRST_DATE = "1994-07-01"
BCB_WINDOW_YEARS = 10
BCB_ATTEMPTS = 5
BCB_RETRY_SECONDS = 10
BCB_FILENAME = "bcb_sgs_1_brl_usd_daily.parquet"

YEAR_COLUMN = "year"
MONTH_COLUMN = "month"
MONTHLY_COLUMNS = (
    YEAR_COLUMN, MONTH_COLUMN, "gold_usd_oz", "brl_per_usd", "gold_brl_oz", "log_gold_usd", "log_gold_brl",
)
ANNUAL_COLUMNS = (YEAR_COLUMN, "gold_usd_oz", "brl_per_usd", "gold_brl_oz", "log_gold_usd", "log_gold_brl", "months")

DEFAULT_MONTHLY_OUTPUT_PATH = str(processed_dir(".", "commodity_prices") / "gold_price_monthly.parquet")
DEFAULT_ANNUAL_OUTPUT_PATH = str(processed_dir(".", "commodity_prices") / "gold_price_annual.parquet")


def commodity_prices_dir(root_dir: str | Path = ".") -> Path:
    return source_root(root_dir, "commodity_prices")


def raw_dir(root_dir: str | Path = ".") -> Path:
    return _shared_raw_dir(root_dir, "commodity_prices")


def pink_sheet_path(root_dir: str | Path = ".") -> Path:
    return raw_dir(root_dir) / PINK_SHEET_FILENAME


def bcb_path(root_dir: str | Path = ".") -> Path:
    return raw_dir(root_dir) / BCB_FILENAME
