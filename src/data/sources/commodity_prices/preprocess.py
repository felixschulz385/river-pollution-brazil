import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import (
    ANNUAL_COLUMNS,
    DEFAULT_ANNUAL_OUTPUT_PATH,
    DEFAULT_MONTHLY_OUTPUT_PATH,
    MONTH_COLUMN,
    MONTHLY_COLUMNS,
    PINK_SHEET_GOLD_COLUMN,
    PINK_SHEET_SHEET,
    YEAR_COLUMN,
    bcb_path,
    pink_sheet_path,
)


logger = logging.getLogger(__name__)


def read_pink_sheet_gold(path):
    """Monthly gold price (USD/oz) from the Pink Sheet: `year`, `month`, `gold_usd_oz`.

    The sheet has a title block, then commodity names on one row and units on
    the next, then one row per month labelled `YYYYMmm`.
    """
    sheet = pd.read_excel(path, sheet_name=PINK_SHEET_SHEET, header=None)
    names = sheet.apply(lambda row: row.astype(str).str.strip().eq(PINK_SHEET_GOLD_COLUMN).any(), axis=1)
    if not names.any():
        raise ValueError(f"No '{PINK_SHEET_GOLD_COLUMN}' column in {path}.")
    header_row = names.idxmax()
    column = sheet.columns[sheet.loc[header_row].astype(str).str.strip().eq(PINK_SHEET_GOLD_COLUMN)][0]
    labels = sheet[0].astype(str).str.extract(r"^(\d{4})M(\d{2})$")
    rows = labels.notna().all(axis=1)
    return pd.DataFrame({
        YEAR_COLUMN: labels.loc[rows, 0].astype(int).to_numpy(),
        MONTH_COLUMN: labels.loc[rows, 1].astype(int).to_numpy(),
        "gold_usd_oz": pd.to_numeric(sheet.loc[rows, column], errors="coerce").to_numpy(),
    }).dropna()


def monthly_exchange_rate(daily):
    """Mean BRL per USD per calendar month of the daily rates."""
    daily = daily.assign(**{YEAR_COLUMN: daily["date"].dt.year, MONTH_COLUMN: daily["date"].dt.month})
    return daily.groupby([YEAR_COLUMN, MONTH_COLUMN], as_index=False)["brl_per_usd"].mean()


def gold_price_tables(gold, rates):
    """Monthly gold prices in USD and BRL with logs, and their annual means over complete months.

    A year's annual value is the mean of its monthly prices; `months` counts
    the months with both series, and years with fewer than 12 are dropped.
    """
    monthly = gold.merge(rates, on=[YEAR_COLUMN, MONTH_COLUMN], how="inner", validate="one_to_one")
    monthly["gold_brl_oz"] = monthly["gold_usd_oz"] * monthly["brl_per_usd"]
    monthly["log_gold_usd"] = np.log(monthly["gold_usd_oz"])
    monthly["log_gold_brl"] = np.log(monthly["gold_brl_oz"])
    monthly = monthly.sort_values([YEAR_COLUMN, MONTH_COLUMN]).reset_index(drop=True)[list(MONTHLY_COLUMNS)]

    annual = monthly.groupby(YEAR_COLUMN).agg(
        gold_usd_oz=("gold_usd_oz", "mean"),
        brl_per_usd=("brl_per_usd", "mean"),
        gold_brl_oz=("gold_brl_oz", "mean"),
        months=(MONTH_COLUMN, "size"),
    ).reset_index()
    annual = annual[annual["months"] == 12].copy()
    annual["log_gold_usd"] = np.log(annual["gold_usd_oz"])
    annual["log_gold_brl"] = np.log(annual["gold_brl_oz"])
    return monthly, annual[list(ANNUAL_COLUMNS)].reset_index(drop=True)


def build_gold_price_tables(root_dir=".", monthly_output_path=None, annual_output_path=None):
    """Write the monthly and annual gold price tables."""
    monthly, annual = gold_price_tables(
        read_pink_sheet_gold(pink_sheet_path(root_dir)),
        monthly_exchange_rate(pd.read_parquet(bcb_path(root_dir))),
    )
    for frame, output_path in (
        (monthly, monthly_output_path or DEFAULT_MONTHLY_OUTPUT_PATH),
        (annual, annual_output_path or DEFAULT_ANNUAL_OUTPUT_PATH),
    ):
        resolved = Path(root_dir) / output_path
        resolved.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(resolved, index=False)
        logger.info("Wrote %s (shape=%s)", resolved, frame.shape)
    return monthly, annual
