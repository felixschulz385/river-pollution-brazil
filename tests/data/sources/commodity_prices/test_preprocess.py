from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.sources.commodity_prices.constants import PINK_SHEET_SHEET
from src.data.sources.commodity_prices.preprocess import gold_price_tables, monthly_exchange_rate, read_pink_sheet_gold


def test_read_pink_sheet_gold_finds_the_column_below_the_title_block(tmp_path):
    rows = [
        ["World Bank Commodity Price Data", None, None],
        [None, None, None],
        [None, "Crude oil, average", "Gold"],
        [None, "($/bbl)", "($/troy oz)"],
        ["1960M01", 1.63, 35.27],
        ["2024M12", 73.8, 2648.0],
        ["2025M01", 79.0, ".."],
    ]
    path = tmp_path / "pink.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame([["Description"]]).to_excel(writer, sheet_name="AFOSHEET", header=False, index=False)
        pd.DataFrame(rows).to_excel(writer, sheet_name=PINK_SHEET_SHEET, header=False, index=False)

    gold = read_pink_sheet_gold(path)

    assert gold.values.tolist() == [[1960, 1, 35.27], [2024, 12, 2648.0]]


def test_gold_price_tables_convert_to_brl_and_keep_complete_years():
    gold = pd.DataFrame({"year": [2020] * 12 + [2021], "month": [*range(1, 13), 1], "gold_usd_oz": [100.0] * 13})
    daily = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-02", "2020-01-03", *(f"2020-{m:02d}-15" for m in range(2, 13)), "2021-01-04"]),
        "brl_per_usd": [4.0, 6.0, *[5.0] * 11, 5.5],
    })

    rates = monthly_exchange_rate(daily)
    monthly, annual = gold_price_tables(gold, rates)

    assert rates.loc[(rates.year == 2020) & (rates.month == 1), "brl_per_usd"].item() == 5.0
    assert monthly["gold_brl_oz"].iloc[-1] == pytest.approx(550.0)
    assert monthly["log_gold_brl"].iloc[0] == pytest.approx(np.log(500.0))
    assert annual["year"].tolist() == [2020]
    assert annual.loc[0, ["gold_brl_oz", "months"]].tolist() == [500.0, 12]
