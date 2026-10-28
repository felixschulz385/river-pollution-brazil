from __future__ import annotations

import pandas as pd

from src.data.sources.commodity_prices import fetch as fetch_module


def test_bcb_windows_cover_the_range_in_ten_year_pieces():
    windows = fetch_module.bcb_windows("1994-07-01", "2026-10-27")

    assert windows[0] == (pd.Timestamp("1994-07-01"), pd.Timestamp("2004-06-30"))
    assert windows[1][0] == pd.Timestamp("2004-07-01")
    assert windows[-1] == (pd.Timestamp("2024-07-01"), pd.Timestamp("2026-10-27"))
    assert all(end + pd.Timedelta(days=1) == nxt for (_, end), (nxt, _) in zip(windows, windows[1:]))


class _Response:
    def __init__(self, ok_json):
        self.ok = True
        self.headers = {"Content-Type": "application/json" if ok_json else "text/html"}

    def json(self):
        return [{"data": "02/01/2020", "valor": "4.0213"}]


def test_bcb_window_retries_html_error_pages(monkeypatch):
    responses = iter([_Response(False), _Response(True)])
    monkeypatch.setattr(fetch_module.requests, "get", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(fetch_module.time, "sleep", lambda seconds: None)

    rows = fetch_module._bcb_window(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-31"))

    assert rows == [{"data": "02/01/2020", "valor": "4.0213"}]
