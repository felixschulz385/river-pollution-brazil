from __future__ import annotations

import pandas as pd
import pytest

from src.data.sources.sensor_data.fetch.data import access_reader


class _FakeCursor:
    def tables(self, tableType=None):
        return [type("Row", (), {"table_name": "Readings"})()]


class _FakeConnection:
    def __init__(self):
        self.closed = False

    def cursor(self):
        return _FakeCursor()

    def close(self):
        self.closed = True


def test_normalize_object_columns_coerces_plain_decimal_comma():
    frame = pd.DataFrame({"value": ["12,5", "3,25"]})

    result = access_reader.normalize_object_columns(frame)

    assert result["value"].tolist() == [12.5, 3.25]


def test_normalize_object_columns_coerces_pt_br_thousands_and_decimal():
    # "1.234,56" is unambiguously pt-BR formatted: "." groups thousands,
    # "," is the decimal separator -> 1234.56, not 1.23456.
    frame = pd.DataFrame({"value": ["1.234,56", "2.000,00"]})

    result = access_reader.normalize_object_columns(frame)

    assert result["value"].tolist() == [1234.56, 2000.00]


def test_load_mdb_tables_closes_connection_even_on_success(monkeypatch: pytest.MonkeyPatch):
    connection = _FakeConnection()
    monkeypatch.setattr(access_reader, "connect_access_database", lambda mdb_path: connection)
    monkeypatch.setattr(
        access_reader,
        "read_access_table",
        lambda conn, table: pd.DataFrame({"value": [1]}),
    )

    access_reader.load_mdb_tables(
        "station.mdb",
        archive_name="archive.zip",
        station_code="S1",
        table_map={},
        column_map={},
    )

    assert connection.closed is True


def test_load_mdb_tables_closes_connection_when_reading_raises(monkeypatch: pytest.MonkeyPatch):
    connection = _FakeConnection()
    monkeypatch.setattr(access_reader, "connect_access_database", lambda mdb_path: connection)

    def _raise(conn, table):
        raise RuntimeError("boom")

    monkeypatch.setattr(access_reader, "read_access_table", _raise)

    with pytest.raises(RuntimeError):
        access_reader.load_mdb_tables(
            "station.mdb",
            archive_name="archive.zip",
            station_code="S1",
            table_map={},
            column_map={},
        )

    assert connection.closed is True
