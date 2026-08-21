from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.sources.health.preprocess.preprocess import _coerce_tabnet_numeric, _read_datasus_csv


def test_coerce_tabnet_numeric_treats_bare_dash_as_missing():
    series = pd.Series(["-", "10", "5,3"])

    result = _coerce_tabnet_numeric(series)

    assert result.tolist() == [0.0, 10.0, 5.3]


def test_coerce_tabnet_numeric_preserves_negative_values():
    series = pd.Series(["-5,3", "-12"])

    result = _coerce_tabnet_numeric(series)

    assert result.tolist() == [-5.3, -12.0]


def test_read_datasus_csv_drops_all_trailing_total_rows(tmp_path: Path):
    csv_path = tmp_path / "export.csv"
    csv_path.write_text(
        '"Município";"Valor"\n'
        '"110001 Alta Floresta D\'Oeste";"10"\n'
        '"110002 Ariquemes";"20"\n'
        '"Total";"15"\n'
        '"Total";"30"\n',
        encoding="latin1",
    )

    frame = _read_datasus_csv(str(csv_path))

    assert "Total" not in frame.iloc[:, 0].tolist()
    assert len(frame) == 2
