from __future__ import annotations

import pandas as pd
import pytest

from src.data.sources.sensor_data.fetch.database import (
    append_dataframe_table,
    read_dataframe_table,
)


def test_append_dataframe_table_retypes_all_null_column_on_type_conflict(tmp_path):
    """A column DuckDB first saw as all-NULL is inferred as INTEGER; a later
    batch bringing real string values for it must not blow up on the cast --
    the column carried no data, so it gets retyped to match."""
    root_dir = str(tmp_path)

    # First batch: `note` is entirely NULL, so DuckDB creates it as INTEGER.
    append_dataframe_table(
        root_dir,
        "widgets",
        pd.DataFrame([{"id": 1, "note": None}, {"id": 2, "note": None}]),
    )

    # Second batch: a genuine string for the same column.
    append_dataframe_table(
        root_dir,
        "widgets",
        pd.DataFrame([{"id": 3, "note": "first real note"}]),
    )

    frame = read_dataframe_table(root_dir, "widgets").sort_values("id").reset_index(drop=True)
    assert frame["note"].tolist()[:2] == [None, None] or frame["note"].isna().tolist()[:2] == [True, True]
    assert frame.loc[frame["id"] == 3, "note"].item() == "first real note"


def test_append_dataframe_table_leaves_populated_column_conflict_to_fail(tmp_path):
    """The retype only applies to columns that have only ever held NULLs -- a
    real type clash on a populated column is a genuine error and must still
    surface rather than be silently widened."""
    root_dir = str(tmp_path)

    append_dataframe_table(
        root_dir,
        "widgets",
        pd.DataFrame([{"id": 1, "count": 10}, {"id": 2, "count": 20}]),
    )

    with pytest.raises(Exception):
        append_dataframe_table(
            root_dir,
            "widgets",
            pd.DataFrame([{"id": 3, "count": "not a number"}]),
        )
