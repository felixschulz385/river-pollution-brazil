"""Diagnose why verification's `sensor_data` check reports missing columns.

`data verify --source sensor_data` (and `data summary`) flag
`sensor_data.parquet` as failed with:

    required_columns is missing required columns: ['datetime', 'station_code']

`src/data/verification/sources.py::_sensor_data_check_outputs` reads the
parquet with a plain `pd.read_parquet()` and looks for `station_code`/
`datetime` as *columns*. `assembly.py` writes the file with
`.set_index([STATION_CODE_COLUMN, DATE_COLUMN]).to_parquet(..., index=True)`,
so `station_code` becomes an index level, not a column -- explaining that
half of the error. `datetime` is *not* part of that index in the current
code (only used as a sort key), so it should still be a real column; if it's
also missing, that's either a stale file from an older assembly run or a
genuine gap worth a closer look.

This script reads the actual on-disk file and reports exactly what's there,
so the two possibilities can be told apart without guessing.

Usage:
    python scripts/diagnose_sensor_data_columns.py [--root-dir .] [--path PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pyarrow.parquet as pq

from src.data.sources.sensor_data.constants import get_processed_dir
from src.data.sources.sensor_data.schema import ASSEMBLED_SENSOR_DATA_PARQUET


def diagnose(path: Path) -> None:
    print(f"Path: {path}")
    if not path.exists():
        print("  Does NOT exist -- nothing to diagnose.")
        return

    stat = path.stat()
    print(f"  Size: {stat.st_size:,} bytes")
    print(f"  Modified: {pd.Timestamp(stat.st_mtime, unit='s')}")

    schema = pq.read_schema(path)
    print(f"\nParquet schema ({len(schema.names)} fields):")
    for name in schema.names:
        print(f"  - {name}")
    index_cols = schema.pandas_metadata.get("index_columns", []) if schema.pandas_metadata else []
    print(f"\nPandas index columns recorded in parquet metadata: {index_cols}")

    frame = pd.read_parquet(path)
    print(f"\npd.read_parquet() result: shape={frame.shape}")
    print(f"  Index names: {list(frame.index.names)}")
    print(f"  Columns: {list(frame.columns)}")

    for target in ("station_code", "datetime"):
        in_columns = target in frame.columns
        in_index = target in (frame.index.names or [])
        if in_columns:
            state = "present as a COLUMN (verification's check should pass on this one)"
        elif in_index:
            state = "present as an INDEX LEVEL, not a column -- this is why the check fails"
        else:
            state = "MISSING entirely -- not a column or an index level"
        print(f"\n'{target}': {state}")

    print("\nFirst 3 rows (index reset for readability):")
    print(frame.reset_index().head(3).to_string())

    reset = frame.reset_index()
    still_missing = [c for c in ("station_code", "datetime") if c not in reset.columns]
    print("\nVerdict:")
    if not still_missing:
        print(
            "  Both columns exist once the index is reset. The verification check "
            "should reset_index() before checking columns (or the assembly step "
            "should write with index=False) -- this is a verification-side bug, "
            "not a data problem."
        )
    else:
        print(
            f"  {still_missing} still missing even after reset_index() -- this file "
            "is genuinely missing data (likely produced by an older/incomplete "
            "assembly run). Re-running `data preprocess --source sensor_data` "
            "should regenerate it correctly."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default=".")
    parser.add_argument(
        "--path",
        default=None,
        help="Override the parquet path (default: <root-dir>'s sensor_data aggregate output).",
    )
    args = parser.parse_args()

    path = Path(args.path) if args.path else get_processed_dir(args.root_dir, stage="aggregate") / ASSEMBLED_SENSOR_DATA_PARQUET
    diagnose(path)


if __name__ == "__main__":
    main()
