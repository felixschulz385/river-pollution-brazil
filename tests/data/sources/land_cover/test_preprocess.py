from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.sources.land_cover.constants import LAND_COVER_TOTAL_COLUMN
from src.data.sources.land_cover.preprocess import _accumulate_mapped_counts, process_year


def test_accumulate_mapped_counts_excludes_unmapped_nodata_from_total():
    """Unmapped raster codes (e.g. MapBiomas class 0 not flagged as raster
    nodata) must not inflate TOTAL beyond the sum of the mapped class
    columns."""
    values = np.array([1, 3, 0])
    counts = np.array([2, 5, 100])

    class_mapper = lambda arr: np.array(  # noqa: E731
        [{1: 1.0, 3: 3.0}.get(v, np.nan) for v in arr]
    )
    subclass_mapper = lambda arr: np.array(  # noqa: E731
        [{1: 1.0, 3: 3.0}.get(v, np.nan) for v in arr]
    )

    column_positions = {
        LAND_COVER_TOTAL_COLUMN: 0,
        "land_cover_class_1": 1,
        "land_cover_class_3": 2,
    }
    row_data = np.zeros(3, dtype=np.int64)

    _accumulate_mapped_counts(
        row_data,
        values,
        counts,
        (class_mapper, subclass_mapper),
        column_positions,
    )

    assert row_data[column_positions[LAND_COVER_TOTAL_COLUMN]] == 7
    assert row_data[column_positions["land_cover_class_1"]] == 2
    assert row_data[column_positions["land_cover_class_3"]] == 5


def test_process_year_raises_instead_of_returning_all_zero_year(tmp_path: Path):
    """A raster that fails to open must abort the year, not silently fall
    through to an all-zero output frame indistinguishable from genuine
    zero-overlap coverage."""
    missing_raster = tmp_path / "brazil_coverage_1999.tif"

    with pytest.raises(Exception):
        process_year(
            1999,
            missing_raster,
            polygon_items=[(1, None)],
            output_columns=[LAND_COVER_TOTAL_COLUMN],
            legend_mappers=(),
        )
