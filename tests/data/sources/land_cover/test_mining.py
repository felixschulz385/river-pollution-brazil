from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.data.sources.land_cover.mining import (
    build_mining_legend,
    compute_mining_composition,
    mining_column_names,
    mining_group,
    pixel_area_km2,
)
from src.data.sources.land_cover.preprocess import _accumulate_mapped_counts, create_mappers
from src.data.sources.land_cover.schema import get_output_columns


@pytest.mark.parametrize(
    ("code", "group"),
    [(1115, 1), (1215, 1), (1159, 2), (1258, 2), (1114, 3), (2115, 4), (2259, 5), (2114, 5)],
)
def test_mining_group(code, group):
    assert mining_group(code) == group


def test_legend_extracts_groups_phases_and_full_area(tmp_path):
    legend_path = tmp_path / "legend.xlsx"
    build_mining_legend([1115, 1215, 1159, 2114]).to_excel(legend_path, index=False)
    columns = get_output_columns(legend_path)
    positions = {column: index for index, column in enumerate(columns)}
    row = np.zeros(len(columns), dtype=np.int64)

    values = np.array([0, 1115, 1215, 1159, 2114])
    counts = np.array([90, 4, 3, 2, 1])
    _accumulate_mapped_counts(row, values, counts, create_mappers(legend_path), positions)
    result = dict(zip(columns, row))

    assert result["land_cover_total"] == 100
    assert result["land_cover_class_0"] == 90
    assert result["land_cover_class_1"] == 7
    assert result["land_cover_class_11"] == 4
    assert result["land_cover_class_12"] == 3
    assert result["land_cover_class_21"] == 2
    assert result["land_cover_class_5"] == 1


def test_compute_mining_composition_weights_rings_by_kernel():
    def rows(bucket, gold, total):
        return [
            {"mun_id": "1", "year": 2020, "bucket": bucket, "land_cover_class": -1, "share": total},
            {"mun_id": "1", "year": 2020, "bucket": bucket, "land_cover_class": 1, "share": gold},
            {"mun_id": "1", "year": 2020, "bucket": bucket, "land_cover_class": 11, "share": gold},
        ]

    bucket_map = {0: ("0_25km", 12.5), 25: ("25_50km", 37.5)}
    buckets = pd.DataFrame(rows(0, 0.02, 1.0) + rows(25, 0.0, 1.0))
    result = compute_mining_composition(buckets, entity_columns=["mun_id", "year"], bucket_map=bucket_map)

    w0, w25 = 1 / math.sqrt(12.5), 1 / math.sqrt(37.5)
    assert result.loc[0, "mining_garimpo_gold"] == pytest.approx(0.02 * w0 / (w0 + w25))
    assert result.loc[0, "mining_garimpo_gold_active"] == pytest.approx(0.02 * w0 / (w0 + w25))
    assert result.loc[0, "mining_industrial_gold"] == 0.0
    assert set(mining_column_names().values()) <= set(result.columns)


def test_compute_mining_composition_skips_rings_without_area():
    buckets = pd.DataFrame(
        [
            {"mun_id": "1", "year": 2020, "bucket": 0, "land_cover_class": 1, "share": 0.05},
            {"mun_id": "1", "year": 2020, "bucket": 25, "land_cover_class": 1, "share": np.nan},
        ]
    )
    result = compute_mining_composition(buckets, entity_columns=["mun_id", "year"])
    assert result.loc[0, "mining_garimpo_gold"] == pytest.approx(0.05)


def test_pixel_area_sums_to_sphere():
    resolution = 1.0
    rows = pixel_area_km2(90.0, resolution, 180)
    assert rows.sum() * 360 == pytest.approx(4 * math.pi * 6371.0072**2)
