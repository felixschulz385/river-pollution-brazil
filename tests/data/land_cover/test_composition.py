from __future__ import annotations

import math

import pandas as pd
import pytest

from src.data.land_cover.composition import compute_kernel_weighted_composition
from src.data.land_cover.constants import LAND_COVER_ALR_CLASSES, LAND_COVER_LEAF_CLASSES


def _rows_for_bucket(station_code, year, bucket, *, forest, nonforest_nat, pasture, agriculture, urban, mining, other_raw, water):
    """Build raw MapBiomas-coded rows for one (station, year, bucket)."""
    c3 = pasture + agriculture + 0.02  # leaves 0.02 of unattributed farming mass
    c4 = urban + mining + other_raw + 0.04  # leaves 0.04 of unattributed urban-parent mass
    return [
        {"station_code": station_code, "year": year, "bucket": bucket, "land_cover_class": 1, "share": forest},
        {"station_code": station_code, "year": year, "bucket": bucket, "land_cover_class": 2, "share": nonforest_nat},
        {"station_code": station_code, "year": year, "bucket": bucket, "land_cover_class": 3, "share": c3},
        {"station_code": station_code, "year": year, "bucket": bucket, "land_cover_class": 30, "share": pasture},
        {"station_code": station_code, "year": year, "bucket": bucket, "land_cover_class": 31, "share": agriculture},
        {"station_code": station_code, "year": year, "bucket": bucket, "land_cover_class": 4, "share": c4},
        {"station_code": station_code, "year": year, "bucket": bucket, "land_cover_class": 40, "share": urban},
        {"station_code": station_code, "year": year, "bucket": bucket, "land_cover_class": 41, "share": mining},
        {"station_code": station_code, "year": year, "bucket": bucket, "land_cover_class": 42, "share": other_raw},
        {"station_code": station_code, "year": year, "bucket": bucket, "land_cover_class": 5, "share": water},
    ]


def test_compute_kernel_weighted_composition_matches_hand_derived_weights():
    bucket_0 = _rows_for_bucket(
        "S1", 2020, 0,
        forest=0.4, nonforest_nat=0.1, pasture=0.12, agriculture=0.05,
        urban=0.05, mining=0.03, other_raw=0.02, water=0.1,
    )
    bucket_25 = _rows_for_bucket(
        "S1", 2020, 25,
        forest=0.5, nonforest_nat=0.2, pasture=0.06, agriculture=0.02,
        urban=0.03, mining=0.02, other_raw=0.01, water=0.1,
    )
    bucket_df = pd.DataFrame(bucket_0 + bucket_25)

    result = compute_kernel_weighted_composition(
        bucket_df, entity_columns=["station_code", "year"]
    )

    assert result.shape[0] == 1
    row = result.iloc[0]
    assert row["station_code"] == "S1"
    assert row["year"] == 2020

    # Independently reproduce the inverse-sqrt-distance kernel weighting and
    # c3/c4 mismatch resolution the SQL transform performs, then compare.
    def resolved_leaf_shares(forest, nonforest_nat, pasture, agriculture, urban, mining, other_raw, water):
        c3 = pasture + agriculture + 0.02
        c4 = urban + mining + other_raw + 0.04
        farming_unclassified = max(c3 - pasture - agriculture, 0.0)
        other = other_raw + max(c4 - urban - mining - other_raw, 0.0)
        raw_shares = {
            "forest": forest,
            "nonforest_nat": nonforest_nat,
            "pasture": pasture,
            "agriculture": agriculture,
            "farming_unclassified": farming_unclassified,
            "urban": urban,
            "mining": mining,
            "other": other,
            "water": water,
        }
        leaf_total = sum(raw_shares.values())
        return {leaf_class: share / leaf_total for leaf_class, share in raw_shares.items()}

    shares_0 = resolved_leaf_shares(0.4, 0.1, 0.12, 0.05, 0.05, 0.03, 0.02, 0.1)
    shares_25 = resolved_leaf_shares(0.5, 0.2, 0.06, 0.02, 0.03, 0.02, 0.01, 0.1)
    assert math.isclose(sum(shares_0.values()), 1.0)
    assert math.isclose(sum(shares_25.values()), 1.0)

    raw_weight_0 = 1.0 / math.sqrt(12.5)
    raw_weight_25 = 1.0 / math.sqrt(37.5)
    weight_0 = raw_weight_0 / (raw_weight_0 + raw_weight_25)
    weight_25 = raw_weight_25 / (raw_weight_0 + raw_weight_25)

    expected_lc = {
        leaf_class: shares_0[leaf_class] * weight_0 + shares_25[leaf_class] * weight_25
        for leaf_class in LAND_COVER_LEAF_CLASSES
    }

    for leaf_class in LAND_COVER_LEAF_CLASSES:
        assert row[f"lc_{leaf_class}"] == pytest.approx(expected_lc[leaf_class], abs=1e-9)

    expected_lc_nat = expected_lc["forest"] + expected_lc["nonforest_nat"]
    assert row["lc_nat"] == pytest.approx(expected_lc_nat, abs=1e-9)

    pseudocount = 1e-4
    for alr_class in LAND_COVER_ALR_CLASSES:
        expected_alr = math.log(
            (expected_lc[alr_class] + pseudocount) / (expected_lc_nat + pseudocount)
        )
        assert row[f"alr_{alr_class}"] == pytest.approx(expected_alr, abs=1e-9)


def test_compute_kernel_weighted_composition_supports_adm2_entity_columns():
    bucket_df = pd.DataFrame(
        _rows_for_bucket(
            "1234567", 2019, 0,
            forest=0.5, nonforest_nat=0.1, pasture=0.1, agriculture=0.1,
            urban=0.05, mining=0.05, other_raw=0.02, water=0.08,
        )
    )
    bucket_df = bucket_df.rename(columns={"station_code": "mun_id"})

    result = compute_kernel_weighted_composition(bucket_df, entity_columns=["mun_id", "year"])

    assert list(result[["mun_id", "year"]].itertuples(index=False, name=None)) == [
        ("1234567", 2019)
    ]
    # Single bucket -> renormalized within the leaf-class total: forest=0.5 out of a
    # 1.06 leaf total (farming_unclassified=0.02, other=0.06 resolved from c3/c4).
    assert result.loc[0, "lc_forest"] == pytest.approx(0.5 / 1.06, abs=1e-9)
