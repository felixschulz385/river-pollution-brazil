from __future__ import annotations

import pandas as pd
import pytest

from src.data.shared.sensor_upstream import explode_list_matches, prepare_trench_adm2_matches


def test_explode_list_matches_raises_clear_error_on_mismatched_list_lengths():
    frame = pd.DataFrame(
        {
            "trench_id": [1, 2],
            "adm2_list": [["A", "B"], ["C"]],
            "intersection_lengths": [[10.0, 20.0], [5.0, 6.0]],
        }
    )

    with pytest.raises(ValueError, match="Mismatched list lengths"):
        explode_list_matches(
            frame,
            id_columns=["trench_id"],
            values_column="adm2_list",
            value_name="adm2",
            weights_column="intersection_lengths",
            weight_name="weight",
        )


class _RnModule:
    SYSTEM_ID_KEY = "system_id"
    ADM2_COLUMN = "adm2"


class _FakeNetwork:
    def __init__(self):
        self.trenches = pd.DataFrame(
            {
                "trench_id": [101, 102],
                "system_id": [1, 1],
                "adm2": [None, None],
            }
        )
        self.trench_adm2_table = pd.DataFrame(
            {
                "trench_id": [101, 102],
                "adm2": ["A", "B"],
            }
        )


def test_prepare_trench_adm2_matches_prefers_persisted_relation_table():
    network = _FakeNetwork()

    matches = prepare_trench_adm2_matches(
        network,
        rn_module=_RnModule,
        trench_id_column="trench_id",
    )

    assert matches[["trench_id", "adm2"]].to_dict("records") == [
        {"trench_id": 101, "adm2": "A"},
        {"trench_id": 102, "adm2": "B"},
    ]


def test_prepare_trench_adm2_matches_falls_back_to_trench_adm2_column():
    network = _FakeNetwork()
    network.trench_adm2_table = None
    network.trenches["adm2"] = ["A", "B"]

    matches = prepare_trench_adm2_matches(
        network,
        rn_module=_RnModule,
        trench_id_column="trench_id",
    )

    assert sorted(matches["adm2"].tolist()) == ["A", "B"]
