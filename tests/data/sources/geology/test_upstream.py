from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from src.data.sources.geology.constants import OTTO_GROUPS, UPSTREAM_COLUMNS, otto_column, upstream_count_column
from src.data.sources.geology.upstream import (
    assign_trench,
    load_occurrence_groups,
    otto_exposure,
    pfafstetter_upstream,
    reach_counts,
    trench_counts,
)
from src.data.verification.sources import _geology_upstream_checks


@pytest.mark.parametrize(
    "code_a, code_b, expected",
    [
        ("4587", "4583", True),  # headwater interbasin above a main-stem interbasin
        ("4584", "4583", True),  # tributary entering above interbasin 3
        ("4582", "4583", False),  # tributary entering below it
        ("4589", "4584", False),  # nothing in the parent drains into a tributary
        ("4587", "4683", False),  # different level-3 basin
        ("4583", "4583", False),
        ("458", "4583", False),
    ],
)
def test_pfafstetter_upstream_follows_the_digit_rule(code_a, code_b, expected):
    assert pfafstetter_upstream(code_a, code_b) is expected


def test_otto_exposure_sets_upstream_and_downstream_binaries():
    exposure = otto_exposure(
        {"a": {"4583"}, "b": {"4589"}, "c": {"4684", "4583"}},
        {"gold": {"4587"}, "tin": {"4581"}},
    )

    assert exposure.loc["a", otto_column("gold", "upstream")] == 1
    assert exposure.loc["a", otto_column("gold", "downstream")] == 0
    assert exposure.loc["b", otto_column("gold", "upstream")] == 0
    assert exposure.loc["b", otto_column("gold", "downstream")] == 1
    assert exposure.loc["a", otto_column("tin", "downstream")] == 1
    assert exposure.loc["a", otto_column("tin", "upstream")] == 0
    assert exposure.loc["c", otto_column("gold", "upstream")] == 1
    assert exposure.loc["a", otto_column("gold_unexploited", "upstream")] == 0
    assert set(exposure.columns) == {otto_column(g, d) for g in OTTO_GROUPS for d in ("upstream", "downstream")}


def test_reach_counts_skip_own_trenches_and_cap_the_reach():
    counts = pd.DataFrame(
        {"gold": [1, 2, 4, 8], "gold_unexploited": [0, 1, 0, 1], "gold_garimpo": 0, "gold_mine": 0,
         "gold_alluvial": 0, "tin": [0, 0, 0, 3]},
        index=pd.Index([10, 11, 12, 13], name="trench_id"),
    )
    # Trench 10 is the municipality's own body (negative distance); 11 is listed twice.
    membership = pd.DataFrame({"trench_id": [10, 11, 11, 12, 13, 99], "adjusted_distance": [-5.0, 30.0, 70.0, 120.0, 500.0, 1.0]})

    row = reach_counts(membership, counts)

    assert row[upstream_count_column("gold", "50")] == 2
    assert row[upstream_count_column("gold", "200")] == 6
    assert row[upstream_count_column("gold", "all")] == 14
    assert row[upstream_count_column("gold_unexploited", "all")] == 2
    assert row[upstream_count_column("tin", "200")] == 0
    assert row[upstream_count_column("tin", "all")] == 3


def test_occurrence_groups_trench_assignment_and_counts(tmp_path):
    gpkg = tmp_path / "sgb.gpkg"
    gpd.GeoDataFrame(
        {
            "SUBSTANCIAS": ["Ouro", "Ouro, Prata", "Estanho", "Cobre", "Ouro"],
            "STATUS_ECONOMICO": ["Garimpo", "Não explotado", "Garimpo", "Mina", "Mina"],
            "ROCHAS_HOSPEDEIRAS": ["Alúvio", None, None, None, None],
            "geometry": [Point(-55.5, -5.5), Point(-55.4, -5.5), Point(-54.5, -5.5), Point(-54.4, -5.5),
                         Point(-50.0, -5.5)],
        },
        crs=4326,
    ).to_file(gpkg, layer="occurrences", driver="GPKG")
    drainage = gpd.GeoDataFrame(
        {"trench_id": [1, 2], "geometry": [box(-56, -6, -55, -5), box(-55, -6, -54, -5)]}, crs=4326
    )

    occurrences = load_occurrence_groups(gpkg)
    trench_ids = assign_trench(occurrences, drainage)
    counts = trench_counts(occurrences, trench_ids)

    assert len(occurrences) == 4  # copper is in no group
    assert occurrences["gold_alluvial"].tolist() == [True, False, False, False]
    assert trench_ids.tolist()[:3] == [1, 1, 2] and pd.isna(trench_ids.iloc[3])  # far point dropped
    assert counts.loc[1, ["gold", "gold_garimpo", "gold_unexploited", "tin"]].tolist() == [2, 1, 1, 0]
    assert counts.loc[2, "tin"] == 1


def test_geology_upstream_checks_flag_shrinking_counts_and_non_binary_exposure():
    frame = pd.DataFrame({column: [0, 0] for column in UPSTREAM_COLUMNS})
    frame["mun_id"] = ["150010", "150020"]
    frame[upstream_count_column("gold", "all")] = [3, 1]
    assert all(check.ok for check in _geology_upstream_checks(frame))

    frame[upstream_count_column("gold", "50")] = [5, 0]
    frame[otto_column("tin", "upstream")] = [2, 0]
    failed = {check.name for check in _geology_upstream_checks(frame) if not check.ok}
    assert failed == {"counts_grow_with_reach", "otto_binary"}
