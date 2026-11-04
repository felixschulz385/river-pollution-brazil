from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from src.data.sources.enforcement.constants import PRIORITY_COLUMNS, priority_path
from src.data.sources.enforcement.preprocess import build_priority_list, fold_name, priority_list_adm2
from src.data.verification.sources import _enforcement_priority_checks


GADM = [
    ("Buritis", "1100452"),  # also a municipality in MG
    ("Buritis", "3109303"),
    ("Rio Branco", "1200401"),
    ("Rorainópolis", "1400472"),
    ("Machadinho D'Oeste", "1100130"),
    ("São Félix do Xingu", "1507300"),
]
COLUMNS = ["biome", "uf", "name_muni", "code_muni", "year", "priority", "monitored", "portaria"]


def _row(uf, name, code, year, priority=1, monitored=0, biome="Amazônia"):
    return [biome, uf, name, code, year, priority, monitored, "Portaria MMA"]


def _table():
    """The record's errors: a name-only join that duplicates Buritis RO with MG's code and gives Rio Branco AC a
    code in MT, Rorainópolis under RO, a curly apostrophe, a municipality that moves to the monitored list, and
    a Cerrado row."""
    rows = [
        _row("RO", "Buritis", "1100452", 2017), _row("RO", "Buritis", "3109303", 2017),
        _row("RO", "Buritis", "1100452", 2018), _row("RO", "Buritis", "3109303", 2018),
        _row("AC", "Rio Branco", "5107206", 2018),
        _row("RO", "Rorainópolis", "1400472", 2017),
        _row("RO", "Machadinho D’oeste", "1100130", 2016),
        _row("PA", "São Félix do Xingu", "1507300", 2016),
        _row("PA", "São Félix do Xingu", "1507300", 2017, priority=0, monitored=1),
        _row("MG", "Buritis", "3109303", 2018, biome="Cerrado"),
    ]
    return pd.DataFrame(rows, columns=COLUMNS).astype({"code_muni": str})


def _municipalities():
    keys = [code[:2] + "|" + fold_name(name) for name, code in GADM]
    return pd.Series([code[:6] for _, code in GADM], index=keys)


def test_fold_name_unifies_accents_case_apostrophes_and_hyphens():
    assert fold_name("Machadinho D’oeste") == fold_name("Machadinho D'Oeste") == "machadinho d'oeste"
    assert fold_name("São  Félix-do Xingu ") == "sao felix do xingu"


def test_rows_are_recoded_from_state_and_name_on_a_complete_grid():
    result = priority_list_adm2(_table(), _municipalities())
    assert list(result.columns) == list(PRIORITY_COLUMNS)
    assert sorted(result["mun_id"].unique()) == ["110013", "110045", "120040", "140047", "150730"]
    assert len(result) == 5 * 3  # every listed municipality in 2016-2018
    flags = result.set_index(["mun_id", "year"])
    assert flags.loc[("110045", 2017), "priority_list"] == 1.0  # Buritis RO, once despite the duplicate
    assert flags.loc[("110045", 2016), "priority_list"] == 0.0
    assert flags.loc[("120040", 2018), "priority_list"] == 1.0  # Rio Branco AC, not the MT code
    assert flags.loc[("140047", 2017), "priority_list"] == 1.0  # Rorainópolis, in RR
    assert flags.loc[("150730", 2017), ["priority_list", "priority_monitored"]].tolist() == [0.0, 1.0]
    assert _enforcement_priority_checks(result) and all(check.ok for check in _enforcement_priority_checks(result))


def test_unmatched_names_and_both_lists_raise():
    unknown = pd.concat([_table(), pd.DataFrame([_row("PA", "Atlântida", "1500000", 2018)], columns=COLUMNS)])
    with pytest.raises(ValueError, match="Atlântida"):
        priority_list_adm2(unknown, _municipalities())
    both = pd.concat([_table(), pd.DataFrame([_row("PA", "São Félix do Xingu", "1507300", 2017)], columns=COLUMNS)])
    with pytest.raises(ValueError, match="both lists"):
        priority_list_adm2(both, _municipalities())


def test_build_reads_the_download_and_gadm(tmp_path):
    path = priority_path(tmp_path)
    path.parent.mkdir(parents=True)
    _table().to_csv(path, index=False)
    gadm_path = tmp_path / "gadm.gpkg"
    gpd.GeoDataFrame({"NAME_2": [name for name, _ in GADM], "CC_2": [code for _, code in GADM]},
                     geometry=[box(i, 0, i + 1, 1) for i in range(len(GADM))], crs=4326).to_file(
        gadm_path, layer="ADM_ADM_2", driver="GPKG")
    result = build_priority_list(root_dir=tmp_path, gadm_path=gadm_path)
    written = pd.read_parquet(tmp_path / "data" / "enforcement" / "processed" / "priority_list_adm2.parquet")
    pd.testing.assert_frame_equal(written, result)
    assert result["priority_list"].sum() == 6


def test_verification_flags_a_municipality_on_both_lists():
    result = priority_list_adm2(_table(), _municipalities())
    result.loc[0, ["priority_list", "priority_monitored"]] = 1.0
    failed = {check.name for check in _enforcement_priority_checks(result) if not check.ok}
    assert failed == {"flags_exclusive"}
