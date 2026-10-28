from __future__ import annotations

import zipfile

import pandas as pd
import pytest

from src.data.sources.mining.constants import ADM2_COLUMNS, scm_archive_path
from src.data.sources.mining.preprocess import build_adm2_permits
from src.data.verification.sources import _mining_permit_checks


def _write_archive(path):
    """Four processes in two municipalities: a gold PLG request later granted, a tin request,
    a gold PLG without an expiry date, and a research permit (not a PLG)."""
    tables = {
        "Processo.txt": [
            "DSProcesso;NRProcesso;IDTipoRequerimento;DTProtocolo;QTAreaHA",
            "850.001/2012;1;2;2012-05-02 00:00:00;50,00",  # gold, in two municipalities
            "850.002/2013;2;24;2013-08-01 00:00:00;-10,00",  # tin, change of regime to PLG, bad area
            "850.003/2008;3;9;2008-01-15 00:00:00;100,00",  # gold PLG title, not a PLG request type
            "850.004/2013;4;1;2013-03-01 00:00:00;1000,00",  # research permit
        ],
        "ProcessoTitulo.txt": [
            "DSProcesso;NRTitulo;IDTipoDocumentoLegal;DTPublicacao;DTVencimento",
            "850.001/2012;10;2;2014-02-01;2019-02-01",
            "850.001/2012;11;2;2019-03-01;2024-03-01",  # renewal: extends the expiry, not a new grant
            "850.003/2008;12;2;2010-06-01;",  # no expiry: grant + 5 years
            "850.004/2013;13;1;2014-01-01;2017-01-01",  # research permit title
        ],
        "ProcessoMunicipio.txt": [
            "DSProcesso;IDMunicipio",
            "850.001/2012;1500107",
            "850.001/2012;1500206",
            "850.002/2013;1100015",
            "850.003/2008;1500107",
            "850.004/2013;1500107",
        ],
        "ProcessoSubstancia.txt": [
            "DSProcesso;IDSubstancia;IDTipoUsoSubstancia",
            "850.001/2012;104901;99",
            "850.002/2013;101104;99",
            "850.003/2008;104900;99",
            "850.003/2008;100307;99",  # orpiment: not gold
            "850.004/2013;104901;99",
        ],
        "Municipio.txt": ["IDMunicipio;NMMunicipio;SGUF", "1500107;A;PA", "1500206;B;PA", "1100015;C;RO", "1500305;D;PA"],
    }
    with zipfile.ZipFile(path, "w") as bundle:
        for name, lines in tables.items():
            bundle.writestr(name, ("\n".join(lines) + "\n").encode("cp1252"))


@pytest.fixture
def permits(tmp_path):
    snapshot = tmp_path / "data" / "mining" / "raw" / "anm_scm_microdados_2026-10-26"
    snapshot.mkdir(parents=True)
    _write_archive(snapshot / "microdados-scm.zip")
    result = build_adm2_permits(root_dir=str(tmp_path))
    return tmp_path, result.set_index(["mun_id", "year"])


def test_requests_count_plg_request_types_and_split_areas(permits):
    _, result = permits

    assert result.loc[("150010", 2012), "plg_requests_gold"] == 1
    assert result.loc[("150020", 2012), "plg_requests_gold"] == 1  # counted in each municipality
    assert result.loc[("150010", 2012), "plg_request_area_ha_gold"] == pytest.approx(25.0)  # area split
    assert result.loc[("110001", 2013), "plg_requests_tin"] == 1
    assert result.loc[("110001", 2013), "plg_request_area_ha_tin"] == 0.0  # negative area dropped
    assert result.loc[("150010", 2008), "plg_requests_all"] == 0  # not a PLG request type
    assert result.loc[("150010", 2013), "plg_requests_all"] == 0  # research permit


def test_grants_use_the_earliest_title_and_active_spans_run_to_the_latest_expiry(permits):
    _, result = permits

    assert result.loc[("150010", 2014), "plg_grants_gold"] == 1
    assert result.loc[("150010", 2019), "plg_grants_gold"] == 0  # renewal
    active = result.loc["150020", "plg_active_gold"]
    assert active.loc[2013] == 0 and active.loc[2014] == 1 and active.loc[2024] == 1 and active.loc[2025] == 0
    # No expiry: granted 2010, in force 2010-2015; the orpiment substance changes nothing.
    gold = result.loc["150010", "plg_active_gold"]
    assert gold.loc[2010] == 1 and gold.loc[2015] == 2 and gold.loc[2016] == 1
    assert result.loc[("150010", 2015), "plg_active_area_ha_gold"] == pytest.approx(125.0)
    assert result.loc[("150010", 2016), "plg_active_all"] == 1  # research permit titles never count


def test_output_is_a_full_zero_filled_grid_that_passes_verification(permits):
    root, result = permits

    assert list(result.reset_index().columns) == list(ADM2_COLUMNS)
    assert len(result) == 4 * len(range(1985, 2027))
    assert result.loc["150030"].sum().sum() == 0  # municipality without processes
    assert scm_archive_path(root).name == "microdados-scm.zip"
    assert all(check.ok for check in _mining_permit_checks(result.reset_index()))


def test_verification_flags_groups_exceeding_all():
    frame = pd.DataFrame({column: [0, 0] for column in ADM2_COLUMNS})
    frame["mun_id"], frame["year"] = ["150010", "150010"], [2012, 2013]
    frame["plg_requests_gold"] = [1, 0]

    failed = {check.name for check in _mining_permit_checks(frame) if not check.ok}

    assert failed == {"groups_within_all"}
