from __future__ import annotations

import csv
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from src.data.sources.enforcement.constants import ADM2_COLUMNS, RECORD_COLUMNS, snapshot_path
from src.data.sources.enforcement.preprocess import build_enforcement
from src.data.verification.sources import _enforcement_adm2_checks, _enforcement_record_checks


FINE_HEADER = [
    "SEQ_AUTO_INFRACAO", "NUM_AUTO_INFRACAO", "SER_AUTO_INFRACAO", "SIT_CANCELADO", "DAT_HORA_AUTO_INFRACAO",
    "COD_MUNICIPIO", "UF", "NUM_LONGITUDE_AUTO", "NUM_LATITUDE_AUTO", "VAL_AUTO_INFRACAO", "OPERACAO",
    "OPERACAO_SOL_RECURSO", "CD_ACAO_FISCALIZATORIA", "SEQ_ACAO_FISCALIZATORIA", "DES_AUTO_INFRACAO",
    "DES_INFRACAO", "NOME_INFRATOR",
]
EMBARGO_HEADER = [
    "SEQ_TAD", "NUM_TAD", "SER_TAD", "SIT_CANCELADO", "DAT_EMBARGO", "COD_MUNICIPIO", "UF", "NUM_LONGITUDE_TAD",
    "NUM_LATITUDE_TAD", "QTD_AREA_EMBARGADA", "DAT_DESEMBARGO", "OPERACAO", "OPERACAO_SOL_RECURSO",
    "CD_ACAO_FISCALIZATORIA", "SEQ_ACAO_FISCALIZATORIA", "SEQ_AUTO_INFRACAO", "NUM_AUTO_INFRACAO", "DES_TAD",
]
SEIZURE_HEADER = [
    "SEQ_TAD", "CD_TAD", "CD_SERIE_TAD", "ST_CANCELADO", "DT_APREENSAO", "CD_MUNICIPIO", "SG_UF",
    "NU_LONGITUDE_TAD", "NU_LATITUDE_TAD", "VL_TAD", "NO_OPERACAO", "NO_OPERACAO_SOL_RECURSO",
    "CD_ACAO_FISCALIZATORIA", "SEQ_ACAO_FISCALIZATORIA", "SEQ_AUTO_INFRACAO", "CD_AUTO_INFRACAO", "DS_TAD",
    "DS_COMPLEMENTAR", "DS_AUTO_INFRACAO",
]
DESTRUCTION_HEADER = [
    "SEQ_TAD", "NUM_TAD", "SER_TAD", "SIT_CANCELADO", "DAT_TAD", "COD_MUNICIPIO", "SIG_UF", "NUM_LONGITUDE_TAD",
    "NUM_LATITUDE_TAD", "VALOR_TAD", "OPERACAO_SOL_RECURSO", "CD_ACAO_FISCALIZATORIA", "SEQ_ACAO_FISCALIZATORIA",
    "SEQ_AUTO_INFRACAO", "NUM_AUTO_INFRACAO", "DES_TAD", "DS_INFORMACOES_COMPLEMENTARES", "DES_JUSTIFICATIVA",
]


def _csv(header, rows, quoting=csv.QUOTE_MINIMAL):
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=";", quoting=quoting, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue()


def _zip(path, members):
    with zipfile.ZipFile(path, "w") as bundle:
        for name, text in members.items():
            bundle.writestr(name, text.encode("utf-8"))


def _write_snapshot(snapshot):
    """Fines in two year files, an embargo before its coverage and one with an impossible date, seizures
    and destruction terms whose items are linked by system ID or by printed code, and cancelled records."""
    _zip(snapshot / "auto_infracao__auto_infracao_csv.zip", {
        "auto_infracao_2012.csv": _csv(FINE_HEADER, [
            ["101", "A1", "D", "N", "2012-03-01 10:00:00", "1500107", "PA", "-55,5", "-6,25", "1000,50", "", "", "",
             "", "Extrair ouro sem licença\nem garimpo", "", "Fulano"],
            ["102", "A2", "D", "S", "2012-04-01", "1500107", "PA", "", "", "500,00", "", "", "", "",
             "Garimpo ilegal", "", "Fulano"],
            ["103", "A3", "D", "N", "2012-05-01", "1500107", "PA", "-6,25", "-55,5", "200,00", "", "", "", "",
             "Desmatar 10 ha na Fazenda Ouro Verde e na Fazenda Estrela de Ouro",
             "Importar lâmpadas de vapor de mercúrio", "Fulano"],
        ]),
        "auto_infracao_2023.csv": _csv(FINE_HEADER, [
            ["", "XK12AB34", "", "N", "2023-06-01 09:00:00", "1400100", "RR", "0", "0", "", "OP X", "", "OPC1",
             "74601.0", "Fazer funcionar atividade poluidora",
             "Executar pesquisa, lavra ou extração de minerais sem autorização", "Fulano"],
        ]),
        "auto_infracao_2026.csv": _csv(FINE_HEADER, [
            ["104", "A4", "D", "N", "2026-03-01", "1400100", "RR", "", "", "", "", "", "", "", "Garimpo", "", ""],
        ]),
    })
    (snapshot / "embargo__termo_de_embargo.csv").write_text(_csv(EMBARGO_HEADER, [
        ["201", "E1", "C", "N", "2012-03-02 00:00:00", "1500107", "PA", "-55.5", "-6.25", "24,99",
         "2014-01-01 00:00:00", "", "", "", "", "101.0000000000", "A1",
         "Ficam embargadas as atividades de extração minerária exploradas por garimpeiros"],
        ["202", "E2", "C", "N", "1999-05-01 00:00:00", "1500107", "PA", "", "", "10,0", "", "", "", "", "", "", "",
         "Desmatamento"],
        ["203", "E3", "C", "N", "2925-09-25 00:00:00", "1500107", "PA", "", "", "", "", "", "", "", "", "", "",
         "Desmatamento"],
    ], quoting=csv.QUOTE_ALL), encoding="utf-8-sig")
    _zip(snapshot / "apreensao__apreensao_csv.zip", {"apreensao.csv": _csv(SEIZURE_HEADER, [
        ["301", "S1", "c", "N", "2023-07-01", "1400100", "RR", "", "", "", "XAPIRI", "", "OPC1", "", "", "",
         "Apreensão de equipamentos", "", ""],
        ["", "S2XYZ", "", "N", "2023-08-01", "1400100", "RR", "", "", "", "", "", "", "", "", "", "Apreensão", "", ""],
        ["303", "S3", "C", "N", "2004-05-01", "1500107", "PA", "", "", "", "", "", "", "", "", "", "Madeira", "", ""],
    ])})
    _zip(snapshot / "apreensao__bem_csv.zip", {"bem.csv": _csv(
        ["SEQ_TAD", "CD_TAD", "CD_SERIE_TAD", "DS_TIPO_BEM", "DS_BEM_APREENDIDO"],
        [["301", "S1", "C", "Balsa de garimpo", "balsa"], ["", "s2xyz", "", "Motor Estacionário", "motor"],
         ["", "s2xyz", "", "Bomba", "bomba"]],
    )})
    (snapshot / "destruicao__termo_destruicao_inutilizacao.csv").write_text(_csv(DESTRUCTION_HEADER, [
        ["401", "D1", "E", "N", "2023-06-02 10:00:00", "1400100", "RR", "-63.1", "2.5", "1000.0", "GCDA", "OPC1", "",
         "", "", "Destruição", "", "Local de difícil acesso"],
        ["402", "D2", "", "S", "2023-06-03 10:00:00", "1400100", "RR", "", "", "", "", "", "", "", "", "Garimpo", "",
         ""],
    ]), encoding="utf-8-sig")
    (snapshot / "destruicao__termo_destruicao_bem_destruido.csv").write_text(_csv(
        ["SEQ_TAD", "NUM_TAD", "SER_TAD", "DES_TIPO_BEM", "DES_BEM_APREENDIDO"],
        [["", "D1", "E", "Draga Garimpeira", "draga"]],
    ), encoding="utf-8-sig")


@pytest.fixture
def enforcement(tmp_path):
    snapshot = tmp_path / "data" / "enforcement" / "raw" / "ibama_2026-10-29"
    snapshot.mkdir(parents=True)
    _write_snapshot(snapshot)
    records, adm2 = build_enforcement(root_dir=str(tmp_path))
    return tmp_path, records.set_index(["action", "record_id"]), adm2.set_index(["mun_id", "year"])


def test_records_drop_cancellations_and_classify_text_and_items(enforcement):
    _, records, _ = enforcement

    assert sorted(records.loc["fines"].index) == ["code:XK12AB34-", "seq:101", "seq:103", "seq:104"]  # 102 cancelled
    assert records.loc[("fines", "seq:101"), ["mining", "garimpo"]].tolist() == [True, True]
    assert not records.loc[("fines", "seq:103"), "mining"]  # farm names and mercury lamps
    assert records.loc[("fines", "code:XK12AB34-"), ["mining", "garimpo"]].tolist() == [True, False]
    assert records.loc[("seizures", "seq:301"), "garimpo"]  # through its item, "Balsa de garimpo"
    assert records.loc[("seizures", "code:S2XYZ-"), ["items", "mining"]].tolist() == [2, False]  # linked by code
    assert records.loc[("destructions", "seq:401"), ["items", "garimpo"]].tolist() == [1, True]
    assert ("destructions", "seq:402") not in records.index


def test_records_clean_values_ids_dates_and_coordinates(enforcement):
    _, records, _ = enforcement

    fine = records.loc[("fines", "seq:101")]
    assert (fine["mun_id"], fine["year"], fine["value_brl"]) == ("150010", 2012, pytest.approx(1000.5))
    assert (fine["lon"], fine["lat"]) == (-55.5, -6.25)
    assert (records.loc[("fines", "seq:103"), "lon"], records.loc[("fines", "seq:103"), "lat"]) == (-55.5, -6.25)
    assert np.isnan(records.loc[("fines", "code:XK12AB34-"), "lon"])  # 0/0
    assert records.loc[("fines", "code:XK12AB34-"), ["operation", "operation_seq"]].tolist() == ["OP X", "74601"]
    embargo = records.loc[("embargoes", "seq:201")]
    assert (embargo["fine_seq"], embargo["area_ha"], embargo["garimpo"]) == ("101", pytest.approx(24.99), True)
    assert embargo["lifted_date"] == pd.Timestamp("2014-01-01")
    assert pd.isna(records.loc[("embargoes", "seq:203"), "date"])  # year 2925
    assert list(records.reset_index().columns) == list(RECORD_COLUMNS)
    assert "Fulano" not in records.astype(str).to_numpy().ravel()  # personal data dropped


def test_adm2_counts_by_class_and_leaves_years_before_coverage_missing(enforcement):
    _, _, adm2 = enforcement

    row = adm2.loc[("150010", 2012)]
    assert row[["ibama_fines_all", "ibama_fines_mining", "ibama_fines_garimpo"]].tolist() == [2, 1, 1]
    assert row["ibama_fine_value_brl_all"] == pytest.approx(1200.5)
    assert row[["ibama_embargoes_garimpo", "ibama_embargo_area_ha_garimpo"]].tolist() == [1, pytest.approx(24.99)]
    assert row["ibama_seizures_all"] == 0 and np.isnan(row["ibama_destructions_all"])  # destruction from 2014
    assert np.isnan(adm2.loc[("150010", 1999), "ibama_embargoes_all"])  # embargoes from 2000
    assert adm2.loc[("150010", 1999), "ibama_fines_all"] == 0
    assert np.isnan(adm2.loc[("150010", 2004), "ibama_seizures_all"])  # seizures from 2005
    row = adm2.loc[("140010", 2023)]
    assert row[["ibama_seizures_all", "ibama_seizures_garimpo", "ibama_destructions_garimpo"]].tolist() == [2, 1, 1]
    assert row[["ibama_fines_mining", "ibama_fines_garimpo"]].tolist() == [1, 0]
    assert adm2.index.get_level_values("year").max() == 2023  # 2026, the snapshot's partial year, left out


def test_outputs_pass_verification(enforcement):
    root, records, adm2 = enforcement

    assert snapshot_path(root).name == "ibama_2026-10-29"
    assert list(adm2.reset_index().columns) == list(ADM2_COLUMNS)
    assert all(check.ok for check in _enforcement_record_checks(records.reset_index()))
    assert all(check.ok for check in _enforcement_adm2_checks(adm2.reset_index()))


def test_verification_flags_classes_that_exceed_the_broader_class(enforcement):
    _, _, adm2 = enforcement
    frame = adm2.reset_index()
    frame.loc[0, "ibama_fines_garimpo"] = frame.loc[0, "ibama_fines_mining"] + 1

    failed = {check.name for check in _enforcement_adm2_checks(frame) if not check.ok}

    assert failed == {"classes_nested"}
