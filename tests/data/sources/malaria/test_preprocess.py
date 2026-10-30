from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.sources.malaria.constants import year_path
from src.data.sources.malaria.preprocess import build_malaria_panels, municipality_ids


ITAITUBA, JACAREACANGA, BOA_VISTA, ALTO_ALEGRE = 150360, 150375, 140010, 140005


def _record(res_exam="Vivax", occupation="Garimpagem", infection=ITAITUBA, residence=ITAITUBA,
            notification=None, country="Brasil", lvc="Notificacao nao e de LVC", detection="Deteccao ativa",
            date="2020-03-17"):
    return {
        "notification_year": 2020,
        "dt_notif": pd.Timestamp(date).date(),
        "res_exam": res_exam,
        "id_lvc": lvc,
        "tipo_lam": detection,
        "cod_ocup": occupation,
        "pais_inf": country,
        "mun_infe_code": infection,
        "mun_infe_nome": None if infection is None else "x",
        "mun_resi_code": residence,
        "mun_resi_nome": None if residence is None else "x",
        "mun_noti_code": notification or residence,
        "mun_noti_nome": "x",
    }


def _write_year(root: Path, records):
    path = year_path(root, 2020)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    for column in ("mun_infe_code", "mun_resi_code", "mun_noti_code"):
        frame[column] = frame[column].astype("Int32")
    frame.to_parquet(path, index=False)


def test_municipality_ids_drop_placeholders_and_unmatched_codes():
    codes = pd.Series([150360, 539912, 150000, 140005, None], dtype="Int32")
    names = pd.Series(["Itaituba", "Unmapped: 539912", "x", "Unmapped: 140005", None])
    assert municipality_ids(codes, names).tolist() == ["150360", pd.NA, pd.NA, pd.NA, pd.NA]


def test_build_malaria_panels_splits_local_incoming_and_abroad(tmp_path: Path):
    _write_year(
        tmp_path,
        [
            _record(),  # local garimpo case in Itaituba
            _record(residence=JACAREACANGA),  # incoming, same state
            _record(infection=ALTO_ALEGRE, residence=BOA_VISTA),  # incoming, same state
            _record(infection=ITAITUBA, residence=BOA_VISTA),  # incoming from another state
            _record(infection=None, country="Guiana Francesa", residence=JACAREACANGA),  # abroad
            _record(occupation="Agricultura"),  # non-garimpo case
            _record(res_exam="Negativo", occupation=None, infection=None, residence=None, notification=ITAITUBA),
            _record(lvc="Notificacao de LVC"),  # cure verification: dropped
            _record(detection="LVC"),  # cure verification by the old flag: dropped
        ],
    )

    adm2, flows, monthly = build_malaria_panels(
        root_dir=tmp_path,
        years=(2020,),
        adm2_output_path="adm2.parquet",
        flows_output_path="flows.parquet",
        monthly_output_path="monthly.parquet",
    )
    adm2 = adm2.set_index("mun_id")

    itaituba = adm2.loc[str(ITAITUBA)]
    assert itaituba["inf_cases"] == 4
    assert itaituba["inf_garimpo"] == 3
    assert itaituba["inf_garimpo_local"] == 1
    assert itaituba["inf_garimpo_incoming"] == 2
    assert itaituba["inf_garimpo_incoming_other_state"] == 1
    assert itaituba["noti_exams"] == 3  # Itaituba residents' two cases plus the negative
    assert itaituba["noti_positives"] == 2

    jacareacanga = adm2.loc[str(JACAREACANGA)]
    assert jacareacanga["res_garimpo"] == 2
    assert jacareacanga["res_garimpo_elsewhere"] == 1
    assert jacareacanga["res_garimpo_abroad"] == 1
    assert jacareacanga["inf_cases"] == 0

    assert adm2.loc[str(BOA_VISTA), "res_garimpo_elsewhere"] == 2
    assert adm2.loc[str(ALTO_ALEGRE), "inf_garimpo_incoming"] == 1

    abroad = flows[flows["inf_country"] == "Guiana Francesa"]
    assert abroad["res_mun_id"].tolist() == [str(JACAREACANGA)]
    assert abroad["inf_mun_id"].isna().all()
    assert flows["cases"].sum() == 5
    assert (tmp_path / "adm2.parquet").exists() and (tmp_path / "flows.parquet").exists()


def test_monthly_counts_by_infection_municipality_sum_to_the_year(tmp_path: Path):
    _write_year(
        tmp_path,
        [
            _record(date="2020-03-02"),  # local garimpo case in Itaituba, March
            _record(residence=JACAREACANGA, date="2020-03-30"),  # incoming, March
            _record(residence=JACAREACANGA, date="2020-07-01"),  # incoming, July
            _record(occupation="Agricultura", date="2020-07-15"),  # non-garimpo case, July
            _record(infection=None, country="Guiana Francesa", residence=JACAREACANGA),  # abroad: no row
            _record(res_exam="Negativo", occupation=None, infection=None, residence=None, notification=ITAITUBA),
        ],
    )

    adm2, _, monthly = build_malaria_panels(root_dir=tmp_path, years=(2020,), adm2_output_path="adm2.parquet",
                                            flows_output_path="flows.parquet", monthly_output_path="monthly.parquet")
    itaituba = monthly[monthly["mun_id"] == str(ITAITUBA)].set_index("month")

    assert itaituba.index.tolist() == [pd.Timestamp("2020-03-01"), pd.Timestamp("2020-07-01")]
    assert itaituba[["inf_cases", "inf_garimpo", "inf_garimpo_local", "inf_garimpo_incoming"]].values.tolist() == [
        [2, 2, 1, 1], [2, 1, 0, 1]]
    yearly = adm2.set_index("mun_id").loc[str(ITAITUBA)]
    assert (itaituba.sum()[["inf_cases", "inf_garimpo", "inf_garimpo_incoming"]]
            == yearly[["inf_cases", "inf_garimpo", "inf_garimpo_incoming"]]).all()
    assert (tmp_path / "monthly.parquet").exists()
