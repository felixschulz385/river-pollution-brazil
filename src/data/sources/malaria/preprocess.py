import logging
from pathlib import Path

import pandas as pd

from .constants import (
    ADM2_COLUMNS,
    BRAZIL,
    DEFAULT_ADM2_OUTPUT_PATH,
    DEFAULT_FLOWS_OUTPUT_PATH,
    DEFAULT_MONTHLY_OUTPUT_PATH,
    FLOW_CASES_COLUMN,
    FLOW_COLUMNS,
    FLOW_COUNTRY_COLUMN,
    FLOW_INFECTION_COLUMN,
    FLOW_RESIDENCE_COLUMN,
    GARIMPO_OCCUPATION,
    INFECTION_COLUMNS,
    LVC_DETECTION,
    LVC_NOTIFICATION,
    MONTH_COLUMN,
    MONTHLY_COLUMNS,
    MUN_ID_COLUMN,
    NEGATIVE_RESULT,
    NOTIFICATION_COLUMNS,
    PLACEHOLDER_MUNICIPALITY_PREFIX,
    READ_COLUMNS,
    RESIDENCE_COLUMNS,
    UNMAPPED_NAME_PREFIX,
    YEAR_COLUMN,
    YEARS,
    year_path,
)


logger = logging.getLogger(__name__)


def municipality_ids(codes, names):
    """6-digit IBGE codes as `mun_id` strings; placeholders and unmatched codes become missing."""
    codes = pd.to_numeric(codes, errors="coerce").astype("Int64")
    valid = (
        codes.between(110000, 539999)
        & (codes // 100 != PLACEHOLDER_MUNICIPALITY_PREFIX)
        & (codes % 10000 != 0)
        & ~names.fillna("").str.startswith(UNMAPPED_NAME_PREFIX)
    )
    return codes.where(valid.fillna(False)).astype("string")


def classify_records(frame):
    """Add the sample flags and the three municipality ids to raw SIVEP records."""
    lvc = frame["id_lvc"].eq(LVC_NOTIFICATION) | frame["tipo_lam"].eq(LVC_DETECTION)
    positive = frame["res_exam"].notna() & frame["res_exam"].ne(NEGATIVE_RESULT)
    abroad = frame["pais_inf"].notna() & frame["pais_inf"].ne(BRAZIL)
    infection = municipality_ids(frame["mun_infe_code"], frame["mun_infe_nome"])
    return pd.DataFrame(
        {
            YEAR_COLUMN: frame["notification_year"].astype("int64"),
            MONTH_COLUMN: pd.to_datetime(frame["dt_notif"]).dt.to_period("M").dt.to_timestamp().to_numpy(),
            "exam": (~lvc).to_numpy(),
            "case": (positive & ~lvc).to_numpy(),
            "garimpo": frame["cod_ocup"].eq(GARIMPO_OCCUPATION).to_numpy(),
            "abroad": abroad.to_numpy(),
            "country": frame["pais_inf"].where(abroad, BRAZIL),
            "inf": infection.mask(abroad),
            "res": municipality_ids(frame["mun_resi_code"], frame["mun_resi_nome"]),
            "noti": municipality_ids(frame["mun_noti_code"], frame["mun_noti_nome"]),
        }
    )


def _count(records, mask, by, name):
    return records.loc[mask].groupby([by, YEAR_COLUMN]).size().rename(name).rename_axis([MUN_ID_COLUMN, YEAR_COLUMN])


def aggregate_adm2(records):
    """Case counts per municipality-year by infection, residence and notification site."""
    cases = records["case"]
    garimpo = cases & records["garimpo"]
    infected = records["inf"].notna()
    resident = records["res"].notna()
    local = infected & resident & records["inf"].eq(records["res"])
    incoming = infected & resident & ~local
    other_state = incoming & records["inf"].str[:2].ne(records["res"].str[:2])

    columns = [
        _count(records, cases & infected, "inf", "inf_cases"),
        _count(records, garimpo & infected, "inf", "inf_garimpo"),
        _count(records, garimpo & local, "inf", "inf_garimpo_local"),
        _count(records, garimpo & incoming, "inf", "inf_garimpo_incoming"),
        _count(records, garimpo & other_state, "inf", "inf_garimpo_incoming_other_state"),
        _count(records, cases & resident, "res", "res_cases"),
        _count(records, garimpo & resident, "res", "res_garimpo"),
        _count(records, garimpo & incoming, "res", "res_garimpo_elsewhere"),
        _count(records, garimpo & resident & records["abroad"], "res", "res_garimpo_abroad"),
        _count(records, records["exam"] & records["noti"].notna(), "noti", "noti_exams"),
        _count(records, cases & records["noti"].notna(), "noti", "noti_positives"),
    ]
    count_columns = [*INFECTION_COLUMNS, *RESIDENCE_COLUMNS, *NOTIFICATION_COLUMNS]
    result = pd.concat(columns, axis=1).reindex(columns=count_columns).fillna(0).astype("int64")
    return result.reset_index()


def aggregate_monthly(records):
    """Case counts per infection municipality and notification month (months with a case only)."""
    infected = records[records["case"] & records["inf"].notna() & records[MONTH_COLUMN].notna()]
    garimpo = infected["garimpo"]
    local = infected["res"].notna() & infected["inf"].eq(infected["res"])
    incoming = infected["res"].notna() & ~local
    counts = pd.DataFrame({
        MUN_ID_COLUMN: infected["inf"],
        MONTH_COLUMN: infected[MONTH_COLUMN],
        "inf_cases": 1,
        "inf_garimpo": garimpo.astype("int64"),
        "inf_garimpo_local": (garimpo & local).astype("int64"),
        "inf_garimpo_incoming": (garimpo & incoming).astype("int64"),
    })
    return counts.groupby([MUN_ID_COLUMN, MONTH_COLUMN], as_index=False).sum()[list(MONTHLY_COLUMNS)]


def aggregate_flows(records):
    """Garimpo cases by residence municipality and infection municipality (or country abroad)."""
    garimpo = records[records["case"] & records["garimpo"] & records["res"].notna()]
    garimpo = garimpo[garimpo["abroad"] | garimpo["inf"].notna()]
    flows = (
        garimpo.groupby([YEAR_COLUMN, "res", "inf", "country"], dropna=False)
        .size()
        .rename(FLOW_CASES_COLUMN)
        .reset_index()
        .rename(columns={"res": FLOW_RESIDENCE_COLUMN, "inf": FLOW_INFECTION_COLUMN, "country": FLOW_COUNTRY_COLUMN})
    )
    return flows[list(FLOW_COLUMNS)]


def build_malaria_panels(root_dir=".", years=YEARS, adm2_output_path=None, flows_output_path=None,
                         monthly_output_path=None):
    """Write the ADM2 malaria counts (by year and by month) and the garimpo residence-to-infection flows."""
    adm2, flows, monthly = [], [], []
    for year in years:
        frame = pd.read_parquet(year_path(root_dir, year), columns=list(READ_COLUMNS))
        records = classify_records(frame)
        adm2.append(aggregate_adm2(records))
        flows.append(aggregate_flows(records))
        monthly.append(aggregate_monthly(records))
        logger.info(
            "SIVEP %d: %d records, %d cases, %d garimpo cases",
            year,
            len(records),
            int(records["case"].sum()),
            int((records["case"] & records["garimpo"]).sum()),
        )

    adm2 = pd.concat(adm2, ignore_index=True)[list(ADM2_COLUMNS)].sort_values([MUN_ID_COLUMN, YEAR_COLUMN])
    flows = pd.concat(flows, ignore_index=True).sort_values(list(FLOW_COLUMNS[:4]))
    # A few notifications fall in another year's file; sum them into one row per month.
    monthly = pd.concat(monthly, ignore_index=True).groupby([MUN_ID_COLUMN, MONTH_COLUMN], as_index=False).sum()
    outputs = (
        (adm2, adm2_output_path or DEFAULT_ADM2_OUTPUT_PATH),
        (flows, flows_output_path or DEFAULT_FLOWS_OUTPUT_PATH),
        (monthly, monthly_output_path or DEFAULT_MONTHLY_OUTPUT_PATH),
    )
    for frame, output_path in outputs:
        resolved = Path(root_dir) / output_path
        resolved.parent.mkdir(parents=True, exist_ok=True)
        frame.reset_index(drop=True).to_parquet(resolved, index=False)
        logger.info("Wrote %s (shape=%s)", resolved, frame.shape)
    return adm2.reset_index(drop=True), flows.reset_index(drop=True), monthly
