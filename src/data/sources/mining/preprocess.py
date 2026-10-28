import io
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import (
    ADM2_COLUMNS,
    DEFAULT_ADM2_OUTPUT_PATH,
    GROUPS,
    MUN_ID_COLUMN,
    MUNICIPALITY_LIST_TABLE,
    MUNICIPALITY_TABLE,
    PLG_REQUEST_TYPES,
    PLG_TERM_YEARS,
    PLG_TITLE_TYPE,
    PROCESS_TABLE,
    SCM_ENCODING,
    SCM_SEPARATOR,
    SUBSTANCE_GROUPS,
    SUBSTANCE_TABLE,
    TITLE_TABLE,
    YEAR_COLUMN,
    YEARS,
    measure_column,
    scm_archive_path,
)


logger = logging.getLogger(__name__)


def read_scm_table(archive, name, columns):
    """One table of the SCM archive as text columns."""
    with zipfile.ZipFile(archive) as bundle, bundle.open(name) as handle:
        return pd.read_csv(
            io.TextIOWrapper(handle, encoding=SCM_ENCODING),
            sep=SCM_SEPARATOR,
            usecols=list(columns),
            dtype=str,
        )


def _year(dates):
    return pd.to_numeric(dates.str[:4], errors="coerce")


def plg_processes(processes, titles):
    """One row per process with a PLG request or title.

    `request_year` is the protocol year of PLG requests (`PLG_REQUEST_TYPES`);
    `grant_year` the publication year of the earliest PLG title; `expiry_year`
    the latest PLG expiry, or the grant year plus `PLG_TERM_YEARS` when no
    title has one. `area_ha` is the process area; zero or negative areas are
    missing.
    """
    processes = processes.set_index("DSProcesso")
    area = pd.to_numeric(processes["QTAreaHA"].str.replace(",", ".", regex=False), errors="coerce")
    is_request = processes["IDTipoRequerimento"].isin(PLG_REQUEST_TYPES)

    plg_titles = titles[titles["IDTipoDocumentoLegal"] == PLG_TITLE_TYPE].assign(
        grant=lambda frame: _year(frame["DTPublicacao"]),
        expiry=lambda frame: _year(frame["DTVencimento"]),
    )
    by_process = plg_titles.groupby("DSProcesso").agg(grant_year=("grant", "min"), expiry_year=("expiry", "max"))
    by_process["expiry_year"] = by_process["expiry_year"].fillna(by_process["grant_year"] + PLG_TERM_YEARS)

    result = pd.DataFrame(index=processes.index[is_request].union(by_process.index))
    result["request_year"] = _year(processes.loc[is_request, "DTProtocolo"]).reindex(result.index)
    result = result.join(by_process)
    result["area_ha"] = area.where(area > 0).reindex(result.index)
    return result.rename_axis("DSProcesso")


def municipality_shares(links):
    """`DSProcesso`, `mun_id` (6-digit IBGE) and each municipality's equal share of the process."""
    links = links.assign(**{MUN_ID_COLUMN: links["IDMunicipio"].str[:6]})
    links = links[links[MUN_ID_COLUMN].str.fullmatch(r"\d{6}", na=False)]
    links = links.drop_duplicates(["DSProcesso", MUN_ID_COLUMN])
    links["share"] = 1.0 / links.groupby("DSProcesso")[MUN_ID_COLUMN].transform("size")
    return links[["DSProcesso", MUN_ID_COLUMN, "share"]]


def group_processes(substances):
    """Group name -> the set of processes listing any of its substances; "all" is None (no filter)."""
    groups = {"all": None}
    for group, ids in SUBSTANCE_GROUPS.items():
        groups[group] = set(substances.loc[substances["IDSubstancia"].isin(ids), "DSProcesso"])
    return groups


def _count(rows, year_column, name, area_name=None):
    keyed = rows.rename(columns={year_column: YEAR_COLUMN})
    grouped = keyed.groupby([MUN_ID_COLUMN, YEAR_COLUMN])
    out = grouped.size().rename(name).to_frame()
    if area_name:
        out[area_name] = (keyed["area_ha"].fillna(0.0) * keyed["share"]).groupby(
            [keyed[MUN_ID_COLUMN], keyed[YEAR_COLUMN]]
        ).sum()
    return out


def adm2_permits(plg, shares, groups, municipalities, years=YEARS):
    """Municipality x year PLG requests, grants and PLGs in force, by substance group, zeros filled."""
    linked = plg.reset_index().merge(shares, on="DSProcesso")
    first, last = years[0], years[-1]
    grid = pd.MultiIndex.from_product([sorted(municipalities), list(years)], names=[MUN_ID_COLUMN, YEAR_COLUMN])
    frames = []
    for group in GROUPS:
        members = groups[group]
        rows = linked if members is None else linked[linked["DSProcesso"].isin(members)]
        requests = rows.dropna(subset=["request_year"])
        grants = rows.dropna(subset=["grant_year"])
        spans = grants[grants["expiry_year"] >= first].assign(
            start=lambda frame: frame["grant_year"].clip(lower=first),
            end=lambda frame: frame["expiry_year"].clip(upper=last),
        )
        spans = spans[spans["start"] <= spans["end"]]
        active = spans.loc[spans.index.repeat((spans["end"] - spans["start"] + 1).astype(int))].copy()
        active["active_year"] = active["start"] + active.groupby(level=0).cumcount()
        frames += [
            _count(requests, "request_year", measure_column("plg_requests", group),
                   measure_column("plg_request_area_ha", group)),
            _count(grants, "grant_year", measure_column("plg_grants", group)),
            _count(active, "active_year", measure_column("plg_active", group),
                   measure_column("plg_active_area_ha", group)),
        ]
    for frame in frames:
        frame.index = frame.index.set_levels(frame.index.levels[1].astype(int), level=1)
    result = pd.concat(frames, axis=1).reindex(grid).fillna(0.0)
    count_columns = [column for column in result.columns if "area" not in column]
    result[count_columns] = result[count_columns].astype("int64")
    return result.reset_index()[list(ADM2_COLUMNS)]


def build_adm2_permits(root_dir=".", archive=None, output_path=None):
    """Read the newest SCM snapshot and write the municipality x year PLG table."""
    archive = Path(archive) if archive else scm_archive_path(root_dir)
    if archive is None:
        raise FileNotFoundError(
            "No SCM snapshot under data/mining/raw/anm_scm_microdados_*/microdados-scm.zip. "
            "Download microdados-scm.zip from dadosabertos.anm.gov.br/SCM/microdados/ in a browser "
            "(the ANM firewall blocks scripted downloads) and place it in a dated snapshot folder."
        )
    logger.info("Reading SCM tables from %s", archive)
    processes = read_scm_table(archive, PROCESS_TABLE, ["DSProcesso", "IDTipoRequerimento", "DTProtocolo", "QTAreaHA"])
    titles = read_scm_table(archive, TITLE_TABLE, ["DSProcesso", "IDTipoDocumentoLegal", "DTPublicacao", "DTVencimento"])
    links = read_scm_table(archive, MUNICIPALITY_TABLE, ["DSProcesso", "IDMunicipio"])
    substances = read_scm_table(archive, SUBSTANCE_TABLE, ["DSProcesso", "IDSubstancia"])
    municipalities = read_scm_table(archive, MUNICIPALITY_LIST_TABLE, ["IDMunicipio"])["IDMunicipio"].str[:6]

    plg = plg_processes(processes, titles)
    result = adm2_permits(plg, municipality_shares(links), group_processes(substances),
                          set(municipalities[municipalities.str.fullmatch(r"\d{6}", na=False)]))

    resolved_output_path = Path(root_dir) / (output_path or DEFAULT_ADM2_OUTPUT_PATH)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(resolved_output_path, index=False)
    logger.info(
        "Wrote ADM2 PLG permits to %s (shape=%s; %d PLG processes, %d with a grant)",
        resolved_output_path, result.shape, len(plg), int(plg["grant_year"].notna().sum()),
    )
    return result
