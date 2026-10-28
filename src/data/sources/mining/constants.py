from __future__ import annotations

from pathlib import Path

from src.data.shared.paths import processed_dir, raw_dir as _shared_raw_dir, source_root


# ANM Cadastro Mineiro (SCM) microdata: `dadosabertos.anm.gov.br/SCM/microdados/
# microdados-scm.zip`, updated daily, 1934 onward. The ANM firewall returns 403
# to scripted downloads (also from the transfer node), so the archive is
# downloaded in a browser and placed by hand in a dated snapshot folder.
SCM_SNAPSHOT_GLOB = "anm_scm_microdados_*"
SCM_ARCHIVE_FILENAME = "microdados-scm.zip"
SCM_ENCODING = "cp1252"
SCM_SEPARATOR = ";"

# Tables read from the archive.
PROCESS_TABLE = "Processo.txt"  # DSProcesso, IDTipoRequerimento, DTProtocolo, QTAreaHA, ...
TITLE_TABLE = "ProcessoTitulo.txt"  # DSProcesso, IDTipoDocumentoLegal, DTPublicacao, DTVencimento, ...
MUNICIPALITY_TABLE = "ProcessoMunicipio.txt"  # DSProcesso, IDMunicipio (7-digit IBGE)
SUBSTANCE_TABLE = "ProcessoSubstancia.txt"  # DSProcesso, IDSubstancia, ...
MUNICIPALITY_LIST_TABLE = "Municipio.txt"  # IDMunicipio, NMMunicipio, SGUF

# Request types (`TipoRequerimento`) by which miners ask for a PLG (permissão
# de lavra garimpeira): the PLG request and a change of regime to PLG. The
# availability procedures (25, 34) and the public offer (37) are ANM-initiated
# auctions of free areas (201 gold requests in the Legal Amazon in 2023 alone).
PLG_REQUEST_TYPES = ("2", "24")
# `TipoDocumentoLegal` 2: Permissão de Lavra Garimpeira. A process's grant is
# its earliest PLG title; renewals and extensions are further titles.
PLG_TITLE_TYPE = "2"
# Lei 7.805/1989 art. 7: a PLG runs for up to five years, renewable. Applied
# when a process's PLG titles carry no expiry date.
PLG_TERM_YEARS = 5

# `Substancia` IDs by group; a process belongs to a group if it lists any of
# them. "OURO PIGMENTO" (100307) is orpiment, an arsenic mineral, not gold.
SUBSTANCE_GROUPS = {
    "gold": ("104900", "104901", "104908"),  # minério de ouro, ouro, ouro nativo
    "tin": ("101100", "101101", "101104", "101106"),  # minério de estanho, estanho, cassiterita, pirita de estanho
}
GROUPS = ("all", *SUBSTANCE_GROUPS)

YEARS = tuple(range(1985, 2027))

MUN_ID_COLUMN = "mun_id"
YEAR_COLUMN = "year"
MEASURES = (
    "plg_requests",  # PLG requests filed in the year (by protocol date)
    "plg_request_area_ha",  # their area, split equally across the process's municipalities
    "plg_grants",  # PLGs first granted in the year (earliest PLG title)
    "plg_active",  # PLGs in force at some point in the year
    "plg_active_area_ha",  # their area, split as above
)


def measure_column(measure, group):
    return f"{measure}_{group}"


ADM2_COLUMNS = (
    MUN_ID_COLUMN,
    YEAR_COLUMN,
    *(measure_column(measure, group) for group in GROUPS for measure in MEASURES),
)

DEFAULT_ADM2_OUTPUT_PATH = str(processed_dir(".", "mining") / "mining_permits_adm2.parquet")


def mining_dir(root_dir: str | Path = ".") -> Path:
    return source_root(root_dir, "mining")


def raw_dir(root_dir: str | Path = ".") -> Path:
    return _shared_raw_dir(root_dir, "mining")


def scm_archive_path(root_dir: str | Path = ".") -> Path | None:
    """The newest SCM snapshot's archive (snapshot folders sort by date), or None."""
    snapshots = sorted(raw_dir(root_dir).glob(SCM_SNAPSHOT_GLOB))
    for snapshot in reversed(snapshots):
        archive = snapshot / SCM_ARCHIVE_FILENAME
        if archive.exists():
            return archive
    return None
