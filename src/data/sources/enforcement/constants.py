from __future__ import annotations

from pathlib import Path

from src.data.shared.paths import processed_dir, raw_dir as _shared_raw_dir, source_root


# IBAMA Fiscalização open data (`dadosabertos.ibama.gov.br`, CKAN). The portal
# blocks requests from outside Brazil, so the files are downloaded by hand and
# placed in a dated snapshot folder `data/enforcement/raw/ibama_<YYYY-MM-DD>/`
# with PROVENANCE.md, SCHEMA.md and SHA256SUMS. Files keep their served names
# with a `<dataset>__` prefix. Every table is UTF-8 (some with a BOM),
# `;`-separated, with line breaks inside quoted text.
SNAPSHOT_PREFIX = "ibama_"
ENCODING = "utf-8-sig"
SEPARATOR = ";"

FINES = "fines"
EMBARGOES = "embargoes"
SEIZURES = "seizures"
DESTRUCTIONS = "destructions"
ACTIONS = (FINES, EMBARGOES, SEIZURES, DESTRUCTIONS)

# Per action: the raw file (a zip holds one or more CSVs), the record keys, and
# the columns mapped onto the record table. `seq` is the system ID (empty for
# records from the AIe app), `code` and `series` the printed code; `text` are
# the description fields searched for mining terms (location fields are left
# out: they carry place names such as Lavras, Ouro Preto or the
# Transgarimpeira road). Records with `cancelled` != "N" (cancelled, deleted
# or replaced by another record) are dropped.
TABLES = {
    FINES: dict(
        file="auto_infracao__auto_infracao_csv.zip",
        seq="SEQ_AUTO_INFRACAO", code="NUM_AUTO_INFRACAO", series="SER_AUTO_INFRACAO",
        cancelled="SIT_CANCELADO", date="DAT_HORA_AUTO_INFRACAO",
        municipality="COD_MUNICIPIO", uf="UF", lon="NUM_LONGITUDE_AUTO", lat="NUM_LATITUDE_AUTO",
        value="VAL_AUTO_INFRACAO", area=None, lifted=None,
        operation=("OPERACAO", "OPERACAO_SOL_RECURSO"),
        operation_code="CD_ACAO_FISCALIZATORIA", operation_seq="SEQ_ACAO_FISCALIZATORIA",
        fine_seq="SEQ_AUTO_INFRACAO", fine_code="NUM_AUTO_INFRACAO",
        text=("DES_AUTO_INFRACAO", "DES_INFRACAO"),
    ),
    EMBARGOES: dict(
        file="embargo__termo_de_embargo.csv",
        seq="SEQ_TAD", code="NUM_TAD", series="SER_TAD",
        cancelled="SIT_CANCELADO", date="DAT_EMBARGO",
        municipality="COD_MUNICIPIO", uf="UF", lon="NUM_LONGITUDE_TAD", lat="NUM_LATITUDE_TAD",
        value=None, area="QTD_AREA_EMBARGADA", lifted="DAT_DESEMBARGO",
        operation=("OPERACAO", "OPERACAO_SOL_RECURSO"),
        operation_code="CD_ACAO_FISCALIZATORIA", operation_seq="SEQ_ACAO_FISCALIZATORIA",
        fine_seq="SEQ_AUTO_INFRACAO", fine_code="NUM_AUTO_INFRACAO",
        text=("DES_TAD",),
    ),
    SEIZURES: dict(
        file="apreensao__apreensao_csv.zip",
        seq="SEQ_TAD", code="CD_TAD", series="CD_SERIE_TAD",
        cancelled="ST_CANCELADO", date="DT_APREENSAO",
        municipality="CD_MUNICIPIO", uf="SG_UF", lon="NU_LONGITUDE_TAD", lat="NU_LATITUDE_TAD",
        value="VL_TAD", area=None, lifted=None,
        operation=("NO_OPERACAO", "NO_OPERACAO_SOL_RECURSO"),
        operation_code="CD_ACAO_FISCALIZATORIA", operation_seq="SEQ_ACAO_FISCALIZATORIA",
        fine_seq="SEQ_AUTO_INFRACAO", fine_code="CD_AUTO_INFRACAO",
        text=("DS_TAD", "DS_COMPLEMENTAR", "DS_AUTO_INFRACAO"),
    ),
    DESTRUCTIONS: dict(
        file="destruicao__termo_destruicao_inutilizacao.csv",
        seq="SEQ_TAD", code="NUM_TAD", series="SER_TAD",
        cancelled="SIT_CANCELADO", date="DAT_TAD",
        municipality="COD_MUNICIPIO", uf="SIG_UF", lon="NUM_LONGITUDE_TAD", lat="NUM_LATITUDE_TAD",
        value="VALOR_TAD", area=None, lifted=None,
        operation=("OPERACAO_SOL_RECURSO",),
        operation_code="CD_ACAO_FISCALIZATORIA", operation_seq="SEQ_ACAO_FISCALIZATORIA",
        fine_seq="SEQ_AUTO_INFRACAO", fine_code="NUM_AUTO_INFRACAO",
        text=("DES_TAD", "DS_INFORMACOES_COMPLEMENTARES", "DES_JUSTIFICATIVA"),
    ),
}

# Seized and destroyed items, linked to their term by `seq` or, when either
# side lacks it, by `code` + `series`. Their type and description are searched
# like the term's own text ("Balsa de garimpo", "Draga Garimpeira", "Mercúrio").
ITEM_TABLES = {
    SEIZURES: dict(
        file="apreensao__bem_csv.zip", seq="SEQ_TAD", code="CD_TAD", series="CD_SERIE_TAD",
        text=("DS_TIPO_BEM", "DS_BEM_APREENDIDO"),
    ),
    DESTRUCTIONS: dict(
        file="destruicao__termo_destruicao_bem_destruido.csv", seq="SEQ_TAD", code="NUM_TAD", series="SER_TAD",
        text=("DES_TIPO_BEM", "DES_BEM_APREENDIDO"),
    ),
}

# Mining terms, matched on lower-cased, accent-folded text. Names are removed
# first: places (Ouro Preto, São José do Ouro), farms ("Fazenda Estrela de
# Ouro", but not "fazenda com extração de ouro"), the colour, mercury-vapour
# lamps and a company. `lavra` is singular only (Lavras is a town in MG); bare
# "minerais" is left out (it is mostly industry registration and the transport
# of minerals), extraction of minerals is kept.
PLACE_NAME_PATTERN = (
    r"\bouro (?:verde|preto|fino|branco|velho)\b|\bsao jose do ouro\b"
    r"|\b(?:fazenda|faz\.|sitio|chacara|estancia|imovel|rancho|gleba)\s+"
    r"(?:(?!extra|lavra|garimp|minera|minerio|explora|retira|comerci|vend|compr)[a-z]+\s+){0,2}"
    r"(?:d[aeo]s?\s+)?ouro\b"
    r"|\bcor (?:de )?ouro\b|\blampadas?\b[^|]{0,150}?\bmercurio\b|\bmercurio alimentos\b"
)
GARIMPO_PATTERN = r"garimp|\bouro\b|\bmercurio\b|\bcassiterita\b"
MINING_PATTERN = (
    GARIMPO_PATTERN
    + r"|\blavra\b|\bminera(?:cao|coes|dora?s?|rias?|rios?)\b|\bminerios?\b"
    + r"|\bextra\w*\s+(?:de\s+)?(?:recursos\s+|produtos\s+|bens\s+)?minera(?:l|is)\b"
)
# Classes are nested: garimpo within mining within all.
CLASSES = ("all", "mining", "garimpo")

# Brazil's bounding box; coordinates outside it are swapped if that puts them
# inside, and dropped otherwise (0/0 and typos).
LON_RANGE = (-74.1, -28.8)
LAT_RANGE = (-33.8, 5.3)
FIRST_DATE = "1977-01-01"

# Years each action is recorded from (SCHEMA.md section 4): fines from SICAFI
# (1996), embargoes from about 2000, seizures from 2005, destruction terms from
# 2014. The municipality table stops at the last full year before the snapshot.
COVERAGE_START = {FINES: 1996, EMBARGOES: 2000, SEIZURES: 2005, DESTRUCTIONS: 2014}

MUN_ID_COLUMN = "mun_id"
YEAR_COLUMN = "year"
RECORD_COLUMNS = (
    "action", "record_id", "date", YEAR_COLUMN, MUN_ID_COLUMN, "uf", "lon", "lat",
    "value_brl", "area_ha", "lifted_date", "items", "operation", "operation_code", "operation_seq",
    "fine_seq", "fine_code", "mining", "garimpo",
)
# Per action: the record count and, where the table has one, its summed amount.
ACTION_MEASURES = {
    FINES: (("ibama_fines", None), ("ibama_fine_value_brl", "value_brl")),
    EMBARGOES: (("ibama_embargoes", None), ("ibama_embargo_area_ha", "area_ha")),
    SEIZURES: (("ibama_seizures", None),),
    DESTRUCTIONS: (("ibama_destructions", None),),
}


def measure_column(measure, group):
    return f"{measure}_{group}"


def action_columns(action):
    return tuple(measure_column(measure, group) for group in CLASSES for measure, _ in ACTION_MEASURES[action])


ADM2_COLUMNS = (MUN_ID_COLUMN, YEAR_COLUMN, *(column for action in ACTIONS for column in action_columns(action)))

DEFAULT_RECORDS_OUTPUT_PATH = str(processed_dir(".", "enforcement") / "ibama_records.parquet")
DEFAULT_ADM2_OUTPUT_PATH = str(processed_dir(".", "enforcement") / "enforcement_adm2.parquet")


def enforcement_dir(root_dir: str | Path = ".") -> Path:
    return source_root(root_dir, "enforcement")


def raw_dir(root_dir: str | Path = ".") -> Path:
    return _shared_raw_dir(root_dir, "enforcement")


def snapshot_path(root_dir: str | Path = ".") -> Path | None:
    """The newest IBAMA snapshot folder holding every raw file (folders sort by date), or None."""
    required = [spec["file"] for spec in (*TABLES.values(), *ITEM_TABLES.values())]
    for snapshot in sorted(raw_dir(root_dir).glob(f"{SNAPSHOT_PREFIX}*"), reverse=True):
        if all((snapshot / name).exists() for name in required):
            return snapshot
    return None
