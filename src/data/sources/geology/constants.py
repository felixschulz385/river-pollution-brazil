from __future__ import annotations

from pathlib import Path

from src.data.shared.paths import processed_dir, raw_dir as _shared_raw_dir, source_root


# Geological Survey of Brazil (SGB) geoportal, ArcGIS REST. Each layer is
# returned in a single request: the services reject pagination, and the
# occurrence layer's maxRecordCount (100,000) is above its size.
SGB_REST_URL = "https://geoportal.sgb.gov.br/server/rest/services"
SGB_LAYERS = {
    # Geobank mineral occurrences, all substances (points).
    "occurrences": "geologia/ocorrencias/MapServer/0",
    # "Províncias e Distritos Auríferos do Brasil" (Informe de Recursos
    # Minerais, Série Ouro 18, 2022), polygons.
    "gold_provinces": "Províncias_e_Distritos_Auríferos_do_Brasil/MapServer/0",
    "gold_districts": "Províncias_e_Distritos_Auríferos_do_Brasil/MapServer/1",
}
OCCURRENCE_FIELDS = (
    "OBJECTID", "ID_OCORRENCIA", "MUNICIPIO", "UF", "TOPONIMIA", "PROVINCIA",
    "STATUS_ECONOMICO", "IMPORTANCIA", "SITUACAO_MINA", "SITUACAO_GARIMPO",
    "MOTIVO_INATIVIDADE_GARIMPO", "SUBSTANCIAS", "MORFOLOGIA", "CLASSES_UTILITARIAS",
    "ROCHAS_HOSPEDEIRAS", "DATA_CADASTRO", "METODO_GEOPOSICIONAMENTO",
)
SGB_GOLD_FILENAME = "sgb_gold.gpkg"

GOLD_SUBSTANCE = "Ouro"
ALLUVIAL_HOSTS = r"Alúvio|Aluvi|Cascalho|Elúvio|Colúvio|Sedimento detrit"

# Occurrence positions are mostly digitized from 1:250,000 and 1:100,000
# sheets, so points just outside every (simplified) municipality are assigned
# to the nearest one within this distance; the rest are dropped.
NEAREST_MUNICIPALITY_MAX_DISTANCE_M = 5_000

MUN_ID_COLUMN = "mun_id"
AREA_COLUMN = "area_km2"
GOLD_OCCURRENCE_COLUMN = "gold_occurrences"
ALLUVIAL_COLUMN = "gold_occurrences_alluvial"
# `STATUS_ECONOMICO` value -> count column. Only the never-exploited
# occurrences are predetermined geology: garimpo and mine occurrences were
# largely recorded where mining already was.
STATUS_COLUMNS = {
    "Não explotado": "gold_occurrences_unexploited",
    "Garimpo": "gold_occurrences_garimpo",
    "Mina": "gold_occurrences_mine",
    "Indeterminado": "gold_occurrences_undetermined",
}
DISTRICT_SHARE_COLUMN = "gold_district_share"
PROVINCE_SHARE_COLUMN = "gold_province_share"
COUNT_COLUMNS = (GOLD_OCCURRENCE_COLUMN, *STATUS_COLUMNS.values(), ALLUVIAL_COLUMN)
ADM2_COLUMNS = (MUN_ID_COLUMN, AREA_COLUMN, *COUNT_COLUMNS, DISTRICT_SHARE_COLUMN, PROVINCE_SHARE_COLUMN)

DEFAULT_ADM2_OUTPUT_PATH = str(processed_dir(".", "geology") / "geology_adm2.parquet")


def geology_dir(root_dir: str | Path = ".") -> Path:
    return source_root(root_dir, "geology")


def raw_dir(root_dir: str | Path = ".") -> Path:
    return _shared_raw_dir(root_dir, "geology")


def gpkg_path(root_dir: str | Path = ".") -> Path:
    return raw_dir(root_dir) / SGB_GOLD_FILENAME
