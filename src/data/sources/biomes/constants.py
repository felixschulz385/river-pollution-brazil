from __future__ import annotations

from pathlib import Path

from src.data.shared.paths import processed_dir, raw_dir as _shared_raw_dir, source_root
from src.data.sources.gadm.constants import DEFAULT_SIMPLIFIED_GADM_PATH as DEFAULT_GADM_PATH


BIOMES_ARCHIVE_URL = (
    "https://geoftp.ibge.gov.br/informacoes_ambientais/estudos_ambientais/biomas/"
    "vetores/2025_Biomas-e-Sistema-Costeiro-Marinho-do-Brasil-1-250000_shp.zip"
)
BIOMES_ARCHIVE_FILENAME = "2025_Biomas-e-Sistema-Costeiro-Marinho-do-Brasil-1-250000_shp.zip"

# The IBGE archive bundles the terrestrial biome layer alongside a separate
# coastal/marine system layer; only the former is relevant here, identified by
# this substring in its shapefile name (case-insensitive), excluding the
# coastal/marine one.
BIOME_SHAPEFILE_NAME_HINT = "bioma"
BIOME_SHAPEFILE_EXCLUDE_HINT = "costeiro"

# Candidate column names for the biome label in the raw IBGE shapefile,
# in preference order.
BIOME_NAME_COLUMN_CANDIDATES = (
    "NM_BIOMA",
    "Bioma",
    "bioma",
    "NOM_BIOMA",
    "nome_bioma",
    "nm_bioma",
)

BIOME_COLUMN = "biome"
MUN_ID_COLUMN = "mun_id"
STATION_CODE_COLUMN = "station_code"

DEFAULT_ADM2_LAYER = "ADM_ADM_2"
DEFAULT_ADM2_ID_COLUMN = "CC_2"
BRAZIL_PROJECTED_CRS = 5641

DEFAULT_ADM2_OUTPUT_PATH = str(processed_dir(".", "biomes") / "biomes_adm2.parquet")
DEFAULT_SENSOR_OUTPUT_PATH = str(processed_dir(".", "biomes") / "biomes_sensor.parquet")


def biomes_dir(root_dir: str | Path = ".") -> Path:
    return source_root(root_dir, "biomes")


def raw_dir(root_dir: str | Path = ".") -> Path:
    return _shared_raw_dir(root_dir, "biomes")


def archive_path(root_dir: str | Path = ".") -> Path:
    return raw_dir(root_dir) / BIOMES_ARCHIVE_FILENAME
