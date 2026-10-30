from __future__ import annotations

from pathlib import Path

from src.data.shared.paths import processed_dir, raw_dir as _shared_raw_dir, source_root


# INPE TerraBrasilis GeoServer (WFS). It serves the CNUC conservation-unit
# polygons clipped to the Legal Amazon and to each biome; the fetch takes every
# layer and preprocess merges each unit's pieces. The layers carry each unit's
# creation year (units created up to about 2020).
WFS_URL = "https://terrabrasilis.dpi.inpe.br/geoserver/ows"
INDIGENOUS = "indigenous_land"
CONSERVATION = "conservation_unit"
KINDS = (INDIGENOUS, CONSERVATION)
_REGIONS = (
    ("prodes-legal-amz", "legal_amazon"),
    ("prodes-amazon-nb", "amazon_biome"),
    ("prodes-cerrado-nb", "cerrado_biome"),
    ("prodes-caatinga-nb", "caatinga_biome"),
    ("prodes-mata-atlantica-nb", "mata_atlantica_biome"),
    ("prodes-pampa-nb", "pampa_biome"),
    ("prodes-pantanal-nb", "pantanal_biome"),
)
# Kind -> the WFS type names of its layers.
LAYERS = {CONSERVATION: tuple(f"{workspace}:conservation_units_{region}" for workspace, region in _REGIONS)}
FIELDS = {CONSERVATION: ("nome", "categoria", "grupo", "esfera", "ano_cria")}
RAW_FILENAME = "terrabrasilis_protected_areas.gpkg"

# FUNAI's Indigenous-land polygons with the date and act of each stage, placed
# by hand in `raw/funai_<YYYY-MM-DD>/` (FUNAI's GeoServer does not answer from
# the HPC; see the snapshot's PROVENANCE.md). One feature per land, keyed by
# `terrai_cod`; EPSG:4674, DBF field names cut to 10 characters.
FUNAI_SNAPSHOT_PREFIX = "funai_"
FUNAI_FILENAME = "tis_poligonais_portarias.zip"
FUNAI_FIELDS = {
    "terrai_cod": "code", "terrai_nom": "name", "fase_ti": "stage", "modalidade": "modality",
    "data_decla": "declared", "data_homol": "homologated", "data_regul": "registered",
}

# Where mining has no legal option. Indigenous lands from the declaration
# onward (mining there needs a law that was never passed). Conservation units
# of integral protection (`grupo` "PI": parks, ecological stations, biological
# reserves, wildlife refuges, natural monuments) and extractive reserves
# (SNUC, Lei 9.985/2000, art. 18 §6). APAs, national forests (FLONAs, where
# mining is licensed), RDS, ARIEs and RPPNs are left out.
BARRED_STAGES = ("Declarada", "Homologada", "Encaminhada RI", "Regularizada")
BARRED_GROUPS = ("PI",)
BARRED_CATEGORIES = ("Reserva Extrativista",)
# In force by the end of `CUTOFF_YEAR`, before the 2013 good-faith rule: the
# year mining became barred is a unit's creation year and a land's first
# declaration, homologation or registry date (older lands were homologated
# without a declaratory ordinance). Lands without any of these dates count as
# in force when homologated or regularised (old reserves and three old Mato
# Grosso lands), not when a reserve is still being acquired (Encaminhada RI).
CUTOFF_YEAR = 2012
ESTABLISHED_STAGES = ("Homologada", "Regularizada")

AREA_COLUMNS = (
    "area_id", "kind", "name", "category", "group", "sphere", "stage", "year_created", "date_declared",
    "date_homologated", "year_barred", "mining_barred", "in_force_2012", "geometry",
)

MUN_ID_COLUMN = "mun_id"
INDIGENOUS_SHARE_COLUMN = "indigenous_land_share"
CONSERVATION_SHARE_COLUMN = "conservation_barred_share"
PROTECTED_SHARE_COLUMN = "protected_share"
PROTECTED_2012_SHARE_COLUMN = "protected_2012_share"
ADM2_COLUMNS = (
    MUN_ID_COLUMN, INDIGENOUS_SHARE_COLUMN, CONSERVATION_SHARE_COLUMN, PROTECTED_SHARE_COLUMN,
    PROTECTED_2012_SHARE_COLUMN,
)

# The `trenches` product: the share of each river trench's drainage polygon
# inside each Indigenous land (at any stage, pending ones included) and each
# conservation unit that bars mining, for measures by area from the trench
# tables (e.g. C11 garimpo inside a land).
AREAS_PRODUCT = "areas"
TRENCHES_PRODUCT = "trenches"
PRODUCTS = (AREAS_PRODUCT, TRENCHES_PRODUCT)
TRENCH_COLUMNS = ("area_id", "trench_id", "trench_share")
MIN_TRENCH_SHARE = 1e-6

DEFAULT_AREAS_OUTPUT_PATH = str(processed_dir(".", "protected_areas") / "protected_areas.parquet")
DEFAULT_ADM2_OUTPUT_PATH = str(processed_dir(".", "protected_areas") / "protected_areas_adm2.parquet")
DEFAULT_TRENCHES_OUTPUT_PATH = str(processed_dir(".", "protected_areas") / "protected_areas_trenches.parquet")


def protected_areas_dir(root_dir: str | Path = ".") -> Path:
    return source_root(root_dir, "protected_areas")


def raw_dir(root_dir: str | Path = ".") -> Path:
    return _shared_raw_dir(root_dir, "protected_areas")


def gpkg_path(root_dir: str | Path = ".") -> Path:
    return raw_dir(root_dir) / RAW_FILENAME


def layer_name(type_name):
    """GeoPackage layer name of a WFS type name (`workspace:layer` -> `layer`)."""
    return type_name.split(":", 1)[1]


def funai_path(root_dir: str | Path = ".") -> Path | None:
    """The FUNAI layer in the newest snapshot folder holding it (folders sort by date), or None."""
    for snapshot in sorted(raw_dir(root_dir).glob(f"{FUNAI_SNAPSHOT_PREFIX}*"), reverse=True):
        if (snapshot / FUNAI_FILENAME).exists():
            return snapshot / FUNAI_FILENAME
    return None
