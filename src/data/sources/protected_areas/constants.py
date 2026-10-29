from __future__ import annotations

from pathlib import Path

from src.data.shared.paths import processed_dir, raw_dir as _shared_raw_dir, source_root


# INPE TerraBrasilis GeoServer (WFS). It serves the FUNAI Indigenous-land and
# the CNUC conservation-unit polygons clipped to the Legal Amazon and to each
# biome; the fetch takes every layer and preprocess merges each area's pieces.
# FUNAI's own GeoServer does not answer from the HPC. The layers carry the
# Indigenous lands' current stage but no stage dates, and the conservation
# units' creation year (units created up to about 2020).
WFS_URL = "https://terrabrasilis.dpi.inpe.br/geoserver/ows"
INDIGENOUS = "indigenous_land"
CONSERVATION = "conservation_unit"
KINDS = (INDIGENOUS, CONSERVATION)
_LAYER_STEMS = {INDIGENOUS: "indigenous_area", CONSERVATION: "conservation_units"}
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
LAYERS = {
    kind: tuple(f"{workspace}:{stem}_{region}" for workspace, region in _REGIONS)
    for kind, stem in _LAYER_STEMS.items()
}
FIELDS = {
    INDIGENOUS: ("terrai_cod", "terrai_nom", "etnia_nome", "fase_ti"),
    CONSERVATION: ("nome", "categoria", "grupo", "esfera", "ano_cria"),
}
RAW_FILENAME = "terrabrasilis_protected_areas.gpkg"

# Where mining has no legal option. Indigenous lands from the declaration
# onward (mining there needs a law that was never passed). Conservation units
# of integral protection (`grupo` "PI": parks, ecological stations, biological
# reserves, wildlife refuges, natural monuments) and extractive reserves
# (SNUC, Lei 9.985/2000, art. 18 §6). APAs, national forests (FLONAs, where
# mining is licensed), RDS, ARIEs and RPPNs are left out.
BARRED_STAGES = ("Declarada", "Homologada", "Encaminhada RI", "Regularizada")
BARRED_GROUPS = ("PI",)
BARRED_CATEGORIES = ("Reserva Extrativista",)
# In force by the end of `CUTOFF_YEAR`, before the 2013 good-faith rule:
# conservation units created by then; Indigenous lands homologated or further
# on (the stage dates are not in the layer, so a land homologated after 2012
# counts as in force).
CUTOFF_YEAR = 2012
ESTABLISHED_STAGES = ("Homologada", "Encaminhada RI", "Regularizada")

AREA_COLUMNS = (
    "area_id", "kind", "name", "category", "group", "sphere", "stage", "year_created",
    "mining_barred", "in_force_2012", "geometry",
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

DEFAULT_AREAS_OUTPUT_PATH = str(processed_dir(".", "protected_areas") / "protected_areas.parquet")
DEFAULT_ADM2_OUTPUT_PATH = str(processed_dir(".", "protected_areas") / "protected_areas_adm2.parquet")


def protected_areas_dir(root_dir: str | Path = ".") -> Path:
    return source_root(root_dir, "protected_areas")


def raw_dir(root_dir: str | Path = ".") -> Path:
    return _shared_raw_dir(root_dir, "protected_areas")


def gpkg_path(root_dir: str | Path = ".") -> Path:
    return raw_dir(root_dir) / RAW_FILENAME


def layer_name(type_name):
    """GeoPackage layer name of a WFS type name (`workspace:layer` -> `layer`)."""
    return type_name.split(":", 1)[1]
