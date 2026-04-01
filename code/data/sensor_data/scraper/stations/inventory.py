import pandas as pd
import geopandas as gpd
import requests
from xml.etree import ElementTree as ET

from ..database import (
    RAW_STATIONS_TABLE,
    STATIONS_TABLE,
    STATION_RIVERS_TABLE,
    read_geodataframe_table,
    write_dataframe_table,
    write_geodataframe_table,
)
from ..paths import (
    ensure_water_quality_dirs,
    get_root_path,
)


STATION_INVENTORY_URL = (
    "http://telemetriaws1.ana.gov.br/ServiceANA.asmx/HidroInventario?"
    "codEstDE=&codEstATE=&tpEst=&nmEst=&nmRio=&codSubBacia=&codBacia=&"
    "nmMunicipio=&nmEstado=&sgResp=&sgOper=&telemetrica="
)
DEFAULT_BRAZIL_BOUNDARY_PATH = "data/misc/gadm41_BRA.gpkg"
DEFAULT_BRAZIL_BOUNDARY_LAYER = "ADM_ADM_0"
DEFAULT_RIVER_NETWORK_DIR = "data/river_network"
BRAZIL_PROJECTED_CRS = 5641
RAW_TO_ENGLISH_COLUMN_MAP = {
    "BaciaCodigo": "basin_code",
    "SubBaciaCodigo": "sub_basin_code",
    "RioCodigo": "river_code",
    "RioNome": "river_name",
    "EstadoCodigo": "state_code",
    "nmEstado": "state_name",
    "MunicipioCodigo": "municipality_code",
    "nmMunicipio": "municipality_name",
    "ResponsavelCodigo": "responsible_agency_code",
    "ResponsavelSigla": "responsible_agency_acronym",
    "ResponsavelUnidade": "responsible_unit_code",
    "ResponsavelJurisdicao": "responsible_jurisdiction_code",
    "OperadoraCodigo": "operator_agency_code",
    "OperadoraSigla": "operator_agency_acronym",
    "OperadoraUnidade": "operator_unit_code",
    "OperadoraSubUnidade": "operator_subunit_code",
    "TipoEstacao": "station_type_code",
    "Codigo": "station_code",
    "Nome": "station_name",
    "CodigoAdicional": "additional_code",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Altitude": "altitude_m",
    "AreaDrenagem": "drainage_area_km2",
    "TipoEstacaoEscala": "has_staff_gauge",
    "TipoEstacaoRegistradorNivel": "has_water_level_recorder",
    "TipoEstacaoDescLiquida": "has_discharge_measurement",
    "TipoEstacaoSedimentos": "has_sediment_monitoring",
    "TipoEstacaoQualAgua": "has_water_quality_monitoring",
    "TipoEstacaoPluviometro": "has_rain_gauge",
    "TipoEstacaoRegistradorChuva": "has_rain_recorder",
    "TipoEstacaoTanqueEvapo": "has_evaporation_pan",
    "TipoEstacaoClimatologica": "has_climatological_monitoring",
    "TipoEstacaoPiezometria": "has_piezometric_monitoring",
    "TipoEstacaoTelemetrica": "has_telemetry",
    "PeriodoEscalaInicio": "staff_gauge_start_date",
    "PeriodoEscalaFim": "staff_gauge_end_date",
    "PeriodoRegistradorNivelInicio": "water_level_recorder_start_date",
    "PeriodoRegistradorNivelFim": "water_level_recorder_end_date",
    "PeriodoDescLiquidaInicio": "discharge_measurement_start_date",
    "PeriodoDescLiquidaFim": "discharge_measurement_end_date",
    "PeriodoSedimentosInicio": "sediment_monitoring_start_date",
    "PeriodoSedimentosFim": "sediment_monitoring_end_date",
    "PeriodoQualAguaInicio": "water_quality_start_date",
    "PeriodoQualAguaFim": "water_quality_end_date",
    "PeriodoPluviometroInicio": "rain_gauge_start_date",
    "PeriodoPluviometroFim": "rain_gauge_end_date",
    "PeriodoRegistradorChuvaInicio": "rain_recorder_start_date",
    "PeriodoRegistradorChuvaFim": "rain_recorder_end_date",
    "PeriodoTanqueEvapoInicio": "evaporation_pan_start_date",
    "PeriodoTanqueEvapoFim": "evaporation_pan_end_date",
    "PeriodoClimatologicaInicio": "climatology_start_date",
    "PeriodoClimatologicaFim": "climatology_end_date",
    "PeriodoPiezometriaInicio": "piezometry_start_date",
    "PeriodoPiezometriaFim": "piezometry_end_date",
    "PeriodoTelemetricaInicio": "telemetry_start_date",
    "PeriodoTelemetricaFim": "telemetry_end_date",
    "TipoRedeBasica": "is_basic_network",
    "TipoRedeEnergetica": "is_energy_network",
    "TipoRedeNavegacao": "is_navigation_network",
    "TipoRedeCursoDagua": "is_watercourse_network",
    "TipoRedeEstrategica": "is_strategic_network",
    "TipoRedeCaptacao": "is_water_intake_network",
    "TipoRedeSedimentos": "is_sediment_network",
    "TipoRedeQualAgua": "is_water_quality_network",
    "TipoRedeClasseVazao": "is_discharge_class_network",
    "UltimaAtualizacao": "last_updated_at",
    "Operando": "is_operating",
    "Descricao": "description",
    "NumImagens": "image_count",
    "DataIns": "created_at",
    "DataAlt": "updated_at",
}


def _legacy_clean_column_name(name: str) -> str:
    return "".join(character.lower() for character in str(name) if character.isalnum())


def _modern_clean_column_name(name: str) -> str:
    with_boundaries = pd.Series([name]).str.replace(
        r"([a-z0-9])([A-Z])", r"\1_\2", regex=True
    ).iloc[0]
    return (
        "".join(character.lower() if character.isalnum() else "_" for character in with_boundaries)
        .strip("_")
        .replace("__", "_")
    )


STATION_OUTPUT_COLUMN_MAP = {
    alias: english_name
    for raw_name, english_name in RAW_TO_ENGLISH_COLUMN_MAP.items()
    for alias in {_legacy_clean_column_name(raw_name), _modern_clean_column_name(raw_name)}
}
STATION_OUTPUT_COLUMN_MAP.update(
    {
        "in_bounds": "within_brazil",
        "geometry": "geometry_wkt",
        "trench_id": "trench_id",
    }
)


def parse_station_inventory_xml(xml_content):
    """Convert the ANA XML response into a plain tabular DataFrame."""
    root = ET.fromstring(xml_content)
    station_rows = []

    for table in root.findall(".//Table"):
        row = {}
        for child in table:
            row[child.tag] = child.text
        station_rows.append(row)

    return pd.DataFrame(station_rows)


def fetch_station_inventory(root_dir="."):
    """Fetch the raw ANA station inventory and cache it in DuckDB."""
    ensure_water_quality_dirs(root_dir)
    response = requests.get(url=STATION_INVENTORY_URL, timeout=60)
    response.raise_for_status()

    stations = parse_station_inventory_xml(response.content)
    stations_geo = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(
            pd.to_numeric(stations["Longitude"], errors="coerce"),
            pd.to_numeric(stations["Latitude"], errors="coerce"),
        ),
        crs=4326,
    )
    # Persist the raw inventory immediately so the preprocess step can be rerun
    # without touching the remote API again.
    write_geodataframe_table(root_dir, RAW_STATIONS_TABLE, stations_geo)
    return stations_geo


def resolve_brazil_boundary_path(root_dir=".", brazil_boundary_path=None):
    if brazil_boundary_path is None:
        return get_root_path(root_dir) / DEFAULT_BRAZIL_BOUNDARY_PATH
    return get_root_path(root_dir) / brazil_boundary_path


def resolve_river_network_dir(root_dir=".", river_network_dir=None):
    if river_network_dir is None:
        return get_root_path(root_dir) / DEFAULT_RIVER_NETWORK_DIR
    return get_root_path(root_dir) / river_network_dir


def filter_station_inventory(stations_geo, brazil_boundary_path):
    """Apply the geographic sanity checks used by the downstream pipeline."""
    stations_geo = stations_geo.copy()
    stations_geo["longitude"] = pd.to_numeric(stations_geo["longitude"], errors="coerce")
    stations_geo["latitude"] = pd.to_numeric(stations_geo["latitude"], errors="coerce")
    stations_geo = stations_geo.loc[stations_geo["longitude"] > -100].copy()
    brazil = gpd.read_file(
        brazil_boundary_path,
        layer=DEFAULT_BRAZIL_BOUNDARY_LAYER,
        engine="pyogrio",
    )
    brazil_geometry = brazil.union_all()
    stations_geo["in_bounds"] = stations_geo.within(brazil_geometry)
    return stations_geo.loc[stations_geo["in_bounds"]].copy()


def rename_station_output_columns(stations_rivers):
    """Translate the curated station output schema into readable English names."""
    geometry_wkt = stations_rivers.geometry.to_wkt()
    renamed = stations_rivers.rename(columns=STATION_OUTPUT_COLUMN_MAP).copy()
    renamed["geometry_wkt"] = geometry_wkt
    if "geometry" in renamed.columns:
        renamed = renamed.drop(columns="geometry")
    return renamed.reset_index(drop=True)


def preprocess_station_inventory(
    root_dir=".",
    brazil_boundary_path=None,
    river_network_dir=None,
):
    try:
        from ...river_network import RiverNetwork
    except ImportError:
        from river_network import RiverNetwork

    # The fetch step populates the raw station inventory in DuckDB. Preprocess
    # narrows that raw feed down to a single curated station-to-trench table.
    stations_geo = read_geodataframe_table(root_dir, RAW_STATIONS_TABLE)
    stations_geo = filter_station_inventory(
        stations_geo,
        resolve_brazil_boundary_path(root_dir, brazil_boundary_path),
    )

    network = RiverNetwork()
    network.load(resolve_river_network_dir(root_dir, river_network_dir))
    if network.trenches is None:
        raise ValueError("River network trenches are required before preprocessing stations.")

    trenches = network.trenches[["trench_id", "geometry"]].copy()
    station_matches = gpd.sjoin_nearest(
        stations_geo[["codigo", "geometry"]].to_crs(BRAZIL_PROJECTED_CRS),
        trenches.to_crs(BRAZIL_PROJECTED_CRS),
        how="left",
        distance_col="distance_to_river",
    )
    station_matches = pd.DataFrame(
        station_matches[["codigo", "trench_id", "distance_to_river"]]
    ).sort_values(["codigo", "distance_to_river"]).drop_duplicates(
        subset=["codigo"], keep="first"
    )
    stations_with_trench = stations_geo.merge(
        station_matches.drop(columns=["distance_to_river"]),
        on="codigo",
        how="left",
    )
    stations_table = rename_station_output_columns(stations_geo)
    stations_rivers = rename_station_output_columns(stations_with_trench)

    # `stations` captures the cleaned, translated station inventory. The
    # companion `stations_rivers` table adds only the trench join key used by
    # downstream scraping and analysis tasks.
    write_dataframe_table(
        root_dir,
        STATIONS_TABLE,
        stations_table,
    )
    write_dataframe_table(root_dir, STATION_RIVERS_TABLE, stations_rivers)
    return stations_rivers
