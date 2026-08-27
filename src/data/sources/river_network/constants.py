PROCESSED_DIR = "data/river_network/processed"

# ANA's "Base Hidrográfica Ourinhos" 2017 v.01.05, 1:5k-scale hydrography
# geopackage -- the raw input `river_network preprocess` expects by default,
# manually placed (river_network has no automated fetch step).
DEFAULT_RAW_GPKG_PATH = "data/river_network/raw/bho_2017_v_01_05_5k.gpkg"
# Matches load_trenches()'s own default `layer` -- duplicated here (rather
# than imported from core.py) so verification's raw-artifact check doesn't
# have to import the whole RiverNetwork class just for this string.
DEFAULT_RAW_GPKG_TRENCHES_LAYER = "pgh_output.geoft_bho_trecho_drenagem"

TRENCHES_FILENAME = "river_trenches.parquet"
DRAINAGE_AREAS_FILENAME = "drainage_areas.parquet"
SYSTEM_MATRICES_FILENAME = "river_system_matrices.pkl"
TRENCH_ADM2_TABLE_FILENAME = "trench_adm2_matches.parquet"
ADM2_DOMINANT_SYSTEM_TABLE_FILENAME = "adm2_dominant_systems.parquet"
DEFAULT_ADM2_LAYER = "ADM_ADM_2"
BRAZIL_PROJECTED_CRS = 5641

TRENCH_ID_COLUMN = "trench_id"
SYSTEM_ID_KEY = "system_id"
NODE_ID_INDEX_NAME = "node_id"
UPSTREAM_NODE_COLUMN = "upstream_node"
DOWNSTREAM_NODE_COLUMN = "downstream_node"
DISTANCE_COLUMN = "distance"
ESTUARY_DISTANCE_COLUMN = "estuary_distance"
UPSTREAM_NODE_INDEX_COLUMN = "upstream_node_index"
DOWNSTREAM_NODE_INDEX_COLUMN = "downstream_node_index"
TRENCH_INDEX_COLUMN = "trench_index"
NODE_REACHABILITY_KEY = "node_reachability_matrices"
NODE_DISTANCE_KEY = "node_distance_matrices"
TRENCH_REACHABILITY_KEY = "trench_reachability_matrices"
TRENCH_DISTANCE_KEY = "trench_distance_matrices"
ADM2_COLUMN = "adm2"
