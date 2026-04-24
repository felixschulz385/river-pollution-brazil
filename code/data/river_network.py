"""
River network data processing module.

This module loads Brazilian hydrography data, derives disconnected river
systems, and persists a compact on-disk representation used by downstream
pipelines such as land-cover aggregation.

Saved output files
------------------
`river_trenches.parquet`
    River trench records, including the canonical per-system matrix index
    columns used to interpret `river_system_matrices.pkl`.
`drainage_areas.parquet`
    Drainage polygons deduplicated to one row per `trench_id`.
`trench_adm2_matches.parquet`
    Exploded trench-to-ADM2 join table with one row per intersecting match.
`river_system_matrices.pkl`
    Matrix bundle keyed by `system_id`. The row and column ordering metadata
    is not duplicated in the pickle; it is derived from `river_trenches.parquet`.

Matrix semantics
----------------
Node matrices are ordered by the per-system node indices stored indirectly via
`river_trenches.parquet` (`upstream_node_index` and `downstream_node_index`).
For a node matrix entry `(i, j)`, row `i` is an upstream node and column `j` is
the downstream node reachable from it.

Trench matrices are ordered by `trench_index` within each `system_id`. Trench
matrix rows are the queried downstream trench and columns are candidate
upstream trenches. `trench_reachability_matrices[system_id][i, j] == 1` means
that trench `j` can drain into trench `i`, and
`trench_distance_matrices[system_id][i, j]` stores the along-network distance
from upstream trench `j` to downstream trench `i`.

Downstream consumers such as `code.data.land_cover.assembly` therefore look up a
target trench's `system_id` and `trench_index`, read that sparse matrix row, map
the returned column indices back to trench identifiers using
`river_trenches.parquet`, and bucket the resulting upstream distances.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from scipy.sparse import csr_matrix
from pathlib import Path
from typing import Optional
import pickle
import logging


TRENCHES_FILENAME = "river_trenches.parquet"
DRAINAGE_AREAS_FILENAME = "drainage_areas.parquet"
SYSTEM_MATRICES_FILENAME = "river_system_matrices.pkl"
TRENCH_ADM2_TABLE_FILENAME = "trench_adm2_matches.parquet"
DEFAULT_GADM_PATH = "/scicore/home/meiera/schulz0022/projects/river-pollution-brazil/data/gadm/gadm41_BRA.gpkg"
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


logger = logging.getLogger(__name__)


class RiverNetwork:
    """
    Container for river network data with disk storage utilities.
    
    Attributes
    ----------
    trenches : gpd.GeoDataFrame
        River trench (segment) data
    drainage_areas : gpd.GeoDataFrame
        Drainage area polygons
    node_reachability_matrices : dict
        Node reachability matrices keyed by `system_id`
    node_distance_matrices : dict
        Node distance matrices keyed by `system_id`
    trench_reachability_matrices : dict
        Trench reachability matrices keyed by `system_id`
    trench_distance_matrices : dict
        Trench distance matrices keyed by `system_id`
    """
    
    def __init__(self):
        """Initialize empty river network."""
        self.trenches = None
        self.drainage_areas = None
        self.trench_adm2_table = None
        self.node_reachability_matrices = {}
        self.node_distance_matrices = {}
        self.trench_reachability_matrices = {}
        self.trench_distance_matrices = {}

    def _deduplicate_drainage_areas(self) -> None:
        """Keep only the first drainage polygon for each trench_id."""
        if self.drainage_areas is not None:
            self.drainage_areas = self.drainage_areas.drop_duplicates(
                subset=[TRENCH_ID_COLUMN],
                keep="first",
            )

    def _deduplicate_trenches(self) -> None:
        """Keep only the first trench record for each trench_id."""
        if self.trenches is not None:
            self.trenches = self.trenches.drop_duplicates(
                subset=[TRENCH_ID_COLUMN],
                keep="first",
            )

    def _require_trenches(self) -> gpd.GeoDataFrame:
        """Return trenches or raise a clear error when unavailable."""
        if self.trenches is None:
            raise ValueError("Trenches data not loaded. Call load_trenches() first.")
        return self.trenches

    def _empty_trench_adm2_table(self) -> pd.DataFrame:
        """Return an empty exploded trench-to-ADM2 join table."""
        return pd.DataFrame(columns=[TRENCH_ID_COLUMN, ADM2_COLUMN])

    def _build_directed_graph(self) -> nx.DiGraph:
        """Build the directed flow graph from the trench table."""
        trenches = self._require_trenches()
        graph = nx.DiGraph()
        # Every trench encodes a single directed connection in the flow graph.
        graph.add_edges_from(
            trenches[[UPSTREAM_NODE_COLUMN, DOWNSTREAM_NODE_COLUMN]].itertuples(
                index=False,
                name=None,
            )
        )
        return graph

    def _build_node_estuary_distances(self) -> dict[int, float]:
        """Map each node identifier to its estuary distance."""
        trenches = self._require_trenches()
        node_estuary_distances = {}
        for row in trenches[
            [
                UPSTREAM_NODE_COLUMN,
                DOWNSTREAM_NODE_COLUMN,
                DISTANCE_COLUMN,
                ESTUARY_DISTANCE_COLUMN,
            ]
        ].itertuples(index=False):
            upstream_node, downstream_node, trench_distance, estuary_distance = row
            # Downstream-node distances come directly from the source data.
            node_estuary_distances[downstream_node] = estuary_distance
            # Upstream-node distances are inferred by adding the trench length.
            node_estuary_distances.setdefault(
                upstream_node,
                estuary_distance + trench_distance,
            )
        return node_estuary_distances

    def _system_components(self, graph: nx.DiGraph) -> list[set]:
        """Return undirected connected components for the flow graph."""
        return list(nx.connected_components(graph.to_undirected()))

    def _assign_system_ids_from_components(self, components: list[set]) -> None:
        """Write one `system_id` per trench based on the upstream node component."""
        node_to_system = {
            node_id: system_id
            for system_id, component in enumerate(components)
            for node_id in component
        }
        trenches = self._require_trenches().copy()
        trenches[SYSTEM_ID_KEY] = trenches[UPSTREAM_NODE_COLUMN].map(node_to_system).astype(int)
        self.trenches = trenches

    def _initialize_matrix_index_columns(self) -> None:
        """Reset matrix index columns before recomputing system matrices."""
        trenches = self._require_trenches().copy()
        for column in (
            UPSTREAM_NODE_INDEX_COLUMN,
            DOWNSTREAM_NODE_INDEX_COLUMN,
            TRENCH_INDEX_COLUMN,
        ):
            trenches[column] = pd.Series(pd.NA, index=trenches.index, dtype="Int64")
        self.trenches = trenches

    def _update_system_matrix_indices(
        self,
        system_id: int,
        system_trenches: pd.DataFrame,
        node_positions: dict[int, int],
    ) -> None:
        """Persist canonical per-system matrix indices onto the trench table."""
        trenches = self._require_trenches().copy()
        ordered_trenches = system_trenches.copy()
        ordered_trenches[UPSTREAM_NODE_INDEX_COLUMN] = (
            ordered_trenches[UPSTREAM_NODE_COLUMN].map(node_positions).astype(np.int64)
        )
        ordered_trenches[DOWNSTREAM_NODE_INDEX_COLUMN] = (
            ordered_trenches[DOWNSTREAM_NODE_COLUMN].map(node_positions).astype(np.int64)
        )
        ordered_trenches[TRENCH_INDEX_COLUMN] = np.arange(len(ordered_trenches), dtype=np.int64)

        trenches.loc[ordered_trenches.index, SYSTEM_ID_KEY] = int(system_id)
        trenches.loc[
            ordered_trenches.index,
            [
                UPSTREAM_NODE_INDEX_COLUMN,
                DOWNSTREAM_NODE_INDEX_COLUMN,
                TRENCH_INDEX_COLUMN,
            ],
        ] = ordered_trenches[
            [
                UPSTREAM_NODE_INDEX_COLUMN,
                DOWNSTREAM_NODE_INDEX_COLUMN,
                TRENCH_INDEX_COLUMN,
            ]
        ].to_numpy()
        self.trenches = trenches

    def _ordered_trenches_for_system(self, system_id: int) -> pd.DataFrame:
        """Return trenches for one system in canonical trench-matrix order."""
        trenches = self._require_trenches()
        required_columns = {SYSTEM_ID_KEY, TRENCH_INDEX_COLUMN}
        missing_columns = required_columns.difference(trenches.columns)
        if missing_columns:
            raise ValueError(
                "Trench matrix indices are missing required columns: "
                f"{sorted(missing_columns)}. Call compute_distance_matrices() first."
            )

        system_trenches = trenches.loc[
            trenches[SYSTEM_ID_KEY] == system_id
        ].sort_values(TRENCH_INDEX_COLUMN)

        if len(system_trenches) == 0:
            return system_trenches

        trench_indices = system_trenches[TRENCH_INDEX_COLUMN].to_numpy(dtype=np.int64)
        expected_indices = np.arange(len(system_trenches), dtype=np.int64)
        if not np.array_equal(trench_indices, expected_indices):
            raise ValueError(
                f"Trench matrix indices for system {system_id} are not contiguous."
            )
        return system_trenches

    def _ordered_nodes_for_system(self, system_id: int) -> pd.DataFrame:
        """Return node identifiers for one system in canonical node-matrix order."""
        system_trenches = self._ordered_trenches_for_system(system_id)
        if len(system_trenches) == 0:
            return pd.DataFrame(columns=[NODE_ID_INDEX_NAME, "node_index"])

        upstream_nodes = system_trenches[
            [UPSTREAM_NODE_COLUMN, UPSTREAM_NODE_INDEX_COLUMN]
        ].rename(
            columns={
                UPSTREAM_NODE_COLUMN: NODE_ID_INDEX_NAME,
                UPSTREAM_NODE_INDEX_COLUMN: "node_index",
            }
        )
        downstream_nodes = system_trenches[
            [DOWNSTREAM_NODE_COLUMN, DOWNSTREAM_NODE_INDEX_COLUMN]
        ].rename(
            columns={
                DOWNSTREAM_NODE_COLUMN: NODE_ID_INDEX_NAME,
                DOWNSTREAM_NODE_INDEX_COLUMN: "node_index",
            }
        )
        ordered_nodes = (
            pd.concat([upstream_nodes, downstream_nodes], ignore_index=True)
            .drop_duplicates(subset=["node_index"], keep="first")
            .sort_values("node_index")
            .reset_index(drop=True)
        )

        expected_indices = np.arange(len(ordered_nodes), dtype=np.int64)
        actual_indices = ordered_nodes["node_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(actual_indices, expected_indices):
            raise ValueError(f"Node matrix indices for system {system_id} are not contiguous.")
        return ordered_nodes

    def get_system_trench_ids(self, system_id: int) -> np.ndarray:
        """Return trench identifiers in the saved trench-matrix order."""
        system_trenches = self._ordered_trenches_for_system(system_id)
        return system_trenches[TRENCH_ID_COLUMN].to_numpy(dtype=np.int64)

    def get_system_node_ids(self, system_id: int) -> np.ndarray:
        """Return node identifiers in the saved node-matrix order."""
        ordered_nodes = self._ordered_nodes_for_system(system_id)
        return ordered_nodes[NODE_ID_INDEX_NAME].to_numpy(dtype=np.int64)
    
    def load_trenches(
        self,
        gpkg_path: str,
        bbox: Optional[gpd.GeoSeries] = None,
        layer: str = "pgh_output.geoft_bho_trecho_drenagem"
    ) -> None:
        """
        Load river trench (segment) data from GeoPackage.
        
        Parameters
        ----------
        gpkg_path : str
            Path to the GeoPackage file
        bbox : gpd.GeoSeries, optional
            Bounding box for spatial filtering
        layer : str
            Layer name in the GeoPackage
        """
        logger.info("Loading trenches from %s layer %s", gpkg_path, layer)
        trenches = gpd.read_file(gpkg_path, bbox=bbox, layer=layer)
        
        # filter out coastline
        trenches = trenches[~trenches.nuordemcda.isna()]
        
        trenches = trenches[[
            "cotrecho", "nodestino", "noorigem",
            "nucomptrec", "nudistbact", "geometry"
        ]].copy()
        
        trenches.rename(
            columns={
                "cotrecho": TRENCH_ID_COLUMN,
                "nodestino": DOWNSTREAM_NODE_COLUMN,
                "noorigem": UPSTREAM_NODE_COLUMN,
                "nucomptrec": DISTANCE_COLUMN,
                "nudistbact": ESTUARY_DISTANCE_COLUMN,
            },
            inplace=True
        )
        
        self.trenches = trenches
        self._deduplicate_trenches()
        logger.info("Loaded %d deduplicated trenches", len(self.trenches))
    
    def load_drainage_areas(
        self,
        gpkg_path: str,
        bbox: Optional[gpd.GeoSeries] = None,
        layer: str = "pgh_output.geoft_bho_area_drenagem"
    ) -> None:
        """
        Load drainage area polygons from GeoPackage.
        
        Parameters
        ----------
        gpkg_path : str
            Path to the GeoPackage file
        bbox : gpd.GeoSeries, optional
            Bounding box for spatial filtering
        layer : str
            Layer name in the GeoPackage
        """
        logger.info("Loading drainage areas from %s layer %s", gpkg_path, layer)
        drainage = gpd.read_file(gpkg_path, bbox=bbox, layer=layer)
        
        drainage = drainage[["cotrecho", "nuareacont", "geometry"]].rename(
            columns={
                "nuareacont": "drainage_area",
                "cotrecho": TRENCH_ID_COLUMN,
            }
        )
        
        self.drainage_areas = drainage
        self._deduplicate_drainage_areas()
        logger.info("Loaded %d deduplicated drainage areas", len(self.drainage_areas))

    def compute_subsystems(self) -> None:
        """
        Compute subsystem IDs for disconnected river networks.
        
        This method identifies separate river systems by finding connected
        components in the undirected version of the river network graph.
        Adds 'system_id' column to trenches.
        """
        graph = self._build_directed_graph()
        components = self._system_components(graph)
        logger.info("Computed %d disconnected river systems", len(components))
        self._assign_system_ids_from_components(components)
    
    def compute_distance_matrices(self) -> None:
        """
        Compute reachability and distance matrices for each river subsystem.

        For each subsystem, computes:
        - node reachability: 1 if upstream node `i` can reach downstream node `j`
        - node distance: along-network distance from node `i` to node `j`
        - trench reachability: 1 if trench `j` drains into downstream trench `i`
        - trench distance: along-network distance from upstream trench `j` to
          downstream trench `i`

        The trench matrices intentionally use the orientation expected by
        downstream lookup code: callers take the row indexed by a target
        trench's `trench_index` and interpret the nonzero columns as all
        reachable upstream trenches in that same system.

        Populates both node-level and trench-level reachability/distance
        matrices together with their identifier orders written back to
        `self.trenches`.
        """
        logger.info("Computing node and trench matrices")
        graph = self._build_directed_graph()
        node_estuary_dist = self._build_node_estuary_distances()
        components = self._system_components(graph)
        self._assign_system_ids_from_components(components)

        self.node_reachability_matrices = {}
        self.node_distance_matrices = {}
        self.trench_reachability_matrices = {}
        self.trench_distance_matrices = {}
        self._initialize_matrix_index_columns()

        trenches = self._require_trenches()

        for system_id, component in enumerate(components):
            logger.info("Computing matrices for system %d with %d nodes", system_id, len(component))
            subgraph = graph.subgraph(component)
            nodes = list(subgraph.nodes())

            # Create sparse adjacency matrix (memory efficient)
            adj_sparse = nx.to_scipy_sparse_array(subgraph, nodelist=nodes, format='csr')
            
            # Use transitive closure to compute reachability.
            reach_sparse = adj_sparse.copy()
            # Compute transitive closure by matrix squaring
            reach_result = reach_sparse
            for _ in range(int(np.log2(len(nodes))) + 1):
                new_reach = reach_result + reach_result @ reach_sparse
                new_reach.data = np.where(new_reach.data > 0, 1, 0)
                if new_reach.nnz == reach_result.nnz:
                    break
                reach_result = new_reach.asformat('csr')
            
            reach_sparse = (reach_result > 0).astype(int)
            self.node_reachability_matrices[system_id] = reach_sparse
            
            # Reachability tells us which downstream paths exist. Distance is then
            # just the estuary-distance difference along those valid paths.
            rows, cols = reach_sparse.nonzero()
            distances = np.array([
                node_estuary_dist[nodes[i]] - node_estuary_dist[nodes[j]]
                for i, j in zip(rows, cols)
            ])
            dist_matrix_sparse = csr_matrix(
                (distances, (rows, cols)), 
                shape=reach_sparse.shape
            )
            
            self.node_distance_matrices[system_id] = dist_matrix_sparse

            system_trenches = trenches.loc[trenches[SYSTEM_ID_KEY] == system_id].copy()
            system_trenches = system_trenches.sort_values(
                [ESTUARY_DISTANCE_COLUMN, TRENCH_ID_COLUMN],
                ascending=[False, True],
            )
            node_positions = {node_id: idx for idx, node_id in enumerate(nodes)}
            self._update_system_matrix_indices(system_id, system_trenches, node_positions)
            system_trenches = self._ordered_trenches_for_system(system_id)
            trenches = self._require_trenches()

            if len(system_trenches) == 0:
                shape = (0, 0)
                self.trench_reachability_matrices[system_id] = csr_matrix(
                    shape, dtype=np.int8
                )
                self.trench_distance_matrices[system_id] = csr_matrix(
                    shape, dtype=np.float64
                )
                continue

            upstream_positions = system_trenches[UPSTREAM_NODE_INDEX_COLUMN].to_numpy(
                dtype=np.int64
            )
            downstream_positions = system_trenches[DOWNSTREAM_NODE_INDEX_COLUMN].to_numpy(
                dtype=np.int64
            )

            trench_reachability = (
                reach_sparse[upstream_positions][:, downstream_positions]
                .transpose()
                .astype(np.int8)
                .tocsr()
            )

            trench_rows, trench_cols = trench_reachability.nonzero()
            trench_estuary_distances = system_trenches[ESTUARY_DISTANCE_COLUMN].to_numpy(
                dtype=float
            )
            trench_distances = (
                trench_estuary_distances[trench_cols] - trench_estuary_distances[trench_rows]
            )
            trench_distance = csr_matrix(
                (trench_distances, (trench_rows, trench_cols)),
                shape=trench_reachability.shape,
            )

            self.trench_reachability_matrices[system_id] = trench_reachability
            self.trench_distance_matrices[system_id] = trench_distance
        logger.info("Finished computing matrices for %d systems", len(components))

    def get_upstream_trenches(self, trench_id: int) -> pd.DataFrame:
        """
        Return all upstream trenches and distances for a downstream trench.

        Parameters
        ----------
        trench_id : int
            Trench identifier to query.

        Returns
        -------
        pd.DataFrame
            Table with columns `trench_id`, `upstream_distance`, and `system_id`.
            Includes the queried trench itself with distance 0. Internally this
            reads one row from `trench_reachability_matrices[system_id]` and
            `trench_distance_matrices[system_id]`, where the row is keyed by the
            queried trench's `trench_index` and the returned columns identify
            upstream trenches in that same saved trench order.
        """
        if self.trenches is None or TRENCH_INDEX_COLUMN not in self.trenches.columns:
            raise ValueError(
                "Trench matrices not available. Call compute_distance_matrices() first."
            )

        trench_row = self.trenches.loc[
            self.trenches[TRENCH_ID_COLUMN] == trench_id,
            [SYSTEM_ID_KEY, TRENCH_INDEX_COLUMN],
        ].drop_duplicates()
        if len(trench_row) == 0:
            raise KeyError(f"Unknown trench_id: {trench_id}")
        if len(trench_row) > 1:
            raise ValueError(f"Expected one trench row for trench_id {trench_id}.")

        target_system_id = int(trench_row.iloc[0][SYSTEM_ID_KEY])
        target_position = int(trench_row.iloc[0][TRENCH_INDEX_COLUMN])
        system_trenches = self._ordered_trenches_for_system(target_system_id)
        trench_ids_arr = system_trenches[TRENCH_ID_COLUMN].to_numpy(dtype=np.int64)
        reach_row = self.trench_reachability_matrices[target_system_id].getrow(target_position)
        dist_row = self.trench_distance_matrices[target_system_id].getrow(target_position)

        dist_lookup = dict(zip(dist_row.indices.tolist(), dist_row.data.tolist()))
        upstream_records = [
            {
                TRENCH_ID_COLUMN: int(trench_ids_arr[col_idx]),
                "upstream_distance": float(dist_lookup.get(col_idx, 0.0)),
                SYSTEM_ID_KEY: int(target_system_id),
            }
            for col_idx in reach_row.indices.tolist()
        ]

        if trench_id not in [record[TRENCH_ID_COLUMN] for record in upstream_records]:
            upstream_records.append(
                {
                    TRENCH_ID_COLUMN: int(trench_id),
                    "upstream_distance": 0.0,
                    SYSTEM_ID_KEY: int(target_system_id),
                }
            )

        upstream_df = pd.DataFrame(upstream_records)
        return upstream_df.sort_values(
            by=["upstream_distance", TRENCH_ID_COLUMN]
        ).reset_index(drop=True)

    def build_trench_adm2_table(
        self,
        gadm_path: str = DEFAULT_GADM_PATH,
        layer: str = DEFAULT_ADM2_LAYER,
        adm2_column: Optional[str] = "CC_2",
        projected_crs: int = BRAZIL_PROJECTED_CRS,
    ) -> pd.DataFrame:
        """
        Build a trench-to-ADM2 relation table.

        Parameters
        ----------
        gadm_path : str
            Path to the GADM GeoPackage.
        layer : str
            GADM layer name. Defaults to `ADM_ADM_2`.
        adm2_column : str, optional
            Boundary column to use as the ADM2 identifier. If missing, falls back
            to `GID_2`, then `CC_2`, then the boundary row index.
        projected_crs : int
            Projected CRS used for the spatial join.

        Returns
        -------
        pd.DataFrame
            Exploded join table with columns `trench_id` and `adm2`.
        """
        if self.trenches is None:
            raise ValueError("Trenches data not loaded. Call load_trenches() first.")

        logger.info("Building trench-to-ADM2 table from %s layer %s", gadm_path, layer)
        boundaries = gpd.read_file(gadm_path, layer=layer)

        if adm2_column not in boundaries.columns:
            fallback_columns = ["GID_2", "CC_2"]
            selected_adm2_column = next(
                (column for column in fallback_columns if column in boundaries.columns),
                None,
            )
            if selected_adm2_column is None:
                boundaries[ADM2_COLUMN] = boundaries.index.astype(str)
            else:
                boundaries[ADM2_COLUMN] = boundaries[selected_adm2_column]
        else:
            boundaries[ADM2_COLUMN] = boundaries[adm2_column]
        
        boundaries["geometry"] = boundaries.simplify(.01)
        
        trenches_projected = self.trenches[[TRENCH_ID_COLUMN, "geometry"]].to_crs(projected_crs)
        boundaries_projected = boundaries[[ADM2_COLUMN, "geometry"]].to_crs(projected_crs)

        joined = gpd.sjoin(
            trenches_projected,
            boundaries_projected,
            how="left",
            predicate="intersects",
        ).dropna(subset=["index_right"])

        if len(joined) == 0:
            self.trench_adm2_table = self._empty_trench_adm2_table()
        else:
            self.trench_adm2_table = (
                joined[[TRENCH_ID_COLUMN, ADM2_COLUMN]]
                .drop_duplicates()
                .reset_index(drop=True)
            )

        logger.info("Computed %d trench-to-ADM2 matches", len(self.trench_adm2_table))
        
        return self.trench_adm2_table
    
    def arrange_by_systems(self) -> None:
        """
        Arrange trenches by systems and distances.
        
        Sorts the trenches data by system_id, then by estuary_distance in descending order
        (upstream to downstream). Updates trenches in-place.
        """
        if self.trenches is None:
            raise ValueError("Trenches data not loaded. Call load_trenches() first.")
        
        if 'system_id' not in self.trenches.columns:
            raise ValueError("System IDs not computed. Call compute_subsystems() first.")
        
        # Sort by system_id first, then by estuary_distance descending (upstream to downstream)
        self.trenches = self.trenches.sort_values(
            by=['system_id', 'estuary_distance'],
            ascending=[True, False]
        ).reset_index(drop=True)

    def sort_trenches_by_system(self) -> None:
        """Sort trenches from upstream to downstream within each system."""
        self.arrange_by_systems()
    
    def annotate_drainage_areas_with_country_membership(
        self,
        gadm_path: str,
        layer: str = "ADM_ADM_0",
    ) -> None:
        """
        Add a `within_brazil` flag to drainage areas using the country boundary.
        
        Parameters
        ----------
        gadm_path : str
            Path to GADM GeoPackage file
        layer : str
            Layer name for country boundary (default: "ADM_ADM_0")
        """
        if self.drainage_areas is None:
            raise ValueError("Drainage areas not loaded. Call load_drainage_areas() first.")

        logger.info("Loading country boundary from %s layer %s", gadm_path, layer)
        brazil = gpd.read_file(gadm_path, layer=layer).union_all().simplify(0.01)
        
        logger.info("Annotating %d drainage areas with country membership", len(self.drainage_areas))
        self.drainage_areas['within_brazil'] = self.drainage_areas.intersects(brazil)
        
        n_within = self.drainage_areas['within_brazil'].sum()
        logger.info(
            "Found %d/%d drainage areas within Brazil",
            n_within,
            len(self.drainage_areas),
        )
    
    def save(self, output_dir: str) -> None:
        """
        Save river network data to disk.
        
        Saves trenches, drainage areas, ADM2 matches, and matrices using
        explicit output file names.
        
        Parameters
        ----------
        output_dir : str
            Directory to save files to
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info("Saving river network outputs to %s", output_path)
        
        if self.trenches is not None:
            self.trenches.to_parquet(output_path / TRENCHES_FILENAME)

        if self.trench_adm2_table is not None:
            self.trench_adm2_table.to_parquet(output_path / TRENCH_ADM2_TABLE_FILENAME)
        
        if self.drainage_areas is not None:
            self._deduplicate_drainage_areas()
            self.drainage_areas.to_parquet(output_path / DRAINAGE_AREAS_FILENAME)
        
        matrices_data = {
            NODE_REACHABILITY_KEY: self.node_reachability_matrices,
            NODE_DISTANCE_KEY: self.node_distance_matrices,
            TRENCH_REACHABILITY_KEY: self.trench_reachability_matrices,
            TRENCH_DISTANCE_KEY: self.trench_distance_matrices,
        }
        
        with open(output_path / SYSTEM_MATRICES_FILENAME, "wb") as f:
            pickle.dump(matrices_data, f)
        logger.info("Saved river network outputs")
    
    def load(self, input_dir: str) -> None:
        """
        Load river network data from disk.
        
        Parameters
        ----------
        input_dir : str
            Directory containing saved files
        """
        input_path = Path(input_dir)
        logger.info("Loading river network outputs from %s", input_path)
        
        trenches_file = input_path / TRENCHES_FILENAME
        if trenches_file.exists():
            self.trenches = gpd.read_parquet(trenches_file)

        trench_adm2_table_file = input_path / TRENCH_ADM2_TABLE_FILENAME
        if trench_adm2_table_file.exists():
            self.trench_adm2_table = pd.read_parquet(trench_adm2_table_file)
        else:
            self.trench_adm2_table = self._empty_trench_adm2_table()
        
        drainage_file = input_path / DRAINAGE_AREAS_FILENAME
        if drainage_file.exists():
            self.drainage_areas = gpd.read_parquet(drainage_file)
            self._deduplicate_drainage_areas()
        
        matrices_file = input_path / SYSTEM_MATRICES_FILENAME
        if matrices_file.exists():
            with open(matrices_file, "rb") as f:
                matrices_data = pickle.load(f)
                self.node_reachability_matrices = matrices_data[NODE_REACHABILITY_KEY]
                self.node_distance_matrices = matrices_data[NODE_DISTANCE_KEY]
                self.trench_reachability_matrices = matrices_data[
                    TRENCH_REACHABILITY_KEY
                ]
                self.trench_distance_matrices = matrices_data[
                    TRENCH_DISTANCE_KEY
                ]
        logger.info("Loaded river network outputs")
