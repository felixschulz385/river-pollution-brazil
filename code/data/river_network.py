"""
River network data processing module.

This module provides a class for loading and processing river network data
from Brazilian hydrography datasets (BHO - Base Hidrográfica Ottocodificada).
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from scipy.sparse.csgraph import shortest_path
from pathlib import Path
from typing import Optional, Dict
import pickle


class RiverNetwork:
    """
    Container for river network data with disk storage utilities.
    
    Attributes
    ----------
    trenches : gpd.GeoDataFrame
        River trench (segment) data
    drainage_areas : gpd.GeoDataFrame
        Drainage area polygons
    reachability_matrices : dict
        Reachability matrices for each subsystem
    distance_matrices : dict
        Distance matrices for each subsystem
    node_lists : dict
        Node lists for each subsystem
    """
    
    def __init__(self):
        """Initialize empty river network."""
        self.trenches = None
        self.drainage_areas = None
        self.reachability_matrices = {}
        self.distance_matrices = {}
        self.node_lists = {}
    
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
        trenches = gpd.read_file(gpkg_path, bbox=bbox, layer=layer)
        
        trenches = trenches[[
            "cotrecho", "nodestino", "noorigem", 
            "nucomptrec", "nudistbact", "geometry"
        ]].copy()
        
        trenches.rename(
            columns={
                "cotrecho": "trench_id",
                "nodestino": "downstream_node",
                "noorigem": "upstream_node",
                "nucomptrec": "distance",
                "nudistbact": "estuary_distance"
            },
            inplace=True
        )
        
        self.trenches = trenches
    
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
        drainage = gpd.read_file(gpkg_path, bbox=bbox, layer=layer)
        
        drainage = drainage[["cotrecho", "nuareacont", "geometry"]].rename(
            columns={
                "nuareacont": "drainage_area",
                "cotrecho": "trench_id"
            }
        )
        
        self.drainage_areas = drainage
    
    def compute_subsystems(self) -> None:
        """
        Compute subsystem IDs for disconnected river networks.
        
        This method identifies separate river systems by finding connected
        components in the undirected version of the river network graph.
        Adds 'system_id' column to trenches.
        """
        if self.trenches is None:
            raise ValueError("Trenches data not loaded. Call load_trenches() first.")
        
        # Build directed graph
        G = nx.DiGraph()
        for _, row in self.trenches.iterrows():
            G.add_edge(row['upstream_node'], row['downstream_node'])
        
        # Find connected components in undirected version
        components = list(nx.connected_components(G.to_undirected()))
        
        # Create mapping from node to component ID
        node_to_component = {}
        for comp_id, component in enumerate(components):
            for node in component:
                node_to_component[node] = comp_id
        
        # Assign system_id based on upstream_node
        self.trenches['system_id'] = self.trenches.apply(
            lambda row: node_to_component.get(row['upstream_node'], -1),
            axis=1
        )
    
    def compute_distance_matrices(self) -> None:
        """
        Compute reachability and distance matrices for each river subsystem.
        
        For each subsystem, computes:
        - Reachability matrix: 1 if node i can reach node j, 0 otherwise
        - Distance matrix: Distance from node i to node j (based on estuary distance)
        
        Populates reachability_matrices, distance_matrices, and node_lists.
        """
        if self.trenches is None:
            raise ValueError("Trenches data not loaded. Call load_trenches() first.")
        
        # Build directed graph
        G = nx.DiGraph()
        for _, row in self.trenches.iterrows():
            G.add_edge(row['upstream_node'], row['downstream_node'])
        
        # Build node to estuary distance mapping
        node_estuary_dist = {}
        for _, row in self.trenches.iterrows():
            node_estuary_dist[row['downstream_node']] = row['estuary_distance']
            if row['upstream_node'] not in node_estuary_dist:
                node_estuary_dist[row['upstream_node']] = (
                    row['estuary_distance'] + row['distance']
                )
        
        # Get connected components
        components = list(nx.connected_components(G.to_undirected()))
        
        for comp_id, component in enumerate(components):
            subgraph = G.subgraph(component)
            nodes = list(subgraph.nodes())
            self.node_lists[comp_id] = nodes
            
            # Create dense adjacency matrix
            adj_dense = nx.to_numpy_array(subgraph, nodelist=nodes)
            
            # Compute shortest path distances
            dist_dense = shortest_path(
                adj_dense, directed=True, return_predecessors=False
            )
            
            # Create reachability matrix
            reach = np.where(np.isfinite(dist_dense), 1, 0)
            self.reachability_matrices[comp_id] = reach
            
            # Compute distance matrix using estuary distances
            dist_matrix = np.full_like(reach, np.inf, dtype=float)
            for i in range(len(nodes)):
                for j in range(len(nodes)):
                    if reach[i, j] == 1:
                        dist_matrix[i, j] = (
                            node_estuary_dist[nodes[i]] - node_estuary_dist[nodes[j]]
                        )
            
            self.distance_matrices[comp_id] = dist_matrix
    
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
    
    def save(self, output_dir: str) -> None:
        """
        Save river network data to disk.
        
        Saves trenches, drainage areas, and matrices to separate files.
        
        Parameters
        ----------
        output_dir : str
            Directory to save files to
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.trenches is not None:
            self.trenches.to_parquet(output_path / "trenches.parquet")
        
        if self.drainage_areas is not None:
            self.drainage_areas.to_parquet(output_path / "drainage_areas.parquet")
        
        matrices_data = {
            "reachability_matrices": self.reachability_matrices,
            "distance_matrices": self.distance_matrices,
            "node_lists": self.node_lists
        }
        
        with open(output_path / "matrices.pkl", "wb") as f:
            pickle.dump(matrices_data, f)
    
    def load(self, input_dir: str) -> None:
        """
        Load river network data from disk.
        
        Parameters
        ----------
        input_dir : str
            Directory containing saved files
        """
        input_path = Path(input_dir)
        
        trenches_file = input_path / "trenches.parquet"
        if trenches_file.exists():
            self.trenches = gpd.read_parquet(trenches_file)
        
        drainage_file = input_path / "drainage_areas.parquet"
        if drainage_file.exists():
            self.drainage_areas = gpd.read_parquet(drainage_file)
        
        matrices_file = input_path / "matrices.pkl"
        if matrices_file.exists():
            with open(matrices_file, "rb") as f:
                matrices_data = pickle.load(f)
                self.reachability_matrices = matrices_data["reachability_matrices"]
                self.distance_matrices = matrices_data["distance_matrices"]
                self.node_lists = matrices_data["node_lists"]
