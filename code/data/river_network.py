import os
from functools import lru_cache, partial
import pandas as pd
import numpy as np
import geopandas as gpd
import shapely
from sklearn.cluster import MiniBatchKMeans
import multiprocessing as mp
from multiprocessing import Pool
from joblib import Parallel, delayed
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
import pickle
from scipy import sparse

def preprocess_rivers(root_dir, area):
    """
    Preprocess river data by loading from a shapefile and saving to a parquet file.
    
    Parameters
    ----------
    root_dir : str
        Root directory path for data files.
    area : str
        Area identifier (not used in current implementation).
    """
    if os.path.exists('data/misc/raw/GEOFT_BHO_REF_RIO.shp'):
        # Load the boundaries data
        rivers = gpd.read_file(root_dir + 'data/misc/raw/GEOFT_BHO_REF_RIO.shp', engine = "pyogrio")

        # Save the preprocessed data
        rivers.to_parquet(root_dir + 'data/misc/msc_rivers.parquet')

class river_network:
    """
    A class representing a river network, handling loading, processing, and analysis of river geometries and topologies.
    
    Attributes
    ----------
    shapefile : gpd.GeoDataFrame
        GeoDataFrame containing river geometries.
    boundaries : gpd.GeoDataFrame or None
        Administrative boundaries for clipping.
    coastline : gpd.GeoDataFrame or None
        Coastline geometries for estuary detection.
    edges_fl : pd.Series
        First and last vertices of each edge.
    points : np.ndarray
        Flattened array of all vertices.
    points_indices : np.ndarray
        Indices mapping points back to original edges.
    network_nodes : np.ndarray
        Nodes where multiple edges connect.
    border_nodes : np.ndarray
        Nodes at administrative boundaries.
    end_nodes : np.ndarray
        Terminal nodes (sources or estuaries).
    nodes : np.ndarray
        Combined array of all node types.
    nodes_types : np.ndarray
        Types corresponding to nodes ('network', 'border', 'end').
    topology : gpd.GeoDataFrame
        Topology data with node properties.
    distance_from_estuary : pd.DataFrame
        Distance data indexed by river segment with columns for distance and offset.
    reachability : dict
        Dictionary mapping node IDs to sets of reachable nodes.
    """
    def __init__(self, shapefile, boundaries, coastline):
        self.shapefile = shapefile
        self.boundaries = boundaries
        self.coastline = coastline
        self.edges_fl = None
        self.points = None
        self.points_indices = None
        self.network_nodes = None
        self.border_nodes = None
        self.end_nodes = None
        self.nodes = None
        self.nodes_types = None
        self.distance_from_estuary = None
        self.reachability = None
    
    @classmethod
    def load_network(cls, shapefile_path, topology_path, distance_path=None, reachability_path=None):
        """
        Load a river network from existing shapefile and topology files.
        
        Parameters
        ----------
        shapefile_path : str
            Path to the shapefile parquet file.
        topology_path : str
            Path to the topology parquet file.
        distance_path : str, optional
            Path to the distance_from_estuary parquet file. If None, attempts to infer from
            shapefile_path directory. Default is None.
        reachability_path : str, optional
            Path to the reachability npz file. If None, attempts to infer from
            shapefile_path directory. Default is None.
        
        Returns
        -------
        river_network
            A river_network instance with loaded data.
        """
        # Create an instance with minimal initialization
        instance = cls.__new__(cls)
        instance.shapefile = gpd.read_parquet(shapefile_path)
        instance.topology = gpd.read_parquet(topology_path)
        instance.boundaries = None
        instance.coastline = None
        instance.edges_fl = None
        instance.points = None
        instance.points_indices = None
        instance.network_nodes = None
        instance.border_nodes = None
        instance.end_nodes = None
        instance.nodes = None
        instance.nodes_types = None
        instance.distance_from_estuary = None
        instance.reachability = None
        
        # Infer base directory from shapefile_path if not provided
        base_dir = os.path.dirname(shapefile_path)
        if not base_dir.endswith(os.sep):
            base_dir += os.sep
        
        # Load optional distance data
        if distance_path is None:
            distance_path = base_dir + "distance_from_estuary.parquet"
        
        if distance_path and os.path.exists(distance_path):
            df = pd.read_parquet(distance_path)
            # Restore multi-index if index columns are present
            index_cols = [c for c in df.columns if c.startswith('level_') or c == 'index']
            if index_cols:
                df = df.set_index(index_cols)
            instance.distance_from_estuary = df
        
        # Load optional reachability data
        if reachability_path is None:
            reachability_path = base_dir
        
        if reachability_path and os.path.exists(reachability_path + "reachability.npz"):
            instance.reachability = instance.load_reachability(reachability_path)
        
        return instance
    
    @staticmethod
    def load_reachability(reachability_path):
        """
        Load reachability data from a sparse matrix npz file.
        
        Parameters
        ----------
        reachability_path : str
            Path to the reachability directory (without filename).
        
        Returns
        -------
        dict
            Dictionary containing 'matrix' (sparse CSR matrix of distances) and 
            'edge_ids' (array mapping matrix indices to edge IDs).
        """
        # Ensure path ends with separator
        if not reachability_path.endswith(os.sep):
            reachability_path += os.sep
        
        # Load sparse matrix
        matrix_path = reachability_path + "reachability.npz"
        matrix = sparse.load_npz(matrix_path)
        
        # Load edge_ids
        indices_path = reachability_path + "reachability_indices.npy"
        edge_ids = np.load(indices_path, allow_pickle=True)
        
        return {
            'matrix': matrix,
            'edge_ids': edge_ids,
        }
                
    def explode_rivers(self):
        """
        Explode MultiLineString geometries into individual LineString segments.
        """
        self.shapefile = self.shapefile.explode(index_parts=True)
        #self.shapefile.index = self.shapefile.index.map(lambda x: x[0] * 100 + x[1])
        
    def update_vertices(self):
        """
        Update vertices by extracting coordinates from geometries.
        Computes first/last vertices, flattened points, and indices.
        """
        print("*** Updating vertices ***")
        # get vertices as numpy array of line coordinates
        vertices = self.shapefile.geometry.map(lambda x: np.array(x.coords))
        # get first and last vertex of each line
        self.edges_fl = self.shapefile.geometry.map(lambda x: np.array(x.coords)[[0,-1]])
        # get indices of flat list of vertices
        self.points_indices = np.repeat(vertices.index, vertices.map(len))
        # flatten vertices
        self.points = np.vstack(vertices)
        
    def update_nodes(self):
        """
        Combine all node types into unified arrays.
        """
        self.nodes = np.concatenate([self.network_nodes, self.border_nodes, self.end_nodes])
        self.nodes_types = np.concatenate([np.repeat("network", len(self.network_nodes)), np.repeat("border", len(self.border_nodes)), np.repeat("end", len(self.end_nodes))])
        
    def update_network_nodes(self):
        """
        Identify network nodes as points appearing more than once.
        """
        print("*** Updating network nodes ***")
        # get unique points and their counts
        points_counts = np.unique(self.points, return_counts = True, axis = 0)
        # get network nodes (points with more than one occurrence)
        network_nodes = points_counts[0][points_counts[1] > 1]
        # combine nodes and create type array
        self.network_nodes = network_nodes
        
    def update_border_nodes(self):
        """
        Identify border nodes by finding intersections with administrative boundaries using spatial clustering and multiprocessing.
        """
        print("*** Updating border nodes ***")
        ## split up the shapes into smaller chunks of 10000 geometries
        #   (any speed up here requires the order of geometries in the shapefile to correspond to proximity in space
        #   hence we cluster the geometries to obtain cluster with spatial proximity)
        # extract representative points
        representative_points = self.shapefile.centroid
        coords = np.array(list(zip(representative_points.x, representative_points.y)))
        # perform spatial clustering
        mbkmeans = MiniBatchKMeans(n_clusters=((self.shapefile.shape[0] // 1000) + 1), batch_size=10000, random_state=42)
        cluster_labels = mbkmeans.fit_predict(coords)
        # split the shapefile into chunks
        chunks = [self.shapefile[cluster_labels == i].copy() for i in range(mbkmeans.n_clusters)]

        # a function to expand bounds by a factor
        def expand_bounds(bounds, factor = 1.1):
            return [bounds[0] - (bounds[2] - bounds[0]) * (factor - 1) / 2, bounds[1] - (bounds[3] - bounds[1]) * (factor - 1) / 2, bounds[2] + (bounds[2] - bounds[0]) * (factor - 1) / 2, bounds[3] + (bounds[3] - bounds[1]) * (factor - 1) / 2]
        
        # prepare data for multiprocessing
        data_for_processing = []
        #dissolved_shapefile = self.boundaries.dissolve().boundary#.geometry.iloc[0]
        for chunk in chunks:
            cbounds = expand_bounds(chunk.total_bounds)
            cboundaries_line = gpd.clip(self.boundaries.boundary, cbounds).to_frame()
            cboundaries_polygon = gpd.clip(self.boundaries.geometry, cbounds).to_frame()
            if cboundaries_line.empty:
                continue
            #cboundaries = cboundaries.dissolve().geometry.iloc[0]
            data_for_processing.append((chunk.copy(), cboundaries_line, cboundaries_polygon))

        # use Pool to process data in parallel
        #results = [add_intersections_chunk(data_for_processing[i]) for i in range(8)]
        with Pool(4) as pool:
            results = list(tqdb(pool.imap(add_intersections_chunk, data_for_processing), total=len(data_for_processing), unit="chunk"))
        
        # Post-process results
        # Replace geometries with added vertices
        self.shapefile["intersecting_boundary_ids"] = pd.NA
        self.shapefile.update(pd.concat([x[0] for x in results]))
        # Flatten list of intersections and update `self.border_nodes`
        self.border_nodes = np.concatenate([x[1] for x in results if not x[1] == []])

    def update_end_nodes(self):
        """
        Identify end nodes as points appearing only once and at line ends.
        """
        # Note: All estuary nodes are also end nodes
        print("*** Updating end nodes ***")
        # get unique points and their counts
        points_counts = np.unique(self.points, return_counts = True, axis = 0)
        # get those that only appear once
        points_ends = points_counts[0][points_counts[1] == 1]
        ## Get indices of single occurrence points as end candidates
        # Convert arrays to lists of tuples
        points_ends_hashed = [tuple(row) for row in points_ends]
        points_hashed = [tuple(row) for row in self.points]
        # Create a dictionary from elements of the points array to their indices
        dict_points = {elem: i for i, elem in enumerate(points_hashed)}
        # Find matching indices
        points_ends_indices = np.array([dict_points.get(elem, None) for elem in points_ends_hashed])
        ## Get end candidates that are at end of their respective line
        points_ends_reshaped = self.points[points_ends_indices][:, np.newaxis, :]
        edges_fl_subset = np.stack(self.edges_fl.loc[self.points_indices[points_ends_indices]])
        node_external_boolean = np.any(np.all(points_ends_reshaped == edges_fl_subset, axis=2), axis=1)
        # get ends (points with exactly one occurrence at end of line)
        self.end_nodes = self.points[points_ends_indices[node_external_boolean]]
        
    def update_estuary_nodes(self):
        """
        Identify estuary nodes as points intersecting the coastline.
        """
        # turn coastline into dictionary for faster lookup
        coastline_dict = {tuple(x): i for i, x in enumerate(np.array(self.coastline.iloc[0].geometry.coords))}
        # get indices of points that are at coastline
        points_coastline_indices = np.array([coastline_dict.get(tuple(x), np.nan) for x in self.points])
        # get estuary nodes
        self.estuary_nodes = self.points[np.logical_not(np.isnan(points_coastline_indices))]
        
    def break_lines_at_nodes(self):
        """
        Break river lines at node intersections to create separate segments.
        Uses multiprocessing for efficiency.
        """
        print("*** Breaking lines at nodes ***")
        ## get indices of points that are nodes
        # create a dictionary from elements of the points array to their indices
        dict_points = defaultdict(list)
        # Iterate over pairs of corresponding tuples from both arrays
        for point, index in zip(self.points, self.points_indices):
            # Convert numpy arrays to tuples for hashing
            point_tuple = tuple(point)
            # Append the index to the list corresponding to the point
            dict_points[point_tuple].append(index)
        # Convert defaultdict to dict if necessary
        mapping_dict = dict(dict_points)
        
        # get indices of nodes with node
        vertices_w_nodes_indices = np.vstack([dict_points.get(elem, np.nan) for elem in [tuple(row) for row in self.nodes]])
        # get unique indices
        vertices_w_nodes_indices = np.unique(vertices_w_nodes_indices, axis = 0)
        # create dictionary from elements of the nodes array to True
        dict_nodes = {tuple(x): True for x in self.nodes}
        
        data_for_processing = []
        for vertice_id in vertices_w_nodes_indices:
            # get the current vertice
            cvertice = self.shapefile.loc[tuple(vertice_id)]
            #
            if type(cvertice.geometry) == shapely.geometry.linestring.LineString:
                cvertice_coordinates = cvertice.geometry.coords
            elif type(cvertice.geometry) == shapely.geometry.linestring.MultiLineString:
                cvertice_coordinates = np.concatenate([x.coords for x in cvertice.geometry.geoms])
            # get the indices of the points that are nodes
            csplit_ids = np.where([dict_nodes.get(x, 0) for x in cvertice_coordinates])[0]
            csplit_ids = csplit_ids[((csplit_ids!=0) &(csplit_ids!=(len(cvertice_coordinates)-1)))]
            # if there are nodes to split at, add to data for processing
            if (not csplit_ids.size == 0):
                data_for_processing.append((cvertice.copy(), csplit_ids))
    
        # use Pool to process data in parallel
        with Pool(4) as pool:
            results = list(tqdb(pool.imap(split_vertice, data_for_processing), total=len(data_for_processing), unit="vertices"))
        #results = [split_vertice(x) for x in data_for_processing[:1000]]
        
        # Replace geometries with added vertices
        self.shapefile.update(pd.DataFrame().from_records(results, index = [x.name for x in results])) 
                
    def clean_shapefile(self):
        """
        Clean and assign topology attributes to the shapefile, including node IDs, estuary flags, and hierarchical river/segment IDs.
        Builds topology GeoDataFrame and traverses the network recursively.
        """
        # a dict from end point to river end id(s)
        rivers_ends = self.shapefile.copy().geometry.map(lambda x: tuple(np.array(x.coords)[-1]))
        rivers_ends = rivers_ends.reset_index()
        rivers_ends["index"] = rivers_ends[["level_0", "level_1", "level_2"]].apply(tuple, axis = 1)
        rivers_ends_dict = rivers_ends.groupby(rivers_ends["geometry"]).index.apply(list).to_dict()
        # a dict from river id to start point
        rivers_starts = self.shapefile.geometry.map(lambda x: tuple(np.array(x.coords)[0]))
        rivers_starts_dict = dict(zip(rivers_starts.index, rivers_starts))
        # a dict to check if a point is a river source node
        end_nodes_dict = dict(zip([tuple(x) for x in self.end_nodes], range(len(self.end_nodes))))
        # a dict to check if a point is a river network node
        network_nodes_dict = dict(zip([tuple(x) for x in self.network_nodes], range(len(self.network_nodes))))
        # a dict to check if a point is a river border node
        border_nodes_dict = dict(zip([tuple(x) for x in self.border_nodes], range(len(self.border_nodes))))
        # a dict to check if a point is a river estuary node
        coastline_dict = {tuple(x): i for i, x in enumerate(np.array(self.coastline.iloc[0].geometry.coords))}
        
        # prepare a database for nodes
        self.shapefile[["downstream_node_id", "upstream_node_id", "adm2", "estuary", "river", "segment", "subsegment"]] = np.nan
        self.topology = gpd.GeoDataFrame(index = range(len(self.nodes)), columns = ["estuary", "confluence", "source", "border", "geometry"])
        self.topology["estuary"] = np.logical_not(np.isnan(np.array([coastline_dict.get(tuple(x), np.nan) for x in self.nodes])))  # get indices of points that are at coastline
        self.topology["confluence"] = self.nodes_types == "network"
        self.topology["source"] = ((self.nodes_types == "end") & ~(self.nodes_types == "estuary"))
        self.topology["border"] = self.nodes_types == "border"
        self.topology["geometry"] = [shapely.geometry.Point(x) for x in self.nodes]
        
        # function to parse river
        def worker(edge_id, downstream_node_id, estuary, river, segment, subsegment):
            # get end point of current node
            cend_node = rivers_starts_dict.get(edge_id, [])
            # get adm2 with largest intersection
            if not self.shapefile.loc[edge_id].isna().intersecting_boundary_ids:
                if self.shapefile.loc[edge_id].intersecting_boundary_ids.size > 1:
                    cadm_2 = self.shapefile.loc[edge_id].intersecting_boundary_ids[np.argmax(shapely.intersection(self.shapefile.loc[edge_id].geometry, boundaries.loc[self.shapefile.loc[edge_id].intersecting_boundary_ids].geometry).length)]
                else:
                    cadm_2 = self.shapefile.loc[edge_id].intersecting_boundary_ids
            else:
                cadm_2 = np.nan
            # update topology
            cis_border_node = border_nodes_dict.get(cend_node, None)
            cis_network_node = network_nodes_dict.get(cend_node, None)
            cis_end_node = end_nodes_dict.get(cend_node, None)
            # if end point is a network node
            if cis_network_node is not None:
                cnodeid = cis_network_node
                self.shapefile.loc[edge_id, ["downstream_node_id", "upstream_node_id", "adm2", "estuary", "river", "segment", "subsegment"]] = downstream_node_id, cnodeid, cadm_2, estuary, river, segment, subsegment
                # get all upstream river ids
                cnext_ids = rivers_ends_dict.get(cend_node, [])
                if len(cnext_ids) == 1:
                    # get next river id
                    cnext_id = cnext_ids[0]
                    # recursively update all upstream arms
                    estuary, river, segment, subsegment = worker(cnext_id, cnodeid, estuary, river, segment, subsegment + 1)
                else:
                    # check if there are rivers with the same name
                    same_name = [((self.shapefile.loc[cnext_id].NORIOCOMP == self.shapefile.loc[edge_id].NORIOCOMP) and (self.shapefile.loc[cnext_id].NORIOCOMP is not None)) for cnext_id in cnext_ids]
                    # get first river with same name
                    # Find the index of the first True value
                    if any(same_name):
                        start_index = same_name.index(True)
                    else:
                        start_index = 0
                    # Loop over items starting from the first True item, wrapping around as needed
                    for i in range(len(cnext_ids)):
                        # Calculate the current index, wrapping around using modulo
                        current_index = (start_index + i) % len(cnext_ids)
                        # if we are at the start index, maintain river id
                        if current_index == start_index:
                            estuary, river, segment, subsegment = worker(cnext_ids[current_index], cnodeid, estuary, river, segment + 1, 0)
                        # else increase river id
                        else:
                            estuary, river, segment, subsegment = worker(cnext_ids[current_index], cnodeid, estuary, river + 1, 0, 0)
            # if end point is a border node
            elif cis_border_node is not None:
                cnodeid = len(self.network_nodes) + cis_border_node
                self.shapefile.loc[edge_id, ["downstream_node_id", "upstream_node_id", "adm2", "estuary", "river", "segment", "subsegment"]] = downstream_node_id, cnodeid, cadm_2, estuary, river, segment, subsegment
                # get next river id
                cnext_id = rivers_ends_dict.get(cend_node, [])[0]
                # recursively update all upstream arms
                estuary, river, segment, subsegment = worker(cnext_id, cnodeid, estuary, river, segment + 1, 0)
            # if end point is a end node, return
            elif cis_end_node is not None:
                cnodeid = len(self.network_nodes) + len(self.border_nodes) + cis_end_node
                self.shapefile.loc[edge_id, ["downstream_node_id", "upstream_node_id", "adm2", "estuary", "river", "segment", "subsegment"]] = downstream_node_id, cnodeid, cadm_2, estuary, river, segment, subsegment
            return estuary, river, segment, subsegment
        
        # iterate over estuary nodes
        for i in tqdm(range(len(self.estuary_nodes))):#
            init_vertice = rivers_ends_dict.get(tuple(self.estuary_nodes[i]), [])[0]
            worker(init_vertice, np.nan, i, 0, 0, 0)
            
    def calculate_distance_from_estuary(self, segment_length=1000):
        """
        Calculate the distance of river segments from the nearest estuary.
        
        This method traces the river network upstream from estuary points to compute 
        cumulative distances for each segment. Results are stored in `self.distance_from_estuary`.

        Parameters
        ----------
        segment_length : int, optional
            The base segment length used to calculate segment offsets. Default is 1000.

        Returns
        -------
        pd.DataFrame
            DataFrame with `distance_from_estuary` and `segment_offset` columns indexed by river segment.

        Notes
        -------
        - Requires `self.shapefile` and `self.topology` to be populated.
        - The function assumes all data are clean and correctly formatted.
        """
        rivers = self.shapefile
        topology = self.topology
        
        # create a dict from end point to upstream river ids
        dict_points = defaultdict(list)
        # Iterate over pairs of corresponding tuples from both arrays
        for point, index in zip(rivers.dropna(subset = "downstream_node_id").downstream_node_id.astype(np.int32), rivers.dropna(subset = "downstream_node_id").index):
            # Append the index to the list corresponding to the point
            dict_points[point].append(index)
        # Convert defaultdict to dict
        downstream_lookup = dict(dict_points)
        # create a dict from river id to end node
        upstream_lookup = {key: value for key, value in zip(rivers.dropna(subset = "upstream_node_id").index, rivers.dropna(subset = "upstream_node_id").upstream_node_id.astype(np.int32))}
        # create a dict from river id to length
        length_lookup = {key: value for key, value in zip(rivers.dropna(subset = "upstream_node_id").index, rivers.dropna(subset = "upstream_node_id").length)}
        # create a dict to check if a point is a river source node
        end_node_lookup = {x: True for x in topology[topology.source & ~topology.estuary].index}

        # get all river ids of nodes at estuaries
        estuary_ids = rivers.query("river==0 & segment==0 & subsegment==0").index.to_list()

        # prepare data for multiprocessing
        datasets = []
        for i in range(len(estuary_ids)):
            datasets.append([{key: 0 for key in rivers[rivers.estuary == rivers.loc[estuary_ids[i]].estuary].dropna(subset = "upstream_node_id").index}, (estuary_ids[i], 0)])

        # function to calculate segment offsets
        def calculate_distance_worker(dataset):
            query = [dataset[1]]
            out = dataset[0]
            while query:
                tmp = query.pop()
                out[tmp[0]] = tmp[1]
                if not end_node_lookup.get(upstream_lookup[tmp[0]], False):
                    query += [(x, ((tmp[1] + length_lookup[tmp[0]]))) for x in downstream_lookup.get(upstream_lookup[tmp[0]], [])]
            return out
        # compute segment offsets
        distance_from_estuary = [calculate_distance_worker(x) for x in datasets]
        
        # Create DataFrame from results
        result_df = pd.DataFrame()
        result_df = pd.concat([pd.DataFrame({"distance_from_estuary": x}) for x in distance_from_estuary])
        result_df["distance_from_estuary"] = pd.to_numeric(result_df.distance_from_estuary.round(0))
        result_df["segment_offset"] = pd.to_numeric(result_df.distance_from_estuary % segment_length)
        
        # Store results
        self.distance_from_estuary = result_df
        
        return result_df
    
    def compute_reachability(self, n_workers=None):
        """
        Compute reachability distances for each node in the river network.
        
        Stores results as a sparse matrix where entry (i, j) contains the 
        river distance from edge i to edge j (if reachable). Results stored 
        in `self.reachability` as a dict with 'matrix' and 'edge_ids'.
        
        Uses NetworkX for graph operations and the distance_from_estuary data
        to compute actual river distances between nodes. Processes estuaries
        in parallel using multiprocessing.
        
        Parameters
        ----------
        n_workers : int, optional
            Number of worker processes. Defaults to min(4, cpu_count).
        
        Returns
        -------
        dict
            Dictionary with 'matrix' (sparse CSR matrix) and 'edge_ids' (edge ID array).
        """
        import logging
        logger = logging.getLogger(__name__)
        
        if self.distance_from_estuary is None:
            raise ValueError("distance_from_estuary must be computed before reachability")
        
        rivers = self.shapefile
        
        # Join shapefile with distance_from_estuary to get distances per edge
        rivers_with_dist = rivers.join(self.distance_from_estuary, how="inner")
        
        estuaries = rivers_with_dist.estuary.dropna().unique()
        logger.info(f"Processing {len(estuaries)} estuaries")
        
        # Prepare data for multiprocessing - extract edges for each estuary
        node_indices = []
        estuary_data = []
        for i in estuaries:
            estuary_rivers = rivers_with_dist.query(f"estuary=={i}").dropna(subset=["downstream_node_id", "upstream_node_id"])
            node_indices.append(np.vstack(estuary_rivers.index.values))
            edge_list = estuary_rivers.loc[:, ["downstream_node_id", "upstream_node_id", "distance_from_estuary"]].values.tolist()
            # Pack edges with attributes for NetworkX
            packed_edges = [[int(a), int(b), {"idx": ix, "distance_from_estuary": c}] for ix, (a, b, c) in enumerate(edge_list)]
            estuary_data.append(packed_edges)
        
        # Determine number of workers
        if n_workers is None:
            n_workers = min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4)))
        
        logger.info(f"Using {n_workers} worker processes")
        
        # Process estuaries in parallel
        all_nodes = []
        reachability_matrices = []
        failed_estuaries = []
        
        with Pool(n_workers) as pool:
            results = list(tqdm(
                pool.imap(compute_estuary_reachability, estuary_data),
                total=len(estuary_data),
                unit="estuary"
            ))
        
        # Aggregate results - each result is (reachability_matrix, error)
        for idx, (reachability_matrix, error) in enumerate(results):
            if error:
                logger.error(f"Estuary {estuaries[idx]}: {error}")
                failed_estuaries.append((estuaries[idx], error))
            else:
                all_nodes.append(node_indices[idx])
                reachability_matrices.append(reachability_matrix)
        
        # Combine all CSR matrices using block diagonal structure
        # Each estuary's reachability is independent, so we can stack them diagonally
        if reachability_matrices:
            all_nodes = np.vstack(all_nodes)
            combined_matrix = sparse.block_diag(reachability_matrices, format='csr', dtype=np.float32)
        else:
            combined_matrix = sparse.csr_matrix((0, 0), dtype=np.float32)
        
        # Log summary
        logger.info(f"Reachability computation completed")
        logger.info(f"  Successful estuaries: {len(estuaries) - len(failed_estuaries)}/{len(estuaries)}")
        logger.info(f"  Matrix shape: {combined_matrix.shape}")
        logger.info(f"  Non-zero entries: {combined_matrix.nnz}")
        
        # Store results
        self.reachability = {
            'matrix': combined_matrix,
            'edge_ids': all_nodes
        }
        
        return self.reachability
                
    def store_network(self, path):
        """
        Store all processed network data to files.
        Calls individual store methods for each component.
        
        Parameters
        ----------
        path : str
            Directory path to save files.
        """
        # Ensure path ends with separator
        if not path.endswith(os.sep):
            path += os.sep
        
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        self.store_shapefile(path)
        self.store_topology(path)
        self.store_distance_from_estuary(path)
        self.store_reachability(path)
        self.store_pickle(path)
    
    def store_shapefile(self, path):
        """
        Store the shapefile data to a parquet file.
        
        Parameters
        ----------
        path : str
            Directory path to save file.
        """
        if not path.endswith(os.sep):
            path += os.sep
        os.makedirs(path, exist_ok=True)
        
        filepath = path + "shapefile.parquet"
        if not os.path.exists(filepath):
            self.shapefile[["NORIOCOMP", "CORIO", "downstream_node_id", "upstream_node_id", "adm2", "estuary", "river", "segment", "subsegment", "geometry"]].to_parquet(filepath)
    
    def store_topology(self, path):
        """
        Store the topology data to a parquet file.
        
        Parameters
        ----------
        path : str
            Directory path to save file.
        """
        if not path.endswith(os.sep):
            path += os.sep
        os.makedirs(path, exist_ok=True)
        
        filepath = path + "topology.parquet"
        if not os.path.exists(filepath):
            self.topology.to_parquet(filepath)
    
    def store_distance_from_estuary(self, path):
        """
        Store the distance_from_estuary data to a parquet file.
        
        Parameters
        ----------
        path : str
            Directory path to save file.
        """
        if not path.endswith(os.sep):
            path += os.sep
        os.makedirs(path, exist_ok=True)
        
        filepath = path + "distance_from_estuary.parquet"
        if self.distance_from_estuary is not None and not os.path.exists(filepath):
            # Reset index to store multi-index as columns
            df_to_save = self.distance_from_estuary.reset_index()
            df_to_save.to_parquet(filepath)
    
    def store_reachability(self, path):
        """
        Store the reachability data as a sparse matrix npz file.
        
        Parameters
        ----------
        path : str
            Directory path to save file.
        """
        if not path.endswith(os.sep):
            path += os.sep
        os.makedirs(path, exist_ok=True)
        
        if self.reachability is not None and not os.path.exists(path + "reachability.npz"):
            np.save(path + "reachability_indices.npy", self.reachability['edge_ids'])
            sparse.save_npz(path + "reachability.npz", self.reachability['matrix'])
    
    def store_pickle(self, path):
        """
        Store the entire network object as a pickle file.
        
        Parameters
        ----------
        path : str
            Directory path to save file.
        """
        if not path.endswith(os.sep):
            path += os.sep
        os.makedirs(path, exist_ok=True)
        
        filepath = path + "network.pkl"
        if not os.path.exists(filepath):
            with open(filepath, "wb") as f:
                pickle.dump(self, f)
                
def add_intersections_chunk(chunk_data):
    """
    This function add vertices to a chunk of river geometries at the intersections with boundaries.
    It is designed to be used with Pool.map() to process data in parallel.
    
    Parameters
    ----------
    chunk_data : tuple
        A tuple containing a GeoDataFrame of river geometries and a GeoSeries of boundaries.
    
    Returns
    -------
    oshapefile : GeoDataFrame
        A GeoDataFrame containing the river geometries with added vertices.
    ointersections : list
        A list of tuples containing the coordinates of the added vertices.
    """
    chunk, cboundaries_line, cboundaries_polygon = chunk_data
    chunk["intersecting_boundary_ids"] = pd.NA
    ointersections = [None] * len(chunk)
    # get intersections with boundaries
    #cintersections = chunk.intersection(cboundaries)
    # iterate over chunk
    for j in range(len(chunk)):
        # get intersections with boundaries for current geometry
        ccintersections_boundary = gpd.GeoSeries(shapely.intersection(chunk.iloc[j].geometry, cboundaries_line)[0])#cintersections.iloc[j]
        ccintersections_full = gpd.GeoSeries(shapely.intersection(chunk.iloc[j].geometry, cboundaries_polygon)["geometry"])
        # if there are intersections
        if not all(ccintersections_full.is_empty):
            # write all intersecting boundary ids to shapefile
            chunk.at[chunk.index[j], "intersecting_boundary_ids"] = ccintersections_full[~ccintersections_full.is_empty].index.values
        if not all(ccintersections_boundary.is_empty):
            cintersections = []
            # extract points from MultiPoint geometries
            for i in range(len(ccintersections_boundary[~ccintersections_boundary.is_empty])):
                if ccintersections_boundary[~ccintersections_boundary.is_empty].iloc[i].geom_type == "MultiPoint":
                    cintersections += [list(ccintersections_boundary[~ccintersections_boundary.is_empty].iloc[i].geoms)]
                if ccintersections_boundary[~ccintersections_boundary.is_empty].iloc[i].geom_type == "Point":
                    cintersections += [[ccintersections_boundary[~ccintersections_boundary.is_empty].iloc[i]]]
            # create a GeoDataFrame of line points and intersections
            cvertice_points = pd.concat([
                gpd.GeoDataFrame(dict(geometry = [shapely.geometry.Point(x) for x in chunk.iloc[j].geometry.coords], type = "internal")),
                gpd.GeoDataFrame(dict(geometry = np.concatenate(cintersections), type = "external"))
            ], ignore_index = True)
            ## sort points by distance along line
            # add distance along line to points
            # credit: https://gis.stackexchange.com/a/413748
            cvertice_points['dist_along_line'] = cvertice_points['geometry'].apply(lambda p: chunk.iloc[j].geometry.project(p))
            # sort points by distance along line
            cvertice_points = cvertice_points.sort_values(by='dist_along_line')
            # write back to shapefile
            chunk.loc[chunk.index[j],"geometry"] = shapely.geometry.LineString(cvertice_points['geometry'].values)
            # store border_nodes
            ointersections[j] = [x.coords[0] for x in np.concatenate(cintersections)]
    if not all([x is None for x in ointersections]):
        ointersections = np.concatenate([x for x in ointersections if x is not None])
    else:
        ointersections = []
    return chunk, ointersections

def split_vertice(split_data):
    """
    This function splits a river geometry at the nodes provided.
    It is designed to be used with Pool.map() to process data in parallel.

    Parameters
    ----------
    split_data : tuple
        A tuple containing a GeoSeries of river geometries and a numpy array of indices of nodes to split at.

    Returns
    -------
    cvertice : GeoSeries
        A GeoSeries containing the river geometry with added vertices.
    """
    cvertice, csplit_ids = split_data
    while csplit_ids.size != 0:
        # get node to split at
        if type(cvertice.geometry) == shapely.geometry.linestring.LineString:
            # get coordinates of river
            ccoords = np.array(cvertice.geometry.coords)
            # pop node to split at
            isplit = csplit_ids[0]
            csplit_ids = csplit_ids[1:]
            # split river geometry
            cvertice["geometry"] = shapely.geometry.MultiLineString([shapely.geometry.LineString(ccoords[:isplit+1]), shapely.geometry.LineString(ccoords[(isplit):])])
            # fix indices
            csplit_ids[csplit_ids > isplit] += 1
        elif type(cvertice.geometry) == shapely.geometry.multilinestring.MultiLineString:
            # get coordinates of sub-LineStrings in river
            ccoords = np.concatenate([x.coords for x in cvertice.geometry.geoms])
            # get index of sub-LineString at coord
            csubindex = np.repeat(np.arange(len(cvertice.geometry.geoms)), [len(x.coords) for x in cvertice.geometry.geoms])
            # pop node to split at
            isplit = csplit_ids[0]
            csplit_ids = csplit_ids[1:]
            # fix indices
            csplit_ids[csplit_ids > isplit] += 1
            # get index of sub-LineString to split
            iisplit = csubindex[isplit]
            # get index of node in sub-LineString
            isplit -= np.sum([len(x.coords) for x in list(cvertice.geometry.geoms)[:iisplit]])
            # skip if node is at end of sub-LineString
            if (isplit == 0 or isplit == len(ccoords[csubindex==iisplit]) - 1):
                continue
            # split river geometry
            ccoords = list(cvertice.geometry.geoms)[:iisplit] + [shapely.geometry.LineString(ccoords[csubindex==iisplit][:isplit+1])] +\
                [shapely.geometry.LineString(ccoords[csubindex==iisplit][isplit:])] + list(cvertice.geometry.geoms)[iisplit+1:]
            cvertice["geometry"] = shapely.geometry.MultiLineString([shapely.geometry.LineString(x) for x in ccoords])
    return cvertice

def find_reachable_nodes_nx(G, node):
    """ Find all reachable nodes using NetworkX's optimized descendants. """
    reachable = nx.descendants(G, node)
    reachable.add(node)  # Include the node itself
    return node, reachable

def compute_estuary_reachability(edges):
    """
    Compute reachability distances for a single estuary.
    Designed for use with multiprocessing Pool.
    
    Parameters
    ----------
    edges : list
        List of edges with attributes for NetworkX graph.
    
    Returns
    -------
    tuple
        (reachability_matrix, error) where reachability_matrix is a scipy.sparse CSR matrix
    """
    
    import numpy as np
    from scipy.sparse import coo_array
    from collections import defaultdict
    import functools
    
    try:
        if len(edges) == 0:
            return None, ""
        
        # Create directed graph (upstream -> downstream direction for reachability)
        G = nx.DiGraph()
        G.add_edges_from(edges)

        # Extract edge list and build mappings
        edge_list = list(G.edges)
        E = len(edge_list)

        edge_idx = np.empty(E, dtype=int)
        edge_dist = np.empty(E, dtype=float)

        # Extract edge indices and distances by position
        for pos, e in enumerate(edge_list):
            edge_idx[pos] = G.edges[e]["idx"]
            edge_dist[pos] = G.edges[e]["distance_from_estuary"]

        # Map nodes to positions in edge_list
        node_edges_pos = defaultdict(list)
        for pos, (u, v) in enumerate(edge_list):
            node_edges_pos[u].append(pos)

        # Precompute descendants for all nodes
        node_descendants = {u: nx.descendants(G, u) for u in G.nodes}

        # Cached function to get downstream edge positions for a given edge position
        @functools.lru_cache(maxsize=None)
        def downstream_edge_positions(pos):
            src_node = edge_list[pos][0]
            result = []

            # Edges from src_node itself
            for p in node_edges_pos.get(src_node, ()):
                result.append(p)

            # Edges from descendant nodes
            for dn in node_descendants.get(src_node, ()):
                for p in node_edges_pos.get(dn, ()):
                    result.append(p)

            return tuple(result)

        # Build sparse matrix in COO format
        rows = []
        cols = []
        data = []

        for pos in range(E):
            row = edge_idx[pos]
            base_dist = edge_dist[pos]

            for up_pos in downstream_edge_positions(pos):
                col = edge_idx[up_pos]
                rows.append(row)
                cols.append(col)
                data.append(edge_dist[up_pos] - base_dist)

        # Create sparse matrix with appropriate shape to accommodate max edge index
        n = int(max(edge_idx)) + 1 if E > 0 else 0
        reachability_matrix = coo_array((data, (rows, cols)), shape=(n, n)).tocsr()
                    
        return reachability_matrix, ""
        
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"