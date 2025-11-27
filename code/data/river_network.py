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
import shelve

def preprocess_rivers(root_dir, area):
    if os.path.exists('data/misc/raw/GEOFT_BHO_REF_RIO.shp'):
        # Load the boundaries data
        rivers = gpd.read_file(root_dir + 'data/misc/raw/GEOFT_BHO_REF_RIO.shp', engine = "pyogrio")

        # Save the preprocessed data
        rivers.to_feather(root_dir + 'data/misc/msc_rivers.feather')

class river_network:
    """
    A class representing a river network.
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
                
    def explode_rivers(self):
        self.shapefile = self.shapefile.explode(index_parts=True)
        #self.shapefile.index = self.shapefile.index.map(lambda x: x[0] * 100 + x[1])
        
    def update_vertices(self):
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
        self.nodes = np.concatenate([self.network_nodes, self.border_nodes, self.end_nodes])
        self.nodes_types = np.concatenate([np.repeat("network", len(self.network_nodes)), np.repeat("border", len(self.border_nodes)), np.repeat("end", len(self.end_nodes))])
        
    def update_network_nodes(self):
        print("*** Updating network nodes ***")
        # get unique points and their counts
        points_counts = np.unique(self.points, return_counts = True, axis = 0)
        # get network nodes (points with more than one occurrence)
        network_nodes = points_counts[0][points_counts[1] > 1]
        # combine nodes and create type array
        self.network_nodes = network_nodes
        
    def update_border_nodes(self):
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
            results = list(tqdm(pool.imap(add_intersections_chunk, data_for_processing), total=len(data_for_processing), unit="chunk"))
        
        # Post-process results
        # Replace geometries with added vertices
        self.shapefile["intersecting_boundary_ids"] = pd.NA
        self.shapefile.update(pd.concat([x[0] for x in results]))
        # Flatten list of intersections and update `self.border_nodes`
        self.border_nodes = np.concatenate([x[1] for x in results if not x[1] == []])

    def update_end_nodes(self):
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
        # turn coastline into dictionary for faster lookup
        coastline_dict = {tuple(x): i for i, x in enumerate(np.array(self.coastline.iloc[0].geometry.coords))}
        # get indices of points that are at coastline
        points_coastline_indices = np.array([coastline_dict.get(tuple(x), np.nan) for x in self.points])
        # get estuary nodes
        self.estuary_nodes = self.points[np.logical_not(np.isnan(points_coastline_indices))]
        
    def break_lines_at_nodes(self):
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
            results = list(tqdm(pool.imap(split_vertice, data_for_processing), total=len(data_for_processing), unit="vertices"))
        #results = [split_vertice(x) for x in data_for_processing[:1000]]
        
        # Replace geometries with added vertices
        self.shapefile.update(pd.DataFrame().from_records(results, index = [x.name for x in results])) 
                
    def clean_shapefile(self):
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
                
    def store_network(self, path):
        self.shapefile[["NORIOCOMP", "CORIO", "downstream_node_id", "upstream_node_id", "adm2", "estuary", "river", "segment", "subsegment", "geometry"]].to_feather(path + "shapefile.feather")
        self.topology.to_feather(path + "topology.feather")
        with open(path + "network.pkl", "wb") as f:
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

def calculate_distance_from_estuary(rivers, topology, segment_length = 1000):
    """
    Calculates the distance of river segments from the nearest estuary. This function updates
    the `rivers` DataFrame by adding two new columns: `distance_from_estuary` and `segment_offset`.
    The calculation is done by tracing the river network upstream, starting from estuary points,
    to compute cumulative distances for each segment.

    Parameters:
    - rivers (pd.DataFrame): A DataFrame containing river network data. It must include columns
      for `downstream_node_id`, `upstream_node_id`, `length`, and identifying attributes for
      estuaries (e.g., 'river', 'segment', 'subsegment').
    - topology (pd.DataFrame): A DataFrame describing the topology of the network. It should
      contain Boolean columns `source` and `estuary` to identify river sources and estuaries.
    - segment_length (int, optional): The base segment length used to calculate segment offsets.
      Default value is 1000.

    Returns:
    - pd.DataFrame: The updated `rivers` DataFrame with added `distance_from_estuary` and
      `segment_offset` columns, representing the distance from the nearest estuary and the
      distance modulo the `segment_length`, respectively.

    Notes:
    - This function uses multiprocessing for calculating the distance from the estuary for each
      segment efficiently.
    - The function assumes all data are clean and correctly formatted. Missing values in the
      necessary columns may lead to failures or incorrect results.
    """
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
    def calculate_distance_from_estuary(dataset):
        query = [dataset[1]]
        out = dataset[0]
        while query:
            tmp = query.pop()
            out[tmp[0]] = tmp[1]
            if not end_node_lookup.get(upstream_lookup[tmp[0]], False):
                query += [(x, ((tmp[1] + length_lookup[tmp[0]]))) for x in downstream_lookup.get(upstream_lookup[tmp[0]], [])] # % 1000
        return out
    # compute segment offsets
    distance_from_estuary = [calculate_distance_from_estuary(x) for x in datasets]
    
    # write segment offsets to shapefile
    rivers["distance_from_estuary"] = pd.NA
    rivers.update(pd.concat([pd.DataFrame({"distance_from_estuary": x}) for x in distance_from_estuary]))
    rivers["distance_from_estuary"] = pd.to_numeric(rivers.distance_from_estuary.round(0))
    rivers["segment_offset"] = pd.to_numeric(rivers.distance_from_estuary % segment_length)
    
    return rivers


def create_graph(edge_list):
    """ Create a directed graph from a list of edges. """
    G = nx.DiGraph()
    G.add_edges_from(edge_list)
    return G


def find_reachable_nodes_nx(G, node):
    """ Find all reachable nodes using NetworkX's optimized descendants. """
    reachable = nx.descendants(G, node)
    reachable.add(node)  # Include the node itself
    return node, reachable


def compute_reachability_parallel(graph):
    """ Compute reachability for each node in the graph in parallel. """
    n_nodes = len(graph)
    
    # For small graphs, no parallelization needed
    if n_nodes < 500:
        return {node: nx.descendants(graph, node) | {node} for node in graph.nodes()}
    
    # For larger graphs, use parallel processing
    n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
    
    results = Parallel(n_jobs=n_cores, prefer="threads")(
        delayed(find_reachable_nodes_nx)(graph, node) for node in graph.nodes()
    )
    
    return dict(results)


def compute_reachability():
    # Import the river network shapefile
    rivers_brazil_shapefile = gpd.read_feather(
        "/Users/felixschulz/Library/CloudStorage/OneDrive-Personal/Dokumente/Uni/Masterthesis/data/river_network/shapefile.feather"
    )
    
    db_path = "/Users/felixschulz/Library/CloudStorage/OneDrive-Personal/Dokumente/Uni/Masterthesis/data/river_network/reachability.db"
    
    estuaries = rivers_brazil_shapefile.estuary.dropna().unique()
    
    # Open shelve once for entire computation (much more efficient)
    with shelve.open(db_path, flag='c', protocol=-1) as reachability_db:
        for i in tqdm(estuaries):
            edges = (
                rivers_brazil_shapefile
                .query(f"estuary=={i}")
                .loc[:, ["downstream_node_id", "upstream_node_id"]]
                .dropna()
                .values
            )
            
            G = create_graph(edges)
            reachability_dict = compute_reachability_parallel(G)
            
            # Batch write results
            for node, reachable_set in reachability_dict.items():
                reachability_db[str(int(node))] = reachable_set
        
        # Sync to disk at the end
        reachability_db.sync()