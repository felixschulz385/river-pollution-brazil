# Standard library imports
import os
import pickle
import sqlite3
import sys
from collections import defaultdict
from itertools import chain, product
import warnings
import tempfile

# Suppress CRS warnings (buffer(0))
warnings.filterwarnings("ignore", category=UserWarning)

# Third-party imports for data handling and computation
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
from tqdm import tqdm
import sparse

# Geospatial and image processing libraries
import cv2
import shapely
from pysheds.grid import Grid
from rasterio.io import MemoryFile
from rasterio.features import rasterize

# Multiprocessing imports
from multiprocessing import Pool

# Local module imports for specific functionality
sys.path.append("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/code")
from data.river_network import river_network, calculate_distance_from_estuary

# ===========================================================
# Fix ANA drainage polygons
# ===========================================================

def build_graph(edges):
    graph = {}
    for u, v in edges:
        if u not in graph:
            graph[u] = []
        if v not in graph:
            graph[v] = []
        graph[u].append(v)
    return graph
  
def find_source_nodes(edges):
    # Initialize an empty set to keep track of all nodes that have incoming edges
    incoming = set()
    # Initialize an empty set for all nodes (including those with only outgoing edges)
    all_nodes = set()
    
    # Fill the sets based on the edges
    for u, v in edges:
        incoming.add(v)
        all_nodes.add(u)
        all_nodes.add(v)
    
    # Source nodes are all nodes that are not in the incoming set
    source_nodes = all_nodes - incoming
    return source_nodes

def get_catchment_polygon(query_node, grid, fdir, dirmap):
    """
    Generates a catchment area polygon for a given query node using specified flow direction data and direction mapping.

    Parameters:
    - query_node (shapely.geometry.Point): The query node as a shapely Point object for which the catchment area is calculated.
    - grid (pysheds.Grid): A grid object from the pysheds library that contains elevation data and supports hydrological analysis.
    - fdir (numpy.ndarray): Flow direction array where each cell points to its downhill neighbor.
    - dirmap (tuple): Mapping of flow direction values that corresponds to the D8 flow model.

    Returns:
    - shapely.geometry.Polygon: A polygon representing the catchment area transformed to geospatial coordinates.
    """
    # get catchment area
    t_catchment = grid.catchment(x=query_node.x, y=query_node.y, fdir=fdir, dirmap=dirmap, xytype='coordinate')
    # get polygons in image space
    t_polygon_image_space = cv2.findContours(np.array(t_catchment).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #
    if t_polygon_image_space[0][0].shape[0] > 2:
        t_polygon = shapely.geometry.Polygon(t_polygon_image_space[0][0].squeeze())
    else:
        t_polygon = shapely.geometry.Polygon()
    # affine transform to geospatial coordinates
    polygon_affine_space = shapely.affinity.affine_transform(t_polygon, np.array(grid.affine)[[0,1,3,4,2,5]])
    return polygon_affine_space

def calculate_separation_score(distributions):
    """
    Computes a separation score based on the balance and separation of class distributions
    across two measurement points for each class. The function evaluates the difference in
    class distributions and balances this difference by how evenly the classes are spread
    across the measurement points.

    Parameters:
    - distributions (np.ndarray): A numpy array where each row represents a class, and the
      two columns represent class counts at two different measurement points.

    Returns:
    - np.ndarray: A numpy array of separation scores for the distributions. Higher scores
      indicate a greater separation and balance across measurement points.
    """
    # Calculate the absolute difference in class distribution between measuring points for each class
    class_diffs = np.abs(distributions[:, 0] - distributions[:, 1])
    # Calculate the balance for each class across measuring points
    total = distributions.sum(axis=1, keepdims=True) + 1e-9
    balances = 1 - np.abs(distributions / total - 0.5)
    # Combine the separation and balance scores
    separation_scores = class_diffs.mean(axis=0)
    balance_scores = balances.mean(axis=(0, 1))
    # Overall score considers both separation and balance, adjusting weights if necessary
    overall_scores = separation_scores * balance_scores
    return np.nan_to_num(overall_scores)

def get_upstream_drainage_polygon_ids(rivers, drainage_polygons, estuary_id, river_id, segment_id, n = 3):
    """
    Retrieves IDs of upstream drainage polygons for a given river segment, starting from the
    specified segment ID and moving upstream.

    Parameters:
    - rivers (geopandas.GeoDataFrame): GeoDataFrame containing river data.
    - drainage_polygons (geopandas.GeoDataFrame): GeoDataFrame containing drainage polygons data.
    - estuary_id (int): Identifier for the estuary.
    - river_id (int): Identifier for the river.
    - segment_id (int): Starting segment identifier for the search.
    - n (int, optional): Number of upstream IDs to return. Default is 3.

    Returns:
    - numpy.ndarray: Array of IDs for the upstream drainage polygons.
    """
    # get the next river segments
    t_river_segments = rivers.query(f"estuary=={estuary_id} & river=={river_id} & segment>={segment_id}").to_crs(4326)
    # get the drainage polygons that intersect with the river segments
    t_drainage_polygons = t_river_segments.sjoin(gpd.clip(drainage_polygons, t_river_segments.total_bounds)).sort_values("segment")
    # return the n next upstream drainage polygons
    return t_drainage_polygons.index_right.unique()[:n]

def split_polygon(rivers, drainage_polygons, c_payload):
    """
    Performs river network analysis and modifies drainage polygons based on river confluence
    points within a specified drainage polygon. This function uses several GIS operations to
    compute river segments, merge them, and potentially split the drainage polygon based on
    the river confluence analysis.

    Parameters:
    - rivers (geopandas.GeoDataFrame): GeoDataFrame containing river geometries.
    - drainage_polygons (geopandas.GeoDataFrame): GeoDataFrame containing drainage polygon data.
    - c_payload (object): Object containing necessary identifiers and data for analysis, such as estuary_ids, river_ids, and polygon_id.

    Returns:
    - list: A list of either single or multiple polygon geometries representing the split
      or modified drainage polygons based on the river confluence analysis.
    
    Notes:
    - This function relies on geopandas for spatial operations and assumes that the input data
      is in the correct coordinate reference system (CRS) and formatted appropriately.
    - Extensive use of GIS operations such as clipping, dissolving, and spatial joins is made,
      and thus the function can be resource-intensive.
    """
    # dissolve river geometries of rivers at confluence
    t_rivers = pd.concat([rivers.query(f"estuary=={estuary_id} & river=={river_id}") for estuary_id, river_id in zip(c_payload.estuary_ids, c_payload.river_ids)])
    t_rivers_dissolved = t_rivers.dissolve(["estuary", "river"]).to_crs(4326)
    # get current drainage polygon
    c_drainage_polygon = drainage_polygons.loc[c_payload.polygon_id].geometry
    
    # get the river ids of rivers that are not entirely contained in the drainage polygon
    c_river_ids_to_query = t_rivers_dissolved.index[~t_rivers_dissolved.within(c_drainage_polygon)].values
    # if all are entirely contained we have two end nodes in the drainage polygon
    if len(c_river_ids_to_query) == 0:
        c_drainage_polygons = [c_payload.polygon_id]
    else:
        # get the geometries of the rivers to query, clipped to the drainage polygon
        t_rivers_to_query = gpd.clip(pd.concat([rivers.query(f"estuary=={estuary_id} & river=={river_id}") for estuary_id, river_id in c_river_ids_to_query]).to_crs(4326), c_drainage_polygon)
        # for each unique river, get the most upstream segment
        t_rivers_to_query_max_segment = t_rivers_to_query.groupby(["estuary", "river"]).segment.max()
        # get all upstream drainage polygons to include
        c_drainage_polygons = np.unique(np.concatenate([get_upstream_drainage_polygon_ids(rivers, drainage_polygons, i[0], i[1], x) for i, x in t_rivers_to_query_max_segment.items()]))

    # load subset of height profile and turn into pysheds grid via memory file
    with MemoryFile() as memfile:
        t_height_profile = load_height_profile(expand_bounds(drainage_polygons.loc[c_drainage_polygons].total_bounds))
        t_height_profile = t_height_profile.rio.clip([drainage_polygons.loc[c_drainage_polygons].buffer(.1).unary_union])
        t_height_profile.fillna(t_height_profile.max()).rio.write_nodata(-32767).rio.to_raster(memfile.name)
        c_grid = Grid.from_raster(memfile.name)
        c_dem = c_grid.read_raster(memfile.name)
        
    # define direction vectors
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    # get interval of height grid
    grid_interval = t_height_profile.coords["y"][1].values - t_height_profile.coords["y"][0].values
        
    # rivershed workflow @https://github.com/mdbartos/pysheds
    # fill pits
    t_pit_filled_dem = c_grid.fill_pits(c_dem)
    # fill depressions
    t_flooded_dem = c_grid.fill_depressions(t_pit_filled_dem)
    # resolve flats
    t_inflated_dem = c_grid.resolve_flats(t_flooded_dem)
    # compute flow direction
    t_fdir = c_grid.flowdir(t_inflated_dem, dirmap=dirmap)
    c_acc = c_grid.accumulation(t_fdir, dirmap=dirmap)
    # extract river networks
    extracted_river_network = gpd.GeoDataFrame(c_grid.extract_river_network(t_fdir, c_acc > 500, dirmap=dirmap)["features"], crs = 4326)
    # clip for current drainage area only
    extracted_river_network = gpd.clip(extracted_river_network, drainage_polygons.loc[[c_payload.polygon_id]].buffer(.1))
    # merge multilinestrings
    extracted_river_network = extracted_river_network.explode(index_parts=True)
    if extracted_river_network.empty:
        return [c_drainage_polygon]
    # get downstream nodes
    extracted_river_network["end_node"] = extracted_river_network.apply(lambda x: shapely.Point(x["geometry"].coords[-1]), axis = 1)
    # get upstream nodes
    extracted_river_network["query_node"] = extracted_river_network.apply(lambda x: shapely.Point(x["geometry"].coords[-2]), axis = 1)
    #extracted_river_network["start_node"] = extracted_river_network.apply(lambda x: shapely.Point(x["geometry"].coords[0]), axis = 1)
    # compute and merge in occurence counts (confluence can be infered from double occurence of downstream node)
    extracted_river_network = pd.merge(extracted_river_network, extracted_river_network.apply(lambda x: shapely.Point(x["geometry"].coords[-1]), axis = 1).value_counts().sort_values(), left_on="end_node", right_index=True)

    # get multiple occurrences
    query_points = extracted_river_network[extracted_river_network["count"] > 1].groupby("end_node").query_node.apply(lambda x: x.to_list()).reset_index()
    # get catchment polygons for each query node
    query_points["drainage_polygons"] = query_points.query_node.apply(lambda x: [get_catchment_polygon(y, c_grid, t_fdir, dirmap).buffer(0) for y in x])
    if query_points.empty:
        return [c_drainage_polygon]
    t_clip = [shapely.Polygon()] * len(c_payload.fix_confluence_points)
    for confluence_idx in range(len(c_payload.fix_confluence_points)):
        # skip if there were no query points found
        
        # check length of intersection between catchment polygons and dissolved river geometries
        query_points["drainage_river_distribution"] = query_points.drainage_polygons.apply(lambda x: np.array([gpd.GeoSeries(y.intersection(t_rivers_dissolved.loc[c_payload.fix_confluence_rivers[confluence_idx]].geometry), crs=4326).to_crs(5641).length for y in x]))
        # normalize to distribution
        query_points["drainage_river_distribution_p"] = query_points["drainage_river_distribution"] / query_points["drainage_river_distribution"].apply(lambda x: np.sum(x, axis = 0))
        # calculate separation score
        query_points["distribution_score"] = [calculate_separation_score(x) for x in query_points["drainage_river_distribution_p"]]

        ## there is an edge case where the distribution score is perfect but only a tiny segment of one of the rivers is covered
        ## (this happens when a river crosses into the drainage of another - i.e. when the river data is imprecise)
        ## I circumvent this by excluding those that do not cover at least 50% of the river length-wise
        # assign upstream segments to rivers based on optimal share distribution
        query_points["river_polygon_correspondence"] = query_points["drainage_river_distribution_p"].apply(lambda x: np.argmax(x, axis=0))
        # get the length of rivers in upstream polygons for each split node
        t_length_at_optimal_dist = query_points.apply(lambda x: np.array([x.drainage_river_distribution[x.river_polygon_correspondence[i], i] for i in range(len(c_payload.fix_confluence_rivers[confluence_idx]))]), axis = 1)
        # calculate the length of the rivers in the drainage area
        t_river_info_at_confluence = rivers[rivers.downstream_node_id == c_payload.fix_confluence_points_ids[confluence_idx]][["estuary", "river", "segment"]]
        t_river_info_at_confluence = t_river_info_at_confluence.set_index(["estuary","river"]).loc[c_payload.fix_confluence_rivers[confluence_idx]].reset_index()
        t_river_length_upstream_drainage_area = t_river_info_at_confluence.apply(lambda x: gpd.clip(t_rivers[((t_rivers.estuary == x.estuary) & (t_rivers.river == x.river) & (t_rivers.segment >= x.segment))],
                                                                                                    drainage_polygons.loc[[c_payload.polygon_id]].to_crs(5641).geometry).length.sum(), axis = 1)
        # check whether assignment covers at least 50% of the river length-wise
        t_covers_50 = t_length_at_optimal_dist.apply(lambda x: np.all(x > 0.2 * t_river_length_upstream_drainage_area))
        #t_covers_threshold = t_length_at_optimal_dist.apply(lambda x: np.all(x > 200))
        # get id of top score, excluding those not covering enough of smaller river
        # if none cover enough, continue
        if not t_covers_50.any():
            continue
        # if none are above .45 distribution score, continue
        if not (query_points[t_covers_50].distribution_score > .45).any():
            continue
        t_idx_top_score = query_points[t_covers_50].distribution_score.idxmax()
        
        ## determine which river to clip from the residual
        ## I do by checking which river is least present in the downstream polygon
        ## this river will be clipped from the residual
        # get downstream polygon at optimal node
        t_downstream_polygon = drainage_polygons.loc[c_payload.polygon_id].geometry.difference(shapely.unary_union(query_points.loc[t_idx_top_score].drainage_polygons))
        # get river lengths in drainage polygon
        t_downstream_intersection_lengths = t_downstream_polygon.intersection(t_rivers_dissolved.loc[c_payload.fix_confluence_rivers[confluence_idx]].geometry).length
        # get id of river that is least present in downstream polygon
        t_idx_subordinate_river = t_downstream_intersection_lengths.reset_index(drop=True).idxmin()
        
        ## clip the subordinate river from the residual
        # get the id of the polygon to clip
        t_idx_polygon_to_clip = query_points["river_polygon_correspondence"][t_idx_top_score][t_idx_subordinate_river]
        # get the polygon to clip
        t_clip[confluence_idx] = c_drainage_polygon.intersection(query_points.loc[t_idx_top_score].drainage_polygons[t_idx_polygon_to_clip])
  
    # get the residual area left in the drainage polygon when differencing out all clipped polygons
    t_residual = [c_drainage_polygon.difference(shapely.ops.unary_union(t_clip)).buffer(0)]
    # # clean residual: remove small polygons
    # if isinstance(t_residual[0], shapely.geometry.multipolygon.MultiPolygon):
    #     t_residual = np.array(t_residual[0].geoms)[np.array([x.area > .1 * c_drainage_polygon.area for x in t_residual[0].geoms])].tolist()
    #     #t_residual[0] = t_residual[0].buffer(0)
    
    if not any(t_clip):
        return t_residual
    else:
        ## post-process polygons in graph
        ## difference out the clipped polygons iteratively down the graph
        
        # create a graph of river segments
        # sort river segments at confluence points by distance to estuary
        river_origin_distance_to_estuary = {(estuary_id, river_id): rivers.query(f"estuary=={estuary_id} & river=={river_id}").distance_from_estuary.min() for estuary_id, river_id in zip(c_payload.estuary_ids, c_payload.river_ids)}
        t_sorted_confluence_rivers = [sorted(x, key = lambda y: river_origin_distance_to_estuary[y]) for x in c_payload.fix_confluence_rivers]
        # get integer representation of river segments
        int_representation = {x: i for i, x in enumerate([tuple(x) for x in np.unique(np.array(c_payload.fix_confluence_rivers).reshape(-1, 2), axis = 0)])}
        int_representation_r = {i: x for i, x in enumerate([tuple(x) for x in np.unique(np.array(c_payload.fix_confluence_rivers).reshape(-1, 2), axis = 0)])}
        # transform to integer representation
        t_sorted_confluence_rivers = [[int_representation.get(z) for z in y] for y in t_sorted_confluence_rivers]
        
        ## there can only be one source node claiming the residual
        # get the source nodes
        t_source_nodes = find_source_nodes(t_sorted_confluence_rivers)
        # get all rivers that are local source nodes and are not entirely contained in the drainage polygon
        t_nodes_no_info = [x for x in [int_representation_r[x] for x in t_source_nodes] if x not in list(c_river_ids_to_query)]
        t_source_nodes = [x for x in [int_representation_r[x] for x in t_source_nodes] if x in list(c_river_ids_to_query)]
        # if there are multiple source nodes, choose the one with the longest river length
        if len(t_source_nodes) > 1:
            t_source_river_lengths = [gpd.clip(t_rivers.query(f"estuary=={estuary_id} & river=={river_id}"), gpd.GeoSeries(c_drainage_polygon, crs=4326).to_crs(5641).iloc[0]).length.sum() for estuary_id, river_id in t_source_nodes]
            t_id_max_river_length = t_source_nodes[np.argmax(t_source_river_lengths)]
            t_nodes_no_info += [x for x in t_source_nodes if x != t_id_max_river_length]
            t_source_nodes = [t_id_max_river_length]
        
        # build dictionary of polygons by id
        c_polygons = {t_sorted_confluence_rivers[i][1]: [t_clip[i]] for i in range(len(c_payload.fix_confluence_rivers))}
        c_polygons |= {int_representation.get(x): [shapely.Polygon()] for x in t_nodes_no_info}
        if t_source_nodes:
          c_polygons |= {int_representation.get(t_source_nodes[0]): [t_residual[0]]}
        
        def dfs(node, graph, node_values, visited):
            # Mark the node as visited
            visited.add(node)
            # Iterate over successors
            total_polygon = []
            count = 0
            for successor in graph[node]:
                if successor not in visited:
                    dfs(successor, graph, node_values, visited)
                total_polygon += node_values[successor]
                count += 1
            if count > 0:
                # Update node value based on successors
                node_values[node] = [node_values[node][0].difference(shapely.unary_union(total_polygon))]

        def update_node_values(graph, node_values):
            visited = set()
            for node in graph:
                if node not in visited:
                    dfs(node, graph, node_values, visited)
        
        # build graph and iterate differences down the graph
        graph = build_graph(t_sorted_confluence_rivers)
        update_node_values(graph, c_polygons)
        
        o_polygons = []
        # unpack and make sure all polygons are valid
        for key in c_polygons:
            if not c_polygons[key][0].is_valid:
                o_polygons += [c_polygons[key][0].buffer(0)]
            else:
                o_polygons += [c_polygons[key][0]]
        
        if not t_source_nodes:
            o_polygons += [t_residual[0].buffer(0)]
        
        return o_polygons
    


# a function to get shared downstream nodes given a list of rivers
def find_shared_nodes(rivers, topology, estuary_ids, river_ids):
    """
    Identifies shared downstream nodes where river confluences occur within the provided estuary
    and river identifiers. This function compiles a list of river segments based on their estuary
    and river IDs, counts downstream node occurrences, and identifies nodes shared by multiple rivers.

    Parameters:
    - rivers (geopandas.GeoDataFrame): GeoDataFrame containing river segments.
    - topology (geopandas.GeoDataFrame): GeoDataFrame containing topological data of river nodes.
    - estuary_ids (list[int]): List of estuary identifiers.
    - river_ids (list[int]): List of river identifiers.

    Returns:
    - dict: A dictionary containing lists of estuary IDs, river IDs, shared downstream node geometries,
      node IDs, and associated river IDs at each confluence point, structured as follows:
      {
        "estuary_ids": list of estuary IDs,
        "river_ids": list of river IDs,
        "fix_confluence_points": list of geometries of confluence points,
        "fix_confluence_points_ids": list of node IDs of confluence points,
        "fix_confluence_rivers": list of river details at each confluence point
      }
    
    Note:
    - This function assumes that the river dataframes have been properly indexed and contain relevant
      information about estuaries and rivers to effectively find confluences.
    """
    # get all river segments for the given estuary and river IDs
    t_rivers = pd.concat([rivers.query(f"estuary=={estuary_id} & river=={river_id}") for estuary_id, river_id in zip(estuary_ids, river_ids)])
    # count downstream node occurrences
    t_downstream_node_counts = t_rivers.downstream_node_id.value_counts()
    # get shared downstream nodes with corresponding rivers (where downstream node occurs more than once)
    t_confluence_nodes = t_downstream_node_counts[t_downstream_node_counts > 1].reset_index().downstream_node_id.apply(lambda x: topology.loc[x].geometry)
    # get shared downstream node IDs
    t_confluence_nodes_id = t_downstream_node_counts[t_downstream_node_counts > 1].reset_index().downstream_node_id
    # get shared downstream node rivers
    t_confluence_rivers = t_downstream_node_counts[t_downstream_node_counts > 1].reset_index().downstream_node_id.apply(lambda x: list(zip(t_rivers[t_rivers.downstream_node_id == x].estuary, t_rivers[t_rivers.downstream_node_id == x].river)))
    
    return {"estuary_ids": estuary_ids.to_list(), "river_ids": river_ids.to_list(), "fix_confluence_points": t_confluence_nodes.to_list(), "fix_confluence_points_ids": t_confluence_nodes_id.to_list(), "fix_confluence_rivers": t_confluence_rivers.to_list()}  

def fix_rivers_in_grid(i, rivers, topology, drainage_polygons, drainage_polygons_gridded, threshold = 200):
    """
    Analyzes and potentially modifies river configurations within a specific drainage area of a grid
    based on river confluence data. The function identifies problematic river confluences within the
    drainage area, then attempts to resolve these by adjusting the drainage area geometries.

    Parameters:
    - i (int): Index of the drainage area to be analyzed.
    - rivers (geopandas.GeoDataFrame): GeoDataFrame containing river data.
    - topology (geopandas.GeoDataFrame): GeoDataFrame with river topology data.
    - drainage_polygons (geopandas.GeoDataFrame): GeoDataFrame of drainage areas.
    - drainage_polygons_gridded (geopandas.GeoDataFrame): Grid-based GeoDataFrame of drainage polygons.
    - threshold (float, optional): Length threshold for determining significant river segments within a polygon.

    Returns:
    - gpd.GeoDataFrame or None: A GeoDataFrame containing the updated geometries of the drainage
      area with fixed river configurations, or None if no fixes are needed. The DataFrame is transformed
      to the CRS with code 5641 for the final geometry output.

    Note:
    - The function uses spatial joins to find intersecting rivers within the drainage area, assesses
      the confluence points using 'find_shared_nodes', and modifies the river configurations using
      'split_polygon'. It returns updated river geometries if changes are necessary, otherwise returns None.
    - The input data is expected to be well-indexed and formatted, including proper coordinate reference systems.
    """
    # get the current drainage area
    c_drainage_polygons = drainage_polygons.loc[drainage_polygons_gridded.index[drainage_polygons_gridded.index_left == i].values].to_crs(5641)
    # join with rivers to see which rivers are in each the drainage area
    joined = c_drainage_polygons.sjoin(rivers, how="right", predicate = "intersects")
    joined = joined[joined.index_left.notna()]
    joined["intersection_length"] = joined.intersection(c_drainage_polygons.loc[joined.index_left], align=False).length
    # filter for those with multiple rivers of length longer than threshold in drainage polygon
    to_fix = joined[joined.intersection_length > threshold]
    to_fix = to_fix.drop_duplicates(["index_left", "estuary", "river"])
    to_fix = to_fix[to_fix.index_left.isin(to_fix.groupby("index_left").size().where(lambda x: x > 1).dropna().index)]
    to_fix.dropna(subset=["estuary", "river"], inplace = True)
    # return if there are no rivers to fix
    if to_fix.empty:
        return c_drainage_polygons
    # get shared confluence nodes with corresponding rivers
    to_fix = pd.DataFrame().from_dict(to_fix.groupby("index_left").apply(lambda x: find_shared_nodes(rivers, topology, x['estuary'], x['river']), include_groups=False).to_dict(), orient="index").reset_index(names = "polygon_id")
    
    # split the polygons
    update_set = [split_polygon(rivers, drainage_polygons, x) for x in to_fix.itertuples()]
    update_set = gpd.GeoDataFrame(geometry = list(chain(*update_set)), crs = 4326).to_crs(5641)
    
    # update the drainage polygons, replacing the ones that were fixed
    final_df = pd.concat([c_drainage_polygons, update_set.dropna()]).drop(index=to_fix.polygon_id)
    #final_df["geometry"] = final_df.buffer(0)
    return final_df

# ===========================================================
# Extract drainage polygons
# ===========================================================   

def get_sub_polygon(query_points, c_polygon_halves):
    """
    Calculate a sub-polygon based on provided query points two halves of a polygon surrounding it.
    This works as follows: The function finds the nearest points on the polygon halves to the two query points,
    constructs a polygon using the query points and their corresponding nearest points, and returns the resulting polygon.
    
    Args:
    query_points (list): Points that define the query polygon or segment.
    c_polygon_projected (shapely.Polygon): The main polygon on which operations are to be performed.
    c_polygon_halves (list): Exterior of the main polygon split into two halves at river.
    mode (str, optional): Mode of operation, either 'inner' or 'edge'. Defaults to 'inner'.
    direction (str, optional): Direction for processing in 'edge' mode, either '<-' or '->'. Defaults to '<-'.
    
    Returns:
    shapely.Polygon: A sub-polygon derived based on the mode.
    """
    # Function to extend a line segment slightly at both ends
    def get_extended_line(points):
        extended_line = shapely.LineString(
            np.vstack([
                (np.array(points[0]) + 0.5 * (np.array(points[0]) - np.array(points[1]))),
                np.array(points[1]),
                (np.array(points[2]) + 0.5 * (np.array(points[2]) - np.array(points[1])))
            ])
        )
        return extended_line
            
    
    # Find the nearest points on the polygon halves to the first and last query points
    t_closest_points_on_drainage_limit = [
        shapely.ops.nearest_points(shapely.Point(query_points[0]), c_polygon_halves)[1],
        shapely.ops.nearest_points(shapely.Point(query_points[-1]), c_polygon_halves)[1]
    ]
        
    # Create a polygon using the start and end query points and their corresponding closest points
    t_sub_polygon = shapely.Polygon([
        query_points[0], 
        t_closest_points_on_drainage_limit[0][0], 
        t_closest_points_on_drainage_limit[1][0],
        query_points[-1], 
        t_closest_points_on_drainage_limit[1][1], 
        t_closest_points_on_drainage_limit[0][1]
    ])
    
    return t_sub_polygon


def extract_polygons_edge(c_edge, upstream_polygon, payload, c_polygon_projected, c_polygon_halves, to_snap_points, to_snap_points_projected, extracted_river_network, c_grid, t_fdir):
    """
    Computes drainage area polygons by matching provided river edges with edges computed from the DEM.
    
    Args:
    c_edge (pandas.Series): Edge data containing geometry and other specifics.
    upstream_polygon (shapely.geometry): Pre-computed polygon representing upstream polygons to exclude.
    payload (dict): River data.
    c_polygon_projected (shapely.geometry): The projected polygon in which operations are confined. (EPSG:5641)
    c_polygon_halves (shapely.geometry): Two halves of the projected polygon, split at (extended) river. (EPSG:5641)
    to_snap_points (list): List of points before projection. (EPSG:4326)
    to_snap_points_projected (list): List of shapely Points after projection. (EPSG:5641)
    
    Returns:
    list: A list of polygons representing the drainage areas refined and snapped to actual river paths. (EPSG:4326)
    """
    
    # Initial approximation of the edge bounding box polygon
    t_sub_polygon = get_sub_polygon([c_edge.geometry.coords[0], c_edge.geometry.coords[-1]], c_polygon_halves)
    
    candidate_rivers = gpd.clip(extracted_river_network, expand_bounds(gpd.GeoSeries([t_sub_polygon.envelope], crs=5641).to_crs(4326).total_bounds, 1.5)).dissolve().explode(index_parts=True).reset_index()
    
    # Nested function to extract drainage area polygons given a river point
    def get_polygon_da(point):
        # Generate a binary mask for the catchment area starting from the specified point using the computed flow direction
        mask = c_grid.catchment(x=point.x, y=point.y, fdir=t_fdir, dirmap=(64, 128, 1, 2, 4, 8, 16, 32), xytype='coordinate')

        # Find contours in the mask to delineate the boundaries of the catchment area
        polygon_image_space = cv2.findContours(np.array(mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0][0].squeeze()

        # Return an empty polygon if the contour has less than 4 points
        if polygon_image_space.shape[0] < 4:
            return shapely.Polygon()
        
        # Convert the image-space coordinates of the contour to geographic coordinates using the affine transformation of the grid
        polygon_affine_space = shapely.affinity.affine_transform(
            shapely.geometry.Polygon(polygon_image_space).buffer(0),  # Ensure valid geometry
            np.array(c_grid.affine)[[0,1,3,4,2,5]]  # Apply the grid's affine transformation
        )

        return polygon_affine_space

    # Function to find and snap appropriate drainage polygons to the river network based on proximity and containment
    def get_snapped_polygon_da(ii, threshold=0.9):
        # Find the nearest points on the extracted river segments to a given projected point
        c_nearest_points = [shapely.ops.nearest_points(x, to_snap_points_projected[ii])[0] for x in candidate_rivers.to_crs(5641).geometry]

        # Project the nearest points from local CRS to geographic CRS
        t_nearest_points_projected = gpd.GeoSeries(c_nearest_points, crs=5641).to_crs(4326).values

        # Rank the nearest points by distance to the target projected point
        t_distance_ranks = np.argsort([x.distance(to_snap_points_projected[ii]) for x in c_nearest_points])

        # Iterate over the ranked points, extracting and evaluating drainage polygons
        for idx, iii in enumerate(t_distance_ranks):
            if idx == 20:  # Timeout after 20 attempts
                #warnings.warn(f"Timeout reached for finding fitting snap point: cut #{ii}", category=Warning)
                return shapely.Polygon()  # Return an empty polygon if no suitable point found after 20 tries
            t_drainage_polygon = get_polygon_da(t_nearest_points_projected[iii])
            if np.mean(t_drainage_polygon.contains(to_snap_points[:ii])) > threshold:
                return t_drainage_polygon  # Return the polygon if it sufficiently contains the target points
            if idx == (len(t_distance_ranks) - 1):
                #warnings.warn(f"No suitable snap point: cut #{ii}", category=Warning)
                return shapely.Polygon()  # Return an empty polygon if no suitable points are found

    # Iterate over all projected points to match and refine drainage polygons
    snapped_polygons = [get_snapped_polygon_da(ii, 0.8) if to_snap_points_projected[ii] is not None else None for ii in range(1, len(to_snap_points_projected))]
    
    # Only keep area inside the river polygon
    snapped_polygons_in_bounds = [payload.geometry.intersection(x) if x is not None else None for x in snapped_polygons]
    
    # Difference out upstream polygons to isolate new drainage areas
    snapped_polygons_differenced = []
    snapped_polygons_in_bounds_not_none = [i for i in range(len(snapped_polygons_in_bounds)) if snapped_polygons_in_bounds[i] is not None]
    i = 0
    while i < len(snapped_polygons_in_bounds):
        # If the polygon is None, append None to the list and increment the index
        if snapped_polygons_in_bounds[i] is None:
            snapped_polygons_differenced.append(None)
            i += 1
        # If the polygon is the first in the list, difference it with the upstream polygon and increment the index
        elif i == min(snapped_polygons_in_bounds_not_none):
            snapped_polygons_differenced.append(snapped_polygons_in_bounds[i].difference(upstream_polygon))
            i += 1
        # If the polygon is not the first in the list, difference it with the union of the upstream polygon and all previous polygons
        else:
            snapped_polygons_differenced.append(snapped_polygons_in_bounds[i].difference(shapely.union_all([upstream_polygon] + [x for x in snapped_polygons_differenced if x is not None])))
            i += 1
                
    return snapped_polygons_differenced


def extract_polygons_river(payload, rivers, extracted_river_network, c_grid, t_fdir, cut_length):
    """
    Analyzes river segments within a given polygon, computes drainage areas.
    
    Args:
    payload (DataFrame): Limiting drainage polygon geometry, estuary, river id.
    rivers (GeoDataFrame): River geometries.
    extracted_river_network (GeoDataFrame): Extracted river network from DEM processing.
    c_grid (pysheds.grid.Grid): Pysheds grid object from DEM processing.
    t_fdir (numpy.ndarray): Flow direction array from DEM processing.
    cut_length (int): Maximum length of each segment of the river for processing.
    
    Returns:
    GeoDataFrame: Contains all sub-polygons and their associated river data.
    """
    # Project the payload polygon to EPSG:5641 for processing
    c_polygon = payload.geometry
    c_polygon_projected = gpd.GeoSeries([c_polygon], crs=4326).to_crs(5641).values[0]
    
    # Filter river data based on the specified river and estuary and check intersections with the payload polygon
    c_river = rivers.query(f"river == {payload.river} & estuary == {payload.estuary}")
    # c_river = c_river[c_river.to_crs(4326).bounds.apply(lambda x: shapely.geometry.box(*x).intersection(c_polygon).area > 0, axis=1)]
    
    # Calculate snapping points
    to_snap_points_projected = []
    to_snap_points = []
    # Determine segments and snapping points based on river length and segment offset
    for i in range(c_river.shape[0]):
        c_cuts = int((c_river.iloc[i].geometry.length + c_river.iloc[i].segment_offset) // cut_length)
        if c_cuts == 0:
            to_snap = np.array([c_river.iloc[i].geometry.coords[0], c_river.iloc[i].geometry.coords[-1]])
        else:
            to_snap = np.concatenate([[c_river.iloc[i].geometry.coords[0]], [c_river.iloc[i].geometry.interpolate(cut_length * (ii + 1)).coords[0] for ii in range(c_cuts)], [c_river.iloc[i].geometry.coords[-1]]])
        to_snap_points_projected.append([shapely.Point(x) if shapely.Point(x).intersects(c_polygon_projected) else None for x in to_snap])
        to_snap_points.append(gpd.GeoSeries(to_snap_points_projected[-1], crs=5641).to_crs(4326))
    
    ## Calculate river halves
    # Convert river geometries to a MultiLineString and concatenate coordinates
    c_river_coerced = shapely.geometry.MultiLineString([x if isinstance(x, shapely.geometry.LineString) else list(x.geoms) for x in c_river.geometry.values])
    c_river_coords = np.concatenate([x.coords for x in c_river_coerced.geoms])
    
    # Adjust river endpoints to align with the polygon exterior if they are within
    t_ends_within = [shapely.Point(c_river_coords[0]).within(c_polygon_projected), shapely.Point(c_river_coords[-1]).within(c_polygon_projected)]
    if t_ends_within[0]:
        c_river_coords = np.concatenate([[1e9 * (c_river_coords[0] - c_river_coords[1]) + c_river_coords[0]], c_river_coords])
    if t_ends_within[1]:
        c_river_coords = np.concatenate([c_river_coords, [1e9  * (c_river_coords[-1] - c_river_coords[-2]) + c_river_coords[-1]]])
        
    # Split the polygon using the adjusted river coordinates
    c_split_polygons = list(shapely.ops.split(c_polygon_projected, shapely.geometry.LineString(c_river_coords)).geoms)
    c_polygon_halves = [x.exterior.difference(shapely.geometry.LineString(c_river_coords)) for x in c_split_polygons]
    
    # Merge segments of the polygon half and segmentize it every 10 units to add more vertices
    c_polygon_halves = [shapely.line_merge(x).segmentize(10) for x in c_polygon_halves]
    
    # Compute drainage polygons for each segment
    t_polygons_da = extract_polygons_edge(c_river.iloc[0], shapely.Polygon(), payload, c_polygon_projected, c_polygon_halves, 
                                          to_snap_points[0], to_snap_points_projected[0], extracted_river_network, c_grid, t_fdir)
    polygons_da = [t_polygons_da]
    for i in range(1, c_river.shape[0]):
        t_polygons_da = extract_polygons_edge(c_river.iloc[i], shapely.ops.unary_union(list(chain(*polygons_da[:i]))), payload, c_polygon_projected, c_polygon_halves, 
                                              to_snap_points[i], to_snap_points_projected[i], extracted_river_network, c_grid, t_fdir)
        polygons_da += [t_polygons_da]
    
    # Helper function to compile polygon data into GeoDataFrame format
    def helper(polygons, i):
        return gpd.GeoDataFrame({
            "estuary": c_river.estuary.iloc[i],
            "river": c_river.river.iloc[i],
            "segment": c_river.segment.iloc[i],
            "subsegment": c_river.subsegment.iloc[i],
            "distance_from_estuary": list(np.arange((c_river.distance_from_estuary.iloc[i] // 1000) * 1000 + 1000 * (len(polygons[i]) - 1), (c_river.distance_from_estuary.iloc[i] // 1000) * 1000, -1000)) + [c_river.distance_from_estuary.iloc[i]],
            "geometry": polygons[i]
        })
    
    # Aggregate all projection and drainage polygons into GeoDataFrames and convert to appropriate CRS
    polygons_da = pd.concat([helper(polygons_da, i) for i in range(c_river.shape[0])])
    
    if (polygons_da.is_empty | polygons_da.geometry.isna()).all():
        polygons_da.loc[polygons_da[~polygons_da.geometry.isna()]["distance_from_estuary"].idxmin(), "geometry"] = c_polygon
    
    return polygons_da

def extract_polygons_grid_cell(drainage_polygons, rivers, cut_length=1000):
    """
    Analyzes a DataFrame of drainage polygons associated with rivers by (estuary, id), computes 
    drainage areas for each river in the DataFrame.
    Expects all polygons to be in the vicinity (in grid cell) for loading combined DEM into memory.
    
    Args:
    drainage_polygons (DataFrame): Drainage geometry, associated estuary, river ids.
    rivers (GeoDataFrame): All river geometries.
    cut_length (int): Maximum length of each segment of the river for processing.
    
    Returns:
    list: List of GeoDataFrames (or none), one for each drainage polygon in the input DataFrame.
    """
    
    # Load the DEM into a pysheds raster via rioxarray and a temporary file
    with tempfile.NamedTemporaryFile() as memfile:
        # Load and preprocess height profile within the bounding box of the sub-polygon, expanded by 1.2 units
        t_height_profile = load_height_profile(expand_bounds(drainage_polygons.total_bounds, 1.2))
        # Fill missing values, set no data value, and save as raster
        t_height_profile.fillna(t_height_profile.max()).rio.write_nodata(-32767).rio.to_raster(memfile.name, driver="GTiff")
        # Load this raster into a pysheds grid
        c_grid = Grid.from_raster(memfile.name)
        c_dem = c_grid.read_raster(memfile.name)
    
    # Hydrological processing to refine the DEM for accurate flow direction and accumulation computation
    t_pit_filled_dem = c_grid.fill_pits(c_dem)  # Fill pits
    t_flooded_dem = c_grid.fill_depressions(t_pit_filled_dem)  # Fill depressions
    t_inflated_dem = c_grid.resolve_flats(t_flooded_dem)  # Resolve flat areas
    t_fdir = c_grid.flowdir(t_inflated_dem, dirmap=(64, 128, 1, 2, 4, 8, 16, 32))  # Compute flow direction
    c_acc = c_grid.accumulation(t_fdir, dirmap=(64, 128, 1, 2, 4, 8, 16, 32))  # Compute flow accumulation
    
    # Extract river networks and clip them to the payload geometry, dissolve and explode for unique segments
    t_extracted_river_network = c_grid.extract_river_network(t_fdir, c_acc > 500, dirmap=(64, 128, 1, 2, 4, 8, 16, 32))["features"]
    if t_extracted_river_network == []:
        t_extracted_river_network = c_grid.extract_river_network(t_fdir, c_acc > 100, dirmap=(64, 128, 1, 2, 4, 8, 16, 32))["features"]
    if t_extracted_river_network == []:
        return [shapely.Polygon()] * (len(to_snap_points_projected) - 1)
    extracted_river_network = gpd.GeoDataFrame(t_extracted_river_network, crs=4326)
    
    results = [None] * drainage_polygons.shape[0]
    for i in range(drainage_polygons.shape[0]):
        try: 
            results[i] = extract_polygons_river(drainage_polygons.iloc[i], rivers, extracted_river_network, c_grid, t_fdir, cut_length)
        except:
            pass
        
    return results

# ===========================================================
# Helper Functions
# ===========================================================

def expand_bounds(bounds, factor = 1.5):
    """
    Expands a bounding box by a specified factor outward from its center.

    Parameters:
    - bounds (list[float]): A list of four floats representing the coordinates of the bounding box [xmin, ymin, xmax, ymax].
    - factor (float, optional): The factor by which the bounds should be expanded. Default is 1.5.

    Returns:
    - list[float]: A list of four floats representing the new expanded bounding box.
    """
    return [bounds[0] - (bounds[2] - bounds[0]) * (factor - 1) / 2, bounds[1] - (bounds[3] - bounds[1]) * (factor - 1) / 2, bounds[2] + (bounds[2] - bounds[0]) * (factor - 1) / 2, bounds[3] + (bounds[3] - bounds[1]) * (factor - 1) / 2]


def load_height_profile(bbox, polygon = None, dem_path = "/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/misc/raw/DEM_GLO-90/"):
    """
    Load and return the height profile for a given bounding box.

    Args:
    bbox (tuple): A tuple representing the bounding box with four values (min latitude, min longitude, max latitude, max longitude).

    Returns:
    xarray.DataArray: An array containing the height profile within the specified bounding box.
    """

    # Generate file path suffixes for latitude and longitude using grid conventions
    # Latitudes and longitudes are formatted based on their hemisphere and rounded to the nearest degree
    lat_lon = list(product(
        [lat for lat in np.arange(np.floor(bbox[0]), np.ceil(bbox[2]), 1)],
        [lon for lon in np.arange(np.floor(bbox[1]), np.ceil(bbox[3]), 1)]
    ))
    
    # Filter out files that do not intersect the polygon
    if polygon is not None:
        lat_lon = filter(lambda x: shapely.box(x[0], x[1], x[0] + 1, x[1] + 1).intersects(polygon), lat_lon)
    
    lat_lon = [
        (f"E{int(lat):03}" if lat >= 0 else f"W{-int(lat):03}", 
         f"N{int(lon):02}" if lon >= 0 else f"S{-int(lon):02}") for lat, lon in lat_lon
    ]
    
    # Construct file paths for DEM (Digital Elevation Model) tiles within the bounding box
    files_dem_cop = [
        f"{dem_path}Copernicus_DSM_30_{lon}_00_{lat}_00/DEM/Copernicus_DSM_30_{lon}_00_{lat}_00_DEM.tif"
        for lat, lon in lat_lon
    ]
    
    # Filter out files that do not exist
    files_dem_cop = [file for file in files_dem_cop if os.path.exists(file)]

    # Load the DEM files and combine them into a single DataArray using xarray, keeping the first channel and excluding the last pixel on each edge
    height_profile = xr.combine_by_coords([
        rxr.open_rasterio(file)[0, :-1, :-1] for file in files_dem_cop
    ])

    # # Clip the combined height profile to exactly match the bounding box using rioxarray
    # height_profile = height_profile.rio.clip_box(*bbox)

    return height_profile

def main(mode):
    if mode == "fix":
        # Database connection to fetch grid data
        conn = sqlite3.connect('/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/imagery/imagery.db')
        grid_data = pd.read_sql_query("SELECT * FROM GridCells WHERE Internal=1", conn)
        grid_geoms = gpd.read_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/imagery/grid_cells.feather")
        grid_data = gpd.GeoDataFrame(grid_data.merge(grid_geoms, on="CellID"))
        conn.close()

        # Reading and processing river data
        rivers = gpd.read_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/river_network/msc_rivers.feather").to_crs(5641)
        rivers_subset = gpd.clip(rivers, grid_data.geometry.iloc[525])

        # Loading administrative boundaries and clipping to the grid area
        boundaries = gpd.read_file("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/misc/raw/gadm/gadm41_BRA_2.json", engine="pyogrio").to_crs(5641)
        boundaries_subset = gpd.clip(boundaries, grid_data.geometry.iloc[525])

        # Loading river network data and computing distances from the estuary
        rivers_brazil = pickle.load(open("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/river_network/network.pkl", "rb"))
        rivers_brazil_shapefile = gpd.read_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/river_network/shapefile.feather")
        rivers_brazil_topology = gpd.read_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/river_network/topology.feather")
        rivers_brazil_shapefile = calculate_distance_from_estuary(rivers_brazil_shapefile, rivers_brazil_topology)

        # Reading and processing drainage areas
        drainage_polygons = gpd.read_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/drainage_area.feather")
        drainage_polygons_projected = drainage_polygons.to_crs(5641)
        drainage_polygons_gridded = grid_data.sjoin(gpd.GeoDataFrame(geometry=drainage_polygons_projected.centroid, index=drainage_polygons_projected.index), how="right").dropna(subset=["index_left"])

        # Checking for previously saved work and updating the grid calculations
        processed_grid_cells = os.listdir("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/drainage/temp_fixed_grid_cells")
        processed_grid_cells = [int(i.split(".")[0]) for i in processed_grid_cells]
            
        def wrapper(i):
            try:
                update_set = fix_rivers_in_grid(i, rivers_brazil_shapefile, rivers_brazil_topology, drainage_polygons, drainage_polygons_gridded)
                with open(f"/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/drainage/temp_fixed_grid_cells/{i}.pkl", "wb") as f:
                    pickle.dump(update_set, f)
            except:
                pass  
        
        wrapper(526)
        # with Pool(int(os.environ["SLURM_CPUS_PER_TASK"])) as p:
        #     p.map(wrapper, [i for i in grid_data.index if i not in processed_grid_cells])
            
    if mode == "dissolve":
        # connect to database
        conn = sqlite3.connect('/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/imagery/imagery.db')
        # get grid data
        grid_data = pd.read_sql_query("SELECT * FROM GridCells WHERE Internal=1", conn)
        # get grid geometries
        grid_geoms = gpd.read_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/imagery/grid_cells.feather")
        # merge data
        grid_data = gpd.GeoDataFrame(grid_data.merge(grid_geoms, on = "CellID"))
        # close connection
        conn.close()

        # Import the fixed drainage polygons
        fixed_grid_cells_list = os.listdir("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/drainage/temp_fixed_grid_cells")
        fixed_grid_cells = [pickle.load(open(f"/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/drainage/temp_fixed_grid_cells/{fixed_grid_cells_list[i]}", "rb")) for i in range(len(fixed_grid_cells_list))]
        drainage_polygons = gpd.GeoDataFrame(pd.concat(fixed_grid_cells)).reset_index(drop = True)
        drainage_polygons["geometry"] = drainage_polygons.geometry.buffer(0)

        # Save the fixed drainage polygons
        drainage_polygons.to_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/drainage/drainage_polygons.feather")

        # Read the river network
        rivers_brazil_shapefile = gpd.read_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/river_network/shapefile.feather")

        # Join the polygons with the river network
        joined = drainage_polygons.sjoin(rivers_brazil_shapefile, how="left", predicate = "intersects")
        # Remove polygons with no corresponding river
        joined.dropna(subset = ["index_right0", "index_right1", "index_right2"], inplace = True)
        # Create a tuple of river indices
        joined["index_right"] = joined.apply(lambda x: tuple([int(x.index_right0), int(x.index_right1), int(x.index_right2)]), axis = 1)
        # Calculate the intersection length
        joined["intersection_length"] = joined.apply(lambda x: x.geometry.intersection(rivers_brazil_shapefile.loc[x.index_right].geometry).length, axis = 1)
        # Assign drainage polygons to rivers with longest intersection
        tmp = joined.reset_index().groupby(["index", "estuary", "river"]).intersection_length.sum().groupby("index").idxmax()
        joined_assignment = pd.DataFrame({"estuary": tmp.apply(lambda x: x[1]),
                                        "river": tmp.apply(lambda x: x[2])},
                                        index = tmp.apply(lambda x: x[0]))

        # Join in the assignment and dissolve by river
        drainage_polygons = drainage_polygons.join(joined_assignment).to_crs(4326)
        drainage_polygons["geometry"] = drainage_polygons.geometry.buffer(0)
        drainage_polygons_dissolved = drainage_polygons.dissolve(["estuary", "river"]).reset_index()

        drainage_polygons.to_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/code/experiments/test.feather")


        ## Split large polygons (RAM bottleneck)

        # Identify large polygons
        drainage_polygons_dissolved["too_large"] = gpd.GeoSeries(drainage_polygons_dissolved.bounds.apply(lambda x: shapely.box(*x), axis=1)).area > 1

        # Filter out those polygons that are too large
        to_fix = drainage_polygons_dissolved[drainage_polygons_dissolved.too_large].copy()
        drainage_polygons_dissolved = drainage_polygons_dissolved[~drainage_polygons_dissolved.too_large]
        drainage_polygons_dissolved.drop(columns="too_large", inplace=True)

        # Split large polygons
        to_append = []
        for c_estuary, c_river in zip(to_fix.estuary, to_fix.river):
            # Query the polygons and rivers
            t_query_polygons = drainage_polygons.query(f"estuary == {c_estuary} and river == {c_river}").to_crs(5641)
            t_query_river = rivers_brazil_shapefile.query(f"estuary == {c_estuary} and river == {c_river}").sort_values("segment").dissolve()

            # Sort the polygons by distance along the river
            t_query_polygons["distance_along_river"] = t_query_polygons.centroid.apply(lambda x: t_query_river.project(x))
            t_query_polygons = t_query_polygons.sort_values("distance_along_river")
            
            # Prepare the polygons for grouping
            t_query_polygons["dissolve_group"] = None
            t_query_polygons = t_query_polygons.to_crs(4326)
            t_group = 0
            t_query_polygons["cumulative_bbox"] = t_query_polygons.bounds.apply(lambda x: shapely.box(*x), axis = 1)
            t_query_polygons.loc[:,"dissolve_group"].iloc[0] = t_group

            # Iterate over the polygons, build groups with cumulative bounding box area < 1
            for i in range(1, len(t_query_polygons)):
                t_bbox = shapely.box(*t_query_polygons.cumulative_bbox.iloc[i-1].union(t_query_polygons.cumulative_bbox.iloc[i]).bounds)
                if t_bbox.area < 1:
                    t_query_polygons.iloc[i, t_query_polygons.columns.get_loc("cumulative_bbox")] = t_bbox
                    t_query_polygons.iloc[i, t_query_polygons.columns.get_loc("dissolve_group")] = t_group
                else:
                    t_group += 1
                    t_query_polygons.iloc[i, t_query_polygons.columns.get_loc("dissolve_group")] = t_group
            # Fix and dissolve the polygons
            t_query_polygons["geometry"] = t_query_polygons.buffer(0)
            t_query_polygons = t_query_polygons.dissolve("dissolve_group")
            # Append the fixed polygons to the list
            to_append.append(t_query_polygons.loc[:,["estuary", "river", "geometry"]].reset_index(drop=True))

        # Concatenate the fixed polygons
        drainage_polygons_dissolved = pd.concat([drainage_polygons_dissolved] + to_append).reset_index(drop=True)

        # Filter out those polygons for which their centroid does not intersect the grid data
        drainage_polygons_dissolved_filtered = drainage_polygons_dissolved[drainage_polygons_dissolved.centroid.intersects(grid_data.to_crs(4326).unary_union)].copy()

        # Clip the polygons to the expanded boundaries of Brazil
        boundaries_limits = gpd.read_file("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/misc/gadm_410-BRA.geojson", engine="pyogrio").to_crs(5641).buffer(200 * 1e3).to_crs(4326)
        drainage_polygons_dissolved_filtered = gpd.clip(drainage_polygons_dissolved_filtered, boundaries_limits)

        # Save the dissolved and filtered polygons
        drainage_polygons_dissolved.to_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/drainage/drainage_polygons_dissolved.feather")
        drainage_polygons_dissolved_filtered.to_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/drainage/drainage_polygons_dissolved_filtered.feather")
        
    if mode == "extract":
        # A function to calculate a drainage area
        def worker(grid_cell_index, drainage_polygons_dissolved, rivers_brazil_shapefile, grid_data_projected):
            print(f"--- Processing grid cell {grid_cell_index} ---")
            tmp = drainage_polygons_dissolved[drainage_polygons_dissolved.centroid.within(grid_data_projected.geometry.iloc[grid_cell_index])]
            if tmp.empty:
                return
            results = extract_polygons_grid_cell(tmp, rivers_brazil_shapefile)
            pickle.dump(results, open(f"/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/drainage/temp_extracted_grid_cells/{grid_cell_index}.pkl", "wb"))

        def process_chunk(grid_cell_index):
            worker(grid_cell_index, drainage_polygons_dissolved, rivers_brazil_shapefile, grid_data_projected)
            

        # Database connection to fetch grid data
        conn = sqlite3.connect('/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/imagery/imagery.db')
        grid_data = pd.read_sql_query("SELECT * FROM GridCells WHERE Internal=1", conn)
        grid_geoms = gpd.read_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/imagery/grid_cells.feather")
        grid_data = gpd.GeoDataFrame(grid_data.merge(grid_geoms, on="CellID"))
        grid_data_projected = grid_data.to_crs(4326)
        conn.close()
        
        # Check which grid cells have already been processed
        os.makedirs("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/drainage/temp_extracted_grid_cells", exist_ok=True)
        processed_grid_cells = os.listdir("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/drainage/temp_extracted_grid_cells")
        processed_grid_cells = [int(i.split(".")[0]) for i in processed_grid_cells]

        # Loading river network data and computing distances from the estuary
        rivers_brazil_shapefile = gpd.read_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/river_network/shapefile.feather")
        rivers_brazil_topology = gpd.read_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/river_network/topology.feather")
        rivers_brazil_shapefile = calculate_distance_from_estuary(rivers_brazil_shapefile, rivers_brazil_topology)

        # Get the dissolved drainage polygons
        drainage_polygons_dissolved = gpd.read_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/drainage/drainage_polygons_dissolved_filtered.feather")
        
        with Pool(processes = int(os.environ["SLURM_CPUS_PER_TASK"])) as pool:
            # Apply the function to each chunk using multiprocessing
            pool.map(process_chunk, [x for x in grid_data_projected.index if x not in processed_grid_cells])
            
    if mode == "matrices":
        def rasterize_polygon(payload, shape, transform):
            t_rasterized = rasterize(payload, out_shape=shape, transform=transform, all_touched=True)
            rasterized_sparse = sparse.COO.from_numpy(t_rasterized)
            return rasterized_sparse

        def worker(payload):
            t_template = lc_t.rio.clip_box(*expand_bounds(payload[1].total_bounds))
            polygons_raster = [rasterize_polygon([x], payload[2], payload[3]) for x in payload[1].geometry]
            pickle.dump(polygons_raster, open(f"/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/drainage/temp_extraction_masks/{payload[0]}.pkl", "wb"))
            
        # Load extracted polygons
        extracted_drainage_polygons = gpd.read_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/drainage/extracted_drainage_polygons.feather")
        extracted_drainage_polygons = extracted_drainage_polygons.set_crs(4326)[~extracted_drainage_polygons.is_empty].dropna(subset=["geometry"])
        
        # Load land cover data as template
        lc_t = rxr.open_rasterio(f"/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/land_cover/raw/lc_mapbiomas8_30/mapbiomas_brasil_coverage_{1990}.tif", chunks=True).squeeze()

        # Get grid ids to work on
        grid_ids = extracted_drainage_polygons.index.get_level_values(0).unique()
        # Remove already processed grid cells
        files = os.listdir("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/drainage/temp_extraction_masks")
        files = pd.Series(files).str.extract(r"(\d*)").squeeze().astype(int).unique()
        grid_ids = [x for x in grid_ids if not x in files]
        
        # Prepare payloads
        payloads = [
            (idx, 
            extracted_drainage_polygons.loc[idx], 
            lc_t.rio.clip_box(*expand_bounds(extracted_drainage_polygons.loc[idx].total_bounds)).shape,
            lc_t.rio.clip_box(*expand_bounds(extracted_drainage_polygons.loc[idx].total_bounds)).rio.transform()) 
            for idx in grid_ids]

        with Pool(8) as p:
            p.map(worker, payloads)

main("fix")