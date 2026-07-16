import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
import shapely
import cv2
from rasterio.io import MemoryFile
from pysheds.grid import Grid
import matplotlib.pyplot as plt
from itertools import chain, product
import tempfile
from tqdm import tqdm

from data.preprocess.drainage_polygons.aux_functions import expand_bounds, load_height_profile


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
  