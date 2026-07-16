import os
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
import warnings

import sys
sys.path.append("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/code")
from data.preprocess.drainage_polygons.aux_functions import expand_bounds, load_height_profile

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