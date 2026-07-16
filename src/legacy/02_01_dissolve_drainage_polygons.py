import os
import pickle
import sqlite3
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely

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