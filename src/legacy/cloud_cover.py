import os
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rxr

def main():
    # Load the boundaries of Brazil from a geojson file
    boundaries = gpd.read_file("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/misc/gadm_410-BRA.geojson", engine="pyogrio")

    # Load the drainage polygons from a feather file
    drainage_polygons = gpd.read_feather("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/drainage/extracted_drainage_polygons.feather")

    # Compute the centroids of the non-empty drainage polygons
    drainage_polygons_centroids = drainage_polygons[~drainage_polygons.is_empty].dropna(subset="geometry").centroid

    # Define the path to the directory containing raw cloud cover data
    path = "/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/cloud_cover/raw/"

    # Get a list of files in the directory and filter for NetCDF files
    files = pd.Series(os.listdir(path))

    # Open multiple NetCDF files as a single xarray dataset
    cloud_cover = xr.open_mfdataset(path + files[files.str.contains(".nc")], chunks={"time": len(files[files.str.contains(".nc")])})

    # Write the coordinate reference system (CRS) and set spatial dimensions
    cloud_cover = cloud_cover.rio.write_crs("epsg:4326").rio.set_spatial_dims(x_dim="lon", y_dim="lat").rio.write_coordinate_system(inplace=True)

    # Subset the cloud cover data using the bounding box of the boundaries
    cloud_cover_subset = cloud_cover.cfc.rio.clip_box(*boundaries.total_bounds)

    # Resample the subsetted cloud cover data to yearly means
    cloud_cover_subset_resampled = cloud_cover_subset.resample(time="1YE").mean().load()

    # Apply a lambda function to extract the nearest cloud cover values at the centroids of the drainage polygons
    cloud_cover_values = drainage_polygons_centroids.apply(lambda x: cloud_cover_subset_resampled.sel(lon=x.x, lat=x.y, method="nearest").values)

    # Create a DataFrame with the extracted cloud cover values and corresponding years
    tmp = pd.DataFrame(
        {
            "cloud_cover": cloud_cover_values,
            "year": [cloud_cover_subset_resampled.time.dt.year.values] * len(drainage_polygons_centroids),
        }
    )

    # Explode the DataFrame to ensure each year-cloud cover pair is in its own row and save to a feather file
    tmp = tmp.explode(["cloud_cover", "year"]).reset_index(names=["grid_id", "index"])
    tmp.to_parquet("/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/data/cloud_cover/cloud_cover.parquet")
    
if __name__ == "__main__":
    main()