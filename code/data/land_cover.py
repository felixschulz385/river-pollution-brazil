"""
Land cover processing script for SLURM cluster.
Usage: python land_cover.py --n_jobs 16 --output results.feather
"""

import os
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from pathlib import Path
from odc.geo.xr import ODCExtensionDa
from odc.geo.geom import Geometry
from scipy import sparse

# Configure module logger
logger = logging.getLogger(__name__)


# ============== Configuration ==============
DATADIR = "/scicore/home/meiera/schulz0022/projects/river-pollution-brazil/data/land_cover/raw/lc_mapbiomas8_30/"
DRAINAGE_PATH = "data/drainage/extracted_drainage_polygons_filtered.feather"
LEGEND_PATH = "data/land_cover/mapbiomas_legend.xlsx"
OUTPUT_COLUMNS = [0, 1, 2, 3, 4, 20, 21, 30, 31]


# ============== Kernel Functions ==============
def kernel_exponential(distances, decay_rate=0.001):
    """Exponential decay kernel: exp(-decay_rate * distance)"""
    return np.exp(-decay_rate * distances)


def kernel_gaussian(distances, sigma=10000):
    """Gaussian kernel: exp(-distance^2 / (2 * sigma^2))"""
    return np.exp(-distances**2 / (2 * sigma**2))


def kernel_inverse_distance(distances, power=1, min_distance=1):
    """Inverse distance kernel: 1 / (distance + min_distance)^power"""
    return 1.0 / (distances + min_distance)**power


def kernel_uniform(distances, max_distance=None):
    """Uniform kernel: 1 for all distances (optionally within max_distance)"""
    weights = np.ones_like(distances, dtype=float)
    if max_distance is not None:
        weights[distances > max_distance] = 0
    return weights


def kernel_linear_decay(distances, max_distance=50000):
    """Linear decay kernel: 1 - distance/max_distance, clamped to [0, 1]"""
    weights = 1 - distances / max_distance
    return np.clip(weights, 0, 1)


AVAILABLE_KERNELS = {
    'exponential': kernel_exponential,
    'gaussian': kernel_gaussian,
    'inverse_distance': kernel_inverse_distance,
    'uniform': kernel_uniform,
    'linear_decay': kernel_linear_decay
}


# ============== Helper Functions ==============
def add_crs(geom, crs=4326):
    return Geometry(geom, crs)


def create_mappers(legend_path):
    """Create vectorized mappers from legend file."""
    legend = pd.read_excel(legend_path)
    legend_class_dict = legend.set_index("ID").Class.to_dict()
    legend_subclass_dict = legend.set_index("ID").Subclass.to_dict()
    
    legend_class_dict_mapper = np.vectorize(lambda x: legend_class_dict.get(x, np.nan))
    legend_subclass_dict_mapper = np.vectorize(lambda x: legend_subclass_dict.get(x, np.nan))
    
    return legend_class_dict_mapper, legend_subclass_dict_mapper


def get_files(datadir):
    """Get sorted file list with year index."""
    files = pd.Series(os.listdir(datadir)).sort_values()
    years = files.str.extract(r"(\d{4,})").iloc[:, 0].astype("int")
    return files.set_axis(years)


def process_year(year, file, datadir, drainage_polygons, legend_path, output_columns):
    """Process all polygons for a single year."""
    logger.info(f"Processing year {year}")
    
    try:
        legend_class_mapper, legend_subclass_mapper = create_mappers(legend_path)
        lc = rxr.open_rasterio(datadir + file, chunks=None)
    except Exception as e:
        logger.error(f"Failed to load raster for year {year}: {e}")
        # Return empty DataFrame for this year
        mi = pd.MultiIndex.from_product(
            [drainage_polygons.index.values, [year]], 
            names=['id', 'year']
        )
        return pd.DataFrame(0, index=mi, columns=output_columns).reset_index()
    
    # Create output structure for this year
    mi = pd.MultiIndex.from_product(
        [drainage_polygons.index.values, [year]], 
        names=['id', 'year']
    )
    year_df = pd.DataFrame(0, index=mi, columns=output_columns)
    
    for j in drainage_polygons.index:
        try:
            geometry = drainage_polygons.loc[j, 'geometry']
            
            try:
                cropped_data = np.unique(
                    lc.odc.crop(add_crs(geometry)),
                    return_counts=True
                )
                values, counts = cropped_data
            except Exception as e:
                logger.warning(f"Polygon {j} in year {year} does not overlap with raster extent: {e}")
                values, counts = np.array([]), np.array([])
            
            for mapper in (legend_class_mapper, legend_subclass_mapper):
                if len(values) == 0:
                    continue
                cropped_data_classes = mapper(values)
                valid = ~np.isnan(cropped_data_classes)
                classes = cropped_data_classes[valid].astype(int)
                weights = counts[valid]
                
                if len(classes) == 0:
                    continue
                
                uniq, inv = np.unique(classes, return_inverse=True)
                agg = np.bincount(inv, weights=weights)
                year_df.loc[(j, year), uniq] = agg
                
        except Exception as e:
            logger.error(f"Error processing polygon {j} in year {year}: {e}")
    
    logger.info(f"Completed year {year}")
    return year_df.reset_index()


class land_cover:
    """Land cover data processor with CLI integration."""
    
    def __init__(self, 
                 datadir=DATADIR, 
                 drainage_path=DRAINAGE_PATH, 
                 legend_path=LEGEND_PATH,
                 output_columns=OUTPUT_COLUMNS):
        self.datadir = datadir
        self.drainage_path = drainage_path
        self.legend_path = legend_path
        self.output_columns = output_columns
        logger.debug(f"Initialized land_cover with datadir={datadir}")
    
    def fetch(self):
        """Fetch/download raw land cover data (placeholder for future implementation)."""
        logger.info("Land cover data should be downloaded manually from MapBiomas.")
        logger.info(f"Expected location: {self.datadir}")
    
    def preprocess(self, n_jobs=None, output_path="data/land_cover_results.feather"):
        """
        Preprocess land cover data by extracting values for drainage polygons.
        Parallelizes by year using joblib.
        
        Parameters:
        n_jobs (int): Number of parallel jobs (default: all CPUs)
        output_path (str): Path for output feather file
        """
        if n_jobs is None:
            n_jobs = cpu_count()
        
        logger.info(f"Starting preprocessing with n_jobs={n_jobs}")
        logger.info(f"Loading drainage polygons from {self.drainage_path}")
        drainage_polygons = gpd.read_feather(self.drainage_path)
        logger.info(f"Loaded {len(drainage_polygons)} drainage polygons")
        
        files = get_files(self.datadir)
        logger.info(f"Found {len(files)} land cover files for years {files.index.min()}-{files.index.max()}")
        
        # Process all years in parallel using joblib
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(process_year)(
                year, file, self.datadir, drainage_polygons, 
                self.legend_path, self.output_columns
            )
            for year, file in files.items()
        )
        
        # Merge all results
        logger.info("Merging all results...")
        final_df = pd.concat(results, ignore_index=True)
        final_df = final_df.sort_values(['id', 'year']).reset_index(drop=True)
        
        output_path = Path(output_path)
        final_df.to_feather(output_path)
        logger.info(f"Results saved to {output_path}")
        
        return final_df
    
    def aggregate_along_rivers(
        self,
        land_cover_path,
        river_network_path,
        kernel='exponential',
        kernel_params=None,
        years=None,
        n_jobs=None,
        output_path="land_cover_river_aggregated.feather"
    ):
        """
        Aggregate land cover variables along rivers for each ADM2 unit.
        
        For each ADM2 unit, finds all rivers within it, computes the union of all
        downstream river edges with distances using the reachability matrix, then
        applies a spatial weighting kernel to compute weighted sums.
        
        Parameters
        ----------
        land_cover_path : str
            Path to preprocessed land cover feather file (output of preprocess()).
        river_network_path : str
            Path to river network directory containing shapefile.feather, 
            topology.feather, distance_from_estuary.feather, and reachability.npz.
        kernel : str or callable
            Kernel function for spatial weighting. Options:
            - 'exponential': exp(-decay_rate * distance)
            - 'gaussian': exp(-distance^2 / (2 * sigma^2))
            - 'inverse_distance': 1 / (distance + min_distance)^power
            - 'uniform': equal weight for all distances
            - 'linear_decay': 1 - distance/max_distance
            - callable: custom function taking distances array and returning weights
        kernel_params : dict, optional
            Parameters to pass to the kernel function. Default parameters vary by kernel.
        years : list, optional
            List of years to process. If None, processes all available years.
        n_jobs : int, optional
            Number of parallel jobs. Defaults to all CPUs.
        output_path : str
            Path for output feather file.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with weighted sums of land cover variables for each ADM2 unit and year.
            Columns include ADM2 id, year, and weighted land cover class areas.
        """
        from river_network import river_network
        
        if n_jobs is None:
            n_jobs = cpu_count()
        
        kernel_params = kernel_params or {}
        
        # Get kernel function
        if isinstance(kernel, str):
            if kernel not in AVAILABLE_KERNELS:
                raise ValueError(f"Unknown kernel: {kernel}. Available: {list(AVAILABLE_KERNELS.keys())}")
            kernel_func = AVAILABLE_KERNELS[kernel]
        elif callable(kernel):
            kernel_func = kernel
        else:
            raise ValueError("kernel must be a string or callable")
        
        logger.info(f"Loading land cover data from {land_cover_path}")
        land_cover_df = pd.read_feather(land_cover_path)
        
        # Set multi-index if present
        if 'id' in land_cover_df.columns and 'year' in land_cover_df.columns:
            # id should be a tuple of (estuary, river, segment, subsegment)
            pass
        
        logger.info(f"Loading river network from {river_network_path}")
        network_dir = Path(river_network_path)
        network = river_network.load_network(
            str(network_dir / "shapefile.feather"),
            str(network_dir / "topology.feather"),
            distance_path=str(network_dir / "distance_from_estuary.feather"),
            reachability_path=str(network_dir / "reachability.npz")
        )
        
        if network.reachability is None:
            raise ValueError("River network must have reachability data computed")
        
        rivers = network.shapefile
        reachability = network.reachability
        reachability_matrix = reachability['matrix']
        node_ids = reachability['node_ids']
        node_to_idx = reachability['node_to_idx']
        
        # Get unique ADM2 units
        adm2_units = rivers.dropna(subset=['adm2'])['adm2'].unique()
        logger.info(f"Processing {len(adm2_units)} ADM2 units")
        
        # Get available years
        if years is None:
            years = land_cover_df['year'].unique() if 'year' in land_cover_df.columns else [None]
        
        logger.info(f"Processing years: {years}")
        
        # Build mapping from drainage polygon index to land cover data
        # Drainage polygons are indexed by (estuary, river, segment, subsegment)
        lc_columns = [c for c in land_cover_df.columns if c not in ['id', 'year', 'level_0', 'level_1', 'level_2', 'level_3']]
        
        def process_adm2_year(adm2_id, year):
            """Process a single ADM2 unit for a single year."""
            try:
                # Get all river edges in this ADM2 unit
                adm2_rivers = rivers[rivers['adm2'] == adm2_id].dropna(subset=['upstream_node_id'])
                
                if len(adm2_rivers) == 0:
                    return None
                
                # Collect all downstream edges with distances
                downstream_edges = {}  # edge_index -> distance
                
                for idx, row in adm2_rivers.iterrows():
                    upstream_node = int(row['upstream_node_id'])
                    
                    if upstream_node not in node_to_idx:
                        continue
                    
                    node_idx = node_to_idx[upstream_node]
                    
                    # Get all reachable nodes from this node (row in sparse matrix)
                    row_start = reachability_matrix.indptr[node_idx]
                    row_end = reachability_matrix.indptr[node_idx + 1]
                    
                    reachable_indices = reachability_matrix.indices[row_start:row_end]
                    reachable_distances = reachability_matrix.data[row_start:row_end]
                    
                    # Map back to node IDs and find corresponding edges
                    for r_idx, dist in zip(reachable_indices, reachable_distances):
                        reachable_node_id = node_ids[r_idx]
                        
                        # Find edges that have this node as upstream
                        matching_edges = rivers[
                            rivers['upstream_node_id'] == reachable_node_id
                        ].index.tolist()
                        
                        for edge_idx in matching_edges:
                            # Keep minimum distance if edge appears multiple times
                            if edge_idx not in downstream_edges:
                                downstream_edges[edge_idx] = dist
                            else:
                                downstream_edges[edge_idx] = min(downstream_edges[edge_idx], dist)
                
                if not downstream_edges:
                    return None
                
                # Convert edge indices to drainage polygon indices
                # Edge index is (estuary, river, segment, subsegment) tuple
                edge_indices = list(downstream_edges.keys())
                distances = np.array([downstream_edges[e] for e in edge_indices])
                
                # Apply kernel to get weights
                weights = kernel_func(distances, **kernel_params)
                
                # Normalize weights
                weight_sum = weights.sum()
                if weight_sum > 0:
                    weights = weights / weight_sum
                else:
                    return None
                
                # Get land cover data for these drainage polygons
                result = {col: 0.0 for col in lc_columns}
                
                for edge_idx, weight in zip(edge_indices, weights):
                    if weight == 0:
                        continue
                    
                    # Match edge index to land cover data
                    # edge_idx is a tuple like (level_0, level_1, level_2)
                    if isinstance(edge_idx, tuple):
                        # Filter land cover by matching index components
                        if year is not None:
                            mask = (land_cover_df['year'] == year)
                            if 'level_0' in land_cover_df.columns:
                                mask &= (land_cover_df['level_0'] == edge_idx[0])
                                mask &= (land_cover_df['level_1'] == edge_idx[1])
                                if len(edge_idx) > 2:
                                    mask &= (land_cover_df['level_2'] == edge_idx[2])
                            elif 'id' in land_cover_df.columns:
                                mask &= (land_cover_df['id'] == edge_idx)
                            
                            lc_row = land_cover_df[mask]
                        else:
                            lc_row = land_cover_df[land_cover_df['id'] == edge_idx]
                        
                        if len(lc_row) > 0:
                            for col in lc_columns:
                                if col in lc_row.columns:
                                    result[col] += weight * lc_row[col].values[0]
                
                result['adm2'] = adm2_id
                result['year'] = year
                result['n_edges'] = len(edge_indices)
                result['total_weight'] = weight_sum
                
                return result
                
            except Exception as e:
                logger.warning(f"Error processing ADM2 {adm2_id}, year {year}: {e}")
                return None
        
        # Process all ADM2-year combinations
        tasks = [(adm2_id, year) for adm2_id in adm2_units for year in years]
        
        logger.info(f"Processing {len(tasks)} ADM2-year combinations with {n_jobs} workers")
        
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(process_adm2_year)(adm2_id, year) 
            for adm2_id, year in tasks
        )
        
        # Filter None results and create DataFrame
        results = [r for r in results if r is not None]
        
        if not results:
            logger.warning("No results produced")
            return pd.DataFrame()
        
        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values(['adm2', 'year']).reset_index(drop=True)
        
        # Save results
        output_path = Path(output_path)
        result_df.to_feather(output_path)
        logger.info(f"Results saved to {output_path}")
        logger.info(f"Output shape: {result_df.shape}")
        
        return result_df


if __name__ == "__main__":
    import argparse
    
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    parser = argparse.ArgumentParser(description="Process land cover data in parallel")
    parser.add_argument("--n_jobs", type=int, default=None, help="Number of parallel jobs")
    parser.add_argument("--output", type=str, default="land_cover_results.feather", help="Output file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    lc = land_cover()
    lc.preprocess(n_jobs=args.n_jobs, output_path=args.output)