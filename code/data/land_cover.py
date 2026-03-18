"""
Land cover processing for drainage polygons and river-network summaries.

Output files
------------
`land_cover_results.feather`
    One row per (`trench_id`, `year`) with `land_cover_class_*` counts.
`land_cover_river_aggregated.feather`
    One row per (`adm2_id`, `year`) with kernel-weighted upstream trench
    aggregates derived from the saved river-network matrices.

Usage
-----
`python land_cover.py --n_jobs 16 --output results.feather`
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
from tqdm import tqdm

# Try to import river_network with fallback for different execution contexts
try:
    from . import river_network as rn_module
except ImportError:
    try:
        import code.data.river_network as rn_module
    except ImportError:
        import river_network as rn_module

# Configure module logger
logger = logging.getLogger(__name__)


# ============== Configuration ==============
DATADIR = "/scicore/home/meiera/schulz0022/projects/river-pollution-brazil/data/land_cover/raw/lc_mapbiomas8_30/"
DRAINAGE_PATH = "data/river_network/drainage_areas.parquet"
LEGEND_PATH = "data/land_cover/mapbiomas_legend.xlsx"
TRENCH_ID_COLUMN = "trench_id"
YEAR_COLUMN = "year"
ADM2_ID_COLUMN = "adm2_id"
REACHABLE_TRENCH_COUNT_COLUMN = "reachable_trench_count"
TOTAL_WEIGHT_COLUMN = "total_weight"
LAND_COVER_CLASS_PREFIX = "land_cover_class_"
LAND_COVER_TOTAL_COLUMN = "land_cover_total"


def _normalize_optional_int(value):
    """Convert optional integer-like legend values to Python ints or None."""
    if pd.isna(value) or value == "":
        return None
    return int(value)


def _subclass_summary_id(class_id, subclass_id):
    """Build a globally unique subclass summary id from class and subclass."""
    normalized_subclass = _normalize_optional_int(subclass_id)
    if normalized_subclass is None:
        return None
    return int(class_id) * 10 + normalized_subclass


def get_output_columns(legend_path):
    """Derive output columns from the legend file."""
    legend = pd.read_excel(legend_path)
    class_ids = sorted(
        {
            int(class_id)
            for class_id in legend["Class"].dropna().tolist()
        }
    )
    subclass_ids = sorted(
        {
            subclass_summary_id
            for class_id, subclass_id in legend[["Class", "Subclass"]].itertuples(index=False)
            for subclass_summary_id in [_subclass_summary_id(class_id, subclass_id)]
            if subclass_summary_id is not None
        }
    )
    return (
        [LAND_COVER_TOTAL_COLUMN]
        + [f"{LAND_COVER_CLASS_PREFIX}{int(class_id)}" for class_id in class_ids]
        + [f"{LAND_COVER_CLASS_PREFIX}{int(class_id)}" for class_id in subclass_ids]
    )


OUTPUT_COLUMNS = get_output_columns(LEGEND_PATH)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging for standalone land-cover execution."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _trench_ids(drainage_polygons):
    """Return trench identifiers for the deduplicated drainage polygons."""
    if TRENCH_ID_COLUMN not in drainage_polygons.columns:
        raise ValueError(
            f"Drainage polygons must include `{TRENCH_ID_COLUMN}` as an explicit column."
        )
    return drainage_polygons[TRENCH_ID_COLUMN].to_numpy()


def _build_year_multiindex(drainage_polygons, year):
    """Build the standard output index for one processed year."""
    return pd.MultiIndex.from_product(
        [_trench_ids(drainage_polygons), [year]],
        names=[TRENCH_ID_COLUMN, YEAR_COLUMN],
    )


def _class_column_name(class_id):
    """Return the canonical output column name for a land-cover class."""
    return f"{LAND_COVER_CLASS_PREFIX}{int(class_id)}"


def _class_column_names(class_ids):
    """Map numeric class identifiers to canonical output column names."""
    return [_class_column_name(class_id) for class_id in class_ids]


def _mapped_classes_and_weights(values, counts, mapper):
    """Return finite mapped class ids and aligned weights for aggregation."""
    mapped = np.asarray(mapper(values), dtype=float)
    valid = np.isfinite(mapped)
    if not np.any(valid):
        return np.asarray([], dtype=int), np.asarray([], dtype=counts.dtype)
    return mapped[valid].astype(int), counts[valid]


def _deduplicate_drainage_polygons(drainage_polygons):
    """Keep the first drainage polygon for each trench_id."""
    if TRENCH_ID_COLUMN not in drainage_polygons.columns:
        raise ValueError(
            f"Drainage polygons must include `{TRENCH_ID_COLUMN}` as an explicit column."
        )
    drainage_polygons = drainage_polygons.drop_duplicates(
        subset=[TRENCH_ID_COLUMN],
        keep="first",
    )
    return drainage_polygons.set_index(TRENCH_ID_COLUMN, drop=False)


def _validate_land_cover_output_columns(land_cover_df):
    """Validate the expected output schema before downstream aggregation."""
    required_columns = {TRENCH_ID_COLUMN, YEAR_COLUMN}
    missing_columns = required_columns.difference(land_cover_df.columns)
    if missing_columns:
        raise ValueError(
            "Land-cover input is missing required columns: "
            f"{sorted(missing_columns)}. Expected schema uses "
            f"`{TRENCH_ID_COLUMN}` and `{YEAR_COLUMN}`."
        )


def _explode_trench_adm2_matches(rivers):
    """Expand trench-level ADM2 match lists into one row per trench-ADM2 pair."""
    if rn_module.ADM2_LIST_COLUMN not in rivers.columns:
        if "adm2" not in rivers.columns:
            raise ValueError(
                "River trench data must include either `adm2` or "
                f"`{rn_module.ADM2_LIST_COLUMN}` for upstream aggregation."
            )
        trench_adm2 = rivers[[TRENCH_ID_COLUMN, rn_module.SYSTEM_ID_KEY, "adm2"]].copy()
        trench_adm2 = trench_adm2.dropna(subset=["adm2"])
        trench_adm2["intersection_length"] = np.nan
        return trench_adm2

    required_columns = {
        TRENCH_ID_COLUMN,
        rn_module.SYSTEM_ID_KEY,
        rn_module.ADM2_LIST_COLUMN,
        rn_module.ADM2_INTERSECTION_LENGTHS_COLUMN,
    }
    missing_columns = required_columns.difference(rivers.columns)
    if missing_columns:
        raise ValueError(
            "River trench data is missing ADM2 match columns: "
            f"{sorted(missing_columns)}."
        )

    trench_adm2 = rivers[
        [
            TRENCH_ID_COLUMN,
            rn_module.SYSTEM_ID_KEY,
            rn_module.ADM2_LIST_COLUMN,
            rn_module.ADM2_INTERSECTION_LENGTHS_COLUMN,
        ]
    ].copy()
    trench_adm2 = trench_adm2.rename(
        columns={
            rn_module.ADM2_LIST_COLUMN: "adm2",
            rn_module.ADM2_INTERSECTION_LENGTHS_COLUMN: "intersection_length",
        }
    )
    trench_adm2["adm2"] = trench_adm2["adm2"].apply(
        lambda values: values if isinstance(values, list) else []
    )
    trench_adm2["intersection_length"] = trench_adm2["intersection_length"].apply(
        lambda values: values if isinstance(values, list) else []
    )
    trench_adm2 = trench_adm2.explode(
        ["adm2", "intersection_length"],
        ignore_index=True,
    )
    return trench_adm2.dropna(subset=["adm2"])


# ============== Kernel Functions ==============
def distance_weights(distances, kernel="gaussian", h=1000):
    """
    Compute distance-based kernel weights.
    
    Parameters
    ----------
    distances : array-like
        Array of distances in meters.
    kernel : str
        Kernel type: 'uniform', 'triangular', 'epanechnikov', 'gaussian', 'exponential'.
    h : float
        Bandwidth parameter in meters.
    
    Returns
    -------
    np.ndarray
        Array of weights.
    """
    d = np.asarray(distances, dtype=float)
    if kernel == "uniform":
        w = (d <= h).astype(float)
    elif kernel == "triangular":
        w = np.clip(1 - d/h, 0, None)
    elif kernel == "epanechnikov":
        w = np.clip(1 - (d/h)**2, 0, None)
    elif kernel == "gaussian":
        w = np.exp(-(d/h)**2)
    elif kernel == "exponential":
        w = np.exp(-d/h)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    return w


AVAILABLE_KERNELS = ['uniform', 'triangular', 'epanechnikov', 'gaussian', 'exponential']


# ============== Helper Functions ==============
def add_crs(geom, crs=4326):
    return Geometry(geom, crs)


def create_mappers(legend_path):
    """Create vectorized mappers from legend file."""
    legend = pd.read_excel(legend_path)
    legend_class_dict = legend.set_index("ID").Class.to_dict()
    legend_subclass_dict = (
        legend.assign(
            _subclass_summary=legend.apply(
                lambda row: _subclass_summary_id(row["Class"], row["Subclass"]),
                axis=1,
            )
        )
        .set_index("ID")["_subclass_summary"]
        .to_dict()
    )
    
    legend_class_dict_mapper = np.vectorize(lambda x: legend_class_dict.get(x, np.nan))
    legend_subclass_dict_mapper = np.vectorize(
        lambda x: (
            np.nan
            if (subclass_value := legend_subclass_dict.get(x, np.nan)) is None
            else subclass_value
        )
    )
    
    return legend_class_dict_mapper, legend_subclass_dict_mapper


def get_files(datadir):
    """Get sorted file list with year index."""
    files = pd.Series(os.listdir(datadir)).sort_values()
    years = files.str.extract(r"(\d{4,})").iloc[:, 0].astype("int")
    return files.set_axis(years)


def process_year(year, file, datadir, drainage_polygons, legend_path, output_columns, log_level=None):
    """Process all polygons for a single year."""
    if log_level is not None:
        configure_logging(log_level)
    logger.info(f"Processing year {year}")
    
    try:
        legend_class_mapper, legend_subclass_mapper = create_mappers(legend_path)
        lc = rxr.open_rasterio(datadir + file, chunks=None)
    except Exception as e:
        logger.error(f"Failed to load raster for year {year}: {e}")
        # Return empty DataFrame for this year
        mi = _build_year_multiindex(drainage_polygons, year)
        return pd.DataFrame(0, index=mi, columns=output_columns).reset_index()
    
    # Create output structure for this year
    mi = _build_year_multiindex(drainage_polygons, year)
    year_df = pd.DataFrame(0, index=mi, columns=output_columns)
    
    # Track progress
    n_polygons = len(drainage_polygons)
    n_success = 0
    n_no_overlap = 0
    n_errors = 0
    
    for idx, j in enumerate(drainage_polygons.index):
        try:
            geometry = drainage_polygons.loc[j, 'geometry']
            
            # Skip invalid geometries
            if geometry is None or geometry.is_empty:
                n_errors += 1
                continue
            
            # Try to crop the raster with this polygon
            try:
                cropped = lc.odc.crop(add_crs(geometry))
                values, counts = np.unique(cropped, return_counts=True)
                n_success += 1
            except (ValueError, RuntimeError, Exception) as e:
                # Polygon doesn't overlap with raster or other cropping error
                error_msg = str(e)
                if 'overlap' in error_msg.lower() or 'extent' in error_msg.lower():
                    n_no_overlap += 1
                    if n_no_overlap <= 10:  # Only log first 10 to avoid spam
                        logger.debug(f"Polygon {j} doesn't overlap with raster extent")
                else:
                    n_errors += 1
                    logger.warning(f"Error cropping polygon {j}: {e}")
                values, counts = np.array([]), np.array([])
            
            # Process land cover classes if we have data
            if len(values) > 0:
                year_df.loc[(j, year), LAND_COVER_TOTAL_COLUMN] = int(np.sum(counts))
                for mapper in (legend_class_mapper, legend_subclass_mapper):
                    classes, weights = _mapped_classes_and_weights(values, counts, mapper)
                    
                    if len(classes) == 0:
                        continue
                    
                    uniq, inv = np.unique(classes, return_inverse=True)
                    agg = np.bincount(inv, weights=weights)
                    year_df.loc[(j, year), _class_column_names(uniq)] = agg
            
            # Log progress periodically
            if (idx + 1) % 100000 == 0:
                logger.info(f"Year {year}: processed {idx + 1}/{n_polygons} polygons "
                           f"(success: {n_success}, no_overlap: {n_no_overlap}, errors: {n_errors})")
                
        except Exception as e:
            error_msg = str(e)
            if "cannot convert float NaN to integer" in error_msg:
                continue
            n_errors += 1
            logger.error(f"Unexpected error processing polygon {j} in year {year}: {e}")
    
    logger.info(f"Completed year {year}: {n_success} successful, {n_no_overlap} no overlap, {n_errors} errors")
    return year_df.reset_index()


class LandCover:
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
        logger.debug(f"Initialized LandCover with datadir={datadir}")
    
    def fetch(self):
        """Fetch/download raw land cover data (placeholder for future implementation)."""
        logger.info("Land cover data should be downloaded manually from MapBiomas.")
        logger.info(f"Expected location: {self.datadir}")
    
    def preprocess(
        self,
        n_jobs=None,
        river_network_path=None,
        output_path="data/land_cover_results.feather",
        log_level=None,
    ):
        """
        Preprocess land cover data by extracting values for drainage polygons.
        Parallelizes by year using joblib.
        
        Parameters
        ----------
        n_jobs : int, optional
            Number of parallel jobs (default: all CPUs)
        river_network_path : str, optional
            Path to river network directory. If provided, uses drainage_areas from network.
            Otherwise uses drainage_path from configuration.
        output_path : str
            Path for output feather file
        log_level : str, optional
            Logging level to apply inside parallel workers so CLI formatting is
            preserved during extraction.
        """
        if n_jobs is None:
            n_jobs = cpu_count()
        if log_level is not None:
            configure_logging(log_level)
        
        logger.info(f"Starting preprocessing with n_jobs={n_jobs}")
        
        # Load drainage polygons
        if river_network_path:
            logger.info(f"Loading river network from {river_network_path}")
            network = rn_module.RiverNetwork()
            network.load(river_network_path)
            
            if network.drainage_areas is None:
                raise ValueError("River network does not have drainage_areas loaded")
            
            drainage_polygons = network.drainage_areas.to_crs(4326)
            logger.info(f"Loaded {len(drainage_polygons)} drainage areas from network")
            
            # Filter to within_brazil if column exists
            if 'within_brazil' not in drainage_polygons.columns:
                raise ValueError(
                    "Drainage areas missing 'within_brazil' column. "
                    "Run river-network generate with --gadm-path to annotate this column."
                )
            
            n_before = len(drainage_polygons)
            drainage_polygons = drainage_polygons[drainage_polygons['within_brazil']]
            n_after = len(drainage_polygons)
            logger.info(f"Filtered to {n_after}/{n_before} drainage areas within Brazil")
        else:
            logger.info(f"Loading drainage polygons from {self.drainage_path}")
            drainage_polygons = _deduplicate_drainage_polygons(
                gpd.read_feather(self.drainage_path)
            )
            logger.info(f"Loaded {len(drainage_polygons)} drainage polygons")
        
        files = get_files(self.datadir)
        logger.info(f"Found {len(files)} land cover files for years {files.index.min()}-{files.index.max()}")
        
        # Process all years in parallel using joblib
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(process_year)(
                year, file, self.datadir, drainage_polygons, 
                self.legend_path, self.output_columns, log_level
            )
            for year, file in files.items()
        )
        
        # Merge all results
        logger.info("Merging all results...")
        final_df = pd.concat(results, ignore_index=True)
        final_df = final_df.sort_values(
            [TRENCH_ID_COLUMN, YEAR_COLUMN]
        ).reset_index(drop=True)
        
        output_path = Path(output_path)
        final_df.to_feather(output_path)
        logger.info(f"Results saved to {output_path}")
        
        return final_df
    
    def aggregate_along_rivers(
        self,
        land_cover_path,
        river_network_path,
        drainage_polygons_path,
        kernel='gaussian',
        h=1000000,
        years=None,
        n_jobs=None,
        output_path="land_cover_river_aggregated.feather"
    ):
        """
        Aggregate land cover variables upstream of each ADM2 unit.

        For each ADM2 unit, identifies the trenches inside the unit, expands them
        to all upstream trenches using the precomputed trench reachability matrices,
        and applies a distance-decay kernel to
        weighted land-cover sums.
        
        Parameters
        ----------
        land_cover_path : str
            Path to preprocessed land cover feather file (output of preprocess()).
        river_network_path : str
            Path to river network directory containing river-network parquet files
            and `system_matrices.pkl`.
        drainage_polygons_path : str
            Path to drainage polygons feather file. Used only when drainage areas
            are not available inside the saved river-network directory.
        kernel : str
            Kernel function for spatial weighting. Options:
            - 'uniform': weight 1 for d <= h, 0 otherwise
            - 'triangular': 1 - d/h, clamped to [0, 1]
            - 'epanechnikov': 1 - (d/h)^2, clamped to [0, 1]
            - 'gaussian': exp(-(d/h)^2)
            - 'exponential': exp(-d/h)
        h : float
            Bandwidth parameter for the kernel in meters. Default is 1,000,000 (1000 km).
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
            Columns include `adm2_id`, `year`, and weighted land-cover class areas.
        """
        if n_jobs is None:
            n_jobs = cpu_count()
        
        if kernel not in AVAILABLE_KERNELS:
            raise ValueError(f"Unknown kernel: {kernel}. Available: {AVAILABLE_KERNELS}")
        
        logger.info(f"Loading land cover data from {land_cover_path}")
        land_cover_df = pd.read_feather(land_cover_path)
        _validate_land_cover_output_columns(land_cover_df)

        logger.info(f"Loading river network from {river_network_path}")
        network_dir = Path(river_network_path)
        network = rn_module.RiverNetwork()
        network.load(str(network_dir))

        if not network.trench_reachability_matrices:
            raise ValueError("River network must have trench reachability data computed.")
        if network.trenches is None:
            raise ValueError("River network must include trench data.")

        if network.drainage_areas is not None:
            drainage_polygons = _deduplicate_drainage_polygons(network.drainage_areas.copy())
        else:
            raise ValueError("River network must include drainage polygon data.")

        rivers = network.trenches
        drainage_columns = {TRENCH_ID_COLUMN}
        missing_drainage_columns = drainage_columns.difference(drainage_polygons.columns)
        if missing_drainage_columns:
            raise ValueError(
                "Drainage polygons are missing required columns: "
                f"{sorted(missing_drainage_columns)}."
            )

        trench_adm2_matches = _explode_trench_adm2_matches(rivers)
        trench_columns = [TRENCH_ID_COLUMN, rn_module.SYSTEM_ID_KEY, "adm2"]
        trench_lookup = drainage_polygons[[TRENCH_ID_COLUMN]].merge(
            trench_adm2_matches[trench_columns].drop_duplicates(),
            on=TRENCH_ID_COLUMN,
            how="left",
            validate="one_to_many",
        )
        trench_lookup = trench_lookup.dropna(
            subset=[rn_module.SYSTEM_ID_KEY]
        )

        # Get land cover columns (exclude index-like columns)
        lc_columns = [
            c for c in land_cover_df.columns
            if c not in [TRENCH_ID_COLUMN, YEAR_COLUMN]
        ]
        logger.info(f"Land cover columns: {lc_columns}")

        land_cover_by_trench_year = land_cover_df.groupby(
            [TRENCH_ID_COLUMN, YEAR_COLUMN]
        )[lc_columns].sum().sort_index()

        # Get available years
        if years is None:
            years = land_cover_by_trench_year.index.get_level_values(YEAR_COLUMN).unique().tolist()
        
        logger.info(f"Processing years: {years}")
        
        # Get unique ADM2 units
        adm2_units = trench_lookup["adm2"].dropna().unique()
        logger.info(f"Processing {len(adm2_units)} ADM2 units")

        trench_index_columns = {
            TRENCH_ID_COLUMN,
            rn_module.SYSTEM_ID_KEY,
            rn_module.TRENCH_INDEX_COLUMN,
        }
        missing_trench_index_columns = trench_index_columns.difference(rivers.columns)
        if missing_trench_index_columns:
            raise ValueError(
                "River trench data is missing matrix index columns: "
                f"{sorted(missing_trench_index_columns)}. "
                "Recompute river matrices with RiverNetwork.compute_distance_matrices()."
            )

        system_trench_tables = {
            int(system_id): system_trenches[[TRENCH_ID_COLUMN, rn_module.TRENCH_INDEX_COLUMN]]
            .sort_values(rn_module.TRENCH_INDEX_COLUMN)
            .reset_index(drop=True)
            for system_id, system_trenches in rivers.groupby(rn_module.SYSTEM_ID_KEY)
        }
        system_trench_id_arrays = {
            system_id: system_trenches[TRENCH_ID_COLUMN].to_numpy(dtype=np.int64)
            for system_id, system_trenches in system_trench_tables.items()
        }
        system_trench_positions = {
            system_id: dict(
                zip(
                    system_trenches[TRENCH_ID_COLUMN].to_numpy(dtype=np.int64),
                    system_trenches[rn_module.TRENCH_INDEX_COLUMN].to_numpy(dtype=np.int64),
                )
            )
            for system_id, system_trenches in system_trench_tables.items()
        }
        
        def process_adm2(adm2_id):
            """Process a single ADM2 unit for all years."""
            try:
                adm2_trenches = trench_lookup.loc[
                    trench_lookup["adm2"] == adm2_id,
                    [TRENCH_ID_COLUMN, rn_module.SYSTEM_ID_KEY],
                ].drop_duplicates()

                if adm2_trenches.empty:
                    return None

                trench_distance_frames = []
                for system_id, system_adm2_trenches in adm2_trenches.groupby(
                    rn_module.SYSTEM_ID_KEY
                ):
                    system_id = int(system_id)
                    system_trench_ids = system_trench_id_arrays.get(
                        system_id,
                        np.asarray([], dtype=np.int64),
                    )
                    if len(system_trench_ids) == 0:
                        continue

                    trench_position_lookup = system_trench_positions[system_id]
                    seed_positions = np.asarray(
                        [
                            trench_position_lookup[trench_id]
                            for trench_id in system_adm2_trenches[TRENCH_ID_COLUMN]
                            if trench_id in trench_position_lookup
                        ],
                        dtype=np.int64,
                    )

                    if len(seed_positions) == 0:
                        continue

                    system_reachability = network.trench_reachability_matrices[system_id][
                        seed_positions, :
                    ].tocsr()
                    system_distance = network.trench_distance_matrices[system_id][
                        seed_positions, :
                    ].tocsr()

                    min_distances = np.full(len(system_trench_ids), np.inf)
                    for row_idx in range(system_reachability.shape[0]):
                        reach_start = system_reachability.indptr[row_idx]
                        reach_end = system_reachability.indptr[row_idx + 1]
                        reachable_cols = system_reachability.indices[reach_start:reach_end]
                        if len(reachable_cols) == 0:
                            continue

                        dist_start = system_distance.indptr[row_idx]
                        dist_end = system_distance.indptr[row_idx + 1]
                        distance_lookup = dict(
                            zip(
                                system_distance.indices[dist_start:dist_end],
                                system_distance.data[dist_start:dist_end],
                            )
                        )
                        reachable_distances = np.asarray(
                            [distance_lookup.get(col, 0.0) for col in reachable_cols],
                            dtype=float,
                        )
                        np.minimum.at(min_distances, reachable_cols, reachable_distances)

                    reachable_mask = np.isfinite(min_distances)
                    if not np.any(reachable_mask):
                        continue

                    trench_distance_frames.append(
                        pd.DataFrame(
                            {
                                TRENCH_ID_COLUMN: system_trench_ids[
                                    reachable_mask
                                ],
                                "upstream_distance": min_distances[reachable_mask],
                            }
                        )
                    )

                if not trench_distance_frames:
                    return None

                trench_distance_lookup = (
                    pd.concat(trench_distance_frames, ignore_index=True)
                    .groupby(TRENCH_ID_COLUMN, as_index=False)["upstream_distance"]
                    .min()
                    .set_index(TRENCH_ID_COLUMN)["upstream_distance"]
                )

                matched_mask = land_cover_by_trench_year.index.get_level_values(
                    TRENCH_ID_COLUMN
                ).isin(trench_distance_lookup.index)
                df_matched = land_cover_by_trench_year.loc[matched_mask].reset_index()

                if len(df_matched) == 0:
                    return None

                df_matched["upstream_distance"] = df_matched[TRENCH_ID_COLUMN].map(
                    trench_distance_lookup
                )
                weights = distance_weights(
                    df_matched["upstream_distance"].to_numpy(),
                    kernel=kernel,
                    h=h,
                )

                # Compute weighted sums per year
                results = []
                for year in years:
                    try:
                        year_mask = df_matched[YEAR_COLUMN] == year
                        df_year = df_matched.loc[year_mask]
                        w_year = weights[year_mask.to_numpy()]
                        
                        if len(df_year) == 0 or np.sum(w_year) == 0:
                            continue
                        
                        # Weighted sum
                        weighted_sum = np.sum(
                            df_year[lc_columns].to_numpy() * w_year[:, None],
                            axis=0,
                        )
                        result = {col: val for col, val in zip(lc_columns, weighted_sum)}
                        result[ADM2_ID_COLUMN] = adm2_id
                        result[YEAR_COLUMN] = int(year)
                        result[REACHABLE_TRENCH_COUNT_COLUMN] = len(df_year)
                        result[TOTAL_WEIGHT_COLUMN] = float(np.sum(w_year))
                        
                        results.append(result)
                    except Exception as e:
                        logger.warning(f"Error processing ADM2 {adm2_id}, year {year}: {e}")
                        continue
                
                return results
                
            except Exception as e:
                logger.warning(f"Error processing ADM2 {adm2_id}: {e}")
                return None
        
        # Process all ADM2 units in parallel
        logger.info(f"Processing {len(adm2_units)} ADM2 units with {n_jobs} workers")
        
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(process_adm2)(adm2_id) 
            for adm2_id in tqdm(adm2_units, desc="ADM2 units")
        )
        
        # Flatten results and filter None
        all_results = []
        for r in results:
            if r is not None:
                all_results.extend(r)
        
        if not all_results:
            logger.warning("No results produced")
            return pd.DataFrame()
        
        result_df = pd.DataFrame(all_results)
        result_df = result_df.sort_values(
            [ADM2_ID_COLUMN, YEAR_COLUMN]
        ).reset_index(drop=True)
        
        # Save results
        output_path = Path(output_path)
        result_df.to_feather(output_path)
        logger.info(f"Results saved to {output_path}")
        logger.info(f"Output shape: {result_df.shape}")
        
        return result_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process land cover data in parallel")
    parser.add_argument("--n_jobs", type=int, default=None, help="Number of parallel jobs")
    parser.add_argument("--output", type=str, default="land_cover_results.feather", help="Output file path")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for standalone execution (default: INFO)",
    )

    args = parser.parse_args()
    configure_logging(args.log_level)
    logger.info("Starting standalone land-cover preprocessing")

    lc = LandCover()
    lc.preprocess(n_jobs=args.n_jobs, output_path=args.output)
    logger.info("Completed standalone land-cover preprocessing")
