#!/usr/bin/env python3

"""
Data processing CLI for Master Thesis project

This script serves as a command-line interface (CLI) for processing various data modules
related to the Master Thesis project. It allows users to fetch and preprocess health data,
water quality data, generate river network data, and download specific datasets as required
for the thesis research.

Usage:
  python cli.py <module> <action> [options]

Modules:
  health          Process health data
  water-quality   Process water quality data
  river-network   Process river network data
  download        Download datasets

Actions:
  fetch           Fetch data from source
  preprocess      Preprocess the fetched data
  generate        Generate processed river network data

Options:
  -h, --help      Show this help message and exit
  --root-dir      Root directory for data storage (default: /pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/)
  --area          Area identifier (default: BRA)
  --year          Year for data (default: 2010)
  --dataset       Dataset name to fetch (e.g., 'dem')

Examples:
  python cli.py health fetch --subtype mortality
  python cli.py health fetch --subtype hospitalization
  python cli.py health fetch --subtype birth
  python cli.py health fetch --subtype all
  python cli.py health preprocess
  python cli.py water-quality fetch
  python cli.py water-quality preprocess
  python cli.py river-network generate --gpkg-path /path/to/bho_2017.gpkg --output-dir ./river_data
  python cli.py river-network generate --gpkg-path /path/to/bho_2017.gpkg --output-dir ./river_data --min-lon -55 --min-lat -4 --max-lon -47 --max-lat 3
  python cli.py download --dataset dem --area BRA --year 2010
"""

import argparse
import sys
import logging

logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once for CLI and batch execution."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


class DataSourceFactory:
    """
    Factory class for creating data source instances.
    Uses lazy imports to load dependencies only when needed.
    """
    
    @staticmethod
    def create(module: str, **kwargs):
        """
        Create a data source instance based on the module name.
        
        Parameters:
        module (str): Name of the data module ('health', 'water-quality', 'download', 'river-network')
        **kwargs: Additional arguments to pass to the data source constructor
        
        Returns:
        Instance of the requested data source class
        
        Raises:
        ValueError: If the module name is not recognized
        """
        if module == "health":
            from health import health
            return health()
        elif module == "water-quality":
            from water_quality import water_quality
            return water_quality()
        elif module == "river-network":
            from river_network import RiverNetwork
            return RiverNetwork()
        elif module == "land-cover":
            from land_cover import LandCover
            return LandCover()
        elif module == "download":
            from download import download_agent
            return download_agent(
                remote_root_dir=kwargs.get('remote_root_dir', '/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/'),
                local_root_dir=kwargs.get('local_root_dir', '/tmp'),
                area=kwargs.get('area', 'BRA'),
                year=kwargs.get('year', 2010)
            )
        else:
            raise ValueError(f"Unknown module: {module}")


def main():
    """
    Main CLI entrypoint for data processing scripts.
    """
    parser = argparse.ArgumentParser(
        description="Data processing CLI for Master Thesis project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python cli.py health fetch --subtype mortality
        python cli.py health fetch --subtype hospitalization
        python cli.py health fetch --subtype birth
        python cli.py health fetch --subtype all
        python cli.py health preprocess
        python cli.py water-quality fetch
        python cli.py water-quality preprocess
        python cli.py land-cover fetch
        python cli.py land-cover preprocess --n_jobs 16 --output results.feather
        python cli.py land-cover preprocess --n_jobs 16 --output results.feather --river-network-path /path/to/network/
        python cli.py download --dataset dem --area BRA --year 2010
        """
        )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for CLI execution (default: INFO)",
    )
    
    subparsers = parser.add_subparsers(dest="module", help="Data module to process")
    
    # Health module
    health_parser = subparsers.add_parser("health", help="Process health data")
    health_parser.add_argument(
        "action",
        choices=["fetch", "preprocess"],
        help="Action to perform"
    )
    health_parser.add_argument(
        "--subtype",
        choices=["all", "mortality", "hospitalization", "birth"],
        default="all",
        help="Type of health data to fetch (default: all)"
    )
    
    # Water quality module
    wq_parser = subparsers.add_parser("water-quality", help="Process water quality data")
    wq_parser.add_argument(
        "action",
        choices=["fetch", "preprocess"],
        help="Action to perform"
    )
    
    # Land cover module
    lc_parser = subparsers.add_parser("land-cover", help="Process land cover data")
    lc_parser.add_argument(
        "action",
        choices=["fetch", "preprocess"],
        help="Action to perform"
    )
    lc_parser.add_argument(
        "--n_jobs",
        type=int,
        default=None,
        help="Number of parallel jobs (default: all CPUs)"
    )
    lc_parser.add_argument(
        "--output",
        default="land_cover_results.feather",
        help="Output file path (default: land_cover_results.feather)"
    )
    lc_parser.add_argument(
        "--river-network-path",
        default=None,
        help="Path to river network directory. If provided, uses drainage_areas from network instead of feather file"
    )
    
    # Download module
    download_parser = subparsers.add_parser("download", help="Download datasets")
    download_parser.add_argument(
        "--remote-root-dir",
        default="/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/",
        help="Remote root directory for data storage"
    )
    download_parser.add_argument(
        "--local-root-dir",
        default="/tmp",
        help="Local root directory for temporary data"
    )
    download_parser.add_argument(
        "--area",
        default="BRA",
        help="Area identifier (default: BRA)"
    )
    download_parser.add_argument(
        "--year",
        type=int,
        default=2010,
        help="Year for data (default: 2010)"
    )
    download_parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name to fetch (e.g., 'dem')"
    )
    
    # River network module
    river_parser = subparsers.add_parser("river-network", help="Process river network data")
    river_parser.add_argument(
        "action",
        choices=["generate"],
        help="Action to perform"
    )
    river_parser.add_argument(
        "--gpkg-path",
        required=True,
        help="Path to the GeoPackage file (e.g., '/path/to/bho_2017_v_01_05_5k.gpkg')"
    )
    river_parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for processed river network data"
    )
    river_parser.add_argument(
        "--min-lon",
        type=float,
        help="Minimum longitude for spatial filtering (optional)"
    )
    river_parser.add_argument(
        "--min-lat",
        type=float,
        help="Minimum latitude for spatial filtering (optional)"
    )
    river_parser.add_argument(
        "--max-lon",
        type=float,
        help="Maximum longitude for spatial filtering (optional)"
    )
    river_parser.add_argument(
        "--max-lat",
        type=float,
        help="Maximum latitude for spatial filtering (optional)"
    )
    river_parser.add_argument(
        "--gadm-path",
        help="Path to GADM GeoPackage for country filtering (optional)"
    )
    river_parser.add_argument(
        "--gadm-layer",
        default="ADM_ADM_0",
        help="GADM layer name for country boundary (default: ADM_ADM_0)"
    )
    river_parser.add_argument(
        "--gadm-adm2-layer",
        default="ADM_ADM_2",
        help="GADM layer name for ADM2 matches (default: ADM_ADM_2)"
    )
    
    args = parser.parse_args()
    configure_logging(args.log_level)
    
    if args.module is None:
        parser.print_help()
        sys.exit(1)
    
    try:
        # Use factory to create the appropriate data source instance
        if args.module == "download":
            agent = DataSourceFactory.create(
                args.module,
                remote_root_dir=args.remote_root_dir,
                local_root_dir=args.local_root_dir,
                area=args.area,
                year=args.year
            )
        else:
            agent = DataSourceFactory.create(args.module)
        
        # Execute the requested action
        if args.module in ["health", "water-quality"]:
            action = args.action
            logger.info(f"Running {args.module} module: {action}")
            
            if action == "fetch":
                if args.module == "health":
                    logger.info(f"Fetching health data (subtype: {args.subtype})")
                    agent.fetch(subtype=args.subtype)
                else:
                    agent.fetch()
            elif action == "preprocess":
                agent.preprocess()
        
        elif args.module == "land-cover":
            action = args.action
            logger.info(f"Running {args.module} module: {action}")
            
            if action == "fetch":
                agent.fetch()
            elif action == "preprocess":
                logger.info(
                    "Preprocessing with n_jobs=%s, output=%s, river_network_path=%s, log_level=%s",
                    args.n_jobs,
                    args.output,
                    args.river_network_path,
                    args.log_level,
                )
                agent.preprocess(
                    n_jobs=args.n_jobs,
                    river_network_path=args.river_network_path,
                    output_path=args.output,
                    log_level=args.log_level,
                )

        elif args.module == "river-network":
            logger.info("Running %s module: %s", args.module, args.action)
            
            if args.action == "generate":
                import geopandas as gpd
                from shapely.geometry import box
                
                # Create bounding box if coordinates provided
                bbox = None
                if all([args.min_lon, args.min_lat, args.max_lon, args.max_lat]):
                    bbox = gpd.GeoSeries(
                        box(args.min_lon, args.min_lat, args.max_lon, args.max_lat),
                        crs=4326
                    )
                    logger.info(
                        "Loading data with bbox: (%s, %s, %s, %s)",
                        args.min_lon,
                        args.min_lat,
                        args.max_lon,
                        args.max_lat,
                    )
                else:
                    logger.info("Loading full dataset (no spatial filter)")
                
                # Load and process data
                logger.info("Loading trenches")
                agent.load_trenches(args.gpkg_path, bbox=bbox)
                
                logger.info("Loading drainage areas")
                agent.load_drainage_areas(args.gpkg_path, bbox=bbox)
                
                logger.info("Computing subsystems")
                agent.compute_subsystems()
                
                logger.info("Computing distance matrices")
                agent.compute_distance_matrices()
                
                logger.info("Arranging by systems and distances")
                agent.sort_trenches_by_system()
                
                if args.gadm_path:
                    logger.info(
                        "Annotating drainage areas with country membership from %s layer %s",
                        args.gadm_path,
                        args.gadm_layer,
                    )
                    agent.annotate_drainage_areas_with_country_membership(
                        args.gadm_path,
                        layer=args.gadm_layer,
                    )
                    logger.info(
                        "Building trench-to-ADM2 matches from %s layer %s",
                        args.gadm_path,
                        args.gadm_adm2_layer,
                    )
                    agent.build_trench_adm2_table(
                        gadm_path=args.gadm_path,
                        layer=args.gadm_adm2_layer,
                    )
                
                logger.info("Saving to %s", args.output_dir)
                agent.save(args.output_dir)
        
        elif args.module == "download":
            logger.info(f"Running download module for dataset: {args.dataset}")
            # Create a simple dataset dictionary for the fetch method
            dataset = {"name": args.dataset}
            agent.fetch(dataset)
        
        logger.info(f"✓ {args.module} module completed successfully")
        
    except Exception as e:
        logger.error(f"✗ Error running {args.module} module: {str(e)}")
        logger.debug("Traceback:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
