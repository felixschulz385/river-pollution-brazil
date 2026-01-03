#!/usr/bin/env python3

"""
Data processing CLI for Master Thesis project

This script serves as a command-line interface (CLI) for processing various data modules
related to the Master Thesis project. It allows users to fetch and preprocess health data,
water quality data, and download specific datasets as required for the thesis research.

Usage:
  python cli.py <module> <action> [options]

Modules:
  health          Process health data
  water-quality   Process water quality data
  land-cover      Process land cover data
  download         Download datasets

Actions:
  fetch           Fetch data from source
  preprocess      Preprocess the fetched data

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
  python cli.py land-cover fetch
  python cli.py land-cover preprocess --n_jobs 16 --output results.feather
  python cli.py download --dataset dem --area BRA --year 2010
"""

import argparse
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


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
        module (str): Name of the data module ('health', 'water-quality', 'download')
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
        elif module == "land-cover":
            from land_cover import land_cover
            return land_cover()
        elif module == "river-network":
            from river_network import river_network
            return river_network
        elif module == "download":
            from download import download_agent
            return download_agent(
                root_dir=kwargs.get('root_dir', '/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/'),
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
  python cli.py download --dataset dem --area BRA --year 2010
        """
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
    
    # River network module
    rn_parser = subparsers.add_parser("river-network", help="Process river network data")
    rn_parser.add_argument(
        "action",
        choices=["compute-reachability", "debug"],
        help="Action to perform"
    )
    rn_parser.add_argument(
        "--shapefile-path",
        required=False,
        help="Path to shapefile feather file"
    )
    rn_parser.add_argument(
        "--topology-path",
        required=False,
        help="Path to topology feather file"
    )
    rn_parser.add_argument(
        "--distance-path",
        required=False,
        help="Path to distance_from_estuary feather file"
    )
    rn_parser.add_argument(
        "--reachability-path",
        required=False,
        help="Path to reachability npz file"
    )
    rn_parser.add_argument(
        "--output-dir",
        default="./",
        help="Output directory for results (default: ./)"
    )
    
    # Download module
    download_parser = subparsers.add_parser("download", help="Download datasets")
    download_parser.add_argument(
        "--root-dir",
        default="/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/",
        help="Root directory for data storage"
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
    
    args = parser.parse_args()
    
    if args.module is None:
        parser.print_help()
        sys.exit(1)
    
    try:
        # Use factory to create the appropriate data source instance
        if args.module == "download":
            agent = DataSourceFactory.create(
                args.module,
                root_dir=args.root_dir,
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
                logger.info(f"Preprocessing with n_jobs={args.n_jobs}")
                agent.preprocess(
                    n_jobs=args.n_jobs,
                    output_path=args.output
                )

        elif args.module == "river-network":
            action = args.action
            logger.info(f"Running {args.module} module: {action}")
            
            if action == "debug":
                # Debug configuration for testing reachability
                logger.info("Starting debug configuration for reachability computation")
                
                if not args.shapefile_path or not args.topology_path:
                    logger.error("Debug mode requires --shapefile-path and --topology-path")
                    sys.exit(1)
                
                logger.info(f"Loading shapefile from: {args.shapefile_path}")
                logger.info(f"Loading topology from: {args.topology_path}")
                
                from river_network import river_network
                
                try:
                    # Load existing network
                    network = river_network.load_network(
                        args.shapefile_path,
                        args.topology_path,
                        distance_path=args.distance_path,
                        reachability_path=args.reachability_path
                    )
                    logger.info("✓ Network loaded successfully")
                    logger.info(f"  Shapefile shape: {network.shapefile.shape}")
                    logger.info(f"  Topology shape: {network.topology.shape}")
                    if network.distance_from_estuary is not None:
                        logger.info(f"  Distance data shape: {network.distance_from_estuary.shape}")
                    if network.reachability is not None:
                        logger.info(f"  Reachability matrix shape: {network.reachability['matrix'].shape}")
                    
                    # Compute distances from estuaries if not present
                    if network.distance_from_estuary is None:
                        logger.info("Computing distances from estuary...")
                        network.calculate_distance_from_estuary()
                        logger.info(f"✓ Distances computed successfully")
                        
                        # Store distance data
                        logger.info(f"Storing distance_from_estuary to: {args.output_dir}")
                        network.store_distance_from_estuary(args.output_dir)
                        logger.info("✓ Distance data stored successfully")
                    else:
                        logger.info("✓ Distance data already loaded")
                    
                    # Compute reachability
                    logger.info("Computing reachability graph...")
                    reachability_data = network.compute_reachability()
                    logger.info(f"✓ Reachability computed successfully")
                    logger.info(f"  Matrix shape: {reachability_data['matrix'].shape}")
                    logger.info(f"  Non-zero entries: {reachability_data['matrix'].nnz}")
                    
                    # Store reachability results
                    logger.info(f"Storing reachability to: {args.output_dir}")
                    network.store_reachability(args.output_dir)
                    logger.info("✓ Reachability data stored successfully")
                    
                except Exception as e:
                    logger.error(f"Error during debug execution: {str(e)}")
                    logger.debug("Traceback:", exc_info=True)
                    sys.exit(1)
            
            elif action == "compute-reachability":
                logger.info("Computing reachability for river network")
                
                if not args.shapefile_path or not args.topology_path:
                    logger.error("compute-reachability requires --shapefile-path and --topology-path")
                    sys.exit(1)
                
                from river_network import river_network
                
                try:
                    # Load network
                    logger.info(f"Loading network from {args.shapefile_path} and {args.topology_path}")
                    network = river_network.load_network(
                        args.shapefile_path,
                        args.topology_path,
                        distance_path=args.distance_path,
                        reachability_path=args.reachability_path
                    )
                    logger.info("✓ Network loaded")
                    logger.info(f"  Shapefile shape: {network.shapefile.shape}")
                    logger.info(f"  Topology shape: {network.topology.shape}")
                    if network.distance_from_estuary is not None:
                        logger.info(f"  Distance data shape: {network.distance_from_estuary.shape}")
                    
                    # Compute distances from estuaries if not present
                    if network.distance_from_estuary is None:
                        logger.info("Computing distances from estuary...")
                        network.calculate_distance_from_estuary()
                        logger.info("✓ Distances computed")
                        
                        logger.info(f"Storing distance_from_estuary to: {args.output_dir}")
                        network.store_distance_from_estuary(args.output_dir)
                        logger.info("✓ Distance data stored")
                    else:
                        logger.info("✓ Distance data already loaded")
                    
                    # Compute reachability
                    logger.info("Computing reachability...")
                    reachability_data = network.compute_reachability()
                    logger.info("✓ Reachability computed")
                    logger.info(f"  Matrix shape: {reachability_data['matrix'].shape}")
                    logger.info(f"  Non-zero entries: {reachability_data['matrix'].nnz}")
                    
                    # Store reachability results
                    logger.info(f"Storing reachability to: {args.output_dir}")
                    network.store_reachability(args.output_dir)
                    logger.info("✓ Reachability stored")
                    
                except Exception as e:
                    logger.error(f"Error computing reachability: {str(e)}")
                    logger.debug("Traceback:", exc_info=True)
                    sys.exit(1)

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