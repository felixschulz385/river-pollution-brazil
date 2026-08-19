import argparse
import logging

from .core import Climate


logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging for standalone climate execution."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def configure_parser(parser, include_action=True):
    """Add climate CLI arguments to ``parser``."""
    if include_action:
        parser.add_argument("action", choices=["fetch", "preprocess", "assemble"])
    parser.add_argument("--root-dir", default=".")
    parser.add_argument(
        "--subtype",
        default="era5_land_hourly",
        choices=["era5_land_hourly", "era5_land_daily", "era5_land_arco"],
        help="Fetch variant only -- ignored for preprocess, which always processes "
        "every GRIB-origin variant together since they share one zarr store.",
    )
    parser.add_argument("--stage", default="all", choices=["all", "zarr", "parquet"])
    parser.add_argument(
        "--variant",
        default="sensor_upstream_distance_buckets",
        choices=["sensor_upstream_distance_buckets", "adm2_upstream_yearly"],
    )
    parser.add_argument("--climate-path", default=None)
    parser.add_argument("--water-quality-path", default=None)
    parser.add_argument("--stations-rivers-path", default=None)
    parser.add_argument("--river-network-path", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--n_jobs", type=int, default=None)
    return parser


def run(args):
    """Execute the requested climate action for parsed ``args``."""
    agent = Climate(root_dir=args.root_dir)
    if args.action == "fetch":
        agent.fetch(subtype=args.subtype)
    elif args.action == "preprocess":
        agent.preprocess(stage=args.stage, n_jobs=args.n_jobs)
    else:
        agent.assemble(
            variant=args.variant,
            climate_path=args.climate_path,
            water_quality_path=args.water_quality_path,
            stations_rivers_path=args.stations_rivers_path,
            river_network_path=args.river_network_path,
            output_path=args.output,
            n_jobs=args.n_jobs,
        )


def main():
    parser = argparse.ArgumentParser(description="Run climate-data workflows")
    configure_parser(parser)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for standalone execution",
    )

    args = parser.parse_args()
    configure_logging(args.log_level)
    logger.info("Starting standalone climate %s", args.action)
    run(args)
    logger.info("Completed standalone climate %s", args.action)


if __name__ == "__main__":
    main()
