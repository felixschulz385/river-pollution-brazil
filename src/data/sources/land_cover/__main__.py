import argparse
import logging

from .constants import DEFAULT_ASSEMBLY_LAND_COVER_PATH
from .core import LandCover
from .preprocess import configure_logging
from src.data.sources.river_network.constants import PROCESSED_DIR as RIVER_NETWORK_PROCESSED_DIR


logger = logging.getLogger(__name__)


def configure_parser(parser, include_action=True):
    """Add land-cover CLI arguments to ``parser``."""
    if include_action:
        parser.add_argument("action", choices=["fetch", "preprocess", "assemble"])
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--n_jobs", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--river-network-path", default=None)
    parser.add_argument("--datadir", default=None)
    parser.add_argument("--drainage-path", default=None)
    parser.add_argument("--legend-path", default=None)
    parser.add_argument("--variant", default="sensor")
    parser.add_argument("--land-cover-path", default=None)
    parser.add_argument("--water-quality-path", default=None)
    parser.add_argument("--stations-rivers-path", default=None)
    return parser


def run(args):
    """Execute the requested land-cover action for parsed ``args``."""
    agent = LandCover(
        root_dir=args.root_dir,
        datadir=args.datadir,
        drainage_path=args.drainage_path,
        legend_path=args.legend_path,
    )
    if args.action == "fetch":
        agent.fetch()
    elif args.action == "preprocess":
        agent.preprocess(
            n_jobs=args.n_jobs,
            river_network_path=args.river_network_path,
            output_path=args.output or DEFAULT_ASSEMBLY_LAND_COVER_PATH,
            log_level=getattr(args, "log_level", None),
        )
    else:
        agent.assemble(
            variant=args.variant,
            land_cover_path=args.land_cover_path,
            water_quality_path=args.water_quality_path,
            stations_rivers_path=args.stations_rivers_path,
            river_network_path=args.river_network_path or RIVER_NETWORK_PROCESSED_DIR,
            output_path=args.output,
            n_jobs=args.n_jobs,
        )


def main():
    parser = argparse.ArgumentParser(description="Run land-cover workflows")
    configure_parser(parser)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for standalone execution",
    )

    args = parser.parse_args()
    configure_logging(args.log_level)
    logger.info("Starting standalone land-cover %s", args.action)
    run(args)
    logger.info("Completed standalone land-cover %s", args.action)


if __name__ == "__main__":
    main()
