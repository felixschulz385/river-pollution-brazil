import argparse
import logging

from .constants import ADM2_ASSEMBLY_VARIANT, ASSEMBLY_VARIANTS, LAND_COVER_PRODUCT, PRODUCTS
from .core import LandCover
from .preprocess import configure_logging
from src.data.sources.river_network.constants import PROCESSED_DIR as RIVER_NETWORK_PROCESSED_DIR


logger = logging.getLogger(__name__)


def configure_parser(parser, include_action=True):
    """Add land-cover CLI arguments to ``parser``."""
    if include_action:
        parser.add_argument("action", choices=["fetch", "preprocess", "assemble"])
    parser.add_argument("--root-dir", default=".")
    parser.add_argument(
        "--product",
        default=LAND_COVER_PRODUCT,
        choices=PRODUCTS,
        help="Raster set to build: 'land_cover' (MapBiomas Collection 10 coverage, default) or "
        "'mining' (Collection 11 mining substances). Sets the default --datadir, --legend-path, "
        "extract output and aggregate input/output; 'mining' aggregates to ADM2 only.",
    )
    parser.add_argument("--n_jobs", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--river-network-path", default=None)
    parser.add_argument("--datadir", default=None)
    parser.add_argument("--drainage-path", default=None)
    parser.add_argument("--legend-path", default=None)
    parser.add_argument(
        "--variant",
        default="all",
        choices=["all", *ASSEMBLY_VARIANTS],
        help="Assemble variant -- 'all' (default) runs the sensor and ADM2 "
        "upstream panels sequentially (only ADM2 for --product mining); "
        "--output is not supported with 'all' since each variant writes its "
        "own default output path.",
    )
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
        product=args.product,
    )
    if args.action == "fetch":
        agent.fetch()
    elif args.action == "preprocess":
        agent.preprocess(
            n_jobs=args.n_jobs,
            river_network_path=args.river_network_path,
            output_path=args.output,
            log_level=getattr(args, "log_level", None),
        )
    else:
        product_variants = ASSEMBLY_VARIANTS if args.product == LAND_COVER_PRODUCT else (ADM2_ASSEMBLY_VARIANT,)
        variants = list(product_variants) if args.variant == "all" else [args.variant]
        if args.variant == "all" and len(variants) > 1 and args.output is not None:
            raise ValueError("--output is not supported with --variant all; run each variant separately.")
        for variant in variants:
            agent.assemble(
                variant=variant,
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
