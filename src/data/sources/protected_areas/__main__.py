import argparse
import logging

from .constants import AREAS_PRODUCT, PRODUCTS, TRENCHES_PRODUCT
from .core import ProtectedAreas


logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging for standalone protected-areas execution."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def configure_parser(parser, include_action=True):
    """Add protected-areas CLI arguments to ``parser``."""
    if include_action:
        parser.add_argument("action", choices=["fetch", "preprocess"])
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--gpkg", default=None)
    parser.add_argument("--funai", default=None, help="FUNAI tis_poligonais_portarias.zip (default: newest snapshot)")
    parser.add_argument("--gadm-path", default=None)
    parser.add_argument("--gadm-layer", default=None)
    parser.add_argument("--adm2-id-column", default=None)
    parser.add_argument("--areas-output", default=None)
    parser.add_argument("--adm2-output", default=None)
    parser.add_argument(
        "--product",
        default=AREAS_PRODUCT,
        choices=PRODUCTS,
        help="'areas' (default): the merged areas and municipality shares; 'trenches': the share of each river "
        "trench's drainage polygon inside each area (needs the areas and river_network).",
    )
    parser.add_argument("--drainage-path", default=None, help="Drainage polygons (default: river_network output)")
    parser.add_argument("--trenches-output", default=None)
    return parser


def run(args):
    """Execute the requested protected-areas action for parsed ``args``."""
    agent = ProtectedAreas(root_dir=args.root_dir)
    if args.action == "fetch":
        agent.fetch()
    elif args.product == TRENCHES_PRODUCT:
        agent.preprocess_trenches(areas_path=args.areas_output, drainage_path=args.drainage_path,
                                  output_path=args.trenches_output)
    else:
        agent.preprocess(
            gpkg=args.gpkg,
            funai=args.funai,
            gadm_path=args.gadm_path,
            layer=args.gadm_layer,
            adm2_id_column=args.adm2_id_column,
            areas_output_path=args.areas_output,
            adm2_output_path=args.adm2_output,
        )


def main():
    parser = argparse.ArgumentParser(description="Run protected-areas workflows")
    configure_parser(parser)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for standalone execution",
    )

    args = parser.parse_args()
    configure_logging(args.log_level)
    logger.info("Starting standalone protected-areas %s", args.action)
    run(args)
    logger.info("Completed standalone protected-areas %s", args.action)


if __name__ == "__main__":
    main()
