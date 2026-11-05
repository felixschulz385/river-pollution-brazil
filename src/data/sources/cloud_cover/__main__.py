import argparse
import logging

from .constants import ALERTS_PRODUCT, CLOUDS_PRODUCT, PRODUCTS
from .core import CloudCover


logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging for standalone cloud-cover execution."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def configure_parser(parser, include_action=True):
    """Add cloud-cover CLI arguments to ``parser``."""
    if include_action:
        parser.add_argument("action", choices=["preprocess"])
    parser.add_argument("--root-dir", default=".")
    parser.add_argument(
        "--product",
        default=CLOUDS_PRODUCT,
        choices=PRODUCTS,
        help="'clouds' (default): DETER's monthly cloud shares, 2005-2016, from the openICPSR 132281 package; "
        "'deter_alerts': DETER-B alerts per municipality and month from the TerraBrasilis download.",
    )
    parser.add_argument("--raw", default=None, help="The raw file (default: the package, or the newest DETER-B zip)")
    parser.add_argument("--gadm-path", default=None)
    parser.add_argument("--gadm-layer", default=None)
    parser.add_argument("--adm2-id-column", default=None)
    parser.add_argument("--month-output", default=None)
    parser.add_argument("--adm2-output", default=None)
    parser.add_argument("--workers", type=int, default=None, help="Months measured in parallel (default: "
                        "SLURM_CPUS_PER_TASK or 1)")
    return parser


def run(args):
    """Execute the requested cloud-cover action for parsed ``args``."""
    agent = CloudCover(root_dir=args.root_dir)
    options = dict(path=args.raw, gadm_path=args.gadm_path, layer=args.gadm_layer, adm2_id_column=args.adm2_id_column,
                   month_output_path=args.month_output, adm2_output_path=args.adm2_output)
    if args.product == ALERTS_PRODUCT:
        agent.preprocess_alerts(**options)
    else:
        agent.preprocess(**options, workers=args.workers)


def main():
    parser = argparse.ArgumentParser(description="Run DETER cloud-cover and alert workflows")
    configure_parser(parser)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for standalone execution",
    )

    args = parser.parse_args()
    configure_logging(args.log_level)
    logger.info("Starting standalone cloud-cover %s", args.action)
    run(args)
    logger.info("Completed standalone cloud-cover %s", args.action)


if __name__ == "__main__":
    main()
