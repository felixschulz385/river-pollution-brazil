import argparse
import logging

from .core import Gadm


logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging for standalone gadm execution."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def configure_parser(parser, include_action=True):
    """Add gadm CLI arguments to ``parser``."""
    if include_action:
        parser.add_argument("action", choices=["preprocess"])
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--gadm-path", default=None)
    parser.add_argument("--adm0-layer", default=None)
    parser.add_argument("--adm2-layer", default=None)
    parser.add_argument("--tolerance", type=float, default=None)
    parser.add_argument("--output-path", default=None)
    return parser


def run(args):
    """Execute the requested gadm action for parsed ``args``."""
    agent = Gadm(root_dir=args.root_dir)
    agent.preprocess(
        gadm_path=args.gadm_path,
        adm0_layer=args.adm0_layer,
        adm2_layer=args.adm2_layer,
        tolerance=args.tolerance,
        output_path=args.output_path,
    )


def main():
    parser = argparse.ArgumentParser(description="Simplify GADM boundaries")
    configure_parser(parser)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for standalone execution",
    )

    args = parser.parse_args()
    configure_logging(args.log_level)
    run(args)
    logger.info("Completed standalone gadm preprocessing")


if __name__ == "__main__":
    main()
