import argparse
import logging

from .core import CommodityPrices


logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging for standalone commodity-price execution."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def configure_parser(parser, include_action=True):
    """Add commodity-price CLI arguments to ``parser``."""
    if include_action:
        parser.add_argument("action", choices=["fetch", "preprocess"])
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--monthly-output", default=None)
    parser.add_argument("--annual-output", default=None)
    return parser


def run(args):
    """Execute the requested commodity-price action for parsed ``args``."""
    agent = CommodityPrices(root_dir=args.root_dir)
    if args.action == "fetch":
        agent.fetch()
    else:
        agent.preprocess(monthly_output_path=args.monthly_output, annual_output_path=args.annual_output)


def main():
    parser = argparse.ArgumentParser(description="Run commodity-price workflows")
    configure_parser(parser)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for standalone execution",
    )

    args = parser.parse_args()
    configure_logging(args.log_level)
    logger.info("Starting standalone commodity-price %s", args.action)
    run(args)
    logger.info("Completed standalone commodity-price %s", args.action)


if __name__ == "__main__":
    main()
