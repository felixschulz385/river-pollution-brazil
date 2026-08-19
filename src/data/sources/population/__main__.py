import argparse
import logging

from .core import Population


logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging for standalone population execution."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def configure_parser(parser, include_action=True):
    """Add population CLI arguments to ``parser``."""
    if include_action:
        parser.add_argument("action", choices=["fetch", "preprocess"])
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--billing-project", default="river-pollution-499210")
    return parser


def run(args):
    """Execute the requested population action for parsed ``args``."""
    agent = Population(root_dir=args.root_dir, billing_project=args.billing_project)
    if args.action == "fetch":
        agent.fetch()
    else:
        agent.preprocess()


def main():
    parser = argparse.ArgumentParser(description="Run population-data workflows")
    configure_parser(parser)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for standalone execution",
    )

    args = parser.parse_args()
    configure_logging(args.log_level)
    logger.info("Starting standalone population %s", args.action)
    run(args)
    logger.info("Completed standalone population %s", args.action)


if __name__ == "__main__":
    main()
