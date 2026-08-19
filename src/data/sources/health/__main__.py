import argparse
import logging

from .core import Health


logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging for standalone health execution."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def configure_parser(parser, include_action=True):
    """Add health CLI arguments to ``parser``."""
    if include_action:
        parser.add_argument("action", choices=["fetch", "preprocess"])
    parser.add_argument(
        "--subtype",
        choices=["all", "mortality", "hospitalization", "birth"],
        default="all",
    )
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--download-dir", default=None)
    return parser


def run(args):
    """Execute the requested health action for parsed ``args``."""
    agent = Health(
        root_dir=args.root_dir,
        headless=getattr(args, "headless", False),
        download_dir=getattr(args, "download_dir", None),
    )
    if args.action == "fetch":
        agent.fetch(subtype=args.subtype)
    else:
        agent.preprocess(subtype=args.subtype)


def main():
    parser = argparse.ArgumentParser(description="Run health-data workflows")
    configure_parser(parser)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for standalone execution",
    )

    args = parser.parse_args()
    configure_logging(args.log_level)
    logger.info("Starting standalone health %s", args.action)
    run(args)
    logger.info("Completed standalone health %s", args.action)


if __name__ == "__main__":
    main()
