import argparse
import logging

from .core import Mining


logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging for standalone mining execution."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def configure_parser(parser, include_action=True):
    """Add mining CLI arguments to ``parser``."""
    if include_action:
        parser.add_argument("action", choices=["preprocess"])
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--archive", default=None, help="SCM microdata zip (default: the newest snapshot)")
    parser.add_argument("--adm2-output", default=None)
    parser.add_argument("--monthly-output", default=None)
    return parser


def run(args):
    """Execute the requested mining action for parsed ``args``."""
    Mining(root_dir=args.root_dir).preprocess(archive=args.archive, adm2_output_path=args.adm2_output,
                                                monthly_output_path=args.monthly_output)


def main():
    parser = argparse.ArgumentParser(description="Run mining-permit workflows")
    configure_parser(parser)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for standalone execution",
    )

    args = parser.parse_args()
    configure_logging(args.log_level)
    logger.info("Starting standalone mining %s", args.action)
    run(args)
    logger.info("Completed standalone mining %s", args.action)


if __name__ == "__main__":
    main()
