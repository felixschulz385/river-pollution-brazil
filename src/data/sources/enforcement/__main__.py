import argparse
import logging

from .core import Enforcement


logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging for standalone enforcement execution."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def configure_parser(parser, include_action=True):
    """Add enforcement CLI arguments to ``parser``."""
    if include_action:
        parser.add_argument("action", choices=["preprocess"])
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--snapshot", default=None, help="IBAMA snapshot folder (default: the newest complete one)")
    parser.add_argument("--records-output", default=None)
    parser.add_argument("--adm2-output", default=None)
    return parser


def run(args):
    """Execute the requested enforcement action for parsed ``args``."""
    Enforcement(root_dir=args.root_dir).preprocess(
        snapshot=args.snapshot, records_output_path=args.records_output, adm2_output_path=args.adm2_output
    )


def main():
    parser = argparse.ArgumentParser(description="Run IBAMA enforcement workflows")
    configure_parser(parser)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for standalone execution",
    )

    args = parser.parse_args()
    configure_logging(args.log_level)
    logger.info("Starting standalone enforcement %s", args.action)
    run(args)
    logger.info("Completed standalone enforcement %s", args.action)


if __name__ == "__main__":
    main()
