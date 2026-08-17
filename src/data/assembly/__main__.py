import argparse
import logging

from .constants import DEFAULT_CONFIG_PATH
from .core import Assembly


logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def configure_parser(parser):
    """Add assembly CLI arguments to `parser`."""
    parser.add_argument("action", choices=["assemble"])
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset", required=True, help="Dataset id from the assembly config.")
    parser.add_argument("--output", default=None, help="Override the config's output path.")
    return parser


def run(args):
    """Execute the requested assembly action for parsed `args`."""
    agent = Assembly(root_dir=args.root_dir, config_path=args.config)
    agent.assemble(dataset_id=args.dataset, output_path=args.output)


def main():
    parser = argparse.ArgumentParser(description="Run assembly workflows")
    configure_parser(parser)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for standalone execution",
    )

    args = parser.parse_args()
    configure_logging(args.log_level)
    logger.info("Starting standalone assembly %s", args.action)
    run(args)
    logger.info("Completed standalone assembly %s", args.action)


if __name__ == "__main__":
    main()
