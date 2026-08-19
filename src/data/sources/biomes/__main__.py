import argparse
import logging

from .core import Biomes


logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging for standalone biomes execution."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def configure_parser(parser, include_action=True):
    """Add biomes CLI arguments to ``parser``."""
    if include_action:
        parser.add_argument("action", choices=["fetch", "preprocess"])
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--gadm-path", default=None)
    parser.add_argument("--gadm-layer", default=None)
    parser.add_argument("--adm2-id-column", default=None)
    parser.add_argument("--adm2-output", default=None)
    parser.add_argument("--sensor-output", default=None)
    return parser


def run(args):
    """Execute the requested biomes action for parsed ``args``."""
    agent = Biomes(root_dir=args.root_dir)
    if args.action == "fetch":
        agent.fetch()
    else:
        agent.preprocess(
            gadm_path=args.gadm_path,
            layer=args.gadm_layer,
            adm2_id_column=args.adm2_id_column,
            adm2_output_path=args.adm2_output,
            sensor_output_path=args.sensor_output,
        )


def main():
    parser = argparse.ArgumentParser(description="Run biomes workflows")
    configure_parser(parser)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for standalone execution",
    )

    args = parser.parse_args()
    configure_logging(args.log_level)
    logger.info("Starting standalone biomes %s", args.action)
    run(args)
    logger.info("Completed standalone biomes %s", args.action)


if __name__ == "__main__":
    main()
