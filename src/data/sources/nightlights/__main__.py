import argparse
import logging

from .core import Nightlights


logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging for standalone night-lights execution."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def configure_parser(parser, include_action=True):
    """Add night-lights CLI arguments to ``parser``."""
    if include_action:
        parser.add_argument("action", choices=["preprocess"])
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--raw", default=None, help="Folder holding the EOG snapshot folders (default: data/nightlights/raw)")
    parser.add_argument("--gadm-path", default=None)
    parser.add_argument("--gadm-layer", default=None)
    parser.add_argument("--adm2-id-column", default=None)
    parser.add_argument("--protected-areas", default=None, help="Processed protected areas (default: protected_areas.parquet)")
    parser.add_argument("--geology-gpkg", default=None, help="SGB occurrences GeoPackage (default: geology raw)")
    parser.add_argument("--adm2-output", default=None)
    parser.add_argument("--workers", type=int, default=None, help="Files read in parallel (default: SLURM_CPUS_PER_TASK or 1)")
    return parser


def run(args):
    """Execute the requested night-lights action for parsed ``args``."""
    Nightlights(root_dir=args.root_dir).preprocess(
        raw=args.raw,
        gadm_path=args.gadm_path,
        layer=args.gadm_layer,
        adm2_id_column=args.adm2_id_column,
        protected_areas_path=args.protected_areas,
        geology_gpkg=args.geology_gpkg,
        output_path=args.adm2_output,
        workers=args.workers,
    )


def main():
    parser = argparse.ArgumentParser(description="Run VIIRS night-lights workflows")
    configure_parser(parser)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for standalone execution",
    )

    args = parser.parse_args()
    configure_logging(args.log_level)
    logger.info("Starting standalone night-lights %s", args.action)
    run(args)
    logger.info("Completed standalone night-lights %s", args.action)


if __name__ == "__main__":
    main()
