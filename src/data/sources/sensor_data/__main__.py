import argparse
import logging

from .core import SensorData


logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging for standalone sensor-data execution."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def configure_parser(parser, include_action=True):
    """Add sensor-data CLI arguments to ``parser``."""
    if include_action:
        parser.add_argument("action", choices=["fetch", "preprocess"])
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--brazil-boundary-path", default=None)
    parser.add_argument(
        "--river-network-dir",
        "--river-network-path",
        dest="river_network_dir",
        default=None,
    )
    parser.add_argument("--download-dir", default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--keep-browser-on-error", action="store_true")
    parser.add_argument("--single-station", default=None)
    parser.add_argument(
        "--fetch-mode",
        default="default",
        choices=["default", "missing-only", "retry-failed", "redownload-all"],
    )
    parser.add_argument("--preprocess-workers", type=int, default=None)
    parser.add_argument("--source-tables", default=None)
    parser.add_argument(
        "--preprocess-backend",
        default="thread",
        choices=["thread", "process"],
    )
    parser.add_argument("--log-every-tables", type=int, default=None)
    parser.add_argument("--stations-rivers-path", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--n_jobs", type=int, default=None)
    return parser


def run(args):
    """Execute the requested sensor-data action for parsed ``args``."""
    agent = SensorData(
        root_dir=args.root_dir,
        brazil_boundary_path=args.brazil_boundary_path,
        river_network_dir=args.river_network_dir,
        download_dir=args.download_dir,
        headless=args.headless,
        keep_browser_on_error=args.keep_browser_on_error,
        single_station=args.single_station,
        fetch_mode=args.fetch_mode,
        preprocess_workers=args.preprocess_workers,
        source_tables=args.source_tables,
        preprocess_backend=args.preprocess_backend,
        log_every_tables=args.log_every_tables,
    )
    if args.action == "fetch":
        agent.fetch()
    else:
        agent.preprocess(
            stations_rivers_path=args.stations_rivers_path,
            output_path=args.output,
            n_jobs=args.n_jobs,
        )


def main():
    parser = argparse.ArgumentParser(description="Run sensor-data workflows")
    configure_parser(parser)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for standalone execution",
    )

    args = parser.parse_args()
    configure_logging(args.log_level)
    logger.info("Starting standalone sensor-data %s", args.action)
    run(args)
    logger.info("Completed standalone sensor-data %s", args.action)


if __name__ == "__main__":
    main()
