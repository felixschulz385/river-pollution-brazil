"""Repository-level CLI entrypoint."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


logger = logging.getLogger(__name__)

# The CLI calls the sensor-data module "water-quality"; the underlying package
# is src.data.sensor_data. This alias is CLI-facing only.
DATA_MODULES = (
    "health",
    "climate",
    "water-quality",
    "land-cover",
    "population",
    "river-network",
    "download",
)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once for CLI and batch execution."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _add_data_module_parsers(subparsers, module_dest: str) -> None:
    """Register the data workflow parsers under a subparser collection.

    Each submodule owns its own CLI surface via `configure_parser()` in its
    `__main__.py` (also used for standalone `python -m src.data.<module>`
    invocation); this just mounts it under the unified `data <module>` tree.
    """
    from src.data.climate.__main__ import configure_parser as configure_climate_parser
    from src.data.health.__main__ import configure_parser as configure_health_parser
    from src.data.land_cover.__main__ import configure_parser as configure_land_cover_parser
    from src.data.population.__main__ import configure_parser as configure_population_parser
    from src.data.river_network.__main__ import configure_parser as configure_river_network_parser
    from src.data.sensor_data.__main__ import configure_parser as configure_sensor_data_parser

    health_parser = subparsers.add_parser("health", help="Process health data")
    health_parser.set_defaults(**{module_dest: "health"})
    configure_health_parser(health_parser)

    climate_parser = subparsers.add_parser("climate", help="Process climate data")
    climate_parser.set_defaults(**{module_dest: "climate"})
    configure_climate_parser(climate_parser)

    wq_parser = subparsers.add_parser("water-quality", help="Process water quality data")
    wq_parser.set_defaults(**{module_dest: "water-quality"})
    configure_sensor_data_parser(wq_parser)

    lc_parser = subparsers.add_parser("land-cover", help="Process land cover data")
    lc_parser.set_defaults(**{module_dest: "land-cover"})
    configure_land_cover_parser(lc_parser)

    download_parser = subparsers.add_parser("download", help="Download datasets")
    download_parser.set_defaults(**{module_dest: "download"})
    download_parser.add_argument(
        "--remote-root-dir",
        default="/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/",
    )
    download_parser.add_argument("--local-root-dir", default="/tmp")
    download_parser.add_argument("--area", default="BRA")
    download_parser.add_argument("--year", type=int, default=2010)
    download_parser.add_argument("--dataset", required=True)

    population_parser = subparsers.add_parser("population", help="Process population data")
    population_parser.set_defaults(**{module_dest: "population"})
    configure_population_parser(population_parser)

    river_parser = subparsers.add_parser("river-network", help="Process river network data")
    river_parser.set_defaults(**{module_dest: "river-network"})
    configure_river_network_parser(river_parser)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""
    parser = argparse.ArgumentParser(
        description="Project CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for CLI execution (default: INFO).",
    )
    subparsers = parser.add_subparsers(dest="module", required=True)

    analysis_parser = subparsers.add_parser("analysis", help="Run analysis workflows")
    analysis_parser.add_argument(
        "analysis_module",
        choices=["sensor-data"],
        help="Analysis workflow to run.",
    )
    analysis_parser.add_argument(
        "analysis_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the selected analysis module.",
    )

    data_parser = subparsers.add_parser("data", help="Run data workflows")
    data_subparsers = data_parser.add_subparsers(dest="data_module", required=True)
    _add_data_module_parsers(data_subparsers, "data_module")

    # Flat aliases remain for compatibility with older scripts.
    _add_data_module_parsers(subparsers, "module")

    return parser


def _run_data_cli(args: argparse.Namespace) -> int:
    """Dispatch a parsed data-module command to that module's own `run()`."""
    data_module = getattr(args, "data_module", None) or args.module
    try:
        if data_module == "download":
            from src.data.download import download_agent

            agent = download_agent(
                remote_root_dir=args.remote_root_dir,
                local_root_dir=args.local_root_dir,
                area=args.area,
                year=args.year,
            )
            agent.fetch({"name": args.dataset})
        elif data_module == "health":
            from src.data.health.__main__ import run as run_health

            run_health(args)
        elif data_module == "climate":
            from src.data.climate.__main__ import run as run_climate

            run_climate(args)
        elif data_module == "water-quality":
            from src.data.sensor_data.__main__ import run as run_sensor_data

            run_sensor_data(args)
        elif data_module == "land-cover":
            from src.data.land_cover.__main__ import run as run_land_cover

            run_land_cover(args)
        elif data_module == "population":
            from src.data.population.__main__ import run as run_population

            run_population(args)
        elif data_module == "river-network":
            from src.data.river_network.__main__ import run as run_river_network

            run_river_network(args)
        else:
            raise ValueError(f"Unknown data module: {data_module}")

        logger.info("Completed %s module successfully", data_module)
        return 0
    except Exception as exc:
        logger.error("Error running %s module: %s", data_module, exc)
        logger.debug("Traceback:", exc_info=True)
        return 1


def main(argv: list[str] | None = None) -> int:
    """Run the repository CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    if args.module == "analysis":
        if args.analysis_module != "sensor-data":
            parser.error(f"Unsupported analysis module: {args.analysis_module}")
        from src.analysis.cli import main as analysis_main

        return analysis_main(["--log-level", args.log_level, *args.analysis_args])

    if args.module == "data" or args.module in DATA_MODULES:
        return _run_data_cli(args)

    parser.error(f"Unknown module: {args.module}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
