"""Repository-level CLI entrypoint."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


logger = logging.getLogger(__name__)

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


class DataSourceFactory:
    """Factory class for creating data source instances."""

    @staticmethod
    def create(module: str, **kwargs):
        if module == "health":
            from code.data.health import health

            return health()
        if module == "climate":
            from code.data.climate import climate

            return climate(root_dir=kwargs.get("root_dir", "."))
        if module == "water-quality":
            from code.data.sensor_data.sensor_data import sensor_data

            return sensor_data(
                root_dir=kwargs.get("root_dir", "."),
                brazil_boundary_path=kwargs.get("brazil_boundary_path"),
                river_network_dir=kwargs.get("river_network_dir"),
                download_dir=kwargs.get("download_dir"),
                headless=kwargs.get("headless", False),
                keep_browser_on_error=kwargs.get("keep_browser_on_error", False),
                single_station=kwargs.get("single_station"),
                fetch_mode=kwargs.get("fetch_mode", "default"),
                preprocess_workers=kwargs.get("preprocess_workers"),
                source_tables=kwargs.get("source_tables"),
                preprocess_backend=kwargs.get("preprocess_backend", "thread"),
                log_every_tables=kwargs.get("log_every_tables"),
            )
        if module == "river-network":
            from code.data.river_network import RiverNetwork

            return RiverNetwork()
        if module == "land-cover":
            from code.data.land_cover import LandCover

            return LandCover(
                datadir=kwargs.get("datadir"),
                drainage_path=kwargs.get("drainage_path"),
                legend_path=kwargs.get("legend_path"),
            )
        if module == "download":
            from code.data.download import download_agent

            return download_agent(
                remote_root_dir=kwargs.get(
                    "remote_root_dir",
                    "/pfs/work7/workspace/scratch/tu_zxobe27-master_thesis/",
                ),
                local_root_dir=kwargs.get("local_root_dir", "/tmp"),
                area=kwargs.get("area", "BRA"),
                year=kwargs.get("year", 2010),
            )
        if module == "population":
            from code.data.population import population

            return population(
                root_dir=kwargs.get("root_dir", "."),
                billing_project=kwargs.get(
                    "billing_project",
                    "river-pollution-499210",
                ),
            )
        raise ValueError(f"Unknown module: {module}")


def _add_data_module_parsers(subparsers, module_dest: str) -> None:
    """Register the data workflow parsers under a subparser collection."""
    health_parser = subparsers.add_parser("health", help="Process health data")
    health_parser.set_defaults(**{module_dest: "health"})
    health_parser.add_argument("action", choices=["fetch", "preprocess"])
    health_parser.add_argument(
        "--subtype",
        choices=["all", "mortality", "hospitalization", "birth"],
        default="all",
    )

    climate_parser = subparsers.add_parser("climate", help="Process climate data")
    climate_parser.set_defaults(**{module_dest: "climate"})
    climate_parser.add_argument("action", choices=["fetch", "preprocess", "assemble"])
    climate_parser.add_argument("--root-dir", default=".")
    climate_parser.add_argument(
        "--subtype",
        default="cloud_cover",
        choices=["cloud_cover", "era5_land_hourly", "era5_land_daily"],
    )
    climate_parser.add_argument("--stage", default="all", choices=["all", "zarr", "parquet"])
    climate_parser.add_argument(
        "--variant",
        default="sensor_upstream_distance_buckets",
        choices=["sensor_upstream_distance_buckets", "adm2_upstream_yearly"],
    )
    climate_parser.add_argument(
        "--climate-path",
        default="data/climate/processed/era5_land.parquet",
    )
    climate_parser.add_argument(
        "--water-quality-path",
        default="data/sensor_data/water_quality.parquet",
    )
    climate_parser.add_argument(
        "--stations-rivers-path",
        default="data/sensor_data/stations_rivers.parquet",
    )
    climate_parser.add_argument("--river-network-path", default="data/river_network")
    climate_parser.add_argument("--output", default=None)
    climate_parser.add_argument(
        "--kernel",
        default="gaussian",
        choices=["uniform", "triangular", "epanechnikov", "gaussian", "exponential"],
    )
    climate_parser.add_argument("--h", type=float, default=1000000.0)
    climate_parser.add_argument("--n_jobs", type=int, default=None)

    wq_parser = subparsers.add_parser("water-quality", help="Process water quality data")
    wq_parser.set_defaults(**{module_dest: "water-quality"})
    wq_parser.add_argument("action", choices=["fetch", "preprocess", "assemble"])
    wq_parser.add_argument("--root-dir", default=".")
    wq_parser.add_argument("--brazil-boundary-path", default=None)
    wq_parser.add_argument(
        "--river-network-dir",
        "--river-network-path",
        dest="river_network_dir",
        default=None,
    )
    wq_parser.add_argument("--download-dir", default=None)
    wq_parser.add_argument("--headless", action="store_true")
    wq_parser.add_argument("--keep-browser-on-error", action="store_true")
    wq_parser.add_argument("--single-station", default=None)
    wq_parser.add_argument(
        "--fetch-mode",
        default="default",
        choices=["default", "missing-only", "retry-failed", "redownload-all"],
    )
    wq_parser.add_argument("--preprocess-workers", type=int, default=None)
    wq_parser.add_argument("--source-tables", default=None)
    wq_parser.add_argument(
        "--preprocess-backend",
        default="thread",
        choices=["thread", "process"],
    )
    wq_parser.add_argument("--log-every-tables", type=int, default=None)
    wq_parser.add_argument("--water-quality-path", default=None)
    wq_parser.add_argument("--streamflow-path", default=None)
    wq_parser.add_argument("--stations-rivers-path", default=None)
    wq_parser.add_argument("--output", default=None)
    wq_parser.add_argument("--n_jobs", type=int, default=None)

    lc_parser = subparsers.add_parser("land-cover", help="Process land cover data")
    lc_parser.set_defaults(**{module_dest: "land-cover"})
    lc_parser.add_argument("action", choices=["fetch", "preprocess", "assemble"])
    lc_parser.add_argument("--n_jobs", type=int, default=None)
    lc_parser.add_argument("--output", default=None)
    lc_parser.add_argument("--river-network-path", default=None)
    lc_parser.add_argument("--datadir", default=None)
    lc_parser.add_argument("--drainage-path", default=None)
    lc_parser.add_argument("--legend-path", default=None)
    lc_parser.add_argument("--variant", default="sensor")
    lc_parser.add_argument("--land-cover-path", default="data/land_cover/land_cover.feather")
    lc_parser.add_argument(
        "--water-quality-path",
        default="data/sensor_data/water_quality.parquet",
    )
    lc_parser.add_argument(
        "--stations-rivers-path",
        default="data/sensor_data/stations_rivers.parquet",
    )

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
    population_parser.add_argument("action", choices=["fetch", "preprocess"])
    population_parser.add_argument("--root-dir", default=".")
    population_parser.add_argument(
        "--billing-project",
        default="river-pollution-499210",
    )

    river_parser = subparsers.add_parser("river-network", help="Process river network data")
    river_parser.set_defaults(**{module_dest: "river-network"})
    river_parser.add_argument("action", choices=["generate"])
    river_parser.add_argument("--gpkg-path", required=True)
    river_parser.add_argument("--output-dir", required=True)
    river_parser.add_argument("--min-lon", type=float)
    river_parser.add_argument("--min-lat", type=float)
    river_parser.add_argument("--max-lon", type=float)
    river_parser.add_argument("--max-lat", type=float)
    river_parser.add_argument("--gadm-path")
    river_parser.add_argument("--gadm-layer", default="ADM_ADM_0")
    river_parser.add_argument("--gadm-adm2-layer", default="ADM_ADM_2")


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
    """Execute the historical data CLI commands under the unified entrypoint."""
    data_module = getattr(args, "data_module", None) or args.module
    try:
        if data_module == "download":
            agent = DataSourceFactory.create(
                data_module,
                remote_root_dir=args.remote_root_dir,
                local_root_dir=args.local_root_dir,
                area=args.area,
                year=args.year,
            )
        elif data_module == "climate":
            agent = DataSourceFactory.create(data_module, root_dir=args.root_dir)
        elif data_module == "water-quality":
            agent = DataSourceFactory.create(
                data_module,
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
        elif data_module == "population":
            agent = DataSourceFactory.create(
                data_module,
                root_dir=args.root_dir,
                billing_project=args.billing_project,
            )
        else:
            agent = DataSourceFactory.create(
                data_module,
                datadir=getattr(args, "datadir", None),
                drainage_path=getattr(args, "drainage_path", None),
                legend_path=getattr(args, "legend_path", None),
            )

        if data_module in ["health", "water-quality", "population"]:
            if args.action == "fetch":
                if data_module == "health":
                    agent.fetch(subtype=args.subtype)
                else:
                    agent.fetch()
            elif args.action == "preprocess":
                agent.preprocess()
            elif args.action == "assemble":
                agent.assemble(
                    water_quality_path=args.water_quality_path,
                    streamflow_path=args.streamflow_path,
                    stations_rivers_path=args.stations_rivers_path,
                    river_network_path=args.river_network_dir,
                    output_path=args.output,
                    n_jobs=args.n_jobs,
                )
        elif data_module == "climate":
            if args.action == "fetch":
                agent.fetch(subtype=args.subtype)
            elif args.action == "preprocess":
                agent.preprocess(subtype=args.subtype, stage=args.stage, n_jobs=args.n_jobs)
            elif args.action == "assemble":
                agent.assemble(
                    variant=args.variant,
                    climate_path=args.climate_path,
                    water_quality_path=args.water_quality_path,
                    stations_rivers_path=args.stations_rivers_path,
                    river_network_path=args.river_network_path,
                    output_path=args.output,
                    kernel=args.kernel,
                    h=args.h,
                    n_jobs=args.n_jobs,
                )
        elif data_module == "land-cover":
            if args.action == "fetch":
                agent.fetch()
            elif args.action == "preprocess":
                output_path = args.output or "land_cover_results.feather"
                agent.preprocess(
                    n_jobs=args.n_jobs,
                    river_network_path=args.river_network_path,
                    output_path=output_path,
                    log_level=args.log_level,
                )
            elif args.action == "assemble":
                river_network_path = args.river_network_path or "data/river_network"
                agent.assemble(
                    variant=args.variant,
                    land_cover_path=args.land_cover_path,
                    water_quality_path=args.water_quality_path,
                    stations_rivers_path=args.stations_rivers_path,
                    river_network_path=river_network_path,
                    output_path=args.output,
                    n_jobs=args.n_jobs,
                )
        elif data_module == "river-network" and args.action == "generate":
            import geopandas as gpd
            from shapely.geometry import box

            bbox = None
            if all([args.min_lon, args.min_lat, args.max_lon, args.max_lat]):
                bbox = gpd.GeoSeries(
                    box(args.min_lon, args.min_lat, args.max_lon, args.max_lat),
                    crs=4326,
                )

            agent.load_trenches(args.gpkg_path, bbox=bbox)
            agent.load_drainage_areas(args.gpkg_path, bbox=bbox)
            agent.compute_subsystems()
            agent.compute_distance_matrices()
            agent.sort_trenches_by_system()

            if args.gadm_path:
                agent.annotate_drainage_areas_with_country_membership(
                    args.gadm_path,
                    layer=args.gadm_layer,
                )
                agent.build_trench_adm2_table(
                    gadm_path=args.gadm_path,
                    layer=args.gadm_adm2_layer,
                )

            agent.save(args.output_dir)
        elif data_module == "download":
            agent.fetch({"name": args.dataset})

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
        from code.analysis.cli import main as analysis_main

        return analysis_main(["--log-level", args.log_level, *args.analysis_args])

    if args.module == "data" or args.module in DATA_MODULES:
        return _run_data_cli(args)

    parser.error(f"Unknown module: {args.module}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
