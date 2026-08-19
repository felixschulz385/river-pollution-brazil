import argparse
import logging

from .core import RiverNetwork
from .constants import PROCESSED_DIR


logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging for standalone river-network execution."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def configure_parser(parser, include_action=True):
    """Add river-network CLI arguments to ``parser``."""
    if include_action:
        parser.add_argument("action", choices=["preprocess"])
    parser.add_argument("--gpkg-path", required=True)
    parser.add_argument("--output-dir", default=PROCESSED_DIR)
    parser.add_argument("--min-lon", type=float)
    parser.add_argument("--min-lat", type=float)
    parser.add_argument("--max-lon", type=float)
    parser.add_argument("--max-lat", type=float)
    parser.add_argument("--gadm-path")
    parser.add_argument("--gadm-layer", default="ADM_ADM_0")
    parser.add_argument("--gadm-adm2-layer", default="ADM_ADM_2")
    return parser


def run(args):
    """Execute the requested river-network action for parsed ``args``."""
    bbox = None
    if all([args.min_lon, args.min_lat, args.max_lon, args.max_lat]):
        import geopandas as gpd
        from shapely.geometry import box

        bbox = gpd.GeoSeries(
            box(args.min_lon, args.min_lat, args.max_lon, args.max_lat),
            crs=4326,
        )

    network = RiverNetwork()
    network.generate(
        args.gpkg_path,
        args.output_dir,
        bbox=bbox,
        gadm_path=args.gadm_path,
        gadm_layer=args.gadm_layer,
        gadm_adm2_layer=args.gadm_adm2_layer,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate river-network outputs")
    configure_parser(parser)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for standalone execution",
    )

    args = parser.parse_args()
    configure_logging(args.log_level)
    run(args)
    logger.info("Completed standalone river-network generation")


if __name__ == "__main__":
    main()
