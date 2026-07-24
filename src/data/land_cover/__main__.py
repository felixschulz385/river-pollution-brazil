import argparse
import logging

from .core import LandCover
from .preprocess import configure_logging


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Process land cover data in parallel")
    parser.add_argument("--n_jobs", type=int, default=None, help="Number of parallel jobs")
    parser.add_argument(
        "--output",
        type=str,
        default="land_cover_results.feather",
        help="Output file path",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for standalone execution",
    )
    parser.add_argument("--datadir", type=str, default=None, help="Raster directory")
    parser.add_argument(
        "--drainage-path",
        type=str,
        default=None,
        help="Drainage polygons feather/parquet path",
    )
    parser.add_argument("--legend-path", type=str, default=None, help="Legend workbook path")

    args = parser.parse_args()
    configure_logging(args.log_level)
    logger.info("Starting standalone land-cover preprocessing")

    lc = LandCover(
        datadir=args.datadir,
        drainage_path=args.drainage_path,
        legend_path=args.legend_path,
    )
    lc.preprocess(n_jobs=args.n_jobs, output_path=args.output)
    logger.info("Completed standalone land-cover preprocessing")


if __name__ == "__main__":
    main()
