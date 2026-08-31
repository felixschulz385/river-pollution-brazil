import logging

from .aggregation import aggregate_along_rivers as _aggregate_along_rivers
from .assembly import assemble_land_cover as _assemble_land_cover
from .constants import build_paths
from .preprocess import preprocess_land_cover as _preprocess_land_cover
from .schema import get_output_columns


logger = logging.getLogger(__name__)


class LandCover:
    """Land cover data processor with CLI integration."""

    def __init__(
        self,
        root_dir=".",
        datadir=None,
        drainage_path=None,
        legend_path=None,
        output_columns=None,
    ):
        self.root_dir = root_dir
        paths = build_paths(root_dir)
        self.datadir = paths.datadir if datadir is None else datadir
        self.drainage_path = paths.drainage_path if drainage_path is None else drainage_path
        self.legend_path = paths.legend_path if legend_path is None else legend_path
        self.output_columns = (
            get_output_columns(self.legend_path)
            if output_columns is None
            else output_columns
        )
        logger.debug(
            "Initialized LandCover with datadir=%s, drainage_path=%s, legend_path=%s",
            self.datadir,
            self.drainage_path,
            self.legend_path,
        )

    def fetch(self):
        """Fetch/download raw land cover data."""
        logger.info("Land cover data should be downloaded manually from MapBiomas.")
        logger.info("Expected location: %s", self.datadir)

    def preprocess(
        self,
        n_jobs=None,
        river_network_path=None,
        output_path="data/land_cover/processed/extract/land_cover.parquet",
        log_level=None,
    ):
        """Extract per-trench land-cover class shares from raw rasters."""
        return _preprocess_land_cover(
            self,
            n_jobs=n_jobs,
            river_network_path=river_network_path,
            output_path=output_path,
            log_level=log_level,
        )

    def assemble(
        self,
        variant=None,
        land_cover_path=None,
        water_quality_path=None,
        stations_rivers_path=None,
        river_network_path=None,
        output_path=None,
        n_jobs=None,
    ):
        """Assemble analysis-ready land-cover outputs for the requested variant."""
        from .constants import (
            DEFAULT_ASSEMBLY_LAND_COVER_PATH,
            DEFAULT_RIVER_NETWORK_PATH,
            DEFAULT_STATIONS_TRENCHES_PATH,
            DEFAULT_WATER_QUALITY_PATH,
            SENSOR_ASSEMBLY_VARIANT,
        )

        return _assemble_land_cover(
            self,
            variant=variant or SENSOR_ASSEMBLY_VARIANT,
            land_cover_path=land_cover_path or DEFAULT_ASSEMBLY_LAND_COVER_PATH,
            water_quality_path=water_quality_path or DEFAULT_WATER_QUALITY_PATH,
            stations_rivers_path=stations_rivers_path or DEFAULT_STATIONS_TRENCHES_PATH,
            river_network_path=river_network_path or DEFAULT_RIVER_NETWORK_PATH,
            output_path=output_path,
            n_jobs=n_jobs,
        )

    def aggregate_along_rivers(
        self,
        land_cover_path=None,
        river_network_path=None,
        drainage_polygons_path=None,
        years=None,
        n_jobs=None,
        output_path="data/land_cover/processed/aggregate/land_cover_river_aggregated.parquet",
    ):
        """Aggregate land-cover variables upstream of each ADM2 unit."""
        from .constants import DEFAULT_ASSEMBLY_LAND_COVER_PATH, DEFAULT_RIVER_NETWORK_PATH

        return _aggregate_along_rivers(
            self,
            land_cover_path=land_cover_path or DEFAULT_ASSEMBLY_LAND_COVER_PATH,
            river_network_path=river_network_path or DEFAULT_RIVER_NETWORK_PATH,
            drainage_polygons_path=drainage_polygons_path,
            years=years,
            n_jobs=n_jobs,
            output_path=output_path,
        )
