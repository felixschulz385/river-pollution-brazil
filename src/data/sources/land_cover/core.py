import logging
from pathlib import Path

from .aggregation import aggregate_along_rivers as _aggregate_along_rivers
from .assembly import assemble_land_cover as _assemble_land_cover
from .constants import LAND_COVER_PRODUCT, MINING_PRODUCT, PRODUCTS, build_paths
from .preprocess import preprocess_land_cover as _preprocess_land_cover
from .schema import get_output_columns


logger = logging.getLogger(__name__)


def product_paths(product, root_dir="."):
    """Default raster directory, legend, extract and ADM2 upstream paths of one product.

    Raster and legend paths are rooted at ``root_dir``; the processed paths are
    relative, like the other extract/aggregate defaults.
    """
    from .constants import DEFAULT_ADM2_UPSTREAM_OUTPUT_PATH, DEFAULT_ASSEMBLY_LAND_COVER_PATH
    from .mining import (
        MINING_ADM2_UPSTREAM_PATH,
        MINING_DATADIR,
        MINING_EXTRACT_PATH,
        MINING_LEGEND_PATH,
    )

    if product == LAND_COVER_PRODUCT:
        paths = build_paths(root_dir)
        return dict(
            datadir=paths.datadir,
            legend_path=paths.legend_path,
            extract_path=DEFAULT_ASSEMBLY_LAND_COVER_PATH,
            adm2_upstream_path=DEFAULT_ADM2_UPSTREAM_OUTPUT_PATH,
        )
    if product == MINING_PRODUCT:
        return dict(
            datadir=Path(root_dir) / MINING_DATADIR,
            legend_path=Path(root_dir) / MINING_LEGEND_PATH,
            extract_path=MINING_EXTRACT_PATH,
            adm2_upstream_path=MINING_ADM2_UPSTREAM_PATH,
        )
    raise ValueError(f"Unknown land-cover product {product!r}; expected one of {PRODUCTS}.")


class LandCover:
    """Land cover data processor with CLI integration.

    `product` selects the raster set and its default paths (`product_paths`):
    the MapBiomas Collection 10 coverage or the Collection 11 mining substances.
    """

    def __init__(
        self,
        root_dir=".",
        datadir=None,
        drainage_path=None,
        legend_path=None,
        output_columns=None,
        product=LAND_COVER_PRODUCT,
    ):
        self.root_dir = root_dir
        self.product = product
        self.paths = product_paths(product, root_dir)
        self.datadir = self.paths["datadir"] if datadir is None else datadir
        self.drainage_path = build_paths(root_dir).drainage_path if drainage_path is None else drainage_path
        self.legend_path = self.paths["legend_path"] if legend_path is None else legend_path
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
        """Download the Collection 10 coverage rasters (the mining product is staged by hand)."""
        if self.product == MINING_PRODUCT:
            logger.info(
                "The mining-substances rasters are exported from Earth Engine "
                "(scripts/gee_export_mapbiomas_mining.js) and merged with "
                "scripts/prepare_mapbiomas_mining.py; expected location: %s",
                self.datadir,
            )
            return self.datadir
        from .fetch import fetch_mapbiomas_coverage

        return fetch_mapbiomas_coverage(self.datadir)

    def preprocess(
        self,
        n_jobs=None,
        river_network_path=None,
        output_path=None,
        log_level=None,
    ):
        """Extract per-trench land-cover class shares from raw rasters."""
        return _preprocess_land_cover(
            self,
            n_jobs=n_jobs,
            river_network_path=river_network_path,
            output_path=output_path or self.paths["extract_path"],
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
            ADM2_ASSEMBLY_VARIANT,
            DEFAULT_RIVER_NETWORK_PATH,
            DEFAULT_STATIONS_TRENCHES_PATH,
            DEFAULT_WATER_QUALITY_PATH,
            SENSOR_ASSEMBLY_VARIANT,
        )

        variant = variant or SENSOR_ASSEMBLY_VARIANT
        if self.product != LAND_COVER_PRODUCT:
            if variant != ADM2_ASSEMBLY_VARIANT:
                raise ValueError(f"The {self.product} product has only the {ADM2_ASSEMBLY_VARIANT} variant.")
            output_path = output_path or self.paths["adm2_upstream_path"]
        return _assemble_land_cover(
            self,
            variant=variant,
            land_cover_path=land_cover_path or self.paths["extract_path"],
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
        output_path=None,
    ):
        """Aggregate land-cover variables upstream of each ADM2 unit."""
        from .constants import (
            DEFAULT_ADM2_UPSTREAM_OUTPUT_PATH,
            DEFAULT_ASSEMBLY_LAND_COVER_PATH,
            DEFAULT_RIVER_NETWORK_PATH,
        )

        return _aggregate_along_rivers(
            self,
            land_cover_path=land_cover_path or DEFAULT_ASSEMBLY_LAND_COVER_PATH,
            river_network_path=river_network_path or DEFAULT_RIVER_NETWORK_PATH,
            drainage_polygons_path=drainage_polygons_path,
            years=years,
            n_jobs=n_jobs,
            output_path=output_path or DEFAULT_ADM2_UPSTREAM_OUTPUT_PATH,
        )
