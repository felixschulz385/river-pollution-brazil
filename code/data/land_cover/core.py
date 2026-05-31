import logging

from .aggregation import aggregate_along_rivers
from .assembly import assemble_land_cover
from .constants import DATADIR, DRAINAGE_PATH, LEGEND_PATH
from .preprocess import preprocess_land_cover
from .schema import get_output_columns


logger = logging.getLogger(__name__)
OUTPUT_COLUMNS = get_output_columns(LEGEND_PATH)


class LandCover:
    """Land cover data processor with CLI integration."""

    def __init__(
        self,
        datadir=DATADIR,
        drainage_path=DRAINAGE_PATH,
        legend_path=LEGEND_PATH,
        output_columns=OUTPUT_COLUMNS,
    ):
        self.datadir = datadir
        self.drainage_path = drainage_path
        self.legend_path = legend_path
        self.output_columns = output_columns
        logger.debug("Initialized LandCover with datadir=%s", datadir)

    def fetch(self):
        """Fetch/download raw land cover data."""
        logger.info("Land cover data should be downloaded manually from MapBiomas.")
        logger.info("Expected location: %s", self.datadir)

    preprocess = preprocess_land_cover
    assemble = assemble_land_cover
    aggregate_along_rivers = aggregate_along_rivers
