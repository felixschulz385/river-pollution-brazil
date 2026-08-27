import logging


logger = logging.getLogger(__name__)


class Gadm:
    """Simplify the shared GADM ADM0/ADM2 boundary geopackage for downstream sources."""

    def __init__(self, root_dir="."):
        self.root_dir = root_dir

    def preprocess(
        self,
        gadm_path=None,
        adm0_layer=None,
        adm2_layer=None,
        tolerance=None,
        output_path=None,
    ):
        """Simplify the raw GADM boundaries and cache the result."""
        from .preprocess import simplify_gadm

        return simplify_gadm(
            root_dir=self.root_dir,
            gadm_path=gadm_path,
            adm0_layer=adm0_layer,
            adm2_layer=adm2_layer,
            tolerance=tolerance,
            output_path=output_path,
        )


__all__ = ["Gadm"]
