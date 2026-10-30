class Nightlights:
    """Measure EOG's annual VIIRS night lights per ADM2 municipality, inside protected areas and near gold."""

    def __init__(self, root_dir="."):
        self.root_dir = root_dir

    def preprocess(self, raw=None, gadm_path=None, layer=None, adm2_id_column=None, protected_areas_path=None,
                   geology_gpkg=None, output_path=None, workers=None):
        """Sum radiance and count lit pixels per municipality, zone and year, with the mean cloud-free coverage."""
        from .preprocess import build_nightlights

        return build_nightlights(
            root_dir=self.root_dir,
            raw=raw,
            gadm_path=gadm_path,
            layer=layer,
            adm2_id_column=adm2_id_column,
            protected_areas_path=protected_areas_path,
            geology_gpkg=geology_gpkg,
            output_path=output_path,
            workers=workers,
        )


__all__ = ["Nightlights"]
