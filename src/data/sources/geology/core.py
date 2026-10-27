class Geology:
    """Fetch SGB gold geology and aggregate it to ADM2 municipalities."""

    def __init__(self, root_dir="."):
        self.root_dir = root_dir

    def fetch(self):
        """Download the SGB occurrence and gold province/district layers."""
        from .fetch import fetch_sgb_gold

        return fetch_sgb_gold(root_dir=self.root_dir)

    def aggregate(self, gpkg=None, gadm_path=None, layer=None, adm2_id_column=None, adm2_output_path=None):
        """Build the ADM2 gold occurrence counts and province/district shares."""
        from .preprocess import build_adm2_geology

        return build_adm2_geology(
            root_dir=self.root_dir,
            gpkg=gpkg,
            gadm_path=gadm_path,
            layer=layer,
            adm2_id_column=adm2_id_column,
            output_path=adm2_output_path,
        )


__all__ = ["Geology"]
