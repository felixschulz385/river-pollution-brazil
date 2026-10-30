class ProtectedAreas:
    """Read Indigenous lands and conservation units and measure their extent per ADM2 municipality."""

    def __init__(self, root_dir="."):
        self.root_dir = root_dir

    def fetch(self):
        """Download the TerraBrasilis conservation-unit layers (FUNAI's Indigenous lands are placed by hand)."""
        from .fetch import fetch_protected_areas

        return fetch_protected_areas(root_dir=self.root_dir)

    def preprocess(self, gpkg=None, funai=None, gadm_path=None, layer=None, adm2_id_column=None, areas_output_path=None,
                   adm2_output_path=None):
        """Merge the areas, flag where mining is barred and since when, and compute municipality shares."""
        from .preprocess import build_protected_areas

        return build_protected_areas(
            root_dir=self.root_dir,
            gpkg=gpkg,
            funai=funai,
            gadm_path=gadm_path,
            layer=layer,
            adm2_id_column=adm2_id_column,
            areas_output_path=areas_output_path,
            adm2_output_path=adm2_output_path,
        )

    def preprocess_trenches(self, areas_path=None, drainage_path=None, output_path=None):
        """Share of each river trench's drainage polygon inside each Indigenous land or barred unit."""
        from .preprocess import build_trench_overlaps

        return build_trench_overlaps(root_dir=self.root_dir, areas_path=areas_path, drainage_path=drainage_path,
                                     output_path=output_path)


__all__ = ["ProtectedAreas"]
