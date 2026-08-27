class Biomes:
    """Fetch and preprocess IBGE biome polygons into ADM2 and sensor-station mappings."""

    def __init__(self, root_dir="."):
        self.root_dir = root_dir

    def fetch(self):
        """Download and extract the raw IBGE biomes archive."""
        from .fetch import fetch_biomes

        return fetch_biomes(root_dir=self.root_dir)

    def aggregate(
        self,
        gadm_path=None,
        layer=None,
        adm2_id_column=None,
        adm2_output_path=None,
        sensor_output_path=None,
    ):
        """Build the ADM2-to-biome and station-to-biome mapping tables."""
        from .preprocess import build_adm2_biomes, build_station_biomes

        adm2_biomes = build_adm2_biomes(
            root_dir=self.root_dir,
            gadm_path=gadm_path,
            layer=layer,
            adm2_id_column=adm2_id_column,
            output_path=adm2_output_path,
        )
        station_biomes = build_station_biomes(
            root_dir=self.root_dir,
            output_path=sensor_output_path,
        )
        return adm2_biomes, station_biomes


__all__ = ["Biomes"]
