from .scraper.stations.inventory import (
    fetch_station_inventory,
    preprocess_station_inventory,
)


class sensor_stations:
    """Station inventory workflow that prepares inputs for `sensor_data`.

    The workflow is intentionally split into `fetch()` and `preprocess()` so the
    raw ANA inventory can be cached once and enriched later with local boundary
    and river-network assets.
    """

    def __init__(
        self,
        root_dir=".",
        brazil_boundary_path=None,
        river_network_dir=None,
    ):
        self.root_dir = root_dir
        self.brazil_boundary_path = brazil_boundary_path
        self.river_network_dir = river_network_dir

    def fetch(self):
        """Fetch the raw ANA station inventory without downstream filtering."""
        fetch_station_inventory(root_dir=self.root_dir)

    def preprocess(self):
        """Filter and enrich the fetched station inventory for downstream use."""
        preprocess_station_inventory(
            root_dir=self.root_dir,
            brazil_boundary_path=self.brazil_boundary_path,
            river_network_dir=self.river_network_dir,
        )
