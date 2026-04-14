class sensor_data:
    """Run sensor-data fetching, conversion, and cleaning workflows."""

    def __init__(
        self,
        root_dir=".",
        brazil_boundary_path=None,
        river_network_dir=None,
        download_dir=None,
        headless=False,
        keep_browser_on_error=False,
        single_station=None,
        fetch_mode="default",
        preprocess_workers=None,
        source_tables=None,
        preprocess_backend="thread",
        log_every_tables=None,
    ):
        self.root_dir = root_dir
        self.brazil_boundary_path = brazil_boundary_path
        self.river_network_dir = river_network_dir
        self.download_dir = download_dir
        self.headless = headless
        self.keep_browser_on_error = keep_browser_on_error
        self.single_station = single_station
        self.fetch_mode = fetch_mode
        self.preprocess_workers = preprocess_workers
        self.source_tables = source_tables
        self.preprocess_backend = preprocess_backend
        self.log_every_tables = log_every_tables

    def fetch(self):
        """Fetch inventory and archives, then convert both into DuckDB tables."""
        from .fetch.data.download import fetch_station_data
        from .fetch.data.preprocess import preprocess_station_data
        from .fetch.stations.inventory import (
            fetch_station_inventory,
            preprocess_station_inventory,
        )

        fetch_station_inventory(root_dir=self.root_dir)
        preprocess_station_inventory(
            root_dir=self.root_dir,
            brazil_boundary_path=self.brazil_boundary_path,
            river_network_dir=self.river_network_dir,
        )
        fetch_station_data(
            root_dir=self.root_dir,
            download_dir=self.download_dir,
            headless=self.headless,
            keep_browser_on_error=self.keep_browser_on_error,
            single_station=self.single_station,
            fetch_mode=self.fetch_mode,
        )
        return preprocess_station_data(
            root_dir=self.root_dir,
            single_station=self.single_station,
            preprocess_workers=self.preprocess_workers,
            source_tables=self.source_tables,
            preprocess_backend=self.preprocess_backend,
            log_every_tables=self.log_every_tables,
        )

    def preprocess(self):
        """Rename, clean, and export sensor data as parquet files."""
        from .preprocess import preprocess_all

        return preprocess_all(root_dir=self.root_dir)

__all__ = ["sensor_data"]
