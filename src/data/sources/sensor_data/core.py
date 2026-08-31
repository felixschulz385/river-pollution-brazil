from src.data.sources.river_network.constants import PROCESSED_DIR as RIVER_NETWORK_PROCESSED_DIR


class SensorData:
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
        """Scrape station inventory and archives, then export raw parquet tables."""
        from .fetch.data.download import fetch_station_data
        from .fetch.data.preprocess import preprocess_station_data
        from .fetch.export import export_raw_tables
        from .fetch.stations.inventory import (
            fetch_station_inventory,
            preprocess_station_inventory,
        )

        fetch_station_inventory(root_dir=self.root_dir)
        preprocess_station_inventory(root_dir=self.root_dir)
        fetch_station_data(
            root_dir=self.root_dir,
            download_dir=self.download_dir,
            headless=self.headless,
            keep_browser_on_error=self.keep_browser_on_error,
            single_station=self.single_station,
            fetch_mode=self.fetch_mode,
        )
        preprocess_station_data(
            root_dir=self.root_dir,
            single_station=self.single_station,
            preprocess_workers=self.preprocess_workers,
            source_tables=self.source_tables,
            preprocess_backend=self.preprocess_backend,
            log_every_tables=self.log_every_tables,
        )
        return export_raw_tables(root_dir=self.root_dir)

    def preprocess(
        self,
        stations_rivers_path=None,
        output_path=None,
        n_jobs=None,
    ):
        """Clean raw sensor data, then join it with GADM/river_network into
        the final water-quality/streamflow panel. This is the one stage
        where GADM and river_network are required -- fetch has no such
        dependency.

        Cleaned water-quality, streamflow, and station data are passed
        straight from `preprocess_all()` into `assemble_sensor_data()` in
        memory -- none of the three have a consumer outside this method
        (the panel `assemble_sensor_data()` writes is the canonical output
        land_cover/climate read), so none are written to disk as
        intermediate files."""
        from .preprocess import assemble_sensor_data, preprocess_all

        outputs = preprocess_all(root_dir=self.root_dir)
        return assemble_sensor_data(
            root_dir=self.root_dir,
            water_quality_frame=outputs["clean_frame"],
            streamflow_frame=outputs["streamflow"],
            stations_frame=outputs["stations"],
            stations_rivers_path=stations_rivers_path,
            river_network_path=self.river_network_dir or RIVER_NETWORK_PROCESSED_DIR,
            brazil_boundary_path=self.brazil_boundary_path,
            output_path=output_path,
            n_jobs=n_jobs,
        )


__all__ = ["SensorData"]
