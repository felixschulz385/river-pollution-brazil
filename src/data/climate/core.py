class Climate:
    """Run climate-data workflows."""

    def __init__(self, root_dir="."):
        self.root_dir = root_dir

    def fetch(self, subtype="cloud_cover"):
        """Fetch the requested climate subtype."""
        if subtype == "cloud_cover":
            from .fetch.cloud_cover import fetch_cloud_cover

            return fetch_cloud_cover(root_dir=self.root_dir)
        if subtype == "era5_land_hourly":
            from .fetch.era5_land_hourly import fetch_era5_land_hourly

            return fetch_era5_land_hourly(root_dir=self.root_dir)
        if subtype == "era5_land_daily":
            from .fetch.era5_land_daily import fetch_era5_land_daily

            return fetch_era5_land_daily(root_dir=self.root_dir)
        if subtype == "era5_land_arco":
            from .fetch.era5_land_arco import fetch_era5_land_arco

            return fetch_era5_land_arco(root_dir=self.root_dir)
        raise ValueError(f"Unsupported climate fetch subtype: {subtype}")

    def preprocess(self, subtype="cloud_cover", stage="all", n_jobs=None):
        """Preprocess the requested climate subtype."""
        if subtype == "cloud_cover":
            from .preprocess.cloud_cover import preprocess_cloud_cover

            return preprocess_cloud_cover(root_dir=self.root_dir)
        if subtype in {"era5_land_hourly", "era5_land_daily"}:
            from .preprocess.era5_land import preprocess_era5_land_worker

            return preprocess_era5_land_worker(
                root_dir=self.root_dir,
                subtype=subtype,
                stage=stage,
                n_jobs=n_jobs,
            )
        raise ValueError(f"Unsupported climate preprocess subtype: {subtype}")

    def assemble(
        self,
        variant="sensor_upstream_distance_buckets",
        climate_path=None,
        water_quality_path=None,
        stations_rivers_path=None,
        river_network_path=None,
        output_path=None,
        n_jobs=None,
    ):
        """Assemble the requested climate variant."""
        from .assembly import assemble_climate

        return assemble_climate(
            self,
            variant=variant,
            climate_path=climate_path,
            water_quality_path=water_quality_path,
            stations_rivers_path=stations_rivers_path,
            river_network_path=river_network_path,
            output_path=output_path,
            n_jobs=n_jobs,
        )


__all__ = ["Climate"]
