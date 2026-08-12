class climate:
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

    def preprocess(self, subtype="cloud_cover"):
        """Preprocess the requested climate subtype."""
        if subtype == "cloud_cover":
            from .preprocess.cloud_cover import preprocess_cloud_cover

            return preprocess_cloud_cover(root_dir=self.root_dir)
        if subtype in {"era5_land_hourly", "era5_land_daily"}:
            from .preprocess.era5_land import preprocess_era5_land_worker

            return preprocess_era5_land_worker(root_dir=self.root_dir, subtype=subtype)
        raise ValueError(f"Unsupported climate preprocess subtype: {subtype}")


__all__ = ["climate"]
