class Climate:
    """Run climate-data workflows."""

    def __init__(self, root_dir="."):
        self.root_dir = root_dir

    def fetch(self, subtype="era5_land_hourly"):
        """Fetch the requested ERA5-Land variant.

        All variants write into the same shared zarr store
        (DEFAULT_ERA5_LAND_STORE_PATH); this only selects *how* the raw data
        is obtained (CDS GRIB vs. the ARCO cloud store).
        """
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

    def preprocess(self, stage="all", n_jobs=None):
        """Preprocess GRIB-origin ERA5-Land input into the shared zarr store.

        Not split by variant: era5_land_hourly and era5_land_daily both feed
        the same store, so both are always drained together. era5_land_arco
        writes directly during fetch and has no separate preprocess step.
        """
        from .preprocess.era5_land import preprocess_era5_land_worker

        return preprocess_era5_land_worker(root_dir=self.root_dir, stage=stage, n_jobs=n_jobs)

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
