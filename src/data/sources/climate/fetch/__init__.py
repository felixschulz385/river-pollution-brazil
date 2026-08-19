__all__ = [
    "fetch_cloud_cover",
    "fetch_era5_land_daily",
    "fetch_era5_land_hourly",
]


def __getattr__(name):
    if name == "fetch_cloud_cover":
        from .cloud_cover import fetch_cloud_cover as _fetch_cloud_cover

        return _fetch_cloud_cover
    if name == "fetch_era5_land_hourly":
        from .era5_land_hourly import (
            fetch_era5_land_hourly as _fetch_era5_land_hourly,
        )

        return _fetch_era5_land_hourly
    if name == "fetch_era5_land_daily":
        from .era5_land_daily import fetch_era5_land_daily as _fetch_era5_land_daily

        return _fetch_era5_land_daily
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
