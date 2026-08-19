__all__ = [
    "fetch_era5_land_daily",
    "fetch_era5_land_hourly",
]


def __getattr__(name):
    if name == "fetch_era5_land_hourly":
        from .era5_land_hourly import (
            fetch_era5_land_hourly as _fetch_era5_land_hourly,
        )

        return _fetch_era5_land_hourly
    if name == "fetch_era5_land_daily":
        from .era5_land_daily import fetch_era5_land_daily as _fetch_era5_land_daily

        return _fetch_era5_land_daily
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
