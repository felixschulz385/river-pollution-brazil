__all__ = ["preprocess_cloud_cover", "preprocess_era5_land", "preprocess_era5_land_worker"]


def __getattr__(name):
    if name == "preprocess_cloud_cover":
        from .cloud_cover import preprocess_cloud_cover as _preprocess_cloud_cover

        return _preprocess_cloud_cover
    if name == "preprocess_era5_land":
        from .era5_land import preprocess_era5_land as _preprocess_era5_land

        return _preprocess_era5_land
    if name == "preprocess_era5_land_worker":
        from .era5_land import preprocess_era5_land_worker as _preprocess_era5_land_worker

        return _preprocess_era5_land_worker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
