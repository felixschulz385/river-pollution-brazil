__all__ = ["fetch_station_data", "preprocess_station_data"]


def __getattr__(name):
    if name == "fetch_station_data":
        from .download import fetch_station_data as _fetch_station_data

        return _fetch_station_data
    if name == "preprocess_station_data":
        from .preprocess import preprocess_station_data as _preprocess_station_data

        return _preprocess_station_data
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
