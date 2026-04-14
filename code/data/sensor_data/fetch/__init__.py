__all__ = [
    "fetch_station_data",
    "fetch_station_inventory",
    "preprocess_station_data",
    "preprocess_station_inventory",
]


def __getattr__(name):
    if name == "fetch_station_data":
        from .data.download import fetch_station_data as _fetch_station_data

        return _fetch_station_data
    if name == "preprocess_station_data":
        from .data.preprocess import preprocess_station_data as _preprocess_station_data

        return _preprocess_station_data
    if name == "fetch_station_inventory":
        from .stations.inventory import (
            fetch_station_inventory as _fetch_station_inventory,
        )

        return _fetch_station_inventory
    if name == "preprocess_station_inventory":
        from .stations.inventory import (
            preprocess_station_inventory as _preprocess_station_inventory,
        )

        return _preprocess_station_inventory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
