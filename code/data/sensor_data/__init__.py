__all__ = ["sensor_data", "sensor_stations"]


def __getattr__(name):
    if name == "sensor_data":
        from .sensor_data import sensor_data as _sensor_data

        return _sensor_data
    if name == "sensor_stations":
        from .sensor_stations import sensor_stations as _sensor_stations

        return _sensor_stations
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
