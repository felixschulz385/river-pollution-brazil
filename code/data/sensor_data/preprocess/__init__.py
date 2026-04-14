__all__ = [
    "preprocess_all",
    "preprocess_sensor_data",
    "preprocess_stations_rivers",
    "preprocess_streamflow",
]


def __getattr__(name):
    if name == "preprocess_sensor_data":
        from .preprocess import preprocess_sensor_data as _preprocess_sensor_data

        return _preprocess_sensor_data
    if name == "preprocess_stations_rivers":
        from .preprocess import preprocess_stations_rivers as _preprocess_stations_rivers

        return _preprocess_stations_rivers
    if name == "preprocess_streamflow":
        from .preprocess import preprocess_streamflow as _preprocess_streamflow

        return _preprocess_streamflow
    if name == "preprocess_all":
        from .preprocess import preprocess_all as _preprocess_all

        return _preprocess_all
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
