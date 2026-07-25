__all__ = ["SensorData"]


def __getattr__(name):
    if name == "SensorData":
        from .core import SensorData as _SensorData

        return _SensorData
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
