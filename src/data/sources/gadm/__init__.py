__all__ = ["Gadm"]


def __getattr__(name):
    if name == "Gadm":
        from .core import Gadm as _Gadm

        return _Gadm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
