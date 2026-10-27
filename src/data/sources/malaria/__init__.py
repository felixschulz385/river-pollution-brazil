__all__ = ["Malaria"]


def __getattr__(name):
    if name == "Malaria":
        from .core import Malaria as _Malaria

        return _Malaria
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
