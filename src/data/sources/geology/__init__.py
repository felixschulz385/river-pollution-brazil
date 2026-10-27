__all__ = ["Geology"]


def __getattr__(name):
    if name == "Geology":
        from .core import Geology as _Geology

        return _Geology
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
