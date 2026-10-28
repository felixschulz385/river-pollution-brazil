__all__ = ["Mining"]


def __getattr__(name):
    if name == "Mining":
        from .core import Mining as _Mining

        return _Mining
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
