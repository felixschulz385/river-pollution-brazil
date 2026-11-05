__all__ = ["CloudCover"]


def __getattr__(name):
    if name == "CloudCover":
        from .core import CloudCover as _CloudCover

        return _CloudCover
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
