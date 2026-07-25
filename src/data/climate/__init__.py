__all__ = ["Climate"]


def __getattr__(name):
    if name == "Climate":
        from .core import Climate as _Climate

        return _Climate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
