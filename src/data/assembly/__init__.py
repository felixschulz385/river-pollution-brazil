__all__ = ["Assembly"]


def __getattr__(name):
    if name == "Assembly":
        from .core import Assembly as _Assembly

        return _Assembly
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
