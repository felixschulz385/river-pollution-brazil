__all__ = ["Enforcement"]


def __getattr__(name):
    if name == "Enforcement":
        from .core import Enforcement as _Enforcement

        return _Enforcement
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
