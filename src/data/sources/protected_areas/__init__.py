__all__ = ["ProtectedAreas"]


def __getattr__(name):
    if name == "ProtectedAreas":
        from .core import ProtectedAreas as _ProtectedAreas

        return _ProtectedAreas
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
