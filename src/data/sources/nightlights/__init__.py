__all__ = ["Nightlights"]


def __getattr__(name):
    if name == "Nightlights":
        from .core import Nightlights as _Nightlights

        return _Nightlights
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
