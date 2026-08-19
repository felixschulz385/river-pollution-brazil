__all__ = ["Biomes"]


def __getattr__(name):
    if name == "Biomes":
        from .core import Biomes as _Biomes

        return _Biomes
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
