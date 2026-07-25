__all__ = [
    "LandCover",
    "configure_logging",
    "get_output_columns",
    "process_year",
]


def __getattr__(name):
    if name == "LandCover":
        from .core import LandCover as _LandCover

        return _LandCover
    if name in {"configure_logging", "process_year"}:
        from . import preprocess as _preprocess

        return getattr(_preprocess, name)
    if name == "get_output_columns":
        from .schema import get_output_columns as _get_output_columns

        return _get_output_columns
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
