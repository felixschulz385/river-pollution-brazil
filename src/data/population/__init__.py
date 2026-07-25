__all__ = ["Population"]


def __getattr__(name):
    if name == "Population":
        from .core import Population as _Population

        return _Population
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
