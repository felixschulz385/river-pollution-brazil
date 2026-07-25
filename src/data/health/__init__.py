__all__ = ["Health"]


def __getattr__(name):
    if name == "Health":
        from .core import Health as _Health

        return _Health
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
