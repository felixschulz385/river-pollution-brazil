__all__ = ["CommodityPrices"]


def __getattr__(name):
    if name == "CommodityPrices":
        from .core import CommodityPrices as _CommodityPrices

        return _CommodityPrices
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
