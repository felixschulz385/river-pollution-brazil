"""Compatibility import for the shared river-network module."""

try:
    from .. import river_network as rn_module
except ImportError:
    import river_network as rn_module


__all__ = ["rn_module"]
