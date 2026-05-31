try:
    from .. import river_network as rn_module
except ImportError:
    try:
        import code.data.river_network as rn_module
    except ImportError:
        import river_network as rn_module
