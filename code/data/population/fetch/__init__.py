__all__ = ["POPULATION_QUERY", "fetch_population_data"]


def __getattr__(name):
    if name == "POPULATION_QUERY":
        from .population import POPULATION_QUERY as _POPULATION_QUERY

        return _POPULATION_QUERY
    if name == "fetch_population_data":
        from .population import fetch_population_data as _fetch_population_data

        return _fetch_population_data
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
