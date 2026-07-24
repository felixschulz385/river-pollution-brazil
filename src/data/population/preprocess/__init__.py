__all__ = [
    "normalize_age_group",
    "normalize_text",
    "preprocess_population_data",
    "transform_population_frame",
]


def __getattr__(name):
    if name == "normalize_age_group":
        from .preprocess import normalize_age_group as _normalize_age_group

        return _normalize_age_group
    if name == "normalize_text":
        from .preprocess import normalize_text as _normalize_text

        return _normalize_text
    if name == "transform_population_frame":
        from .preprocess import transform_population_frame as _transform_population_frame

        return _transform_population_frame
    if name == "preprocess_population_data":
        from .preprocess import preprocess_population_data as _preprocess_population_data

        return _preprocess_population_data
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
