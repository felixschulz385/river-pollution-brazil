__all__ = [
    "preprocess_birth_outcome_tables",
    "preprocess_health_data",
    "preprocess_hospitalization_tables",
    "preprocess_mortality_age_tables",
]


def __getattr__(name):
    if name == "preprocess_health_data":
        from .preprocess import preprocess_health_data as _preprocess_health_data

        return _preprocess_health_data
    if name == "preprocess_mortality_age_tables":
        from .preprocess import preprocess_mortality_age_tables as _preprocess_mortality_age_tables

        return _preprocess_mortality_age_tables
    if name == "preprocess_hospitalization_tables":
        from .preprocess import preprocess_hospitalization_tables as _preprocess_hospitalization_tables

        return _preprocess_hospitalization_tables
    if name == "preprocess_birth_outcome_tables":
        from .preprocess import preprocess_birth_outcome_tables as _preprocess_birth_outcome_tables

        return _preprocess_birth_outcome_tables
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
