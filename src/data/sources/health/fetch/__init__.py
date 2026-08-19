__all__ = [
    "DatasusTabnetForm",
    "fetch_birth_outcome_tables",
    "fetch_health_data",
    "fetch_hospitalization_tables",
    "fetch_mortality_age_tables",
    "fetch_sih_residence_icd10_chapter_municipality_year",
    "fetch_sih_residence_selected_morbidity_list_municipality_year",
    "fetch_sih_residence_total_municipality_year",
]


def __getattr__(name):
    if name == "DatasusTabnetForm":
        from .forms import DatasusTabnetForm as _DatasusTabnetForm

        return _DatasusTabnetForm
    if name == "fetch_health_data":
        from .datasus import fetch_health_data as _fetch_health_data

        return _fetch_health_data
    if name == "fetch_mortality_age_tables":
        from .datasus import fetch_mortality_age_tables as _fetch_mortality_age_tables

        return _fetch_mortality_age_tables
    if name == "fetch_hospitalization_tables":
        from .datasus import fetch_hospitalization_tables as _fetch_hospitalization_tables

        return _fetch_hospitalization_tables
    if name == "fetch_sih_residence_total_municipality_year":
        from .datasus import (
            fetch_sih_residence_total_municipality_year as _fetch_sih_residence_total_municipality_year,
        )

        return _fetch_sih_residence_total_municipality_year
    if name == "fetch_sih_residence_icd10_chapter_municipality_year":
        from .datasus import (
            fetch_sih_residence_icd10_chapter_municipality_year as _fetch_sih_residence_icd10_chapter_municipality_year,
        )

        return _fetch_sih_residence_icd10_chapter_municipality_year
    if name == "fetch_sih_residence_selected_morbidity_list_municipality_year":
        from .datasus import (
            fetch_sih_residence_selected_morbidity_list_municipality_year as _fetch_sih_residence_selected_morbidity_list_municipality_year,
        )

        return _fetch_sih_residence_selected_morbidity_list_municipality_year
    if name == "fetch_birth_outcome_tables":
        from .datasus import fetch_birth_outcome_tables as _fetch_birth_outcome_tables

        return _fetch_birth_outcome_tables
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
