from __future__ import annotations

from pathlib import Path

from src.data.shared.paths import processed_dir as _shared_processed_dir
from src.data.shared.paths import raw_dir as _shared_raw_dir
from src.data.shared.paths import source_root

HEALTH_DATASET_NAME = "health"


def health_dir(root_dir: str | Path = ".") -> Path:
    return source_root(root_dir, "health")


def raw_dir(root_dir: str | Path = ".") -> Path:
    return _shared_raw_dir(root_dir, "health")


def processed_dir(root_dir: str | Path = ".") -> Path:
    return _shared_processed_dir(root_dir, "health")


# Processed hospitalization outputs (preprocess_hospitalization_tables).
HOSPITALIZATIONS_FILENAME = "health_hospitalizations.parquet"
HOSPITALIZATIONS_ICD10_CHAPTER_FILENAME = "health_hospitalizations_icd10_chapter.parquet"
HOSPITALIZATIONS_SELECTED_MORBIDITY_LIST_FILENAME = "health_hospitalizations_selected_morbidity_list.parquet"

# Processed mortality-age-count outputs (preprocess_mortality_age_tables).
MORTALITY_PRE_1996_FILENAME = "health_scraping_pre_1996.csv"
MORTALITY_POST_1996_FILENAME = "health_scraping_post_1996.csv"


def birth_outcome_filename(outcome_name: str) -> str:
    """Processed birth-outcome output filename (preprocess_birth_outcome_tables)."""
    return f"health_{outcome_name}.parquet"
