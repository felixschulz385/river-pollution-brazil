from __future__ import annotations

from pathlib import Path

from src.data.shared.paths import processed_dir as _shared_processed_dir
from src.data.shared.paths import raw_dir as _shared_raw_dir
from src.data.shared.paths import source_root


def population_dir(root_dir: str | Path) -> Path:
    return source_root(root_dir, "population")


def raw_dir(root_dir: str | Path) -> Path:
    return _shared_raw_dir(root_dir, "population")


def processed_dir(root_dir: str | Path) -> Path:
    return _shared_processed_dir(root_dir, "population")


DEFAULT_POPULATION_OUTPUT_FILENAME = "population.parquet"
DEFAULT_POPULATION_RAW_FILENAME = "population_raw.parquet"
