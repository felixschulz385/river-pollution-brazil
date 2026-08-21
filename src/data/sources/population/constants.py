from __future__ import annotations

from pathlib import Path


def population_dir(root_dir: str | Path) -> Path:
    return Path(root_dir) / "data" / "population"


def raw_dir(root_dir: str | Path) -> Path:
    return population_dir(root_dir) / "raw"


def processed_dir(root_dir: str | Path) -> Path:
    return population_dir(root_dir) / "processed"


DEFAULT_POPULATION_OUTPUT_FILENAME = "population.parquet"
