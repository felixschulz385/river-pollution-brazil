from __future__ import annotations

from pathlib import Path


def population_dir(root_dir: str | Path) -> Path:
    path = Path(root_dir) / "data" / "population"
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_dir(root_dir: str | Path) -> Path:
    path = population_dir(root_dir) / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def processed_dir(root_dir: str | Path) -> Path:
    path = population_dir(root_dir) / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path


DEFAULT_POPULATION_OUTPUT_FILENAME = "population.parquet"
