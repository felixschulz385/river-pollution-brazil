from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd


def _population_dir(root_dir: str | Path) -> Path:
    path = Path(root_dir) / "data" / "population"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _raw_dir(root_dir: str | Path) -> Path:
    path = _population_dir(root_dir) / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_text(value: str) -> str:
    if pd.isna(value):
        return value

    normalized = str(value).strip().lower()
    normalized = unicodedata.normalize("NFKD", normalized)
    normalized = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return re.sub(r"_+", "_", normalized).strip("_")


def transform_population_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the notebook cleaning rules to the raw population extract."""

    return (
        frame.rename(
            columns={
                "ano": "year",
                "id_municipio": "mun_id",
                "id_municipio_nome": "mun_name",
                "sexo": "sex",
                "grupo_idade": "age_group",
                "populacao": "population",
            }
        )
        .assign(
            mun_id=lambda d: d["mun_id"].astype(str).str[:6],
            sex=lambda d: d["sex"].map(normalize_text).replace(
                {
                    "feminino": "female",
                    "masculino": "male",
                }
            ),
            age_group=lambda d: (
                d["age_group"]
                .astype(str)
                .str.replace(" anos", "", regex=False)
                .str.replace("80-mais", "80_plus", regex=False)
                .map(normalize_text)
            ),
            population=lambda d: pd.to_numeric(d["population"], errors="coerce"),
            year=lambda d: pd.to_numeric(d["year"], errors="coerce").astype("Int64"),
        )
        .drop(columns=["mun_name"])
        .loc[:, ["mun_id", "year", "sex", "age_group", "population"]]
    )


def preprocess_population_data(
    root_dir: str | Path = ".",
    raw_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Load the raw population extract, normalize it, and persist the cleaned table."""

    source = Path(raw_path) if raw_path else _raw_dir(root_dir) / "population_raw.parquet"
    if not source.exists():
        raise FileNotFoundError(f"Raw population file not found: {source}")

    destination = Path(output_path) if output_path else _population_dir(root_dir) / "population.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(source)
    cleaned = transform_population_frame(frame)
    cleaned.to_parquet(destination, index=False)
    return destination
