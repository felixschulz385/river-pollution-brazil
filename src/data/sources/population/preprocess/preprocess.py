from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

from ..constants import DEFAULT_POPULATION_OUTPUT_FILENAME, DEFAULT_POPULATION_RAW_FILENAME
from ..constants import processed_dir as _processed_dir
from ..constants import raw_dir as _raw_dir
from src.data.sources.land_cover.constants import derive_mun_id_from_adm2_id


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


def normalize_age_group(value: str) -> str:
    if pd.isna(value):
        return value

    normalized = (
        str(value)
        .replace(" anos", "")
        .replace("80-mais", "80_plus")
    )
    normalized = normalize_text(normalized)
    if pd.isna(normalized):
        return normalized

    match = re.fullmatch(r"(\d{1,2})(?:_a)?_(\d{1,2})", normalized)
    if match:
        start, end = match.groups()
        return f"{int(start):02d}_{int(end):02d}"

    return normalized


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
            # Deliberately not `.astype(str)` first: `derive_mun_id_from_adm2_id`
            # needs to see a real null to reject it -- pre-stringifying would
            # turn a NaN `mun_id` into the literal string "nan", which the null
            # check can no longer catch and which would otherwise get silently
            # truncated into a bogus "na" join key.
            mun_id=lambda d: d["mun_id"].map(derive_mun_id_from_adm2_id),
            sex=lambda d: d["sex"].map(normalize_text).replace(
                {
                    "feminino": "female",
                    "masculino": "male",
                }
            ),
            # Deliberately not `.astype(str)` first: see the `mun_id` comment
            # above -- pre-stringifying would turn a real NaN `age_group`
            # into the literal string "nan", which `normalize_age_group`'s
            # `pd.isna` guard can no longer catch.
            age_group=lambda d: d["age_group"].map(normalize_age_group),
            population=lambda d: pd.to_numeric(d["population"], errors="coerce"),
            year=lambda d: pd.to_numeric(d["year"], errors="coerce").astype("Int64"),
        )
        .drop(columns=["mun_name"])
        .loc[:, ["mun_id", "year", "sex", "age_group", "population"]]
        .sort_values(["mun_id", "year", "age_group", "sex"], ignore_index=True)
    )


def preprocess_population_data(
    root_dir: str | Path = ".",
    raw_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Load the raw population extract, normalize it, and persist the cleaned table."""

    source = Path(raw_path) if raw_path else _raw_dir(root_dir) / DEFAULT_POPULATION_RAW_FILENAME
    if not source.exists():
        raise FileNotFoundError(f"Raw population file not found: {source}")

    destination = (
        Path(output_path)
        if output_path
        else _processed_dir(root_dir) / DEFAULT_POPULATION_OUTPUT_FILENAME
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(source)
    cleaned = transform_population_frame(frame)
    cleaned.to_parquet(destination, index=False)
    return destination
