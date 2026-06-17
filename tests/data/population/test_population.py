from __future__ import annotations

import pandas as pd

import code

from code.data.population.preprocess import (
    normalize_age_group,
    normalize_text,
    preprocess_population_data,
    transform_population_frame,
)


def test_normalize_text_removes_accents_and_punctuation() -> None:
    assert normalize_text("  São José-Do Rio! ") == "sao_jose_do_rio"


def test_normalize_age_group_zero_pads_numeric_ranges() -> None:
    assert normalize_age_group("5 a 9 anos") == "05_09"
    assert normalize_age_group("0_4") == "00_04"
    assert normalize_age_group("20 a 24 anos") == "20_24"
    assert normalize_age_group("70_74") == "70_74"
    assert normalize_age_group("80-mais") == "80_plus"


def test_transform_population_frame_matches_notebook_logic() -> None:
    raw = pd.DataFrame(
        {
            "ano": ["2021", "2020", "2020"],
            "id_municipio": ["7654321", "1234567", "1234567"],
            "id_municipio_nome": ["Bar", "Foo", "Foo"],
            "sexo": ["Masculino", "Masculino", "Feminino"],
            "grupo_idade": ["80-mais", "5 a 9 anos", "20 a 24 anos"],
            "populacao": ["31", "8", "15"],
        }
    )

    result = transform_population_frame(raw)

    assert list(result.columns) == ["mun_id", "year", "sex", "age_group", "population"]
    assert result.to_dict("records") == [
        {
            "mun_id": "123456",
            "year": 2020,
            "sex": "male",
            "age_group": "05_09",
            "population": 8,
        },
        {
            "mun_id": "123456",
            "year": 2020,
            "sex": "female",
            "age_group": "20_24",
            "population": 15,
        },
        {
            "mun_id": "765432",
            "year": 2021,
            "sex": "male",
            "age_group": "80_plus",
            "population": 31,
        },
    ]


def test_preprocess_population_data_writes_expected_output(tmp_path) -> None:
    root_dir = tmp_path
    raw_dir = root_dir / "data" / "population" / "raw"
    raw_dir.mkdir(parents=True)

    raw = pd.DataFrame(
        {
            "ano": ["2022"],
            "id_municipio": ["1100015"],
            "id_municipio_nome": ["Alta Floresta D'Oeste"],
            "sexo": ["Total"],
            "grupo_idade": ["70 a 79 anos"],
            "populacao": ["1234"],
        }
    )
    raw.to_parquet(raw_dir / "population_raw.parquet", index=False)

    output_path = preprocess_population_data(root_dir=root_dir)
    result = pd.read_parquet(output_path)

    assert output_path == root_dir / "data" / "population" / "population.parquet"
    assert result.to_dict("records") == [
        {
            "mun_id": "110001",
            "year": 2022,
            "sex": "total",
            "age_group": "70_79",
            "population": 1234,
        }
    ]
