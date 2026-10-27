from __future__ import annotations

import pytest

from src.data.assembly.schema import get_dataset_config, load_assembly_config

CONFIG_YAML = """
datasets:
  - id: sensor_panel
    mode: sensor
    index: [station_code, datetime]
    output_path: data/assembly/sensor_panel.parquet
    sources:
      - name: water_quality
        path: data/sensor_data/processed/aggregate/sensor_data.parquet
        join_keys: [station_code, datetime]
        variables: [ph]
      - name: land_cover
        path: data/land_cover/processed/aggregate/land_cover_sensor_upstream.parquet
        type: land_cover_bucketed
        join_keys: [station_code, year]
        variables: [lc_forest]
"""


def test_load_assembly_config_parses_datasets_and_sources(tmp_path):
    config_path = tmp_path / "assembly_datasets.yaml"
    config_path.write_text(CONFIG_YAML)

    datasets = load_assembly_config(config_path)

    assert set(datasets) == {"sensor_panel"}
    dataset = datasets["sensor_panel"]
    assert dataset.mode == "sensor"
    assert dataset.index == ("station_code", "datetime")
    assert len(dataset.sources) == 2
    assert dataset.sources[1].type == "land_cover_bucketed"
    assert dataset.sources[1].variables == ("lc_forest",)


def test_get_dataset_config_raises_for_unknown_dataset(tmp_path):
    config_path = tmp_path / "assembly_datasets.yaml"
    config_path.write_text(CONFIG_YAML)

    with pytest.raises(ValueError, match="not found"):
        get_dataset_config(config_path, "does_not_exist")


def test_load_assembly_config_rejects_unsupported_mode(tmp_path):
    config_path = tmp_path / "assembly_datasets.yaml"
    config_path.write_text(
        """
datasets:
  - id: bad
    mode: county
    index: [x]
    output_path: out.parquet
    sources:
      - path: a.parquet
        join_keys: [x]
        variables: [y]
"""
    )

    with pytest.raises(ValueError, match="unsupported mode"):
        load_assembly_config(config_path)


def test_load_assembly_config_rejects_unknown_source_type(tmp_path):
    # A typo'd/misspelled `type:` (e.g. "climate_bucket" instead of
    # "climate_bucketed") must fail loudly at config-load time, not silently
    # fall through `build.py`'s type dispatch and get treated as plain wide
    # data.
    config_path = tmp_path / "assembly_datasets.yaml"
    config_path.write_text(
        """
datasets:
  - id: sensor_panel
    mode: sensor
    index: [station_code, datetime]
    output_path: out.parquet
    sources:
      - name: climate
        path: a.parquet
        type: climate_bucket
        join_keys: [station_code, year]
        variables: [tp]
"""
    )

    with pytest.raises(ValueError, match="unknown type"):
        load_assembly_config(config_path)


def test_load_assembly_config_rejects_local_scope_on_unbucketed_source(tmp_path):
    config_path = tmp_path / "assembly_datasets.yaml"
    config_path.write_text(
        """
datasets:
  - id: adm2_panel
    mode: adm2
    index: [mun_id, year]
    output_path: out.parquet
    sources:
      - name: population
        path: a.parquet
        scope: local
        join_keys: [mun_id, year]
        variables: [population]
"""
    )

    with pytest.raises(ValueError, match="only applies to bucketed sources"):
        load_assembly_config(config_path)


def _write_single_source_config(tmp_path, source_yaml):
    config_path = tmp_path / "assembly_datasets.yaml"
    config_path.write_text(
        """
datasets:
  - id: adm2_panel
    mode: adm2
    index: [mun_id, year]
    output_path: out.parquet
    sources:
"""
        + source_yaml
    )
    return config_path


def test_load_assembly_config_parses_reach_cap_and_fill(tmp_path):
    config_path = _write_single_source_config(
        tmp_path,
        """      - name: c11
        path: a.parquet
        type: mining_bucketed
        max_km: 50
        join_keys: [mun_id, year]
        variables: [mining_garimpo_gold]
      - name: malaria
        path: b.parquet
        fill: {value: 0, years: [2004, 2024]}
        join_keys: [mun_id, year]
        variables: [inf_cases]
""",
    )

    c11, malaria = load_assembly_config(config_path)["adm2_panel"].sources

    assert c11.type == "mining_bucketed" and c11.max_km == 50
    assert malaria.fill_value == 0 and malaria.fill_years == (2004, 2024)


def test_load_assembly_config_rejects_reach_cap_on_unbucketed_or_local_source(tmp_path):
    for extra in ("", "        type: land_cover_bucketed\n        scope: local\n"):
        config_path = _write_single_source_config(
            tmp_path,
            f"""      - name: population
        path: a.parquet
{extra}        max_km: 50
        join_keys: [mun_id, year]
        variables: [population]
""",
        )

        with pytest.raises(ValueError, match="sets max_km"):
            load_assembly_config(config_path)


def test_load_assembly_config_rejects_fill_without_value(tmp_path):
    config_path = _write_single_source_config(
        tmp_path,
        """      - name: malaria
        path: b.parquet
        fill: {years: [2004, 2024]}
        join_keys: [mun_id, year]
        variables: [inf_cases]
""",
    )

    with pytest.raises(ValueError, match="invalid fill"):
        load_assembly_config(config_path)
