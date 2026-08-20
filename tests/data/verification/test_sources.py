from __future__ import annotations

import json

import pandas as pd
import pytest

from src.data.verification.sources import SOURCE_ADAPTERS


# --------------------------------------------------------------------------
# river_network
# --------------------------------------------------------------------------

def test_river_network_list_fetched_absent(tmp_path):
    adapter = SOURCE_ADAPTERS["river_network"]

    listing = adapter.list_fetched(tmp_path)

    assert listing.present == 0
    assert listing.expected == 1


def test_river_network_check_outputs_missing_files(tmp_path):
    adapter = SOURCE_ADAPTERS["river_network"]

    artifacts = adapter.check_outputs(tmp_path)

    assert len(artifacts) == 2
    assert all(not artifact.exists for artifact in artifacts)
    assert all(not artifact.ok for artifact in artifacts)


def test_river_network_check_outputs_flags_duplicate_trench_id(tmp_path):
    river_dir = tmp_path / "data" / "river_network" / "processed"
    river_dir.mkdir(parents=True)
    trenches = pd.DataFrame(
        {
            "trench_id": [1, 1],
            "upstream_node": [1, 2],
            "downstream_node": [2, 3],
            "distance": [1.0, 2.0],
            "system_id": [0, 0],
        }
    )
    trenches.to_parquet(river_dir / "river_trenches.parquet", index=False)

    adapter = SOURCE_ADAPTERS["river_network"]
    artifacts = adapter.check_outputs(tmp_path)

    trench_artifact = next(a for a in artifacts if a.label == "river_trenches")
    assert trench_artifact.exists
    uniqueness_check = next(c for c in trench_artifact.checks if c.name == "unique:trench_id")
    assert not uniqueness_check.ok


def test_river_network_check_outputs_passes_valid_data(tmp_path):
    river_dir = tmp_path / "data" / "river_network" / "processed"
    river_dir.mkdir(parents=True)
    trenches = pd.DataFrame(
        {
            "trench_id": [1, 2],
            "upstream_node": [1, 2],
            "downstream_node": [2, 3],
            "distance": [1.0, 2.0],
            "system_id": [0, 0],
        }
    )
    trenches.to_parquet(river_dir / "river_trenches.parquet", index=False)
    drainage = pd.DataFrame(
        {"trench_id": [1, 2], "drainage_area": [10.0, 20.0], "within_brazil": [True, True]}
    )
    drainage.to_parquet(river_dir / "drainage_areas.parquet", index=False)

    adapter = SOURCE_ADAPTERS["river_network"]
    artifacts = adapter.check_outputs(tmp_path)

    assert all(artifact.ok for artifact in artifacts)


# --------------------------------------------------------------------------
# land_cover
# --------------------------------------------------------------------------

def test_land_cover_list_fetched_no_raw_dir(tmp_path):
    adapter = SOURCE_ADAPTERS["land_cover"]

    listing = adapter.list_fetched(tmp_path)

    assert listing.present == 0
    assert listing.expected == 40  # 1985..2024 inclusive


def test_land_cover_list_fetched_detects_year_gap(tmp_path):
    datadir = tmp_path / "data" / "land_cover" / "raw" / "lc_mapbiomas10_30"
    datadir.mkdir(parents=True)
    for year in (2000, 2001, 2003):
        (datadir / f"brazil_coverage_{year}.tif").write_bytes(b"fake-tif")

    adapter = SOURCE_ADAPTERS["land_cover"]
    listing = adapter.list_fetched(tmp_path)

    assert listing.present == 3
    assert listing.expected == 40  # deterministic 1985..2024 span, not observed span
    assert "2002" in listing.detail


def test_land_cover_check_outputs_missing(tmp_path):
    adapter = SOURCE_ADAPTERS["land_cover"]

    artifacts = adapter.check_outputs(tmp_path)

    assert len(artifacts) == 1
    assert not artifacts[0].exists


def test_land_cover_check_outputs_flags_out_of_range_share(tmp_path):
    """The real output is long-format (land_cover_class as a row value, a
    single "share" column), not per-class "{class}_shr" columns."""
    output_dir = tmp_path / "data" / "land_cover" / "processed" / "aggregate"
    output_dir.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "station_code": ["A"],
            "year": [2020],
            "land_cover_class": ["forest"],
            "share": [1.5],
        }
    )
    frame.to_parquet(output_dir / "land_cover_sensor_upstream.parquet", index=False)

    adapter = SOURCE_ADAPTERS["land_cover"]
    artifacts = adapter.check_outputs(tmp_path)

    assert artifacts[0].exists
    assert not artifacts[0].ok


# --------------------------------------------------------------------------
# sensor_data
# --------------------------------------------------------------------------

def test_sensor_data_list_fetched_no_database(tmp_path):
    adapter = SOURCE_ADAPTERS["sensor_data"]

    listing = adapter.list_fetched(tmp_path)

    assert listing.present == 0
    assert listing.expected == 1


def test_sensor_data_check_outputs_recognizes_columns_kept_as_named_index(tmp_path):
    """The real assembled panel is written with station_code/datetime as a
    named index (`.set_index([...]).to_parquet(..., index=True)`), not plain
    columns. `check_required_columns` only sees `frame.columns`, so those
    index levels must be restored or they're falsely reported as missing."""
    output_dir = tmp_path / "data" / "sensor_data" / "processed" / "aggregate"
    output_dir.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "station_code": ["S1"],
            "datetime": pd.to_datetime(["2020-01-01"]),
            "discharge": [10.0],
        }
    ).set_index(["station_code", "datetime"])
    frame.to_parquet(output_dir / "water_quality_streamflow.parquet", index=True)

    adapter = SOURCE_ADAPTERS["sensor_data"]
    artifacts = adapter.check_outputs(tmp_path)

    required_columns_check = next(c for c in artifacts[0].checks if c.name == "required_columns")
    assert required_columns_check.ok, required_columns_check.message


def test_sensor_data_check_outputs_flags_discharge_out_of_range(tmp_path):
    """The final assembled table renames "discharge" to
    "streamflow_discharge_day"; the plain "discharge" column never survives
    to this output, so the range check must target the renamed column."""
    output_dir = tmp_path / "data" / "sensor_data" / "processed" / "aggregate"
    output_dir.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "station_code": ["S1"],
            "datetime": pd.to_datetime(["2020-01-01"]),
            "streamflow_discharge_day": [2_000_000.0],
        }
    )
    frame.to_parquet(output_dir / "water_quality_streamflow.parquet", index=False)

    adapter = SOURCE_ADAPTERS["sensor_data"]
    artifacts = adapter.check_outputs(tmp_path)

    assert artifacts[0].exists
    assert not artifacts[0].ok


# --------------------------------------------------------------------------
# climate
# --------------------------------------------------------------------------

def test_climate_list_fetched_absent_directory_does_not_raise(tmp_path):
    adapter = SOURCE_ADAPTERS["climate"]

    listing = adapter.list_fetched(tmp_path)

    assert listing.present == 0
    assert listing.expected is not None and listing.expected > 0


def test_climate_list_fetched_counts_zarr_store_variables_not_raw_files(tmp_path):
    """era5_land_hourly, era5_land_daily, and era5_land_arco all write into
    the same shared zarr store, and preprocessing deletes each raw .grib
    once it's folded in -- fetch completeness must be read from which
    variable arrays exist in that store, not from raw-file/manifest
    presence per fetch variant."""
    from src.data.sources.climate.constants import DEFAULT_ERA5_LAND_STORE_PATH

    adapter = SOURCE_ADAPTERS["climate"]
    store_path = tmp_path / DEFAULT_ERA5_LAND_STORE_PATH
    for variable in ("tp", "sro", "2t"):
        (store_path / variable).mkdir(parents=True)
    # A coordinate array, not a data variable -- must not be counted.
    (store_path / "latitude").mkdir(parents=True)

    listing = adapter.list_fetched(tmp_path)

    assert listing.present == 3
    assert listing.expected is not None and listing.expected > 3


def test_climate_check_outputs_missing(tmp_path):
    adapter = SOURCE_ADAPTERS["climate"]

    artifacts = adapter.check_outputs(tmp_path)

    assert len(artifacts) == 2
    assert all(not artifact.exists for artifact in artifacts)


def test_climate_check_outputs_flags_out_of_range_value(tmp_path):
    """Both climate outputs are long-format: the variable code lives as a
    row value in "climate_variable", not baked into the column name, so the
    range check must filter by that column rather than match a column-name
    prefix (which would never match anything in this schema)."""
    output_dir = tmp_path / "data" / "climate" / "processed" / "aggregate"
    output_dir.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "station_code": ["S1"],
            "date": pd.to_datetime(["2020-01-01"]),
            "distance_bucket": ["0"],
            "climate_variable": ["2t"],
            "reachable_trench_count": [1],
            "mean_day": [1000.0],  # far outside the (180, 340) Kelvin range
        }
    )
    frame.to_parquet(output_dir / "climate_sensor_upstream.parquet", index=False)

    adapter = SOURCE_ADAPTERS["climate"]
    artifacts = adapter.check_outputs(tmp_path)

    sensor_artifact = next(a for a in artifacts if a.label == "climate_sensor_upstream")
    assert sensor_artifact.exists
    assert not sensor_artifact.ok


# --------------------------------------------------------------------------
# biomes
# --------------------------------------------------------------------------

def test_biomes_list_fetched_absent(tmp_path):
    adapter = SOURCE_ADAPTERS["biomes"]

    listing = adapter.list_fetched(tmp_path)

    assert listing.present == 0
    assert listing.expected == 1


def test_biomes_check_outputs_missing_required_column(tmp_path):
    output_dir = tmp_path / "data" / "biomes" / "processed"
    output_dir.mkdir(parents=True)
    frame = pd.DataFrame({"mun_id": ["350001"]})  # missing "biome"
    frame.to_parquet(output_dir / "biome_adm2.parquet", index=False)

    adapter = SOURCE_ADAPTERS["biomes"]
    artifacts = adapter.check_outputs(tmp_path)

    adm2_artifact = next(a for a in artifacts if a.label == "biome_adm2")
    assert adm2_artifact.exists
    assert not adm2_artifact.ok


# --------------------------------------------------------------------------
# population
# --------------------------------------------------------------------------

def test_population_list_fetched_absent(tmp_path):
    adapter = SOURCE_ADAPTERS["population"]

    listing = adapter.list_fetched(tmp_path)

    assert listing.present == 0
    assert listing.expected == 1


def test_population_check_outputs_valid(tmp_path):
    output_dir = tmp_path / "data" / "population" / "processed"
    output_dir.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "mun_id": ["350001"],
            "year": [2020],
            "sex": ["M"],
            "age_group": ["0-4"],
            "population": [100],
        }
    )
    frame.to_parquet(output_dir / "population.parquet", index=False)

    adapter = SOURCE_ADAPTERS["population"]
    artifacts = adapter.check_outputs(tmp_path)

    assert artifacts[0].exists
    assert artifacts[0].ok


# --------------------------------------------------------------------------
# health
# --------------------------------------------------------------------------

def test_health_list_fetched_no_manifests(tmp_path):
    adapter = SOURCE_ADAPTERS["health"]

    listing = adapter.list_fetched(tmp_path)

    assert listing.present == 0
    assert listing.expected is None


def test_health_list_fetched_counts_completed_batches(tmp_path):
    manifest_dir = tmp_path / "data" / "health" / "raw" / "SIH_RESIDENCE_TOTAL_MUNICIPALITY_YEAR"
    manifest_dir.mkdir(parents=True)
    entries = [
        {"batch_id": "2020", "status": "completed", "raw_path": "x.csv"},
        {"batch_id": "2021", "status": "pending", "raw_path": "y.csv"},
    ]
    with open(manifest_dir / "manifest.jsonl", "w") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")

    adapter = SOURCE_ADAPTERS["health"]
    listing = adapter.list_fetched(tmp_path)

    assert listing.present == 1
    assert listing.expected == 2


def test_health_check_outputs_all_missing(tmp_path):
    adapter = SOURCE_ADAPTERS["health"]

    artifacts = adapter.check_outputs(tmp_path)

    assert len(artifacts) == 5
    assert all(not artifact.exists for artifact in artifacts)


def test_health_check_outputs_flags_negative_metric_value(tmp_path):
    output_dir = tmp_path / "data" / "health" / "processed"
    output_dir.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "municipality_code": ["350001"],
            "year": [2020],
            "metric_name": ["total_requests"],
            "metric_value": [-5.0],
        }
    )
    frame.to_parquet(output_dir / "hospitalizations.parquet", index=False)

    adapter = SOURCE_ADAPTERS["health"]
    artifacts = adapter.check_outputs(tmp_path)

    artifact = next(a for a in artifacts if a.label == "hospitalizations.parquet")
    assert artifact.exists
    assert not artifact.ok


def test_health_check_outputs_flags_negative_birth_outcome_total(tmp_path):
    output_dir = tmp_path / "data" / "health" / "processed"
    output_dir.mkdir(parents=True)
    frame = pd.DataFrame({"mun_id": ["350001"], "year": [2020], "Total": [-1.0]})
    frame.to_parquet(output_dir / "birth_weight.parquet", index=False)

    adapter = SOURCE_ADAPTERS["health"]
    artifacts = adapter.check_outputs(tmp_path)

    artifact = next(a for a in artifacts if a.label == "birth_weight.parquet")
    assert artifact.exists
    assert not artifact.ok


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

ASSEMBLY_CONFIG_YAML = """
datasets:
  - id: sensor_panel
    mode: sensor
    index: [station_code, datetime]
    output_path: data/assembly/sensor_panel.parquet
    sources:
      - name: water_quality
        path: data/sensor_data/processed/aggregate/water_quality_streamflow.parquet
        join_keys: [station_code, datetime]
        variables: [ph, turbidity]
"""


def test_assembly_list_fetched_missing_config(tmp_path):
    adapter = SOURCE_ADAPTERS["assembly"]

    listing = adapter.list_fetched(tmp_path)

    assert listing.present == 0
    assert listing.expected is None


def test_assembly_check_outputs_missing_required_column(tmp_path, monkeypatch):
    config_path = tmp_path / "assembly_datasets.yaml"
    config_path.write_text(ASSEMBLY_CONFIG_YAML)
    monkeypatch.setattr(
        "src.data.assembly.constants.DEFAULT_CONFIG_PATH", str(config_path.relative_to(tmp_path))
    )

    output_dir = tmp_path / "data" / "assembly"
    output_dir.mkdir(parents=True)
    # Missing the declared "turbidity" column.
    frame = pd.DataFrame({"station_code": ["S1"], "datetime": pd.to_datetime(["2020-01-01"]), "ph": [7.0]})
    frame.to_parquet(output_dir / "sensor_panel.parquet", index=False)

    adapter = SOURCE_ADAPTERS["assembly"]
    artifacts = adapter.check_outputs(tmp_path)

    assert len(artifacts) == 1
    assert artifacts[0].exists
    assert not artifacts[0].ok


def test_assembly_check_outputs_passes_when_columns_present(tmp_path, monkeypatch):
    config_path = tmp_path / "assembly_datasets.yaml"
    config_path.write_text(ASSEMBLY_CONFIG_YAML)
    monkeypatch.setattr(
        "src.data.assembly.constants.DEFAULT_CONFIG_PATH", str(config_path.relative_to(tmp_path))
    )

    output_dir = tmp_path / "data" / "assembly"
    output_dir.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "station_code": ["S1"],
            "datetime": pd.to_datetime(["2020-01-01"]),
            "ph": [7.0],
            "turbidity": [2.0],
        }
    )
    frame.to_parquet(output_dir / "sensor_panel.parquet", index=False)

    adapter = SOURCE_ADAPTERS["assembly"]
    artifacts = adapter.check_outputs(tmp_path)

    assert artifacts[0].ok
