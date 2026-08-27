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

    assert len(artifacts) == 4
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
    trenches.to_parquet(river_dir / "river_network_trenches.parquet", index=False)

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
    trenches.to_parquet(river_dir / "river_network_trenches.parquet", index=False)
    drainage = pd.DataFrame(
        {"trench_id": [1, 2], "drainage_area": [10.0, 20.0], "within_brazil": [True, True]}
    )
    drainage.to_parquet(river_dir / "river_network_drainage_areas.parquet", index=False)
    trench_adm2 = pd.DataFrame({"trench_id": [1, 2], "adm2": ["X", "X"]})
    trench_adm2.to_parquet(river_dir / "river_network_trench_adm2_matches.parquet", index=False)
    dominant_systems = pd.DataFrame({"adm2": ["X"], "system_id": [0]})
    dominant_systems.to_parquet(river_dir / "river_network_adm2_dominant_systems.parquet", index=False)

    adapter = SOURCE_ADAPTERS["river_network"]
    artifacts = adapter.check_outputs(tmp_path)

    assert all(artifact.ok for artifact in artifacts)


def test_river_network_check_fetched_missing_raw_hydrography(tmp_path):
    adapter = SOURCE_ADAPTERS["river_network"]

    artifacts = adapter.check_fetched(tmp_path)

    assert len(artifacts) == 1
    assert not artifacts[0].exists


def test_river_network_check_fetched_valid_gpkg(tmp_path):
    import geopandas as gpd
    from shapely.geometry import Point

    from src.data.sources.river_network.constants import DEFAULT_RAW_GPKG_PATH, DEFAULT_RAW_GPKG_TRENCHES_LAYER

    gpkg_path = tmp_path / DEFAULT_RAW_GPKG_PATH
    gpkg_path.parent.mkdir(parents=True, exist_ok=True)
    frame = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs=4326)
    frame.to_file(gpkg_path, layer=DEFAULT_RAW_GPKG_TRENCHES_LAYER, driver="GPKG")

    adapter = SOURCE_ADAPTERS["river_network"]
    artifacts = adapter.check_fetched(tmp_path)

    assert artifacts[0].exists
    assert artifacts[0].ok


# --------------------------------------------------------------------------
# gadm
# --------------------------------------------------------------------------

def test_gadm_list_fetched_absent(tmp_path):
    adapter = SOURCE_ADAPTERS["gadm"]

    listing = adapter.list_fetched(tmp_path)

    assert listing.present == 0
    assert listing.expected == 1


def test_gadm_check_outputs_missing(tmp_path):
    adapter = SOURCE_ADAPTERS["gadm"]

    artifacts = adapter.check_outputs(tmp_path)

    assert len(artifacts) == 1
    assert not artifacts[0].exists


def test_gadm_check_fetched_valid_raw_file_is_ok_but_output_still_missing(tmp_path):
    """check_fetched checks the raw file; check_outputs checks the simplified
    preprocessing output -- gadm's own preprocess step (src/data/sources/gadm)
    is what produces the latter, so a valid raw file alone doesn't satisfy
    check_outputs."""
    import geopandas as gpd
    from shapely.geometry import Point

    from src.data.sources.gadm.constants import DEFAULT_ADM2_LAYER, RAW_GADM_PATH

    gadm_path = tmp_path / RAW_GADM_PATH
    gadm_path.parent.mkdir(parents=True, exist_ok=True)
    frame = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs=4326)
    frame.to_file(gadm_path, layer=DEFAULT_ADM2_LAYER, driver="GPKG")

    adapter = SOURCE_ADAPTERS["gadm"]

    fetched = adapter.check_fetched(tmp_path)
    outputs = adapter.check_outputs(tmp_path)

    assert fetched[0].ok
    assert not outputs[0].exists


def test_gadm_check_outputs_valid_simplified_file_is_ok(tmp_path):
    """check_outputs requires both layers -- ADM_ADM_0 and ADM_ADM_2 -- to be
    present and readable in the simplified output, since gadm preprocess()
    writes both."""
    import geopandas as gpd
    from shapely.geometry import Point

    from src.data.sources.gadm.constants import DEFAULT_ADM0_LAYER, DEFAULT_ADM2_LAYER, DEFAULT_SIMPLIFIED_GADM_PATH

    gadm_path = tmp_path / DEFAULT_SIMPLIFIED_GADM_PATH
    gadm_path.parent.mkdir(parents=True, exist_ok=True)
    frame = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs=4326)
    frame.to_file(gadm_path, layer=DEFAULT_ADM0_LAYER, driver="GPKG")
    frame.to_file(gadm_path, layer=DEFAULT_ADM2_LAYER, driver="GPKG")

    adapter = SOURCE_ADAPTERS["gadm"]

    outputs = adapter.check_outputs(tmp_path)

    assert outputs[0].ok


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

    assert len(artifacts) == 2
    assert all(not artifact.exists for artifact in artifacts)


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


def test_land_cover_check_outputs_tracks_river_aggregated(tmp_path):
    """land_cover_river_aggregated.parquet is a real, always-produced output
    of aggregate_along_rivers() -- it must be tracked like the other
    aggregate-stage outputs, not silently ignored."""
    output_dir = tmp_path / "data" / "land_cover" / "processed" / "aggregate"
    output_dir.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "mun_id": ["350001"],
            "year": [2020],
            "land_cover_class": ["forest"],
            "share": [0.5],
        }
    )
    frame.to_parquet(output_dir / "land_cover_river_aggregated.parquet", index=False)

    adapter = SOURCE_ADAPTERS["land_cover"]
    artifacts = adapter.check_outputs(tmp_path)

    river_aggregated = next(a for a in artifacts if a.label == "land_cover_river_aggregated")
    assert river_aggregated.exists
    assert river_aggregated.ok


def test_land_cover_check_fetched_missing(tmp_path):
    adapter = SOURCE_ADAPTERS["land_cover"]

    artifacts = adapter.check_fetched(tmp_path)

    assert len(artifacts) == 1
    assert not artifacts[0].exists


def test_land_cover_check_fetched_existing_empty_raw_dir_reports_absent(tmp_path):
    """The raw datadir can exist on disk with zero matching tiles inside --
    exists must reflect that no tile was actually found, not merely that the
    directory happens to be present."""
    datadir = tmp_path / "data" / "land_cover" / "raw" / "lc_mapbiomas10_30"
    datadir.mkdir(parents=True)

    adapter = SOURCE_ADAPTERS["land_cover"]
    artifacts = adapter.check_fetched(tmp_path)

    assert len(artifacts) == 1
    assert not artifacts[0].exists


def _write_fake_tile(path, *, valid: bool):
    if not valid:
        path.write_bytes(b"not-a-real-tif")
        return
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    transform = from_origin(-45.0, -10.0, 0.001, 0.001)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=1,
        width=1,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
    ) as dataset:
        dataset.write(np.array([[1]], dtype="uint8"), 1)


def test_land_cover_check_fetched_valid_tile(tmp_path):
    datadir = tmp_path / "data" / "land_cover" / "raw" / "lc_mapbiomas10_30"
    datadir.mkdir(parents=True)
    _write_fake_tile(datadir / "brazil_coverage_2020.tif", valid=True)

    adapter = SOURCE_ADAPTERS["land_cover"]
    artifacts = adapter.check_fetched(tmp_path)

    assert artifacts[0].exists
    assert artifacts[0].ok


def test_land_cover_check_fetched_flags_corrupt_tile(tmp_path):
    datadir = tmp_path / "data" / "land_cover" / "raw" / "lc_mapbiomas10_30"
    datadir.mkdir(parents=True)
    _write_fake_tile(datadir / "brazil_coverage_2020.tif", valid=False)

    adapter = SOURCE_ADAPTERS["land_cover"]
    artifacts = adapter.check_fetched(tmp_path)

    assert artifacts[0].exists
    assert not artifacts[0].ok


def test_land_cover_check_fetched_caches_unchanged_tiles(tmp_path):
    """The (size, mtime_ns)-keyed cache must skip re-checking a tile whose
    file hasn't changed since the last run."""
    from src.data.verification import checks as checks_module

    datadir = tmp_path / "data" / "land_cover" / "raw" / "lc_mapbiomas10_30"
    datadir.mkdir(parents=True)
    _write_fake_tile(datadir / "brazil_coverage_2020.tif", valid=True)

    adapter = SOURCE_ADAPTERS["land_cover"]
    adapter.check_fetched(tmp_path)  # first run populates the cache

    call_count = 0
    original = checks_module.check_raster_header_readable

    def counting_check(path, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(path, **kwargs)

    import src.data.verification.sources as sources_module

    sources_module.check_raster_header_readable = counting_check
    try:
        adapter.check_fetched(tmp_path)  # unchanged file -> cache hit, no re-check
    finally:
        sources_module.check_raster_header_readable = original

    assert call_count == 0


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
    frame.to_parquet(output_dir / "sensor_data_water_quality_streamflow.parquet", index=True)

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
    frame.to_parquet(output_dir / "sensor_data_water_quality_streamflow.parquet", index=False)

    adapter = SOURCE_ADAPTERS["sensor_data"]
    artifacts = adapter.check_outputs(tmp_path)

    assert artifacts[0].exists
    assert not artifacts[0].ok


def test_sensor_data_check_fetched_missing(tmp_path):
    adapter = SOURCE_ADAPTERS["sensor_data"]

    artifacts = adapter.check_fetched(tmp_path)

    assert len(artifacts) == 2
    assert all(not artifact.exists for artifact in artifacts)


def test_sensor_data_check_fetched_db_file_present_without_stations_table_reports_absent(tmp_path):
    """The duckdb FILE can exist (holding some other table) without the
    "stations" table ever having been populated -- exists must reflect
    whether the table is actually there, not merely whether the file is."""
    import pandas as pd
    from src.data.sources.sensor_data.fetch.database import write_dataframe_table

    write_dataframe_table(str(tmp_path), "some_other_table", pd.DataFrame({"x": [1]}))

    adapter = SOURCE_ADAPTERS["sensor_data"]
    artifacts = adapter.check_fetched(tmp_path)

    station_artifact = next(a for a in artifacts if a.label == "station_inventory")
    assert not station_artifact.exists
    assert station_artifact.checks == []


def test_sensor_data_check_fetched_malformed_stations_table_reports_failed_not_crashed(tmp_path):
    """A `stations` table that exists but doesn't match what
    read_geodataframe_table's stored metadata expects (e.g. left over from an
    interrupted or pre-metadata-tracking write, missing the geometry column
    entirely) must surface as a failed check, not propagate the KeyError and
    crash verification -- see the module docstring's "must degrade
    gracefully... never raised" contract."""
    import pandas as pd
    from src.data.sources.sensor_data.fetch.database import STATIONS_TABLE, write_dataframe_table

    write_dataframe_table(str(tmp_path), STATIONS_TABLE, pd.DataFrame({"station_code": ["1"]}))

    adapter = SOURCE_ADAPTERS["sensor_data"]
    artifacts = adapter.check_fetched(tmp_path)

    station_artifact = next(a for a in artifacts if a.label == "station_inventory")
    assert station_artifact.exists
    assert not station_artifact.ok


def test_sensor_data_check_fetched_empty_raw_dir_reports_absent(tmp_path):
    """archives_dir can exist on disk (created but holding zero .zip archives) --
    exists must reflect that no archive was actually found."""
    from src.data.sources.sensor_data.constants import get_archives_dir

    archives_dir = get_archives_dir(tmp_path)
    archives_dir.mkdir(parents=True, exist_ok=True)
    (archives_dir / "not-an-archive.txt").write_text("stray file")

    adapter = SOURCE_ADAPTERS["sensor_data"]
    artifacts = adapter.check_fetched(tmp_path)

    archive_artifact = next(a for a in artifacts if a.label == "raw_archives")
    assert not archive_artifact.exists
    assert archive_artifact.checks == []


def test_sensor_data_check_fetched_valid_station_inventory_and_archive(tmp_path):
    import zipfile

    import geopandas as gpd
    from src.data.sources.sensor_data.constants import get_archives_dir
    from src.data.sources.sensor_data.fetch.database import STATIONS_TABLE, write_geodataframe_table

    stations = pd.DataFrame({"station_code": ["11111111"]})
    stations_geo = gpd.GeoDataFrame(
        stations, geometry=gpd.points_from_xy([-45.0], [-10.0]), crs=4326
    )
    write_geodataframe_table(tmp_path, STATIONS_TABLE, stations_geo)

    archives_dir = get_archives_dir(tmp_path)
    archives_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archives_dir / "11111111.zip", "w") as archive:
        archive.writestr("data.txt", "hello")

    adapter = SOURCE_ADAPTERS["sensor_data"]
    artifacts = adapter.check_fetched(tmp_path)

    assert all(artifact.exists for artifact in artifacts)
    assert all(artifact.ok for artifact in artifacts)


def test_sensor_data_check_fetched_flags_corrupt_archive(tmp_path):
    from src.data.sources.sensor_data.constants import get_archives_dir

    archives_dir = get_archives_dir(tmp_path)
    archives_dir.mkdir(parents=True, exist_ok=True)
    (archives_dir / "corrupt.zip").write_bytes(b"not-a-zip")

    adapter = SOURCE_ADAPTERS["sensor_data"]
    artifacts = adapter.check_fetched(tmp_path)

    archive_artifact = next(a for a in artifacts if a.label == "raw_archives")
    assert archive_artifact.exists
    assert not archive_artifact.ok


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


def test_climate_check_fetched_missing_store(tmp_path):
    adapter = SOURCE_ADAPTERS["climate"]

    artifacts = adapter.check_fetched(tmp_path)

    assert len(artifacts) == 1
    assert not artifacts[0].exists


def test_climate_check_fetched_flags_out_of_range_value(tmp_path):
    import numpy as np
    import xarray as xr

    from src.data.sources.climate.constants import DEFAULT_ERA5_LAND_STORE_PATH

    store_path = tmp_path / DEFAULT_ERA5_LAND_STORE_PATH
    time = pd.date_range("2020-01-01", periods=3, freq="D")
    dataset = xr.Dataset(
        {"2t": (("time", "latitude", "longitude"), np.full((3, 1, 1), 50.0))},  # far outside [180, 340] K
        coords={"time": time, "latitude": [-10.0], "longitude": [-45.0]},
    )
    dataset.to_zarr(store_path, mode="w", consolidated=False)

    adapter = SOURCE_ADAPTERS["climate"]
    artifacts = adapter.check_fetched(tmp_path)

    assert artifacts[0].exists
    assert not artifacts[0].ok


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
    frame.to_parquet(output_dir / "biomes_adm2.parquet", index=False)

    adapter = SOURCE_ADAPTERS["biomes"]
    artifacts = adapter.check_outputs(tmp_path)

    adm2_artifact = next(a for a in artifacts if a.label == "biome_adm2")
    assert adm2_artifact.exists
    assert not adm2_artifact.ok


def test_biomes_check_fetched_missing(tmp_path):
    adapter = SOURCE_ADAPTERS["biomes"]

    artifacts = adapter.check_fetched(tmp_path)

    assert len(artifacts) == 1
    assert not artifacts[0].exists


def test_biomes_check_fetched_valid_archive(tmp_path):
    import zipfile

    from src.data.sources.biomes.constants import archive_path

    path = archive_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("biomes.shp", "fake-shapefile-bytes" * 100)

    adapter = SOURCE_ADAPTERS["biomes"]
    artifacts = adapter.check_fetched(tmp_path)

    assert artifacts[0].exists
    assert artifacts[0].ok


def test_biomes_check_fetched_flags_corrupt_archive(tmp_path):
    from src.data.sources.biomes.constants import archive_path

    path = archive_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not-a-zip" * 200)

    adapter = SOURCE_ADAPTERS["biomes"]
    artifacts = adapter.check_fetched(tmp_path)

    assert artifacts[0].exists
    assert not artifacts[0].ok


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


def test_population_check_fetched_missing(tmp_path):
    adapter = SOURCE_ADAPTERS["population"]

    artifacts = adapter.check_fetched(tmp_path)

    assert len(artifacts) == 1
    assert not artifacts[0].exists


def test_population_check_fetched_valid_raw_columns(tmp_path):
    """The raw parquet keeps the BigQuery query's original (Portuguese)
    column names -- these are renamed only during preprocessing, so
    check_fetched must check the raw names, not the final output's."""
    from src.data.sources.population.constants import raw_dir as _population_raw_dir

    raw_dir = _population_raw_dir(tmp_path)
    raw_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "ano": [2020],
            "id_municipio": ["350001"],
            "id_municipio_nome": ["Sao Paulo"],
            "sexo": ["M"],
            "grupo_idade": ["0-4"],
            "populacao": [100],
        }
    )
    frame.to_parquet(raw_dir / "population_raw.parquet", index=False)

    adapter = SOURCE_ADAPTERS["population"]
    artifacts = adapter.check_fetched(tmp_path)

    assert artifacts[0].exists
    assert artifacts[0].ok


def test_population_check_fetched_flags_negative_populacao(tmp_path):
    from src.data.sources.population.constants import raw_dir as _population_raw_dir

    raw_dir = _population_raw_dir(tmp_path)
    raw_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "ano": [2020],
            "id_municipio": ["350001"],
            "id_municipio_nome": ["Sao Paulo"],
            "sexo": ["M"],
            "grupo_idade": ["0-4"],
            "populacao": [-1],
        }
    )
    frame.to_parquet(raw_dir / "population_raw.parquet", index=False)

    adapter = SOURCE_ADAPTERS["population"]
    artifacts = adapter.check_fetched(tmp_path)

    assert artifacts[0].exists
    assert not artifacts[0].ok


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
    frame.to_parquet(output_dir / "health_hospitalizations.parquet", index=False)

    adapter = SOURCE_ADAPTERS["health"]
    artifacts = adapter.check_outputs(tmp_path)

    artifact = next(a for a in artifacts if a.label == "health_hospitalizations.parquet")
    assert artifact.exists
    assert not artifact.ok


def test_health_check_outputs_flags_negative_birth_outcome_total(tmp_path):
    output_dir = tmp_path / "data" / "health" / "processed"
    output_dir.mkdir(parents=True)
    frame = pd.DataFrame({"mun_id": ["350001"], "year": [2020], "Total": [-1.0]})
    frame.to_parquet(output_dir / "health_birth_weight.parquet", index=False)

    adapter = SOURCE_ADAPTERS["health"]
    artifacts = adapter.check_outputs(tmp_path)

    artifact = next(a for a in artifacts if a.label == "health_birth_weight.parquet")
    assert artifact.exists
    assert not artifact.ok


def test_health_check_fetched_no_completed_batches(tmp_path):
    adapter = SOURCE_ADAPTERS["health"]

    artifacts = adapter.check_fetched(tmp_path)

    assert len(artifacts) == 3
    assert all(not artifact.exists for artifact in artifacts)


def test_health_check_fetched_existing_raw_dir_with_no_completed_batch_reports_absent(tmp_path):
    """The raw batch DIRECTORY can exist (e.g. holding only pending/failed
    attempts) even when there's no genuinely completed, on-disk batch --
    `exists` must reflect that there's no usable data, not merely that the
    directory happens to be present on disk (which would otherwise report
    exists=True with zero checks, unfalsifiable and indistinguishable from a
    real "verified" result)."""
    table_name = "SIH_RESIDENCE_TOTAL_MUNICIPALITY_YEAR"
    manifest_dir = tmp_path / "data" / "health" / "raw" / table_name
    manifest_dir.mkdir(parents=True)
    entries = [{"batch_id": "2020", "status": "pending", "raw_path": str(manifest_dir / "batch_2020.csv")}]
    with open(manifest_dir / "manifest.jsonl", "w") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")

    adapter = SOURCE_ADAPTERS["health"]
    artifacts = adapter.check_fetched(tmp_path)

    artifact = next(a for a in artifacts if a.label == f"health_batches:{table_name}")
    assert not artifact.exists
    assert artifact.checks == []


def _write_health_batch(tmp_path, table_name, *, valid: bool):
    manifest_dir = tmp_path / "data" / "health" / "raw" / table_name
    manifest_dir.mkdir(parents=True, exist_ok=True)
    csv_path = manifest_dir / "batch_2020.csv"
    if valid:
        csv_path.write_text(
            '"Município";"Valor"\n"110001 Alta Floresta D\'Oeste";"10"\n',
            encoding="latin1",
        )
    else:
        csv_path.write_text("not,a,datasus,csv\n", encoding="latin1")
    entries = [{"batch_id": "2020", "status": "completed", "raw_path": str(csv_path)}]
    with open(manifest_dir / "manifest.jsonl", "w") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")


def test_health_check_fetched_valid_csv_sample(tmp_path):
    table_name = "SIH_RESIDENCE_TOTAL_MUNICIPALITY_YEAR"
    _write_health_batch(tmp_path, table_name, valid=True)

    adapter = SOURCE_ADAPTERS["health"]
    artifacts = adapter.check_fetched(tmp_path)

    artifact = next(a for a in artifacts if a.label == f"health_batches:{table_name}")
    assert artifact.exists
    assert artifact.ok


def test_health_check_fetched_flags_unparseable_csv(tmp_path):
    table_name = "SIH_RESIDENCE_TOTAL_MUNICIPALITY_YEAR"
    _write_health_batch(tmp_path, table_name, valid=False)

    adapter = SOURCE_ADAPTERS["health"]
    artifacts = adapter.check_fetched(tmp_path)

    artifact = next(a for a in artifacts if a.label == f"health_batches:{table_name}")
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
        path: data/sensor_data/processed/aggregate/sensor_data_water_quality_streamflow.parquet
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


def test_assembly_check_fetched_is_a_noop(tmp_path):
    """assembly isn't a fetch source -- it joins the other 7 -- so it uses
    SourceAdapter's default no-op check_fetched rather than a real
    implementation."""
    adapter = SOURCE_ADAPTERS["assembly"]

    assert adapter.check_fetched(tmp_path) == []
