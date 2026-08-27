"""Per-source adapter registry for the verification module.

Each of the 8 pipeline sources gets a `SourceAdapter` bundling:
  - `list_fetched(root_dir, force=False) -> FetchListing`: how much of the
    raw/fetched input is present locally vs. expected.
  - `check_outputs(root_dir) -> list[OutputArtifactCheck]`: sanity checks
    against that source's preprocessed output artifact(s).
  - `fingerprint_paths(root_dir) -> list[Path]`: the paths whose
    (size, mtime) should drive the verification cache's fingerprint.

Every adapter must degrade gracefully when data is absent locally (common for
sources that only run on HPC): missing files/dirs are reported via
`FetchListing`/`OutputArtifactCheck.exists=False`, never raised.

Discovery logic is reused from each source's own module wherever it already
exists (see the plan this module implements) rather than reimplemented here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from .checks import (
    CheckResult,
    check_file_nonempty,
    check_gpkg_layer_readable,
    check_null_fraction,
    check_raster_header_readable,
    check_required_columns,
    check_sampled_files,
    check_value_range,
    check_zip_integrity,
)


@dataclass
class FetchListing:
    present: int
    expected: int | None
    detail: str = ""


@dataclass
class OutputArtifactCheck:
    label: str
    path: Path
    exists: bool
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.exists and all(check.ok for check in self.checks)


def _default_check_fetched(root_dir) -> list[OutputArtifactCheck]:
    """Default `check_fetched`: no raw-artifact content checks for this source.

    Used by sources with no separate raw-fetched-artifact concept (assembly,
    which joins the other 7 sources rather than fetching anything itself).
    `core.py` treats this default specially (identity check) to report
    `fetch_status="not_applicable"` rather than "outstanding".
    """
    return []


@dataclass
class SourceAdapter:
    name: str
    list_fetched: Callable[..., FetchListing]
    check_outputs: Callable[..., list[OutputArtifactCheck]]
    fingerprint_paths: Callable[..., list[Path]]
    # Short, human-readable description of how this source's raw data is
    # actually obtained -- shown as-is in the summary table, not derived from
    # anything computed at runtime.
    fetch_method: str = ""
    # Content-level checks against raw fetched artifacts (as opposed to
    # `check_outputs`, which checks preprocessed/assembled outputs). Optional:
    # defaults to a no-op for sources with nothing raw to check.
    check_fetched: Callable[..., list[OutputArtifactCheck]] = _default_check_fetched


def _safe_read_parquet(path: Path):
    """Read a parquet file, returning None if it's missing or unreadable.

    Some outputs (e.g. sensor_data's assembled panel) are written with a
    meaningful named index (station_code/datetime) rather than a plain
    RangeIndex. `check_required_columns` only looks at `frame.columns`, so a
    named index level would be invisible to it and falsely reported as
    missing -- reset it back into columns here. A plain positional index
    (`names == [None]`) is left alone so `reset_index()` doesn't inject a
    spurious `index` column for sources that don't use a custom index.
    """
    if not path.exists():
        return None
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return None
    if list(frame.index.names) != [None]:
        frame = frame.reset_index()
    return frame


def _non_empty_check(frame, name: str = "non_empty") -> CheckResult:
    ok = not frame.empty
    return CheckResult(name=name, ok=ok, message=f"{len(frame)} rows." if ok else "Frame is empty.")


def _uniqueness_check(frame, column: str, name: str | None = None) -> CheckResult:
    name = name or f"unique:{column}"
    if column not in frame.columns:
        return CheckResult(name=name, ok=False, message=f"Column '{column}' not present.")
    duplicate_count = int(frame[column].duplicated().sum())
    ok = duplicate_count == 0
    message = f"All '{column}' values unique." if ok else f"{duplicate_count} duplicate '{column}' values."
    return CheckResult(name=name, ok=ok, message=message)


def _missing_artifact(label: str, path: Path) -> OutputArtifactCheck:
    return OutputArtifactCheck(label=label, path=path, exists=path.exists(), checks=[])


def _absent_artifact(label: str, path: Path) -> OutputArtifactCheck:
    """Like `_missing_artifact`, but for a caller that already determined the
    artifact is absent via domain logic (an empty listing, a missing table)
    rather than raw path existence. `path` here is often a *container*
    (a directory, or a database file holding other tables) that can exist on
    disk while still holding none of the actual content being checked for --
    e.g. an empty raw directory, or a duckdb file present but missing the
    expected table. Using `_missing_artifact`'s own `path.exists()` in that
    situation would silently report `exists=True` with zero checks, which
    can never fail a check and ends up masquerading as fully verified.
    """
    return OutputArtifactCheck(label=label, path=path, exists=False, checks=[])


def _artifact_check(
    label: str,
    path: Path,
    build_checks: Callable[[pd.DataFrame], list[CheckResult]],
) -> OutputArtifactCheck:
    """Load one parquet artifact and build its checks, or report it missing.

    Centralizes the "load parquet -> missing -> build checks" pattern that
    used to be repeated by hand in every adapter's `check_outputs()`: reads
    `path` via `_safe_read_parquet`, returns a `_missing_artifact` result if
    it's absent or unreadable, otherwise hands the loaded frame to
    `build_checks` and wraps the result. A fix to this "load -> missing ->
    build checks" handling now only needs to change here instead of being
    re-verified in every adapter.
    """
    frame = _safe_read_parquet(path)
    if frame is None:
        return _missing_artifact(label, path)
    return OutputArtifactCheck(label=label, path=path, exists=True, checks=build_checks(frame))


# --------------------------------------------------------------------------
# gadm
# --------------------------------------------------------------------------
#
# The shared Brazil ADM boundary geopackage: not owned by any single source,
# but read (via its simplified output) by river_network (country/ADM2
# annotation), sensor_data (filtering stations to within Brazil), and biomes
# (ADM2 mapping). Modeled as its own pseudo-source (like `assembly`) rather
# than folded into river_network's checks, which is what this repo did
# before -- and which meant river_network's own "fetched" status was
# actually reporting on GADM, never on its own raw hydrography input (see
# river_network below).
#
# Unlike before, gadm now has a real preprocessing step
# (`src/data/sources/gadm`, geometry simplification) with a genuine output
# distinct from the raw input, so `check_fetched` and `check_outputs` check
# two different files.


def _gadm_check_fetched(root_dir) -> list[OutputArtifactCheck]:
    from src.data.sources.gadm.constants import DEFAULT_ADM2_LAYER, RAW_GADM_PATH

    gadm_path = Path(root_dir) / RAW_GADM_PATH
    if not gadm_path.exists():
        return [_missing_artifact("gadm_boundaries_raw", gadm_path)]
    checks = [
        check_file_nonempty(gadm_path, min_size_bytes=1024),
        check_gpkg_layer_readable(gadm_path, DEFAULT_ADM2_LAYER),
    ]
    return [OutputArtifactCheck(label="gadm_boundaries_raw", path=gadm_path, exists=True, checks=checks)]


def _gadm_check_outputs(root_dir) -> list[OutputArtifactCheck]:
    from src.data.sources.gadm.constants import DEFAULT_ADM0_LAYER, DEFAULT_ADM2_LAYER, DEFAULT_SIMPLIFIED_GADM_PATH

    gadm_path = Path(root_dir) / DEFAULT_SIMPLIFIED_GADM_PATH
    if not gadm_path.exists():
        return [_missing_artifact("gadm_boundaries_simplified", gadm_path)]
    checks = [
        check_file_nonempty(gadm_path, min_size_bytes=1024),
        check_gpkg_layer_readable(gadm_path, DEFAULT_ADM0_LAYER),
        check_gpkg_layer_readable(gadm_path, DEFAULT_ADM2_LAYER),
    ]
    return [OutputArtifactCheck(label="gadm_boundaries_simplified", path=gadm_path, exists=True, checks=checks)]


def _gadm_list_fetched(root_dir, force: bool = False) -> FetchListing:
    from src.data.sources.gadm.constants import RAW_GADM_PATH

    gadm_path = Path(root_dir) / RAW_GADM_PATH
    present = 1 if gadm_path.exists() else 0
    return FetchListing(
        present=present,
        expected=1,
        detail=f"GADM boundary file {'present' if present else 'missing'} at {gadm_path}.",
    )


def _gadm_fingerprint_paths(root_dir) -> list[Path]:
    from src.data.sources.gadm.constants import DEFAULT_SIMPLIFIED_GADM_PATH, RAW_GADM_PATH

    return [Path(root_dir) / RAW_GADM_PATH, Path(root_dir) / DEFAULT_SIMPLIFIED_GADM_PATH]


# --------------------------------------------------------------------------
# river_network
# --------------------------------------------------------------------------

def _river_network_list_fetched(root_dir, force: bool = False) -> FetchListing:
    from src.data.sources.river_network.constants import DEFAULT_RAW_GPKG_PATH

    gpkg_path = Path(root_dir) / DEFAULT_RAW_GPKG_PATH
    present = 1 if gpkg_path.exists() else 0
    return FetchListing(
        present=present,
        expected=1,
        detail=f"Raw hydrography geopackage {'present' if present else 'missing'} at {gpkg_path}.",
    )


def _river_network_check_outputs(root_dir) -> list[OutputArtifactCheck]:
    from src.data.sources.river_network.constants import (
        DRAINAGE_AREAS_FILENAME,
        PROCESSED_DIR,
        TRENCHES_FILENAME,
        TRENCH_ID_COLUMN,
    )

    river_dir = Path(root_dir) / PROCESSED_DIR
    return [
        _artifact_check(
            "river_trenches",
            river_dir / TRENCHES_FILENAME,
            lambda frame: [
                check_required_columns(
                    frame,
                    ["trench_id", "upstream_node", "downstream_node", "distance", "system_id"],
                ),
                _non_empty_check(frame),
                _uniqueness_check(frame, TRENCH_ID_COLUMN),
            ],
        ),
        _artifact_check(
            "drainage_areas",
            river_dir / DRAINAGE_AREAS_FILENAME,
            lambda frame: [
                check_required_columns(frame, ["trench_id", "drainage_area", "within_brazil"]),
                _non_empty_check(frame),
            ],
        ),
    ]


def _river_network_check_fetched(root_dir) -> list[OutputArtifactCheck]:
    from src.data.sources.river_network.constants import DEFAULT_RAW_GPKG_PATH, DEFAULT_RAW_GPKG_TRENCHES_LAYER

    gpkg_path = Path(root_dir) / DEFAULT_RAW_GPKG_PATH
    if not gpkg_path.exists():
        return [_missing_artifact("raw_hydrography", gpkg_path)]
    checks = [
        check_file_nonempty(gpkg_path, min_size_bytes=1024),
        check_gpkg_layer_readable(gpkg_path, DEFAULT_RAW_GPKG_TRENCHES_LAYER),
    ]
    return [OutputArtifactCheck(label="raw_hydrography", path=gpkg_path, exists=True, checks=checks)]


def _river_network_fingerprint_paths(root_dir) -> list[Path]:
    from src.data.sources.river_network.constants import (
        DEFAULT_RAW_GPKG_PATH,
        DRAINAGE_AREAS_FILENAME,
        PROCESSED_DIR,
        TRENCHES_FILENAME,
    )

    river_dir = Path(root_dir) / PROCESSED_DIR
    return [
        Path(root_dir) / DEFAULT_RAW_GPKG_PATH,
        river_dir / TRENCHES_FILENAME,
        river_dir / DRAINAGE_AREAS_FILENAME,
    ]


# --------------------------------------------------------------------------
# land_cover
# --------------------------------------------------------------------------

# MapBiomas collection 10 (MAPBIOMAS_COLLECTION in land_cover/constants.py)
# covers 1985-2024; no such range constant exists in that module today, so
# it's declared here as the deterministic expected-years list.
_LAND_COVER_EXPECTED_START_YEAR = 1985
_LAND_COVER_EXPECTED_END_YEAR = 2024


def _land_cover_list_fetched(root_dir, force: bool = False) -> FetchListing:
    from src.data.sources.land_cover.constants import build_paths
    from src.data.sources.land_cover.preprocess import get_files

    expected_years = set(range(_LAND_COVER_EXPECTED_START_YEAR, _LAND_COVER_EXPECTED_END_YEAR + 1))
    paths = build_paths(root_dir)
    try:
        files = get_files(paths.datadir)
    except (FileNotFoundError, OSError) as exc:
        return FetchListing(
            present=0,
            expected=len(expected_years),
            detail=f"Land-cover raw directory unavailable: {exc}",
        )

    years = set(int(year) for year in files.index.unique().tolist())
    present_years = years & expected_years
    missing_years = sorted(expected_years - present_years)
    detail = (
        f"{len(present_years)}/{len(expected_years)} years present "
        f"({_LAND_COVER_EXPECTED_START_YEAR}-{_LAND_COVER_EXPECTED_END_YEAR})."
    )
    if missing_years:
        detail += f" Missing: {missing_years}."
    return FetchListing(present=len(present_years), expected=len(expected_years), detail=detail)


def _land_cover_check_outputs(root_dir) -> list[OutputArtifactCheck]:
    from src.data.sources.land_cover.constants import (
        BUCKET_SHARE_COLUMN,
        DEFAULT_SENSOR_UPSTREAM_OUTPUT_PATH,
    )

    def build_checks(frame):
        checks = [
            check_required_columns(frame, ["station_code", "year"]),
            _non_empty_check(frame),
        ]
        # Long-format table: land_cover_class is a row value, and the fraction is
        # a single "share" column (not per-class "{class}_shr" columns), so one
        # range check covers every class's rows.
        if BUCKET_SHARE_COLUMN in frame.columns:
            checks.append(
                check_value_range(
                    frame, BUCKET_SHARE_COLUMN, lo=-1e-6, hi=1.0 + 1e-6, name=f"value_range:{BUCKET_SHARE_COLUMN}"
                )
            )
        return checks

    path = Path(root_dir) / DEFAULT_SENSOR_UPSTREAM_OUTPUT_PATH
    return [_artifact_check("land_cover_sensor_upstream", path, build_checks)]


def _land_cover_check_fetched(root_dir) -> list[OutputArtifactCheck]:
    from src.data.sources.land_cover.constants import build_paths
    from src.data.sources.land_cover.preprocess import get_files

    paths = build_paths(root_dir)
    try:
        files = get_files(paths.datadir)
    except (FileNotFoundError, OSError) as exc:
        return [
            OutputArtifactCheck(
                label="mapbiomas_tiles",
                path=paths.datadir,
                exists=False,
                checks=[CheckResult(name="raster_header_sample", ok=False, message=str(exc))],
            )
        ]
    if files.empty:
        return [_absent_artifact("mapbiomas_tiles", paths.datadir)]

    check = check_sampled_files(
        files.tolist(),
        check_fn=check_raster_header_readable,
        cache_path=paths.datadir / ".raster_verification_cache.json",
        sample_limit=5,
        name="raster_header_sample",
    )
    return [OutputArtifactCheck(label="mapbiomas_tiles", path=paths.datadir, exists=True, checks=[check])]


def _land_cover_fingerprint_paths(root_dir) -> list[Path]:
    from src.data.sources.land_cover.constants import DEFAULT_SENSOR_UPSTREAM_OUTPUT_PATH, build_paths

    paths = build_paths(root_dir)
    try:
        raw_files = sorted(paths.datadir.glob("*.tif"))
    except OSError:
        raw_files = []
    return [*raw_files, Path(root_dir) / DEFAULT_SENSOR_UPSTREAM_OUTPUT_PATH]


# --------------------------------------------------------------------------
# sensor_data
# --------------------------------------------------------------------------

def _sensor_data_list_fetched(root_dir, force: bool = False) -> FetchListing:
    from src.data.sources.sensor_data.constants import get_download_log_database_path

    db_path = get_download_log_database_path(root_dir)
    present = 1 if db_path.exists() else 0
    return FetchListing(
        present=present,
        expected=1,
        detail=f"sensor_downloads.duckdb {'present' if present else 'missing'} at {db_path}.",
    )


def _sensor_data_check_outputs(root_dir) -> list[OutputArtifactCheck]:
    from src.data.sources.sensor_data.constants import get_processed_dir
    from src.data.sources.sensor_data.preprocess.assembly import STREAMFLOW_DAY_COLUMN
    from src.data.sources.sensor_data.schema import (
        ASSEMBLED_SENSOR_DATA_PARQUET,
        STREAMFLOW_MAX_VALID_DISCHARGE,
    )

    def build_checks(frame):
        checks = [
            check_required_columns(frame, ["station_code", "datetime"]),
            _non_empty_check(frame),
        ]
        # The final assembled table renames "discharge" to "streamflow_discharge_day"
        # (plus rolling-mean columns); the plain "discharge" column never survives
        # to this output.
        if STREAMFLOW_DAY_COLUMN in frame.columns:
            checks.append(
                check_value_range(
                    frame,
                    STREAMFLOW_DAY_COLUMN,
                    lo=0.0,
                    hi=STREAMFLOW_MAX_VALID_DISCHARGE,
                    name=f"value_range:{STREAMFLOW_DAY_COLUMN}",
                )
            )
        return checks

    path = get_processed_dir(root_dir, stage="aggregate") / ASSEMBLED_SENSOR_DATA_PARQUET
    return [_artifact_check("water_quality_streamflow", path, build_checks)]


def _sensor_data_check_fetched(root_dir) -> list[OutputArtifactCheck]:
    from src.data.sources.sensor_data.constants import get_raw_dir, get_sensor_database_path
    from src.data.sources.sensor_data.fetch.data.download import (
        _cached_is_parseable_zip,
        _load_zip_verification_cache,
        _save_zip_verification_cache,
    )
    from src.data.sources.sensor_data.fetch.database import STATIONS_TABLE, read_geodataframe_table, table_exists

    artifacts: list[OutputArtifactCheck] = []

    db_path = get_sensor_database_path(root_dir)
    if not table_exists(root_dir, STATIONS_TABLE):
        artifacts.append(_absent_artifact("station_inventory", db_path))
    else:
        stations = read_geodataframe_table(root_dir, STATIONS_TABLE)
        checks = [_non_empty_check(stations, name="non_empty")]
        if "geometry" in stations.columns:
            null_fraction = float(stations.geometry.isna().mean()) if not stations.empty else 1.0
            checks.append(
                CheckResult(
                    name="null_fraction:geometry",
                    ok=null_fraction == 0.0,
                    message=f"{null_fraction:.2%} null geometries.",
                )
            )
        artifacts.append(OutputArtifactCheck(label="station_inventory", path=db_path, exists=True, checks=checks))

    raw_dir = get_raw_dir(root_dir)
    zip_paths = sorted(raw_dir.glob("*.zip")) if raw_dir.exists() else []
    if not zip_paths:
        artifacts.append(_absent_artifact("raw_archives", raw_dir))
    else:
        cache = _load_zip_verification_cache(raw_dir)
        cache_dirty = False
        corrupt: list[str] = []
        for path in zip_paths:
            ok, updated = _cached_is_parseable_zip(path, path.stat(), cache)
            cache_dirty = cache_dirty or updated
            if not ok:
                corrupt.append(path.name)
        if cache_dirty:
            _save_zip_verification_cache(raw_dir, cache)
        ok = not corrupt
        message = f"{len(zip_paths) - len(corrupt)}/{len(zip_paths)} archives parseable."
        if corrupt:
            message += f" Corrupt: {corrupt[:10]}."
        artifacts.append(
            OutputArtifactCheck(
                label="raw_archives",
                path=raw_dir,
                exists=True,
                checks=[CheckResult(name="zip_integrity", ok=ok, message=message)],
            )
        )
    return artifacts


def _sensor_data_fingerprint_paths(root_dir) -> list[Path]:
    from src.data.sources.sensor_data.constants import (
        get_download_log_database_path,
        get_processed_dir,
        get_sensor_database_path,
    )
    from src.data.sources.sensor_data.schema import ASSEMBLED_SENSOR_DATA_PARQUET

    return [
        get_sensor_database_path(root_dir),
        get_download_log_database_path(root_dir),
        get_processed_dir(root_dir, stage="aggregate") / ASSEMBLED_SENSOR_DATA_PARQUET,
    ]


# --------------------------------------------------------------------------
# climate
# --------------------------------------------------------------------------

def _climate_list_fetched(root_dir, force: bool = False) -> FetchListing:
    # era5_land_hourly, era5_land_daily, and era5_land_arco all write into
    # the *same* shared zarr store (DEFAULT_ERA5_LAND_STORE_PATH) rather than
    # producing separately-countable per-variant artifacts, and successfully
    # preprocessed raw GRIB files are deleted once consumed -- so completeness
    # is judged from the store itself (which expected variable arrays it
    # contains), not by counting raw GRIB/manifest files per variant.
    from src.data.sources.climate.constants import DEFAULT_ERA5_LAND_STORE_PATH
    from src.data.sources.climate.preprocess.era5_land import ERA5L_VAR_CONFIG

    store_path = Path(root_dir) / DEFAULT_ERA5_LAND_STORE_PATH

    expected_variables = set(ERA5L_VAR_CONFIG)
    for var_name, cfg in ERA5L_VAR_CONFIG.items():
        for extra_suffix in cfg.get("aggregation", {}).get("extras", {}):
            expected_variables.add(f"{var_name}_{extra_suffix}")
    expected = len(expected_variables)

    if not store_path.exists():
        return FetchListing(present=0, expected=expected, detail=f"ERA5-Land zarr store missing at {store_path}.")

    try:
        present_variables = {entry.name for entry in store_path.iterdir() if entry.is_dir()} & expected_variables
    except OSError as exc:
        return FetchListing(present=0, expected=expected, detail=f"Could not list ERA5-Land zarr store: {exc}")

    present = len(present_variables)
    return FetchListing(
        present=present,
        expected=expected,
        detail=(
            f"{present}/{expected} ERA5-Land variables present in the shared zarr store "
            f"at {store_path}."
        ),
    )


def _climate_check_outputs(root_dir) -> list[OutputArtifactCheck]:
    from src.data.sources.climate.constants import (
        CLIMATE_VARIABLE_COLUMN,
        DEFAULT_ADM2_UPSTREAM_YEARLY_OUTPUT_PATH,
        DEFAULT_SENSOR_UPSTREAM_OUTPUT_PATH,
    )
    from src.data.sources.climate.fetch.verify import ERA5L_VALUE_RANGES

    def build_checks(frame, value_column):
        checks = [_non_empty_check(frame)]
        if CLIMATE_VARIABLE_COLUMN not in frame.columns:
            checks.append(
                CheckResult(
                    name="value_range",
                    ok=False,
                    message=f"Column '{CLIMATE_VARIABLE_COLUMN}' not present.",
                )
            )
            return checks
        for variable, (lo, hi) in ERA5L_VALUE_RANGES.items():
            subset = frame.loc[frame[CLIMATE_VARIABLE_COLUMN] == variable]
            name = f"value_range:{variable}:{value_column}"
            if subset.empty:
                checks.append(CheckResult(name=name, ok=False, message=f"No rows for variable '{variable}'."))
                continue
            checks.append(check_value_range(subset, value_column, lo=lo, hi=hi, name=name))
        return checks

    # Both output tables are long-format: the variable code lives as a row
    # value in `climate_variable`, not baked into the column name, so each
    # variable's range must be checked against a filtered slice, not a
    # column-name prefix match.
    return [
        _artifact_check(
            label,
            Path(root_dir) / rel_path,
            lambda frame, value_column=value_column: build_checks(frame, value_column),
        )
        for label, rel_path, value_column in (
            ("climate_sensor_upstream", DEFAULT_SENSOR_UPSTREAM_OUTPUT_PATH, "mean_day"),
            ("climate_adm2_upstream_yearly", DEFAULT_ADM2_UPSTREAM_YEARLY_OUTPUT_PATH, "mean_value"),
        )
    ]


# How many of the most recent time steps to inspect in the shared zarr store:
# `xr.open_zarr` is lazy, so an unbounded null-fraction/min/max reduction would
# scan the entire multi-year, multi-decade history on every summary run.
_CLIMATE_RAW_SAMPLE_TIME_STEPS = 30


def _climate_check_fetched(root_dir) -> list[OutputArtifactCheck]:
    import xarray as xr

    from src.data.sources.climate.constants import DEFAULT_ERA5_LAND_STORE_PATH
    from src.data.sources.climate.fetch.verify import ERA5L_VALUE_RANGES, MAX_NULL_FRACTION

    store_path = Path(root_dir) / DEFAULT_ERA5_LAND_STORE_PATH
    if not store_path.exists():
        return [_missing_artifact("era5_land_store", store_path)]

    try:
        dataset = xr.open_zarr(store_path, consolidated=False)
    except Exception as exc:
        return [
            OutputArtifactCheck(
                label="era5_land_store",
                path=store_path,
                exists=True,
                checks=[CheckResult(name="open_store", ok=False, message=str(exc))],
            )
        ]

    checks: list[CheckResult] = []
    try:
        sample = (
            dataset.isel(time=slice(-_CLIMATE_RAW_SAMPLE_TIME_STEPS, None)) if "time" in dataset.dims else dataset
        )
        for variable, (lo, hi) in ERA5L_VALUE_RANGES.items():
            if variable not in sample.data_vars:
                checks.append(
                    CheckResult(name=f"value_range:{variable}", ok=False, message=f"'{variable}' not present in store.")
                )
                continue
            data_array = sample[variable]
            null_fraction = float(data_array.isnull().mean())
            if null_fraction > MAX_NULL_FRACTION:
                checks.append(
                    CheckResult(
                        name=f"null_fraction:{variable}",
                        ok=False,
                        message=f"{null_fraction:.2%} null (max {MAX_NULL_FRACTION:.2%}).",
                    )
                )
                continue
            observed_min = float(data_array.min())
            observed_max = float(data_array.max())
            ok = observed_min >= lo and observed_max <= hi
            checks.append(
                CheckResult(
                    name=f"value_range:{variable}",
                    ok=ok,
                    message=f"Observed [{observed_min}, {observed_max}], expected [{lo}, {hi}].",
                )
            )
    finally:
        dataset.close()

    return [OutputArtifactCheck(label="era5_land_store", path=store_path, exists=True, checks=checks)]


def _climate_fingerprint_paths(root_dir) -> list[Path]:
    from src.data.sources.climate.constants import (
        DEFAULT_ADM2_UPSTREAM_YEARLY_OUTPUT_PATH,
        DEFAULT_ERA5_LAND_STORE_PATH,
        DEFAULT_SENSOR_UPSTREAM_OUTPUT_PATH,
    )

    # Fingerprint the shared zarr store itself rather than raw GRIB input
    # files: those get deleted once consumed, and era5_land_arco updates the
    # store without ever producing GRIB files at all, so per-file GRIB
    # fingerprinting would miss ARCO-driven changes entirely.
    return [
        Path(root_dir) / DEFAULT_ERA5_LAND_STORE_PATH / "zarr.json",
        Path(root_dir) / DEFAULT_SENSOR_UPSTREAM_OUTPUT_PATH,
        Path(root_dir) / DEFAULT_ADM2_UPSTREAM_YEARLY_OUTPUT_PATH,
    ]


# --------------------------------------------------------------------------
# biomes
# --------------------------------------------------------------------------

def _biomes_list_fetched(root_dir, force: bool = False) -> FetchListing:
    from src.data.sources.biomes.constants import archive_path

    path = archive_path(root_dir)
    present = 1 if path.exists() else 0
    return FetchListing(
        present=present, expected=1, detail=f"Biomes archive {'present' if present else 'missing'} at {path}."
    )


def _biomes_check_outputs(root_dir) -> list[OutputArtifactCheck]:
    from src.data.sources.biomes.constants import (
        BIOME_COLUMN,
        DEFAULT_ADM2_OUTPUT_PATH,
        DEFAULT_SENSOR_OUTPUT_PATH,
        MUN_ID_COLUMN,
        STATION_CODE_COLUMN,
    )

    def build_checks(frame, required_columns):
        checks = [check_required_columns(frame, required_columns), _non_empty_check(frame)]
        if BIOME_COLUMN in frame.columns:
            checks.append(check_null_fraction(frame, BIOME_COLUMN, max_null_fraction=0.0))
        return checks

    return [
        _artifact_check(
            label,
            Path(root_dir) / rel_path,
            lambda frame, required_columns=required_columns: build_checks(frame, required_columns),
        )
        for label, rel_path, required_columns in (
            ("biome_adm2", DEFAULT_ADM2_OUTPUT_PATH, [MUN_ID_COLUMN, BIOME_COLUMN]),
            ("biome_sensor", DEFAULT_SENSOR_OUTPUT_PATH, [STATION_CODE_COLUMN, BIOME_COLUMN]),
        )
    ]


def _biomes_check_fetched(root_dir) -> list[OutputArtifactCheck]:
    from src.data.sources.biomes.constants import archive_path

    path = archive_path(root_dir)
    if not path.exists():
        return [_missing_artifact("biomes_archive", path)]
    checks = [check_file_nonempty(path, min_size_bytes=1024), check_zip_integrity(path)]
    return [OutputArtifactCheck(label="biomes_archive", path=path, exists=True, checks=checks)]


def _biomes_fingerprint_paths(root_dir) -> list[Path]:
    from src.data.sources.biomes.constants import (
        DEFAULT_ADM2_OUTPUT_PATH,
        DEFAULT_SENSOR_OUTPUT_PATH,
        archive_path,
    )

    return [
        archive_path(root_dir),
        Path(root_dir) / DEFAULT_ADM2_OUTPUT_PATH,
        Path(root_dir) / DEFAULT_SENSOR_OUTPUT_PATH,
    ]


# --------------------------------------------------------------------------
# population
# --------------------------------------------------------------------------

def _population_list_fetched(root_dir, force: bool = False) -> FetchListing:
    from src.data.sources.population.constants import raw_dir as _population_raw_dir

    path = _population_raw_dir(root_dir) / "population_raw.parquet"
    present = 1 if path.exists() else 0
    return FetchListing(
        present=present, expected=1, detail=f"population_raw.parquet {'present' if present else 'missing'} at {path}."
    )


def _population_check_outputs(root_dir) -> list[OutputArtifactCheck]:
    from src.data.sources.population.constants import DEFAULT_POPULATION_OUTPUT_FILENAME
    from src.data.sources.population.constants import processed_dir as _population_processed_dir

    def build_checks(frame):
        checks = [
            check_required_columns(frame, ["mun_id", "year", "sex", "age_group", "population"]),
            _non_empty_check(frame),
        ]
        for categorical_column in ("sex", "age_group"):
            if categorical_column in frame.columns:
                observed = frame[categorical_column].dropna().unique().tolist()
                checks.append(
                    CheckResult(
                        name=f"categorical:{categorical_column}",
                        ok=bool(observed),
                        message=f"{len(observed)} distinct values observed.",
                    )
                )
        return checks

    path = _population_processed_dir(root_dir) / DEFAULT_POPULATION_OUTPUT_FILENAME
    return [_artifact_check("population", path, build_checks)]


def _population_check_fetched(root_dir) -> list[OutputArtifactCheck]:
    from src.data.sources.population.constants import raw_dir as _population_raw_dir

    def build_checks(frame):
        return [
            check_required_columns(frame, ["ano", "id_municipio", "sexo", "grupo_idade", "populacao"]),
            _non_empty_check(frame),
            check_value_range(frame, "populacao", lo=0.0, hi=float("inf"), name="value_range:populacao"),
        ]

    path = _population_raw_dir(root_dir) / "population_raw.parquet"
    return [_artifact_check("population_raw", path, build_checks)]


def _population_fingerprint_paths(root_dir) -> list[Path]:
    from src.data.sources.population.constants import DEFAULT_POPULATION_OUTPUT_FILENAME
    from src.data.sources.population.constants import processed_dir as _population_processed_dir
    from src.data.sources.population.constants import raw_dir as _population_raw_dir

    return [
        _population_raw_dir(root_dir) / "population_raw.parquet",
        _population_processed_dir(root_dir) / DEFAULT_POPULATION_OUTPUT_FILENAME,
    ]


# --------------------------------------------------------------------------
# health
# --------------------------------------------------------------------------

# The batch-manifest based SIH tables (see src/data/sources/health/fetch/datasus.py);
# mortality tables use a simpler non-manifest scrape and aren't counted here.
_HEALTH_KNOWN_BATCH_TABLES = (
    "SIH_RESIDENCE_TOTAL_MUNICIPALITY_YEAR",
    "SIH_RESIDENCE_ICD10_CHAPTER_MUNICIPALITY_YEAR",
    "SIH_RESIDENCE_SELECTED_MORBIDITY_LIST_MUNICIPALITY_YEAR",
)
_HEALTH_OUTPUT_FILES = (
    "hospitalizations.parquet",
    "hospitalizations_icd10_chapter.parquet",
    "hospitalizations_selected_morbidity_list.parquet",
    "birth_weight.parquet",
    "gestational_duration.parquet",
)
# All three SIH hospitalization tables share this long-format base schema
# (see _empty_total/_icd10/_morbidity_hospitalization_frame in
# src/data/sources/health/preprocess/preprocess.py); each adds its own extra
# category columns (icd10_chapter_*, morbidity_*) on top.
_HEALTH_HOSPITALIZATION_REQUIRED_COLUMNS = ["municipality_code", "year", "metric_name", "metric_value"]
# Birth outcome tables (_clean_birth_outcome_frame) keep a municipality id,
# per-category count columns, and a "Total" column -- all non-negative counts.
_HEALTH_BIRTH_OUTCOME_FILES = {"birth_weight.parquet", "gestational_duration.parquet"}
_HEALTH_BIRTH_OUTCOME_REQUIRED_COLUMNS = ["mun_id", "year", "Total"]


def _health_list_fetched(root_dir, force: bool = False) -> FetchListing:
    from src.data.sources.health.constants import HEALTH_DATASET_NAME
    from src.data.shared.batches import load_manifest

    present = 0
    expected = 0
    any_manifest = False
    for table_name in _HEALTH_KNOWN_BATCH_TABLES:
        entries = load_manifest(root_dir, HEALTH_DATASET_NAME, table_name)
        if entries:
            any_manifest = True
        expected += len(entries)
        present += sum(1 for entry in entries if entry.get("status") == "completed")

    if not any_manifest:
        return FetchListing(present=0, expected=None, detail="No health batch manifests found locally.")
    return FetchListing(
        present=present,
        expected=expected,
        detail=f"{present}/{expected} batches completed across {len(_HEALTH_KNOWN_BATCH_TABLES)} known SIH tables.",
    )


def _health_check_outputs(root_dir) -> list[OutputArtifactCheck]:
    def build_checks(frame, filename):
        checks = [_non_empty_check(frame)]
        if filename in _HEALTH_BIRTH_OUTCOME_FILES:
            checks.append(check_required_columns(frame, _HEALTH_BIRTH_OUTCOME_REQUIRED_COLUMNS))
            if "Total" in frame.columns:
                checks.append(
                    check_value_range(frame, "Total", lo=0.0, hi=float("inf"), name="value_range:Total")
                )
        else:
            checks.append(check_required_columns(frame, _HEALTH_HOSPITALIZATION_REQUIRED_COLUMNS))
            if "metric_value" in frame.columns:
                checks.append(
                    check_value_range(frame, "metric_value", lo=0.0, hi=float("inf"), name="value_range:metric_value")
                )
        return checks

    health_dir = Path(root_dir) / "data" / "health" / "processed"
    return [
        _artifact_check(
            filename,
            health_dir / filename,
            lambda frame, filename=filename: build_checks(frame, filename),
        )
        for filename in _HEALTH_OUTPUT_FILES
    ]


def _check_datasus_csv_parseable(path: Path) -> CheckResult:
    from src.data.sources.health.preprocess.preprocess import _read_datasus_csv

    try:
        frame = _read_datasus_csv(str(path))
    except Exception as exc:
        return CheckResult(name="datasus_csv_parseable", ok=False, message=f"{exc.__class__.__name__}: {exc}")
    ok = not frame.empty
    message = f"{len(frame)} rows, {len(frame.columns)} columns." if ok else "Parsed frame is empty."
    return CheckResult(name="datasus_csv_parseable", ok=ok, message=message)


def _health_check_fetched(root_dir) -> list[OutputArtifactCheck]:
    from src.data.sources.health.constants import HEALTH_DATASET_NAME
    from src.data.shared.batches import batch_table_dir, load_manifest

    artifacts: list[OutputArtifactCheck] = []
    for table_name in _HEALTH_KNOWN_BATCH_TABLES:
        table_dir = Path(batch_table_dir(root_dir, HEALTH_DATASET_NAME, table_name))
        entries = [
            entry
            for entry in load_manifest(root_dir, HEALTH_DATASET_NAME, table_name)
            if entry.get("status") == "completed" and entry.get("raw_path") and Path(entry["raw_path"]).exists()
        ]
        if not entries:
            artifacts.append(_absent_artifact(f"health_batches:{table_name}", table_dir))
            continue

        sample_paths = sorted(
            (Path(entry["raw_path"]) for entry in entries), key=lambda p: p.stat().st_mtime, reverse=True
        )
        check = check_sampled_files(
            sample_paths,
            check_fn=_check_datasus_csv_parseable,
            cache_path=table_dir / ".raw_csv_verification_cache.json",
            sample_limit=1,
            name="datasus_csv_sample",
        )
        artifacts.append(
            OutputArtifactCheck(label=f"health_batches:{table_name}", path=table_dir, exists=True, checks=[check])
        )
    return artifacts


def _health_fingerprint_paths(root_dir) -> list[Path]:
    from src.data.sources.health.constants import HEALTH_DATASET_NAME
    from src.data.shared.batches import manifest_path

    health_dir = Path(root_dir) / "data" / "health" / "processed"
    manifest_paths = [Path(manifest_path(root_dir, HEALTH_DATASET_NAME, table_name)) for table_name in _HEALTH_KNOWN_BATCH_TABLES]
    return [health_dir / filename for filename in _HEALTH_OUTPUT_FILES] + manifest_paths


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

def _load_assembly_datasets_safe(root_dir):
    """Return (config_path, datasets_or_None, error_or_None)."""
    from src.data.assembly.constants import DEFAULT_CONFIG_PATH
    from src.data.assembly.schema import load_assembly_config

    config_path = Path(root_dir) / DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return config_path, None, None
    try:
        return config_path, load_assembly_config(config_path), None
    except Exception as exc:
        return config_path, None, str(exc)


def _assembly_list_fetched(root_dir, force: bool = False) -> FetchListing:
    config_path, datasets, error = _load_assembly_datasets_safe(root_dir)
    if datasets is None and error is None:
        return FetchListing(present=0, expected=None, detail=f"Assembly config not found at {config_path}.")
    if error is not None:
        return FetchListing(present=0, expected=None, detail=f"Could not parse assembly config: {error}")

    upstream_paths = sorted({source.path for dataset in datasets.values() for source in dataset.sources})
    present = sum(1 for path in upstream_paths if (Path(root_dir) / path).exists())
    return FetchListing(
        present=present,
        expected=len(upstream_paths),
        detail=f"{present}/{len(upstream_paths)} declared upstream source files present.",
    )


def _assembly_check_outputs(root_dir) -> list[OutputArtifactCheck]:
    from src.data.assembly.constants import WIDE_SOURCE_TYPE

    config_path, datasets, error = _load_assembly_datasets_safe(root_dir)
    if datasets is None and error is None:
        return [_missing_artifact("assembly_config", config_path)]
    if error is not None:
        return [
            OutputArtifactCheck(
                label="assembly_config",
                path=config_path,
                exists=True,
                checks=[CheckResult(name="parse_config", ok=False, message=error)],
            )
        ]

    def build_checks(frame, dataset):
        # Only wide-type sources' declared `variables`/`categorical_variables`
        # map 1:1 onto output column names; long_pivot/*_bucketed sources
        # produce derived column names (e.g. pivoted or kernel-weighted
        # composition columns), so checking those literally would be a
        # guaranteed false failure rather than a real signal.
        wide_variable_columns = sorted(
            {
                column
                for source in dataset.sources
                if source.type == WIDE_SOURCE_TYPE
                for column in (*source.variables, *source.categorical_variables)
            }
        )
        required_columns = sorted(set(wide_variable_columns) | set(dataset.index))
        checks = [
            check_required_columns(frame, required_columns),
            _non_empty_check(frame),
        ]
        # A left join against a source missing rows for some keys produces an
        # all-NaN column with no error; catch a joined-in column coming back
        # (almost) entirely null, which check_required_columns can't detect
        # since the column is still present.
        for column in wide_variable_columns:
            if column in frame.columns:
                checks.append(
                    check_null_fraction(frame, column, max_null_fraction=0.99, name=f"null_fraction:{column}")
                )
        return checks

    return [
        _artifact_check(
            dataset_id,
            Path(root_dir) / dataset.output_path,
            lambda frame, dataset=dataset: build_checks(frame, dataset),
        )
        for dataset_id, dataset in datasets.items()
    ]


def _assembly_fingerprint_paths(root_dir) -> list[Path]:
    config_path, datasets, _error = _load_assembly_datasets_safe(root_dir)
    paths = [config_path]
    if datasets:
        paths.extend(Path(root_dir) / dataset.output_path for dataset in datasets.values())
    return paths


SOURCE_ADAPTERS: dict[str, SourceAdapter] = {
    "gadm": SourceAdapter(
        name="gadm",
        list_fetched=_gadm_list_fetched,
        check_outputs=_gadm_check_outputs,
        fingerprint_paths=_gadm_fingerprint_paths,
        fetch_method="Manual placement (GADM ADM2 boundary .gpkg)",
        check_fetched=_gadm_check_fetched,
    ),
    "river_network": SourceAdapter(
        name="river_network",
        list_fetched=_river_network_list_fetched,
        check_outputs=_river_network_check_outputs,
        fingerprint_paths=_river_network_fingerprint_paths,
        fetch_method="Manual placement (hydrography .gpkg)",
        check_fetched=_river_network_check_fetched,
    ),
    "land_cover": SourceAdapter(
        name="land_cover",
        list_fetched=_land_cover_list_fetched,
        check_outputs=_land_cover_check_outputs,
        fingerprint_paths=_land_cover_fingerprint_paths,
        fetch_method="Manual placement (MapBiomas GeoTIFF tiles)",
        check_fetched=_land_cover_check_fetched,
    ),
    "sensor_data": SourceAdapter(
        name="sensor_data",
        list_fetched=_sensor_data_list_fetched,
        check_outputs=_sensor_data_check_outputs,
        fingerprint_paths=_sensor_data_fingerprint_paths,
        fetch_method="Scraped (ANA HidroWeb, Selenium)",
        check_fetched=_sensor_data_check_fetched,
    ),
    "climate": SourceAdapter(
        name="climate",
        list_fetched=_climate_list_fetched,
        check_outputs=_climate_check_outputs,
        fingerprint_paths=_climate_fingerprint_paths,
        fetch_method="API (CDS/ARCO, ERA5-Land)",
        check_fetched=_climate_check_fetched,
    ),
    "biomes": SourceAdapter(
        name="biomes",
        list_fetched=_biomes_list_fetched,
        check_outputs=_biomes_check_outputs,
        fingerprint_paths=_biomes_fingerprint_paths,
        fetch_method="Download (IBGE shapefile, HTTP)",
        check_fetched=_biomes_check_fetched,
    ),
    "population": SourceAdapter(
        name="population",
        list_fetched=_population_list_fetched,
        check_outputs=_population_check_outputs,
        fingerprint_paths=_population_fingerprint_paths,
        fetch_method="Query (BigQuery, basedosdados)",
        check_fetched=_population_check_fetched,
    ),
    "health": SourceAdapter(
        name="health",
        list_fetched=_health_list_fetched,
        check_outputs=_health_check_outputs,
        fingerprint_paths=_health_fingerprint_paths,
        fetch_method="Scraped (DATASUS, Selenium)",
        check_fetched=_health_check_fetched,
    ),
    "assembly": SourceAdapter(
        name="assembly",
        list_fetched=_assembly_list_fetched,
        check_outputs=_assembly_check_outputs,
        fingerprint_paths=_assembly_fingerprint_paths,
        fetch_method="N/A (joins the other 7 sources)",
        # No separate raw-fetched-artifact concept: assembly isn't a fetch
        # source, and _assembly_list_fetched/_assembly_check_outputs already
        # cover its two real concerns. Uses SourceAdapter's default no-op.
    ),
}


__all__ = ["FetchListing", "OutputArtifactCheck", "SourceAdapter", "SOURCE_ADAPTERS", "_default_check_fetched"]
