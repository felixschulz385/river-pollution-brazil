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

from .checks import CheckResult, check_required_columns, check_value_range


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


# --------------------------------------------------------------------------
# river_network
# --------------------------------------------------------------------------

def _river_network_list_fetched(root_dir, force: bool = False) -> FetchListing:
    from src.data.sources.river_network.constants import DEFAULT_GADM_PATH

    gadm_path = Path(root_dir) / DEFAULT_GADM_PATH
    present = 1 if gadm_path.exists() else 0
    return FetchListing(
        present=present,
        expected=1,
        detail=f"GADM input {'present' if present else 'missing'} at {gadm_path}.",
    )


def _river_network_check_outputs(root_dir) -> list[OutputArtifactCheck]:
    from src.data.sources.river_network.constants import (
        DRAINAGE_AREAS_FILENAME,
        TRENCHES_FILENAME,
        TRENCH_ID_COLUMN,
    )

    river_dir = Path(root_dir) / "data" / "river_network"
    results = []

    trenches_path = river_dir / TRENCHES_FILENAME
    trenches = _safe_read_parquet(trenches_path)
    if trenches is None:
        results.append(_missing_artifact("river_trenches", trenches_path))
    else:
        checks = [
            check_required_columns(
                trenches,
                ["trench_id", "upstream_node", "downstream_node", "distance", "system_id"],
            ),
            _non_empty_check(trenches),
            _uniqueness_check(trenches, TRENCH_ID_COLUMN),
        ]
        results.append(OutputArtifactCheck(label="river_trenches", path=trenches_path, exists=True, checks=checks))

    drainage_path = river_dir / DRAINAGE_AREAS_FILENAME
    drainage = _safe_read_parquet(drainage_path)
    if drainage is None:
        results.append(_missing_artifact("drainage_areas", drainage_path))
    else:
        checks = [
            check_required_columns(drainage, ["trench_id", "drainage_area", "within_brazil"]),
            _non_empty_check(drainage),
        ]
        results.append(OutputArtifactCheck(label="drainage_areas", path=drainage_path, exists=True, checks=checks))

    return results


def _river_network_fingerprint_paths(root_dir) -> list[Path]:
    from src.data.sources.river_network.constants import (
        DEFAULT_GADM_PATH,
        DRAINAGE_AREAS_FILENAME,
        TRENCHES_FILENAME,
    )

    river_dir = Path(root_dir) / "data" / "river_network"
    return [
        Path(root_dir) / DEFAULT_GADM_PATH,
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
    from src.data.sources.land_cover.constants import DEFAULT_SENSOR_UPSTREAM_OUTPUT_PATH

    path = Path(root_dir) / DEFAULT_SENSOR_UPSTREAM_OUTPUT_PATH
    frame = _safe_read_parquet(path)
    if frame is None:
        return [_missing_artifact("land_cover_sensor_upstream", path)]

    checks = [
        check_required_columns(frame, ["station_code", "year"]),
        _non_empty_check(frame),
    ]
    share_columns = [column for column in frame.columns if column.endswith("_shr")]
    for column in share_columns[:5]:
        checks.append(check_value_range(frame, column, lo=-1e-6, hi=1.0 + 1e-6, name=f"value_range:{column}"))
    return [OutputArtifactCheck(label="land_cover_sensor_upstream", path=path, exists=True, checks=checks)]


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
    from src.data.sources.sensor_data.schema import STREAMFLOW_MAX_VALID_DISCHARGE

    path = Path(root_dir) / "data" / "sensor_data" / "water_quality_streamflow.parquet"
    frame = _safe_read_parquet(path)
    if frame is None:
        return [_missing_artifact("water_quality_streamflow", path)]

    checks = [
        check_required_columns(frame, ["station_code", "datetime"]),
        _non_empty_check(frame),
    ]
    if "discharge" in frame.columns:
        checks.append(
            check_value_range(frame, "discharge", lo=0.0, hi=STREAMFLOW_MAX_VALID_DISCHARGE, name="value_range:discharge")
        )
    return [OutputArtifactCheck(label="water_quality_streamflow", path=path, exists=True, checks=checks)]


def _sensor_data_fingerprint_paths(root_dir) -> list[Path]:
    from src.data.sources.sensor_data.constants import get_download_log_database_path

    return [
        get_download_log_database_path(root_dir),
        Path(root_dir) / "data" / "sensor_data" / "water_quality_streamflow.parquet",
    ]


# --------------------------------------------------------------------------
# climate
# --------------------------------------------------------------------------

def _climate_list_fetched(root_dir, force: bool = False) -> FetchListing:
    import json

    from src.data.sources.climate.preprocess.era5_land import (
        ERA5_OUTPUT_END,
        ERA5_OUTPUT_START,
        _candidate_manifest_paths,
    )

    # Successfully preprocessed raw GRIB files are deleted (see
    # `_delete_raw_input_file` in era5_land.py) -- counting *.grib files
    # would report 0 present once preprocessing has consumed them, even
    # though the month was genuinely fetched. Their `.manifest.json`
    # sidecars survive that deletion and keep `download_status`, so use
    # those to determine what was actually downloaded.
    try:
        manifest_paths = _candidate_manifest_paths(root_dir=root_dir, subtype="era5_land_hourly")
        downloaded = 0
        for manifest_path in manifest_paths:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            download_status = manifest.get("download_status", manifest.get("status"))
            if download_status == "downloaded":
                downloaded += 1
    except Exception as exc:
        return FetchListing(present=0, expected=None, detail=f"Could not discover ERA5-Land input files: {exc}")

    expected = len(pd.period_range(ERA5_OUTPUT_START, ERA5_OUTPUT_END, freq="M"))
    return FetchListing(
        present=downloaded,
        expected=expected,
        detail=(
            f"{downloaded} months downloaded (raw GRIB present or already preprocessed) "
            f"of {expected} expected (year-month, era5_land_hourly)."
        ),
    )


def _climate_check_outputs(root_dir) -> list[OutputArtifactCheck]:
    from src.data.sources.climate.constants import (
        DEFAULT_ADM2_UPSTREAM_YEARLY_OUTPUT_PATH,
        DEFAULT_SENSOR_UPSTREAM_OUTPUT_PATH,
    )
    from src.data.sources.climate.fetch.verify import ERA5L_VALUE_RANGES

    results = []
    for label, rel_path in (
        ("climate_sensor_upstream", DEFAULT_SENSOR_UPSTREAM_OUTPUT_PATH),
        ("climate_adm2_upstream_yearly", DEFAULT_ADM2_UPSTREAM_YEARLY_OUTPUT_PATH),
    ):
        path = Path(root_dir) / rel_path
        frame = _safe_read_parquet(path)
        if frame is None:
            results.append(_missing_artifact(label, path))
            continue
        checks = [_non_empty_check(frame)]
        for variable, (lo, hi) in ERA5L_VALUE_RANGES.items():
            matching_columns = [column for column in frame.columns if column.startswith(f"{variable}_")]
            for column in matching_columns[:2]:
                checks.append(check_value_range(frame, column, lo=lo, hi=hi, name=f"value_range:{column}"))
        results.append(OutputArtifactCheck(label=label, path=path, exists=True, checks=checks))
    return results


def _climate_fingerprint_paths(root_dir) -> list[Path]:
    from src.data.sources.climate.constants import (
        DEFAULT_ADM2_UPSTREAM_YEARLY_OUTPUT_PATH,
        DEFAULT_SENSOR_UPSTREAM_OUTPUT_PATH,
    )
    from src.data.sources.climate.preprocess.era5_land import discover_era5_input_files

    try:
        files = discover_era5_input_files(root_dir=root_dir, subtype="era5_land_hourly")
    except Exception:
        files = []
    return [
        *files,
        Path(root_dir) / DEFAULT_SENSOR_UPSTREAM_OUTPUT_PATH,
        Path(root_dir) / DEFAULT_ADM2_UPSTREAM_YEARLY_OUTPUT_PATH,
    ]


# --------------------------------------------------------------------------
# biomes
# --------------------------------------------------------------------------

def _biomes_list_fetched(root_dir, force: bool = False) -> FetchListing:
    from src.data.sources.biomes.constants import BIOMES_ARCHIVE_FILENAME

    path = Path(root_dir) / "data" / "biomes" / "raw" / BIOMES_ARCHIVE_FILENAME
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

    results = []
    for label, rel_path, required_columns in (
        ("biome_adm2", DEFAULT_ADM2_OUTPUT_PATH, [MUN_ID_COLUMN, BIOME_COLUMN]),
        ("biome_sensor", DEFAULT_SENSOR_OUTPUT_PATH, [STATION_CODE_COLUMN, BIOME_COLUMN]),
    ):
        path = Path(root_dir) / rel_path
        frame = _safe_read_parquet(path)
        if frame is None:
            results.append(_missing_artifact(label, path))
            continue
        checks = [check_required_columns(frame, required_columns), _non_empty_check(frame)]
        results.append(OutputArtifactCheck(label=label, path=path, exists=True, checks=checks))
    return results


def _biomes_fingerprint_paths(root_dir) -> list[Path]:
    from src.data.sources.biomes.constants import (
        BIOMES_ARCHIVE_FILENAME,
        DEFAULT_ADM2_OUTPUT_PATH,
        DEFAULT_SENSOR_OUTPUT_PATH,
    )

    return [
        Path(root_dir) / "data" / "biomes" / "raw" / BIOMES_ARCHIVE_FILENAME,
        Path(root_dir) / DEFAULT_ADM2_OUTPUT_PATH,
        Path(root_dir) / DEFAULT_SENSOR_OUTPUT_PATH,
    ]


# --------------------------------------------------------------------------
# population
# --------------------------------------------------------------------------

def _population_list_fetched(root_dir, force: bool = False) -> FetchListing:
    path = Path(root_dir) / "data" / "population" / "raw" / "population_raw.parquet"
    present = 1 if path.exists() else 0
    return FetchListing(
        present=present, expected=1, detail=f"population_raw.parquet {'present' if present else 'missing'} at {path}."
    )


def _population_check_outputs(root_dir) -> list[OutputArtifactCheck]:
    path = Path(root_dir) / "data" / "population" / "population.parquet"
    frame = _safe_read_parquet(path)
    if frame is None:
        return [_missing_artifact("population", path)]

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
    return [OutputArtifactCheck(label="population", path=path, exists=True, checks=checks)]


def _population_fingerprint_paths(root_dir) -> list[Path]:
    return [
        Path(root_dir) / "data" / "population" / "raw" / "population_raw.parquet",
        Path(root_dir) / "data" / "population" / "population.parquet",
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
    health_dir = Path(root_dir) / "data" / "health"
    results = []
    for filename in _HEALTH_OUTPUT_FILES:
        path = health_dir / filename
        frame = _safe_read_parquet(path)
        if frame is None:
            results.append(_missing_artifact(filename, path))
            continue
        checks = [_non_empty_check(frame)]
        results.append(OutputArtifactCheck(label=filename, path=path, exists=True, checks=checks))
    return results


def _health_fingerprint_paths(root_dir) -> list[Path]:
    health_dir = Path(root_dir) / "data" / "health"
    return [health_dir / filename for filename in _HEALTH_OUTPUT_FILES]


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

    results = []
    for dataset_id, dataset in datasets.items():
        path = Path(root_dir) / dataset.output_path
        frame = _safe_read_parquet(path)
        if frame is None:
            results.append(_missing_artifact(dataset_id, path))
            continue
        # Only wide-type sources' declared `variables`/`categorical_variables`
        # map 1:1 onto output column names; long_pivot/*_bucketed sources
        # produce derived column names (e.g. pivoted or kernel-weighted
        # composition columns), so checking those literally would be a
        # guaranteed false failure rather than a real signal.
        required_columns = sorted(
            {
                column
                for source in dataset.sources
                if source.type == WIDE_SOURCE_TYPE
                for column in (*source.variables, *source.categorical_variables)
            }
            | set(dataset.index)
        )
        checks = [
            check_required_columns(frame, required_columns),
            _non_empty_check(frame),
        ]
        results.append(OutputArtifactCheck(label=dataset_id, path=path, exists=True, checks=checks))
    return results


def _assembly_fingerprint_paths(root_dir) -> list[Path]:
    config_path, datasets, _error = _load_assembly_datasets_safe(root_dir)
    paths = [config_path]
    if datasets:
        paths.extend(Path(root_dir) / dataset.output_path for dataset in datasets.values())
    return paths


SOURCE_ADAPTERS: dict[str, SourceAdapter] = {
    "river_network": SourceAdapter(
        name="river_network",
        list_fetched=_river_network_list_fetched,
        check_outputs=_river_network_check_outputs,
        fingerprint_paths=_river_network_fingerprint_paths,
        fetch_method="Manual placement (GADM/hydrography .gpkg)",
    ),
    "land_cover": SourceAdapter(
        name="land_cover",
        list_fetched=_land_cover_list_fetched,
        check_outputs=_land_cover_check_outputs,
        fingerprint_paths=_land_cover_fingerprint_paths,
        fetch_method="Manual placement (MapBiomas GeoTIFF tiles)",
    ),
    "sensor_data": SourceAdapter(
        name="sensor_data",
        list_fetched=_sensor_data_list_fetched,
        check_outputs=_sensor_data_check_outputs,
        fingerprint_paths=_sensor_data_fingerprint_paths,
        fetch_method="Scraped (ANA HidroWeb, Selenium)",
    ),
    "climate": SourceAdapter(
        name="climate",
        list_fetched=_climate_list_fetched,
        check_outputs=_climate_check_outputs,
        fingerprint_paths=_climate_fingerprint_paths,
        fetch_method="API (CDS/ARCO, ERA5-Land)",
    ),
    "biomes": SourceAdapter(
        name="biomes",
        list_fetched=_biomes_list_fetched,
        check_outputs=_biomes_check_outputs,
        fingerprint_paths=_biomes_fingerprint_paths,
        fetch_method="Download (IBGE shapefile, HTTP)",
    ),
    "population": SourceAdapter(
        name="population",
        list_fetched=_population_list_fetched,
        check_outputs=_population_check_outputs,
        fingerprint_paths=_population_fingerprint_paths,
        fetch_method="Query (BigQuery, basedosdados)",
    ),
    "health": SourceAdapter(
        name="health",
        list_fetched=_health_list_fetched,
        check_outputs=_health_check_outputs,
        fingerprint_paths=_health_fingerprint_paths,
        fetch_method="Scraped (DATASUS, Selenium)",
    ),
    "assembly": SourceAdapter(
        name="assembly",
        list_fetched=_assembly_list_fetched,
        check_outputs=_assembly_check_outputs,
        fingerprint_paths=_assembly_fingerprint_paths,
        fetch_method="N/A (joins the other 7 sources)",
    ),
}


__all__ = ["FetchListing", "OutputArtifactCheck", "SourceAdapter", "SOURCE_ADAPTERS"]
