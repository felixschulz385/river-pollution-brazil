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
    check_column_has_data,
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
        ADM2_DOMINANT_SYSTEM_TABLE_FILENAME,
        ADM2_OTTO_BASINS_FILENAME,
        DRAINAGE_AREAS_FILENAME,
        OTTO_BASINS_FILENAME,
        PROCESSED_DIR,
        TRENCHES_FILENAME,
        TRENCH_ADM2_TABLE_FILENAME,
        TRENCH_ID_COLUMN,
    )
    from src.data.sources.river_network.otto_basins import OTTO_BASIN_COLUMNS

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
        _artifact_check(
            "trench_adm2_matches",
            river_dir / TRENCH_ADM2_TABLE_FILENAME,
            lambda frame: [
                check_required_columns(frame, ["trench_id", "adm2"]),
                _non_empty_check(frame),
            ],
        ),
        _artifact_check(
            "otto_basins",
            river_dir / OTTO_BASINS_FILENAME,
            lambda frame: [
                check_required_columns(frame, [TRENCH_ID_COLUMN, *OTTO_BASIN_COLUMNS]),
                _non_empty_check(frame),
                _uniqueness_check(frame, TRENCH_ID_COLUMN),
            ],
        ),
        _artifact_check(
            "adm2_otto_basins",
            river_dir / ADM2_OTTO_BASINS_FILENAME,
            lambda frame: [
                check_required_columns(frame, ["adm2", *OTTO_BASIN_COLUMNS]),
                _non_empty_check(frame),
                _uniqueness_check(frame, "adm2"),
            ],
        ),
        _artifact_check(
            "adm2_dominant_systems",
            river_dir / ADM2_DOMINANT_SYSTEM_TABLE_FILENAME,
            lambda frame: [
                check_required_columns(frame, ["adm2", "system_id"]),
                _non_empty_check(frame),
                _uniqueness_check(frame, "adm2"),
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
        ADM2_DOMINANT_SYSTEM_TABLE_FILENAME,
        ADM2_OTTO_BASINS_FILENAME,
        DEFAULT_RAW_GPKG_PATH,
        DRAINAGE_AREAS_FILENAME,
        OTTO_BASINS_FILENAME,
        PROCESSED_DIR,
        TRENCHES_FILENAME,
        TRENCH_ADM2_TABLE_FILENAME,
    )

    river_dir = Path(root_dir) / PROCESSED_DIR
    return [
        Path(root_dir) / DEFAULT_RAW_GPKG_PATH,
        river_dir / TRENCHES_FILENAME,
        river_dir / DRAINAGE_AREAS_FILENAME,
        river_dir / TRENCH_ADM2_TABLE_FILENAME,
        river_dir / ADM2_DOMINANT_SYSTEM_TABLE_FILENAME,
        river_dir / OTTO_BASINS_FILENAME,
        river_dir / ADM2_OTTO_BASINS_FILENAME,
    ]


# --------------------------------------------------------------------------
# land_cover
# --------------------------------------------------------------------------

def _land_cover_list_fetched(root_dir, force: bool = False) -> FetchListing:
    from src.data.sources.land_cover.constants import MAPBIOMAS_YEARS, build_paths
    from src.data.sources.land_cover.preprocess import get_files

    expected_years = set(MAPBIOMAS_YEARS)
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
        f"({min(expected_years)}-{max(expected_years)})."
    )
    if missing_years:
        detail += f" Missing: {missing_years}."
    return FetchListing(present=len(present_years), expected=len(expected_years), detail=detail)


def _land_cover_share_range_check(frame, share_column) -> list[CheckResult]:
    # Long-format table: land_cover_class is a row value, and the fraction is
    # a single "share" column (not per-class "{class}_shr" columns), so one
    # range check covers every class's rows.
    if share_column not in frame.columns:
        return []
    return [
        check_value_range(frame, share_column, lo=-1e-6, hi=1.0 + 1e-6, name=f"value_range:{share_column}")
    ]


def _land_cover_check_outputs(root_dir) -> list[OutputArtifactCheck]:
    from src.data.sources.land_cover.constants import (
        BUCKET_SHARE_COLUMN,
        DEFAULT_ADM2_UPSTREAM_OUTPUT_PATH,
        DEFAULT_SENSOR_UPSTREAM_OUTPUT_PATH,
    )

    def build_sensor_checks(frame):
        return [
            check_required_columns(frame, ["station_code", "year"]),
            _non_empty_check(frame),
            *_land_cover_share_range_check(frame, BUCKET_SHARE_COLUMN),
        ]

    def build_adm2_upstream_checks(frame):
        return [
            check_required_columns(frame, ["mun_id", "year"]),
            _non_empty_check(frame),
            *_land_cover_share_range_check(frame, BUCKET_SHARE_COLUMN),
        ]

    sensor_path = Path(root_dir) / DEFAULT_SENSOR_UPSTREAM_OUTPUT_PATH
    adm2_upstream_path = Path(root_dir) / DEFAULT_ADM2_UPSTREAM_OUTPUT_PATH
    return [
        _artifact_check("land_cover_sensor_upstream", sensor_path, build_sensor_checks),
        _artifact_check("land_cover_adm2_upstream", adm2_upstream_path, build_adm2_upstream_checks),
        *_land_cover_mining_check_outputs(root_dir),
    ]


# The C11 mining-substances product (`land_cover --product mining`): every
# pixel is class 0 or one mining group, and each group splits into an active
# and a revegetation phase, so both identities must hold exactly in the
# extract and in every (mun_id, year, bucket) of the ADM2 aggregate. Both
# tables are streamed in record batches (the aggregate has ~36M rows).
_MINING_COUNT_TOLERANCE = 1e-6


def _mining_class_ids():
    from src.data.sources.land_cover.mining import MINING_GROUPS, MINING_PHASES

    return {group: [group * 10 + phase for phase in MINING_PHASES] for group in MINING_GROUPS}


def _identity_check(name, max_gap, what):
    ok = max_gap <= _MINING_COUNT_TOLERANCE
    message = f"{what} add up." if ok else f"{what} differ by up to {max_gap:g} pixels."
    return CheckResult(name=name, ok=ok, message=message)


def _mining_extract_checks(path: Path) -> list[CheckResult]:
    import numpy as np
    import pyarrow.parquet as pq

    from src.data.sources.land_cover.constants import LAND_COVER_TOTAL_COLUMN

    phases = _mining_class_ids()
    column = "land_cover_class_{}".format
    required = ["trench_id", "year", LAND_COVER_TOTAL_COLUMN, column(0)]
    required += [column(class_id) for group, ids in phases.items() for class_id in (group, *ids)]
    parquet = pq.ParquetFile(path)
    missing = sorted(set(required) - set(parquet.schema_arrow.names))
    if missing:
        return [CheckResult(name="required_columns", ok=False, message=f"Missing columns: {missing}.")]

    rows, group_gap, phase_gap, negative = 0, 0.0, 0.0, 0
    for batch in parquet.iter_batches(columns=required[2:], batch_size=1_000_000):
        frame = batch.to_pandas()
        rows += len(frame)
        groups = frame[[column(0), *(column(group) for group in phases)]].sum(axis=1)
        group_gap = max(group_gap, float(np.abs(groups - frame[LAND_COVER_TOTAL_COLUMN]).max()))
        for group, ids in phases.items():
            split = frame[[column(class_id) for class_id in ids]].sum(axis=1)
            phase_gap = max(phase_gap, float(np.abs(split - frame[column(group)]).max()))
        negative += int((frame < 0).to_numpy().sum())
    return [
        CheckResult(name="required_columns", ok=True, message="All required columns present."),
        CheckResult(name="non_empty", ok=rows > 0, message=f"{rows} rows."),
        _identity_check("groups_add_up", group_gap, "Class 0 and the mining groups and the trench total"),
        _identity_check("phases_add_up", phase_gap, "Active and revegetation phases and their group"),
        CheckResult(
            name="non_negative_counts",
            ok=negative == 0,
            message="All counts non-negative." if negative == 0 else f"{negative} negative counts.",
        ),
    ]


def _mining_adm2_upstream_checks(path: Path) -> list[CheckResult]:
    import pyarrow.parquet as pq

    from src.data.sources.land_cover.mining import NO_MINING_CLASS, TOTAL_CLASS_STEM

    phases = _mining_class_ids()
    phase_group = {class_id: group for group, ids in phases.items() for class_id in ids}
    keys = ["mun_id", "year", "bucket"]
    required = [*keys, "land_cover_class", "cnt", "share"]
    parquet = pq.ParquetFile(path)
    missing = sorted(set(required) - set(parquet.schema_arrow.names))
    if missing:
        return [CheckResult(name="required_columns", ok=False, message=f"Missing columns: {missing}.")]

    # Signed sums that are zero when the identities hold. Rows of one
    # (mun_id, year, bucket) can straddle two batches, so the per-batch sums
    # are summed again over all batches at the end.
    group_weight = {TOTAL_CLASS_STEM: -1.0, NO_MINING_CLASS: 1.0, **{group: 1.0 for group in phases}}
    group_parts, phase_parts = [], []
    rows, share_min, share_max = 0, float("inf"), float("-inf")
    for batch in parquet.iter_batches(columns=required, batch_size=2_000_000):
        frame = batch.to_pandas()
        rows += len(frame)
        share = frame["share"].dropna()
        if len(share):
            share_min, share_max = min(share_min, float(share.min())), max(share_max, float(share.max()))
        classes = frame["land_cover_class"]
        signed = frame["cnt"] * classes.map(group_weight).fillna(0.0)
        group_parts.append(signed.groupby([frame[key] for key in keys]).sum())
        in_phase = classes.isin(list(phase_group)) | classes.isin(list(phases))
        part = frame[in_phase]
        group = part["land_cover_class"].map(phase_group).fillna(part["land_cover_class"])
        signed = part["cnt"].where(part["land_cover_class"].isin(list(phase_group)), -part["cnt"])
        phase_parts.append(signed.groupby([*(part[key] for key in keys), group]).sum())
    group_gap = float(pd.concat(group_parts).groupby(level=[0, 1, 2]).sum().abs().max()) if rows else 0.0
    phase_gap = float(pd.concat(phase_parts).groupby(level=[0, 1, 2, 3]).sum().abs().max()) if rows else 0.0
    shares_ok = rows == 0 or (share_min >= -1e-6 and share_max <= 1.0 + 1e-6)
    return [
        CheckResult(name="required_columns", ok=True, message="All required columns present."),
        CheckResult(name="non_empty", ok=rows > 0, message=f"{rows} rows."),
        _identity_check("groups_add_up", group_gap, "Class 0 and the mining groups and the bucket total"),
        _identity_check("phases_add_up", phase_gap, "Active and revegetation phases and their group"),
        CheckResult(
            name="value_range:share",
            ok=shares_ok,
            message=f"share within [{share_min:.4g}, {share_max:.4g}]." if rows else "No shares.",
        ),
    ]


def _land_cover_mining_check_outputs(root_dir) -> list[OutputArtifactCheck]:
    from src.data.sources.land_cover.mining import MINING_ADM2_UPSTREAM_PATH, MINING_EXTRACT_PATH

    artifacts = []
    for label, relative_path, build_checks in (
        ("land_cover_mining", MINING_EXTRACT_PATH, _mining_extract_checks),
        ("land_cover_mining_adm2_upstream", MINING_ADM2_UPSTREAM_PATH, _mining_adm2_upstream_checks),
    ):
        path = Path(root_dir) / relative_path
        if not path.exists():
            artifacts.append(_missing_artifact(label, path))
            continue
        artifacts.append(OutputArtifactCheck(label=label, path=path, exists=True, checks=build_checks(path)))
    return artifacts


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
    from src.data.sources.land_cover.constants import (
        DEFAULT_ADM2_UPSTREAM_OUTPUT_PATH,
        DEFAULT_SENSOR_UPSTREAM_OUTPUT_PATH,
        build_paths,
    )

    from src.data.sources.land_cover.mining import MINING_ADM2_UPSTREAM_PATH, MINING_EXTRACT_PATH

    paths = build_paths(root_dir)
    try:
        raw_files = sorted(paths.datadir.glob("*.tif"))
    except OSError:
        raw_files = []
    return [
        *raw_files,
        Path(root_dir) / DEFAULT_SENSOR_UPSTREAM_OUTPUT_PATH,
        Path(root_dir) / DEFAULT_ADM2_UPSTREAM_OUTPUT_PATH,
        Path(root_dir) / MINING_EXTRACT_PATH,
        Path(root_dir) / MINING_ADM2_UPSTREAM_PATH,
    ]


# --------------------------------------------------------------------------
# sensor_data
# --------------------------------------------------------------------------

def _sensor_data_list_fetched(root_dir, force: bool = False) -> FetchListing:
    from src.data.sources.sensor_data.constants import get_raw_dir
    from src.data.sources.sensor_data.schema import (
        RAW_STATIONS_PARQUET,
        RAW_STREAMFLOW_PARQUET,
        RAW_WATER_QUALITY_PARQUET,
    )

    # fetch's DuckDB files are an internal working store; its public hand-off --
    # and the only thing preprocess (and biomes) actually read -- is the trio of
    # raw export parquets `export_raw_tables` writes into `raw/`. Judge fetch
    # completeness from those, not from the download-log database.
    raw = get_raw_dir(root_dir)
    expected_files = (RAW_STATIONS_PARQUET, RAW_WATER_QUALITY_PARQUET, RAW_STREAMFLOW_PARQUET)
    present_files = [name for name in expected_files if (raw / name).exists()]
    missing = [name for name in expected_files if name not in present_files]
    detail = f"{len(present_files)}/{len(expected_files)} raw export parquet(s) present in {raw}."
    if missing:
        detail += f" Missing: {missing}."
    return FetchListing(present=len(present_files), expected=len(expected_files), detail=detail)


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
    import geopandas as gpd

    from src.data.sources.sensor_data.constants import get_archives_dir, get_raw_dir
    from src.data.sources.sensor_data.fetch.data.download import (
        _cached_is_parseable_zip,
        _load_zip_verification_cache,
        _save_zip_verification_cache,
    )
    from src.data.sources.sensor_data.schema import (
        RAW_STATIONS_PARQUET,
        RAW_STREAMFLOW_PARQUET,
        RAW_WATER_QUALITY_PARQUET,
    )

    artifacts: list[OutputArtifactCheck] = []
    raw = get_raw_dir(root_dir)

    # Station inventory: `export_raw_tables` writes this GeoParquet as fetch's
    # public output -- it's what preprocess and biomes read, not the internal
    # DuckDB `stations` table -- so verify the parquet directly.
    stations_path = raw / RAW_STATIONS_PARQUET
    if not stations_path.exists():
        artifacts.append(_absent_artifact("station_inventory", stations_path))
    else:
        try:
            stations = gpd.read_parquet(stations_path)
        except Exception as exc:
            # A file that exists but isn't a valid GeoParquet (e.g. truncated,
            # or written by an older export without geometry) is a data
            # problem to report, not a reason to crash verification.
            artifacts.append(
                OutputArtifactCheck(
                    label="station_inventory",
                    path=stations_path,
                    exists=True,
                    checks=[
                        CheckResult(
                            name="readable",
                            ok=False,
                            message=(
                                f"Could not read {RAW_STATIONS_PARQUET} as a GeoDataFrame: "
                                f"{exc.__class__.__name__}: {exc}. Try re-running "
                                "`data fetch --source sensor_data` to rewrite it."
                            ),
                        )
                    ],
                )
            )
        else:
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
            else:
                checks.append(
                    CheckResult(name="null_fraction:geometry", ok=False, message="No geometry column.")
                )
            artifacts.append(
                OutputArtifactCheck(
                    label="station_inventory", path=stations_path, exists=True, checks=checks
                )
            )

    # The other two raw export parquets preprocess reads.
    for label, filename in (
        ("raw_water_quality", RAW_WATER_QUALITY_PARQUET),
        ("raw_streamflow", RAW_STREAMFLOW_PARQUET),
    ):
        path = raw / filename
        frame = _safe_read_parquet(path)
        if frame is None:
            artifacts.append(_missing_artifact(label, path))
        else:
            artifacts.append(
                OutputArtifactCheck(label=label, path=path, exists=True, checks=[_non_empty_check(frame)])
            )

    # Raw MDB archives are a fetch-internal working artifact consumed before the
    # parquets above are written. `data fetch` runs locally (Selenium); a
    # preprocess-only checkout (e.g. HPC) legitimately has no `archives/` dir, so
    # only check archive integrity when the directory is actually present.
    archives_dir = get_archives_dir(root_dir)
    if not archives_dir.exists():
        return artifacts
    zip_paths = sorted(archives_dir.glob("*.zip"))
    if not zip_paths:
        artifacts.append(_absent_artifact("raw_archives", archives_dir))
    else:
        cache = _load_zip_verification_cache(archives_dir)
        cache_dirty = False
        corrupt: list[str] = []
        for path in zip_paths:
            ok, updated = _cached_is_parseable_zip(path, path.stat(), cache)
            cache_dirty = cache_dirty or updated
            if not ok:
                corrupt.append(path.name)
        if cache_dirty:
            _save_zip_verification_cache(archives_dir, cache)
        ok = not corrupt
        message = f"{len(zip_paths) - len(corrupt)}/{len(zip_paths)} archives parseable."
        if corrupt:
            message += f" Corrupt: {corrupt[:10]}."
        artifacts.append(
            OutputArtifactCheck(
                label="raw_archives",
                path=archives_dir,
                exists=True,
                checks=[CheckResult(name="zip_integrity", ok=ok, message=message)],
            )
        )
    return artifacts


def _sensor_data_fingerprint_paths(root_dir) -> list[Path]:
    from src.data.sources.sensor_data.constants import get_processed_dir, get_raw_dir
    from src.data.sources.sensor_data.schema import (
        ASSEMBLED_SENSOR_DATA_PARQUET,
        RAW_STATIONS_PARQUET,
        RAW_STREAMFLOW_PARQUET,
        RAW_WATER_QUALITY_PARQUET,
    )

    raw = get_raw_dir(root_dir)
    return [
        raw / RAW_STATIONS_PARQUET,
        raw / RAW_WATER_QUALITY_PARQUET,
        raw / RAW_STREAMFLOW_PARQUET,
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
    from src.data.sources.climate.preprocess.era5_land import ERA5L_VAR_CONFIG

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
            # ERA5L_VALUE_RANGES is expressed in raw-GRIB units (K, m); these
            # output tables hold the converted units (degC, mm/day) that
            # preprocess/era5_land.py writes to the shared zarr store, so the
            # bounds need the same affine transform applied there -- see
            # _era5l_stored_bounds's docstring for the fetched-store analogue.
            agg = ERA5L_VAR_CONFIG[variable]["aggregation"]
            bound_lo, bound_hi = _era5l_affine_bounds(lo, hi, agg["scale_factor"], agg["offset"])
            bound_lo, bound_hi = _era5l_widen_bounds(bound_lo, bound_hi)
            subset = frame.loc[frame[CLIMATE_VARIABLE_COLUMN] == variable]
            name = f"value_range:{variable}:{value_column}"
            if subset.empty:
                checks.append(CheckResult(name=name, ok=False, message=f"No rows for variable '{variable}'."))
                continue
            checks.append(check_value_range(subset, value_column, lo=bound_lo, hi=bound_hi, name=name))
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


def _era5l_affine_bounds(lo: float, hi: float, scale: float, offset: float) -> tuple[float, float]:
    """Re-express a raw-GRIB-unit range as ``raw * scale + offset``, sorted."""
    a, b = lo * scale + offset, hi * scale + offset
    return (a, b) if a <= b else (b, a)


def _era5l_widen_bounds(lo: float, hi: float) -> tuple[float, float]:
    """Widen a converted-unit range to absorb float32 affine-transform noise.

    Preprocessing's ``raw * scale_factor + offset`` cast leaves sub-epsilon
    noise at the edges (e.g. swvl min at ~-1e-20, tp min at ~-1e-05); allow
    slack proportional to the bound magnitude so that alone doesn't fail an
    otherwise-clean variable.
    """
    tol = 1e-6 * max(1.0, abs(lo), abs(hi))
    return lo - tol, hi + tol


def _era5l_stored_bounds(data_array, lo: float, hi: float) -> tuple[float, float]:
    """Translate a raw-GRIB-unit range into the unit the shared zarr store holds.

    `ERA5L_VALUE_RANGES` (imported from the fetch-stage verifier) is expressed in
    raw ERA5-Land units -- Kelvin for temperature, metres for accumulations --
    because that verifier runs on raw GRIB batches before preprocessing.
    `preprocess/era5_land.py` writes this store as ``raw * scale_factor + offset``
    (degC for 2t/2d, mm/day for tp/sro/ssro/pev) and records the constants it used
    on each array as ``scale_factor_applied`` / ``offset_applied``. Re-apply the
    same affine transform to the range so the check compares like with like.
    """
    scale = float(data_array.attrs.get("scale_factor_applied", 1.0))
    offset = float(data_array.attrs.get("offset_applied", 0.0))
    return _era5l_affine_bounds(lo, hi, scale, offset)


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
            bound_lo, bound_hi = _era5l_stored_bounds(data_array, lo, hi)
            bound_lo, bound_hi = _era5l_widen_bounds(bound_lo, bound_hi)
            ok = observed_min >= bound_lo and observed_max <= bound_hi
            units = data_array.attrs.get("units", "?")
            checks.append(
                CheckResult(
                    name=f"value_range:{variable}",
                    ok=ok,
                    message=(
                        f"Observed [{observed_min}, {observed_max}] {units}, "
                        f"expected [{bound_lo}, {bound_hi}] {units}."
                    ),
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
# protected_areas
# --------------------------------------------------------------------------

def _protected_areas_list_fetched(root_dir, force: bool = False) -> FetchListing:
    from src.data.sources.protected_areas.constants import funai_path, gpkg_path

    gpkg, funai = gpkg_path(root_dir), funai_path(root_dir)
    present = int(gpkg.exists()) + int(funai is not None)
    return FetchListing(
        present=present, expected=2,
        detail=f"TerraBrasilis GeoPackage {'present' if gpkg.exists() else 'missing'} at {gpkg}; "
               f"FUNAI layer {funai or 'missing (raw/funai_<date>/, placed by hand)'}.",
    )


def _protected_area_checks(frame) -> list[CheckResult]:
    """Columns, unique area IDs, known kinds, and areas in force by 2012 exactly when barred by then."""
    from src.data.sources.protected_areas.constants import AREA_COLUMNS, CUTOFF_YEAR, KINDS

    checks = [
        check_required_columns(frame, list(AREA_COLUMNS)),
        _non_empty_check(frame),
        _uniqueness_check(frame, "area_id"),
    ]
    if "kind" in frame.columns:
        unknown = sorted(set(frame["kind"]) - set(KINDS))
        checks.append(CheckResult(
            name="known_kinds", ok=not unknown,
            message=f"Unknown kinds: {unknown}" if unknown else "Indigenous lands and conservation units only.",
        ))
    if checks[0].ok:
        barred = frame[frame["mining_barred"].astype(bool) & frame["year_barred"].notna()]
        wrong = barred["in_force_2012"].astype(bool) != barred["year_barred"].astype("int64").le(CUTOFF_YEAR)
        checks.append(CheckResult(
            name="in_force_by_year_barred", ok=not wrong.any(),
            message=f"{int(wrong.sum())} dated barred areas whose 2012 flag disagrees with their year barred."
            if wrong.any() else f"Dated barred areas are in force exactly when barred by {CUTOFF_YEAR}.",
        ))
    return checks


def _protected_areas_adm2_checks(frame) -> list[CheckResult]:
    """Shares in [0, 1], the union at least each kind, and the 2012 areas within all barred areas."""
    from src.data.sources.protected_areas.constants import (
        ADM2_COLUMNS,
        CONSERVATION_SHARE_COLUMN,
        INDIGENOUS_SHARE_COLUMN,
        MUN_ID_COLUMN,
        PROTECTED_2012_SHARE_COLUMN,
        PROTECTED_SHARE_COLUMN,
    )

    checks = [
        check_required_columns(frame, list(ADM2_COLUMNS)),
        _non_empty_check(frame),
        _uniqueness_check(frame, MUN_ID_COLUMN),
    ]
    if not checks[0].ok:
        return checks
    shares = [column for column in ADM2_COLUMNS if column != MUN_ID_COLUMN]
    checks += [check_value_range(frame, column, lo=0.0, hi=1.0) for column in shares]
    tolerance = 1e-6
    nested = (
        (frame[PROTECTED_SHARE_COLUMN] + tolerance >= frame[[INDIGENOUS_SHARE_COLUMN, CONSERVATION_SHARE_COLUMN]].max(axis=1))
        & (frame[PROTECTED_2012_SHARE_COLUMN] <= frame[PROTECTED_SHARE_COLUMN] + tolerance)
    )
    checks.append(CheckResult(
        name="shares_nested", ok=bool(nested.all()),
        message="Union covers each kind; 2012 areas within all." if nested.all()
        else f"{int((~nested).sum())} municipalities with inconsistent shares.",
    ))
    return checks


def _protected_areas_check_outputs(root_dir) -> list[OutputArtifactCheck]:
    from src.data.sources.protected_areas.constants import DEFAULT_ADM2_OUTPUT_PATH, DEFAULT_AREAS_OUTPUT_PATH

    return [
        _artifact_check("protected_areas", Path(root_dir) / DEFAULT_AREAS_OUTPUT_PATH, _protected_area_checks),
        _artifact_check("protected_areas_adm2", Path(root_dir) / DEFAULT_ADM2_OUTPUT_PATH,
                        _protected_areas_adm2_checks),
    ]


def _protected_areas_check_fetched(root_dir) -> list[OutputArtifactCheck]:
    from src.data.sources.protected_areas.constants import FUNAI_FILENAME, LAYERS, funai_path, gpkg_path, layer_name

    path = gpkg_path(root_dir)
    if path.exists():
        checks = [check_file_nonempty(path, min_size_bytes=1024)]
        checks += [check_gpkg_layer_readable(path, layer_name(type_name))
                   for type_names in LAYERS.values() for type_name in type_names]
        results = [OutputArtifactCheck(label="protected_areas_gpkg", path=path, exists=True, checks=checks)]
    else:
        results = [_missing_artifact("protected_areas_gpkg", path)]
    funai = funai_path(root_dir)
    if funai is None:
        results.append(_missing_artifact("protected_areas_funai", path.parent / "funai_<date>" / FUNAI_FILENAME))
    else:
        results.append(OutputArtifactCheck(label="protected_areas_funai", path=funai, exists=True,
                                           checks=[check_file_nonempty(funai, min_size_bytes=1024)]))
    return results


def _protected_areas_fingerprint_paths(root_dir) -> list[Path]:
    from src.data.sources.protected_areas.constants import (
        DEFAULT_ADM2_OUTPUT_PATH,
        DEFAULT_AREAS_OUTPUT_PATH,
        funai_path,
        gpkg_path,
    )

    funai = funai_path(root_dir)
    return [gpkg_path(root_dir), *([funai] if funai else []), Path(root_dir) / DEFAULT_AREAS_OUTPUT_PATH,
            Path(root_dir) / DEFAULT_ADM2_OUTPUT_PATH]


# --------------------------------------------------------------------------
# geology
# --------------------------------------------------------------------------

_GEOLOGY_LAYERS = ("occurrences", "gold_provinces", "gold_districts")


def _geology_list_fetched(root_dir, force: bool = False) -> FetchListing:
    from src.data.sources.geology.constants import gpkg_path

    path = gpkg_path(root_dir)
    present = 1 if path.exists() else 0
    return FetchListing(
        present=present, expected=1, detail=f"SGB GeoPackage {'present' if present else 'missing'} at {path}."
    )


def _geology_check_outputs(root_dir) -> list[OutputArtifactCheck]:
    from src.data.sources.geology.constants import (
        ADM2_COLUMNS,
        DEFAULT_ADM2_OUTPUT_PATH,
        DEFAULT_ADM2_UPSTREAM_OUTPUT_PATH,
        DISTRICT_SHARE_COLUMN,
        GOLD_OCCURRENCE_COLUMN,
        MUN_ID_COLUMN,
        PROVINCE_SHARE_COLUMN,
    )

    def build_checks(frame):
        checks = [
            check_required_columns(frame, list(ADM2_COLUMNS)),
            _non_empty_check(frame),
            _uniqueness_check(frame, MUN_ID_COLUMN),
        ]
        for column in (DISTRICT_SHARE_COLUMN, PROVINCE_SHARE_COLUMN):
            if column in frame.columns:
                checks.append(check_value_range(frame, column, lo=0.0, hi=1.0))
        if GOLD_OCCURRENCE_COLUMN in frame.columns:
            checks.append(check_column_has_data(frame, GOLD_OCCURRENCE_COLUMN))
        return checks

    return [
        _artifact_check("geology_adm2", Path(root_dir) / DEFAULT_ADM2_OUTPUT_PATH, build_checks),
        _artifact_check(
            "geology_adm2_upstream", Path(root_dir) / DEFAULT_ADM2_UPSTREAM_OUTPUT_PATH, _geology_upstream_checks
        ),
    ]


def _geology_upstream_checks(frame) -> list[CheckResult]:
    """Columns, one row per municipality, counts that grow with the reach and 0/1 Otto binaries."""
    from src.data.sources.geology.constants import (
        MUN_ID_COLUMN,
        OTTO_GROUPS,
        UPSTREAM_COLUMNS,
        UPSTREAM_GROUPS,
        otto_column,
        upstream_count_column,
    )

    checks = [
        check_required_columns(frame, list(UPSTREAM_COLUMNS)),
        _non_empty_check(frame),
        _uniqueness_check(frame, MUN_ID_COLUMN),
    ]
    if not checks[0].ok:
        return checks
    shrinking = [
        group
        for group in UPSTREAM_GROUPS
        if (frame[upstream_count_column(group, "50")] > frame[upstream_count_column(group, "200")]).any()
        or (frame[upstream_count_column(group, "200")] > frame[upstream_count_column(group, "all")]).any()
    ]
    checks.append(
        CheckResult(
            name="counts_grow_with_reach",
            ok=not shrinking,
            message=f"Counts shrink with the reach for {shrinking}." if shrinking else "50 <= 200 <= unlimited.",
        )
    )
    binaries = [otto_column(group, direction) for group in OTTO_GROUPS for direction in ("upstream", "downstream")]
    not_binary = [column for column in binaries if not frame[column].isin([0, 1]).all()]
    checks.append(
        CheckResult(
            name="otto_binary",
            ok=not not_binary,
            message=f"Not 0/1: {not_binary}." if not_binary else "Otto-4 exposures are 0/1.",
        )
    )
    checks.append(check_column_has_data(frame, upstream_count_column("gold", "all")))
    return checks


def _geology_check_fetched(root_dir) -> list[OutputArtifactCheck]:
    from src.data.sources.geology.constants import gpkg_path

    path = gpkg_path(root_dir)
    if not path.exists():
        return [_missing_artifact("sgb_gold_gpkg", path)]
    checks = [check_file_nonempty(path, min_size_bytes=1024)]
    checks += [check_gpkg_layer_readable(path, layer) for layer in _GEOLOGY_LAYERS]
    return [OutputArtifactCheck(label="sgb_gold_gpkg", path=path, exists=True, checks=checks)]


def _geology_fingerprint_paths(root_dir) -> list[Path]:
    from src.data.sources.geology.constants import (
        DEFAULT_ADM2_OUTPUT_PATH,
        DEFAULT_ADM2_UPSTREAM_OUTPUT_PATH,
        gpkg_path,
    )

    return [
        gpkg_path(root_dir),
        Path(root_dir) / DEFAULT_ADM2_OUTPUT_PATH,
        Path(root_dir) / DEFAULT_ADM2_UPSTREAM_OUTPUT_PATH,
    ]


# --------------------------------------------------------------------------
# malaria
# --------------------------------------------------------------------------

def _malaria_list_fetched(root_dir, force: bool = False) -> FetchListing:
    from src.data.sources.malaria.constants import YEARS, year_path

    present = sum(year_path(root_dir, year).exists() for year in YEARS)
    return FetchListing(
        present=present, expected=len(YEARS), detail=f"{present}/{len(YEARS)} yearly SIVEP parquet files present."
    )


def _malaria_check_outputs(root_dir) -> list[OutputArtifactCheck]:
    from src.data.sources.malaria.constants import (
        ADM2_COLUMNS,
        DEFAULT_ADM2_OUTPUT_PATH,
        DEFAULT_FLOWS_OUTPUT_PATH,
        FLOW_CASES_COLUMN,
        FLOW_COLUMNS,
        MUN_ID_COLUMN,
        YEAR_COLUMN,
    )

    def adm2_checks(frame):
        checks = [check_required_columns(frame, list(ADM2_COLUMNS)), _non_empty_check(frame)]
        if {MUN_ID_COLUMN, YEAR_COLUMN} <= set(frame.columns):
            duplicates = int(frame.duplicated([MUN_ID_COLUMN, YEAR_COLUMN]).sum())
            checks.append(
                CheckResult(
                    name="unique:mun_id,year",
                    ok=duplicates == 0,
                    message="All (mun_id, year) keys unique." if duplicates == 0 else f"{duplicates} duplicate keys.",
                )
            )
        return checks

    def flow_checks(frame):
        checks = [check_required_columns(frame, list(FLOW_COLUMNS)), _non_empty_check(frame)]
        if FLOW_CASES_COLUMN in frame.columns:
            checks.append(check_value_range(frame, FLOW_CASES_COLUMN, lo=1, hi=float("inf")))
        return checks

    return [
        _artifact_check("malaria_adm2", Path(root_dir) / DEFAULT_ADM2_OUTPUT_PATH, adm2_checks),
        _artifact_check("malaria_garimpo_flows", Path(root_dir) / DEFAULT_FLOWS_OUTPUT_PATH, flow_checks),
    ]


def _malaria_check_fetched(root_dir) -> list[OutputArtifactCheck]:
    from src.data.sources.malaria.constants import YEARS, year_path

    artifacts = []
    for year in YEARS:
        path = year_path(root_dir, year)
        if not path.exists():
            artifacts.append(_missing_artifact(f"sivep_{year}", path))
            continue
        checks = [check_file_nonempty(path, min_size_bytes=1024)]
        artifacts.append(OutputArtifactCheck(label=f"sivep_{year}", path=path, exists=True, checks=checks))
    return artifacts


def _malaria_fingerprint_paths(root_dir) -> list[Path]:
    from src.data.sources.malaria.constants import (
        DEFAULT_ADM2_OUTPUT_PATH,
        DEFAULT_FLOWS_OUTPUT_PATH,
        YEARS,
        year_path,
    )

    return [
        *(year_path(root_dir, year) for year in YEARS),
        Path(root_dir) / DEFAULT_ADM2_OUTPUT_PATH,
        Path(root_dir) / DEFAULT_FLOWS_OUTPUT_PATH,
    ]


# --------------------------------------------------------------------------
# commodity_prices
# --------------------------------------------------------------------------

def _commodity_prices_list_fetched(root_dir, force: bool = False) -> FetchListing:
    from src.data.sources.commodity_prices.constants import bcb_path, pink_sheet_path

    present = sum(path.exists() for path in (pink_sheet_path(root_dir), bcb_path(root_dir)))
    return FetchListing(present=present, expected=2, detail=f"{present}/2 raw files (Pink Sheet, BCB SGS 1) present.")


def _commodity_prices_check_outputs(root_dir) -> list[OutputArtifactCheck]:
    from src.data.sources.commodity_prices.constants import (
        ANNUAL_COLUMNS,
        DEFAULT_ANNUAL_OUTPUT_PATH,
        DEFAULT_MONTHLY_OUTPUT_PATH,
        MONTHLY_COLUMNS,
    )

    def build_checks(frame, columns, key):
        checks = [check_required_columns(frame, list(columns)), _non_empty_check(frame)]
        for column in ("gold_usd_oz", "brl_per_usd", "gold_brl_oz"):
            if column in frame.columns:
                checks.append(check_value_range(frame, column, lo=1e-9, hi=float("inf")))
        if set(key) <= set(frame.columns):
            duplicates = int(frame.duplicated(list(key)).sum())
            checks.append(CheckResult(name=f"unique:{','.join(key)}", ok=duplicates == 0,
                                      message="Keys unique." if duplicates == 0 else f"{duplicates} duplicate keys."))
        return checks

    return [
        _artifact_check("gold_price_monthly", Path(root_dir) / DEFAULT_MONTHLY_OUTPUT_PATH,
                        lambda frame: build_checks(frame, MONTHLY_COLUMNS, ("year", "month"))),
        _artifact_check("gold_price_annual", Path(root_dir) / DEFAULT_ANNUAL_OUTPUT_PATH,
                        lambda frame: build_checks(frame, ANNUAL_COLUMNS, ("year",))),
    ]


def _commodity_prices_check_fetched(root_dir) -> list[OutputArtifactCheck]:
    from src.data.sources.commodity_prices.constants import bcb_path, pink_sheet_path

    artifacts = []
    for label, path in (("pink_sheet", pink_sheet_path(root_dir)), ("bcb_sgs_1", bcb_path(root_dir))):
        if not path.exists():
            artifacts.append(_missing_artifact(label, path))
            continue
        artifacts.append(OutputArtifactCheck(label=label, path=path, exists=True,
                                             checks=[check_file_nonempty(path, min_size_bytes=1024)]))
    return artifacts


def _commodity_prices_fingerprint_paths(root_dir) -> list[Path]:
    from src.data.sources.commodity_prices.constants import (
        DEFAULT_ANNUAL_OUTPUT_PATH,
        DEFAULT_MONTHLY_OUTPUT_PATH,
        bcb_path,
        pink_sheet_path,
    )

    return [
        pink_sheet_path(root_dir),
        bcb_path(root_dir),
        Path(root_dir) / DEFAULT_MONTHLY_OUTPUT_PATH,
        Path(root_dir) / DEFAULT_ANNUAL_OUTPUT_PATH,
    ]


# --------------------------------------------------------------------------
# mining
# --------------------------------------------------------------------------

def _mining_list_fetched(root_dir, force: bool = False) -> FetchListing:
    from src.data.sources.mining.constants import scm_archive_path

    archive = scm_archive_path(root_dir)
    present = 1 if archive is not None else 0
    detail = f"SCM snapshot {archive.parent.name} present." if archive else "No SCM snapshot (manual placement)."
    return FetchListing(present=present, expected=1, detail=detail)


def _mining_permit_checks(frame) -> list[CheckResult]:
    """Keys unique, counts and areas non-negative, and each substance group within all PLGs."""
    from src.data.sources.mining.constants import (
        ADM2_COLUMNS,
        MEASURES,
        MUN_ID_COLUMN,
        SUBSTANCE_GROUPS,
        YEAR_COLUMN,
        measure_column,
    )

    checks = [check_required_columns(frame, list(ADM2_COLUMNS)), _non_empty_check(frame)]
    if not set(ADM2_COLUMNS) <= set(frame.columns):
        return checks
    duplicates = int(frame.duplicated([MUN_ID_COLUMN, YEAR_COLUMN]).sum())
    checks.append(CheckResult(
        name="unique:mun_id,year",
        ok=duplicates == 0,
        message="All (mun_id, year) keys unique." if duplicates == 0 else f"{duplicates} duplicate keys.",
    ))
    values = frame[[column for column in ADM2_COLUMNS if column not in (MUN_ID_COLUMN, YEAR_COLUMN)]]
    negative = int((values < 0).sum().sum())
    checks.append(CheckResult(
        name="non_negative",
        ok=negative == 0,
        message="No negative counts or areas." if negative == 0 else f"{negative} negative values.",
    ))
    exceeding = sorted(
        measure_column(measure, group)
        for group in SUBSTANCE_GROUPS
        for measure in MEASURES
        if (frame[measure_column(measure, group)] > frame[measure_column(measure, "all")] + 1e-6).any()
    )
    checks.append(CheckResult(
        name="groups_within_all",
        ok=not exceeding,
        message="Substance groups never exceed all PLGs." if not exceeding else f"Exceed all PLGs: {exceeding}",
    ))
    return checks


def _mining_monthly_checks(frame, yearly=None) -> list[CheckResult]:
    """Keys unique, counts positive somewhere, groups within all, and months summing to the yearly requests."""
    from src.data.sources.mining.constants import (
        GROUPS,
        MONTH_COLUMN,
        MONTHLY_COLUMNS,
        MUN_ID_COLUMN,
        SUBSTANCE_GROUPS,
        YEAR_COLUMN,
        measure_column,
    )

    checks = [check_required_columns(frame, list(MONTHLY_COLUMNS)), _non_empty_check(frame)]
    if not set(MONTHLY_COLUMNS) <= set(frame.columns):
        return checks
    duplicates = int(frame.duplicated([MUN_ID_COLUMN, MONTH_COLUMN]).sum())
    checks.append(CheckResult(
        name="unique:mun_id,month", ok=duplicates == 0,
        message="All (mun_id, month) keys unique." if duplicates == 0 else f"{duplicates} duplicate keys.",
    ))
    counts =[measure_column("plg_requests", group) for group in GROUPS]
    sparse = bool((frame[counts] >= 0).all().all() and (frame[measure_column("plg_requests", "all")] > 0).all())
    checks.append(CheckResult(
        name="non_negative_sparse", ok=sparse,
        message="Counts non-negative; every row has a request." if sparse
        else "Negative counts or rows without a request.",
    ))
    exceeding = [group for group in SUBSTANCE_GROUPS
                 if (frame[measure_column("plg_requests", group)] > frame[measure_column("plg_requests", "all")]).any()]
    checks.append(CheckResult(
        name="groups_within_all", ok=not exceeding,
        message="Substance groups never exceed all requests." if not exceeding else f"Exceed all: {exceeding}",
    ))
    if yearly is not None and set(counts) <= set(yearly.columns):
        by_year = frame.groupby([frame[MUN_ID_COLUMN], frame[MONTH_COLUMN].dt.year.rename(YEAR_COLUMN)])[counts].sum()
        expected = yearly.set_index([MUN_ID_COLUMN, YEAR_COLUMN])[counts]
        expected = expected[expected.sum(axis=1) > 0]
        aligned = by_year.reindex(expected.index.union(by_year.index), fill_value=0)
        mismatched = int((aligned != expected.reindex(aligned.index, fill_value=0)).any(axis=1).sum())
        checks.append(CheckResult(
            name="months_sum_to_years", ok=mismatched == 0,
            message="Monthly requests sum to the yearly table." if mismatched == 0
            else f"{mismatched} municipality-years whose months do not sum to the yearly requests.",
        ))
    return checks


def _mining_check_outputs(root_dir) -> list[OutputArtifactCheck]:
    from src.data.sources.mining.constants import DEFAULT_ADM2_OUTPUT_PATH, DEFAULT_MONTHLY_OUTPUT_PATH

    yearly = _safe_read_parquet(Path(root_dir) / DEFAULT_ADM2_OUTPUT_PATH)
    return [
        _artifact_check("mining_permits_adm2", Path(root_dir) / DEFAULT_ADM2_OUTPUT_PATH, _mining_permit_checks),
        _artifact_check("mining_requests_monthly", Path(root_dir) / DEFAULT_MONTHLY_OUTPUT_PATH,
                        lambda frame: _mining_monthly_checks(frame, yearly)),
    ]


def _mining_check_fetched(root_dir) -> list[OutputArtifactCheck]:
    from src.data.sources.mining.constants import SCM_ARCHIVE_FILENAME, raw_dir, scm_archive_path

    archive = scm_archive_path(root_dir)
    if archive is None:
        return [_missing_artifact("scm_microdata", raw_dir(root_dir) / SCM_ARCHIVE_FILENAME)]
    return [OutputArtifactCheck(label="scm_microdata", path=archive, exists=True,
                                checks=[check_file_nonempty(archive, min_size_bytes=1 << 20)])]


def _mining_fingerprint_paths(root_dir) -> list[Path]:
    from src.data.sources.mining.constants import (
        DEFAULT_ADM2_OUTPUT_PATH,
        DEFAULT_MONTHLY_OUTPUT_PATH,
        scm_archive_path,
    )

    archive = scm_archive_path(root_dir)
    return [*([archive] if archive else []), Path(root_dir) / DEFAULT_ADM2_OUTPUT_PATH,
            Path(root_dir) / DEFAULT_MONTHLY_OUTPUT_PATH]


# --------------------------------------------------------------------------
# enforcement
# --------------------------------------------------------------------------

def _enforcement_list_fetched(root_dir, force: bool = False) -> FetchListing:
    from src.data.sources.enforcement.constants import snapshot_path

    snapshot = snapshot_path(root_dir)
    present = 1 if snapshot is not None else 0
    detail = f"IBAMA snapshot {snapshot.name} present." if snapshot else "No IBAMA snapshot (manual placement)."
    return FetchListing(present=present, expected=1, detail=detail)


def _enforcement_record_checks(frame) -> list[CheckResult]:
    """Record keys unique within each action, and garimpo records a subset of mining records."""
    from src.data.sources.enforcement.constants import RECORD_COLUMNS

    checks = [check_required_columns(frame, list(RECORD_COLUMNS)), _non_empty_check(frame)]
    if not set(RECORD_COLUMNS) <= set(frame.columns):
        return checks
    duplicates = int(frame.duplicated(["action", "record_id"]).sum())
    checks.append(CheckResult(
        name="unique:action,record_id",
        ok=duplicates == 0,
        message="All (action, record_id) keys unique." if duplicates == 0 else f"{duplicates} duplicate keys.",
    ))
    outside = int((frame["garimpo"] & ~frame["mining"]).sum())
    checks.append(CheckResult(
        name="garimpo_within_mining",
        ok=outside == 0,
        message="Every garimpo record is a mining record." if outside == 0 else f"{outside} garimpo records not mining.",
    ))
    return checks


def _enforcement_adm2_checks(frame) -> list[CheckResult]:
    """Keys unique, values non-negative, missing only before each action's coverage, classes nested."""
    from src.data.sources.enforcement.constants import (
        ACTION_MEASURES,
        ADM2_COLUMNS,
        COVERAGE_START,
        MUN_ID_COLUMN,
        YEAR_COLUMN,
        action_columns,
        measure_column,
    )

    checks = [check_required_columns(frame, list(ADM2_COLUMNS)), _non_empty_check(frame)]
    if not set(ADM2_COLUMNS) <= set(frame.columns):
        return checks
    duplicates = int(frame.duplicated([MUN_ID_COLUMN, YEAR_COLUMN]).sum())
    checks.append(CheckResult(
        name="unique:mun_id,year",
        ok=duplicates == 0,
        message="All (mun_id, year) keys unique." if duplicates == 0 else f"{duplicates} duplicate keys.",
    ))
    values = frame[[column for column in ADM2_COLUMNS if column not in (MUN_ID_COLUMN, YEAR_COLUMN)]]
    negative = int((values < 0).sum().sum())
    checks.append(CheckResult(
        name="non_negative",
        ok=negative == 0,
        message="No negative counts or amounts." if negative == 0 else f"{negative} negative values.",
    ))
    misplaced = sorted(
        action for action, start in COVERAGE_START.items()
        if frame.loc[frame[YEAR_COLUMN] >= start, list(action_columns(action))].isna().any().any()
        or frame.loc[frame[YEAR_COLUMN] < start, list(action_columns(action))].notna().any().any()
    )
    checks.append(CheckResult(
        name="missing_only_before_coverage",
        ok=not misplaced,
        message="Values missing exactly before each action's coverage." if not misplaced
        else f"Missing values misplaced: {misplaced}",
    ))
    exceeding = sorted(
        measure_column(measure, narrow)
        for measures in ACTION_MEASURES.values()
        for measure, _ in measures
        for narrow, broad in (("mining", "all"), ("garimpo", "mining"))
        if (frame[measure_column(measure, narrow)] > frame[measure_column(measure, broad)] + 1e-6).any()
    )
    checks.append(CheckResult(
        name="classes_nested",
        ok=not exceeding,
        message="Garimpo within mining within all." if not exceeding else f"Exceed the broader class: {exceeding}",
    ))
    return checks


def _enforcement_check_outputs(root_dir) -> list[OutputArtifactCheck]:
    from src.data.sources.enforcement.constants import DEFAULT_ADM2_OUTPUT_PATH, DEFAULT_RECORDS_OUTPUT_PATH

    return [
        _artifact_check("ibama_records", Path(root_dir) / DEFAULT_RECORDS_OUTPUT_PATH, _enforcement_record_checks),
        _artifact_check("enforcement_adm2", Path(root_dir) / DEFAULT_ADM2_OUTPUT_PATH, _enforcement_adm2_checks),
    ]


def _enforcement_check_fetched(root_dir) -> list[OutputArtifactCheck]:
    from src.data.sources.enforcement.constants import SNAPSHOT_PREFIX, TABLES, raw_dir, snapshot_path

    snapshot = snapshot_path(root_dir)
    if snapshot is None:
        return [_missing_artifact("ibama_snapshot", raw_dir(root_dir) / f"{SNAPSHOT_PREFIX}<YYYY-MM-DD>")]
    return [
        OutputArtifactCheck(label=f"ibama_{action}", path=snapshot / spec["file"], exists=True,
                            checks=[check_file_nonempty(snapshot / spec["file"], min_size_bytes=1 << 20)])
        for action, spec in TABLES.items()
    ]


def _enforcement_fingerprint_paths(root_dir) -> list[Path]:
    from src.data.sources.enforcement.constants import (
        DEFAULT_ADM2_OUTPUT_PATH,
        DEFAULT_RECORDS_OUTPUT_PATH,
        TABLES,
        snapshot_path,
    )

    snapshot = snapshot_path(root_dir)
    raw = [snapshot / spec["file"] for spec in TABLES.values()] if snapshot else []
    return [*raw, Path(root_dir) / DEFAULT_RECORDS_OUTPUT_PATH, Path(root_dir) / DEFAULT_ADM2_OUTPUT_PATH]


# --------------------------------------------------------------------------
# nightlights
# --------------------------------------------------------------------------

def _nightlights_list_fetched(root_dir, force: bool = False) -> FetchListing:
    from src.data.sources.nightlights.constants import AVERAGE, FIRST_YEAR, raw_dir, raw_files

    years = sorted({year for _, year, variant, _, _ in raw_files(raw_dir(root_dir)) if variant == AVERAGE})
    if not years:
        return FetchListing(present=0, expected=1, detail="No EOG VNL files (manual placement).")
    return FetchListing(present=len(years), expected=years[-1] - FIRST_YEAR + 1,
                        detail=f"average_masked for {years[0]}-{years[-1]} ({len(years)} years).")


def _nightlights_adm2_checks(frame) -> list[CheckResult]:
    """Keys unique, a balanced panel of consecutive years, counts within pixels, zones nested."""
    from src.data.sources.nightlights.constants import (
        ADM2_COLUMNS,
        COVERAGE_COLUMN,
        MEASURE_COLUMNS,
        MUN_ID_COLUMN,
        YEAR_COLUMN,
        zone_columns,
    )

    checks = [check_required_columns(frame, list(ADM2_COLUMNS)), _non_empty_check(frame)]
    if not set(ADM2_COLUMNS) <= set(frame.columns):
        return checks
    duplicates = int(frame.duplicated([MUN_ID_COLUMN, YEAR_COLUMN]).sum())
    checks.append(CheckResult(
        name="unique:mun_id,year",
        ok=duplicates == 0,
        message="All (mun_id, year) keys unique." if duplicates == 0 else f"{duplicates} duplicate keys.",
    ))
    years = sorted(frame[YEAR_COLUMN].unique())
    balanced = len(frame) == frame[MUN_ID_COLUMN].nunique() * len(years) and years == list(
        range(years[0], years[-1] + 1))
    checks.append(CheckResult(
        name="balanced_consecutive_years",
        ok=balanced,
        message=f"Every municipality in {years[0]}-{years[-1]}." if balanced else "Unbalanced panel or gaps in years.",
    ))
    negative = int((frame[list(MEASURE_COLUMNS)] < 0).sum().sum())
    checks.append(CheckResult(
        name="non_negative",
        ok=negative == 0,
        message="No negative measures." if negative == 0 else f"{negative} negative values.",
    ))
    pixels, lit, radiance = zone_columns("")
    exceeding = sorted(
        f"{narrow} > {broad}"
        for zone in ("_protected", "_gold5km", "_gold5km_protected")
        for narrow, broad in zip(zone_columns(zone), (pixels, lit, radiance))
        if (frame[narrow] > frame[broad] + 1e-6).any()
    ) + sorted(
        f"{narrow} > {broad}"
        for zone in ("_gold5km_protected",)
        for base in ("_protected", "_gold5km")
        for narrow, broad in zip(zone_columns(zone), zone_columns(base))
        if (frame[narrow] > frame[broad] + 1e-6).any()
    ) + [f"lit > pixels{zone}" for zone in ("", "_protected", "_gold5km", "_gold5km_protected")
         if (frame[zone_columns(zone)[1]] > frame[zone_columns(zone)[0]]).any()]
    checks.append(CheckResult(
        name="zones_nested",
        ok=not exceeding,
        message="Zones within municipalities, lit pixels within pixels." if not exceeding else f"Exceeding: {exceeding}",
    ))
    # A year without its cf_cvg file has no coverage anywhere; otherwise only municipalities without pixels lack it.
    has_pixels = frame[pixels] > 0
    missing = frame.loc[has_pixels, COVERAGE_COLUMN].isna().groupby(frame.loc[has_pixels, YEAR_COLUMN])
    partial = sorted(int(year) for year, flags in missing if flags.any() and not flags.all())
    checks.append(CheckResult(
        name="coverage_missing_by_whole_year",
        ok=not partial,
        message="Coverage missing only for whole years or municipalities without pixels." if not partial
        else f"Coverage missing for some municipalities in {partial}",
    ))
    return checks


def _nightlights_check_outputs(root_dir) -> list[OutputArtifactCheck]:
    from src.data.sources.nightlights.constants import DEFAULT_ADM2_OUTPUT_PATH

    return [_artifact_check("nightlights_adm2", Path(root_dir) / DEFAULT_ADM2_OUTPUT_PATH, _nightlights_adm2_checks)]


def _nightlights_check_fetched(root_dir) -> list[OutputArtifactCheck]:
    from src.data.sources.nightlights.constants import raw_dir, raw_files

    files = raw_files(raw_dir(root_dir))
    if not files:
        return [_absent_artifact("nightlights_vnl", raw_dir(root_dir))]
    return [
        OutputArtifactCheck(label=f"nightlights_{year}_{variant}", path=path, exists=True,
                            checks=[check_file_nonempty(path, min_size_bytes=100 << 20)])
        for path, year, variant, _, _ in files
    ]


def _nightlights_fingerprint_paths(root_dir) -> list[Path]:
    from src.data.sources.nightlights.constants import DEFAULT_ADM2_OUTPUT_PATH, raw_dir, raw_files

    return [*(path for path, *_ in raw_files(raw_dir(root_dir))), Path(root_dir) / DEFAULT_ADM2_OUTPUT_PATH]


# --------------------------------------------------------------------------
# population
# --------------------------------------------------------------------------

def _population_list_fetched(root_dir, force: bool = False) -> FetchListing:
    from src.data.sources.population.constants import DEFAULT_POPULATION_RAW_FILENAME
    from src.data.sources.population.constants import raw_dir as _population_raw_dir

    path = _population_raw_dir(root_dir) / DEFAULT_POPULATION_RAW_FILENAME
    present = 1 if path.exists() else 0
    return FetchListing(
        present=present,
        expected=1,
        detail=f"{DEFAULT_POPULATION_RAW_FILENAME} {'present' if present else 'missing'} at {path}.",
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
    from src.data.sources.population.constants import DEFAULT_POPULATION_RAW_FILENAME
    from src.data.sources.population.constants import raw_dir as _population_raw_dir

    def build_checks(frame):
        return [
            check_required_columns(frame, ["ano", "id_municipio", "sexo", "grupo_idade", "populacao"]),
            _non_empty_check(frame),
            check_value_range(frame, "populacao", lo=0.0, hi=float("inf"), name="value_range:populacao"),
        ]

    path = _population_raw_dir(root_dir) / DEFAULT_POPULATION_RAW_FILENAME
    return [_artifact_check("population_raw", path, build_checks)]


def _population_fingerprint_paths(root_dir) -> list[Path]:
    from src.data.sources.population.constants import DEFAULT_POPULATION_OUTPUT_FILENAME, DEFAULT_POPULATION_RAW_FILENAME
    from src.data.sources.population.constants import processed_dir as _population_processed_dir
    from src.data.sources.population.constants import raw_dir as _population_raw_dir

    return [
        _population_raw_dir(root_dir) / DEFAULT_POPULATION_RAW_FILENAME,
        _population_processed_dir(root_dir) / DEFAULT_POPULATION_OUTPUT_FILENAME,
    ]


# --------------------------------------------------------------------------
# health
# --------------------------------------------------------------------------

# The batch-manifest based SIH tables (see src/data/sources/health/fetch/datasus.py);
# the SINASC and SIM per-year manifests aren't counted here.
_HEALTH_KNOWN_BATCH_TABLES = (
    "SIH_RESIDENCE_TOTAL_MUNICIPALITY_YEAR",
    "SIH_RESIDENCE_ICD10_CHAPTER_MUNICIPALITY_YEAR",
    "SIH_RESIDENCE_SELECTED_MORBIDITY_LIST_MUNICIPALITY_YEAR",
)


def _health_output_files() -> tuple[str, ...]:
    from src.data.sources.health.constants import (
        HOSPITALIZATIONS_FILENAME,
        HOSPITALIZATIONS_ICD10_CHAPTER_FILENAME,
        HOSPITALIZATIONS_SELECTED_MORBIDITY_LIST_FILENAME,
        BIRTHS_FILENAME,
        INFANT_DEATHS_FILENAME,
        MORTALITY_FILENAME,
    )

    return (
        HOSPITALIZATIONS_FILENAME,
        HOSPITALIZATIONS_ICD10_CHAPTER_FILENAME,
        HOSPITALIZATIONS_SELECTED_MORBIDITY_LIST_FILENAME,
        *sorted(_health_birth_outcome_files()),
        BIRTHS_FILENAME,
        MORTALITY_FILENAME,
        INFANT_DEATHS_FILENAME,
    )


def _health_birth_outcome_files() -> set[str]:
    from src.data.sources.health.fetch.datasus import BIRTH_COLUMN_OPTIONS
    from src.data.sources.health.constants import birth_outcome_filename

    return {birth_outcome_filename(outcome) for outcome in BIRTH_COLUMN_OPTIONS}


def _health_births_summary_file() -> str:
    from src.data.sources.health.constants import BIRTHS_FILENAME

    return BIRTHS_FILENAME


def _health_mortality_file() -> str:
    from src.data.sources.health.constants import MORTALITY_FILENAME

    return MORTALITY_FILENAME


def _health_infant_deaths_file() -> str:
    from src.data.sources.health.constants import INFANT_DEATHS_FILENAME

    return INFANT_DEATHS_FILENAME


_HEALTH_OUTPUT_FILES = _health_output_files()
# All three SIH hospitalization tables share this long-format base schema
# (see _empty_total/_icd10/_morbidity_hospitalization_frame in
# src/data/sources/health/preprocess/preprocess.py); each adds its own extra
# category columns (icd10_chapter_*, morbidity_*) on top.
_HEALTH_HOSPITALIZATION_REQUIRED_COLUMNS = ["municipality_code", "year", "metric_name", "metric_value"]
# Birth outcome tables (_clean_birth_outcome_frame) keep a municipality id,
# per-category count columns, and a "Total" column -- all non-negative counts.
_HEALTH_BIRTH_OUTCOME_FILES = _health_birth_outcome_files()
_HEALTH_BIRTH_OUTCOME_REQUIRED_COLUMNS = ["mun_id", "year", "Total"]
# The derived births summary (write_birth_summary_table) is wide by (mun_id, year).
_HEALTH_BIRTHS_SUMMARY_FILE = _health_births_summary_file()
_HEALTH_BIRTHS_SUMMARY_REQUIRED_COLUMNS = [
    "mun_id",
    "year",
    "births",
    "births_weight_known",
    "births_low_weight",
    "births_gestation_known",
    "births_preterm",
]
# SIM deaths (preprocess_mortality_age_tables), wide by (mun_id, year).
_HEALTH_MORTALITY_FILE = _health_mortality_file()
_HEALTH_MORTALITY_REQUIRED_COLUMNS = ["mun_id", "year", "source_system", "deaths_total", "deaths_under_1"]
# SIM infant deaths by ICD-10 chapter (preprocess_infant_death_tables), wide by (mun_id, year).
_HEALTH_INFANT_DEATHS_FILE = _health_infant_deaths_file()
_HEALTH_INFANT_DEATHS_REQUIRED_COLUMNS = ["mun_id", "year", "infant_deaths", "infant_deaths_external",
                                          "infant_deaths_nonexternal"]


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
        if filename == _HEALTH_BIRTHS_SUMMARY_FILE:
            checks.append(check_required_columns(frame, _HEALTH_BIRTHS_SUMMARY_REQUIRED_COLUMNS))
        elif filename == _HEALTH_MORTALITY_FILE:
            checks.append(check_required_columns(frame, _HEALTH_MORTALITY_REQUIRED_COLUMNS))
            if "deaths_total" in frame.columns:
                checks.append(
                    check_value_range(frame, "deaths_total", lo=0.0, hi=float("inf"), name="value_range:deaths_total")
                )
        elif filename == _HEALTH_INFANT_DEATHS_FILE:
            checks.append(check_required_columns(frame, _HEALTH_INFANT_DEATHS_REQUIRED_COLUMNS))
            if "infant_deaths_nonexternal" in frame.columns:
                checks.append(check_value_range(frame, "infant_deaths_nonexternal", lo=0.0, hi=float("inf"),
                                                name="value_range:infant_deaths_nonexternal"))
        elif filename in _HEALTH_BIRTH_OUTCOME_FILES:
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

    from src.data.sources.health.constants import processed_dir as _health_processed_dir

    health_dir = _health_processed_dir(root_dir)
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

    from src.data.sources.health.constants import processed_dir as _health_processed_dir

    health_dir = _health_processed_dir(root_dir)
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
        # entirely null, which check_required_columns can't detect since the
        # column is still present. Many water-quality analytes are
        # legitimately measured at only a handful of stations/dates across
        # the whole dataset, so any nonzero coverage is accepted -- only a
        # fully empty column indicates a real join problem.
        for column in wide_variable_columns:
            if column in frame.columns:
                checks.append(check_column_has_data(frame, column, name=f"has_data:{column}"))
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
        fetch_method="Download (MapBiomas GCS GeoTIFF)",
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
    "protected_areas": SourceAdapter(
        name="protected_areas",
        list_fetched=_protected_areas_list_fetched,
        check_outputs=_protected_areas_check_outputs,
        fingerprint_paths=_protected_areas_fingerprint_paths,
        fetch_method="Download (TerraBrasilis WFS) + manual (FUNAI)",
        check_fetched=_protected_areas_check_fetched,
    ),
    "geology": SourceAdapter(
        name="geology",
        list_fetched=_geology_list_fetched,
        check_outputs=_geology_check_outputs,
        fingerprint_paths=_geology_fingerprint_paths,
        fetch_method="Query (SGB ArcGIS REST)",
        check_fetched=_geology_check_fetched,
    ),
    "malaria": SourceAdapter(
        name="malaria",
        list_fetched=_malaria_list_fetched,
        check_outputs=_malaria_check_outputs,
        fingerprint_paths=_malaria_fingerprint_paths,
        fetch_method="Download (Zenodo SIVEP parquet)",
        check_fetched=_malaria_check_fetched,
    ),
    "commodity_prices": SourceAdapter(
        name="commodity_prices",
        list_fetched=_commodity_prices_list_fetched,
        check_outputs=_commodity_prices_check_outputs,
        fingerprint_paths=_commodity_prices_fingerprint_paths,
        fetch_method="Download (World Bank, BCB API)",
        check_fetched=_commodity_prices_check_fetched,
    ),
    "mining": SourceAdapter(
        name="mining",
        list_fetched=_mining_list_fetched,
        check_outputs=_mining_check_outputs,
        fingerprint_paths=_mining_fingerprint_paths,
        fetch_method="Manual placement (ANM SCM microdata zip)",
        check_fetched=_mining_check_fetched,
    ),
    "enforcement": SourceAdapter(
        name="enforcement",
        list_fetched=_enforcement_list_fetched,
        check_outputs=_enforcement_check_outputs,
        fingerprint_paths=_enforcement_fingerprint_paths,
        fetch_method="Manual placement (IBAMA snapshot)",
        check_fetched=_enforcement_check_fetched,
    ),
    "nightlights": SourceAdapter(
        name="nightlights",
        list_fetched=_nightlights_list_fetched,
        check_outputs=_nightlights_check_outputs,
        fingerprint_paths=_nightlights_fingerprint_paths,
        fetch_method="Manual placement (EOG VNL)",
        check_fetched=_nightlights_check_fetched,
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
