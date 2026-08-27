"""Reusable sanity checks and result dataclasses for source verification.

`CheckResult`/`VerificationResult` generalize `VerificationResult` from
`src/data/sources/climate/fetch/verify.py` (which checks raw GRIB batches at fetch
time) into a source-agnostic shape used to check preprocessed *outputs*. Most
checks below operate on a pandas DataFrame; `check_file_nonempty`,
`check_zip_integrity`, `check_gpkg_layer_readable`, `check_raster_header_readable`,
and `check_sampled_files` instead validate raw, non-tabular fetched artifacts
(archives, geopackages, rasters) directly on disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class CheckResult:
    name: str
    ok: bool
    message: str = ""


@dataclass
class VerificationResult:
    status: str  # "verified" | "failed" | "outstanding" | "not_present_locally"
    checks: list[CheckResult] = field(default_factory=list)
    fetch_completeness: dict | None = None

    @property
    def ok(self) -> bool:
        return self.status == "verified"


def check_required_columns(frame, required_columns, *, name: str = "required_columns") -> CheckResult:
    """Wrap `validate_required_columns` (assembly/land_cover schema) as a CheckResult."""
    from src.data.assembly.schema import validate_required_columns

    try:
        validate_required_columns(frame, required_columns, name)
    except ValueError as exc:
        return CheckResult(name=name, ok=False, message=str(exc))
    return CheckResult(
        name=name, ok=True, message=f"All {len(list(required_columns))} required columns present."
    )


def check_null_fraction(frame, column, *, max_null_fraction: float = 0.5, name: str | None = None) -> CheckResult:
    """Fail if `column`'s null share exceeds `max_null_fraction`."""
    name = name or f"null_fraction:{column}"
    if column not in frame.columns or frame.empty:
        return CheckResult(name=name, ok=False, message=f"Column '{column}' not present or frame is empty.")
    null_fraction = float(frame[column].isna().mean())
    ok = null_fraction <= max_null_fraction
    return CheckResult(
        name=name,
        ok=ok,
        message=f"{null_fraction:.2%} null (max allowed {max_null_fraction:.2%}).",
    )


def check_value_range(frame, column, *, lo: float, hi: float, name: str | None = None) -> CheckResult:
    """Fail if any observed value in `column` falls outside `[lo, hi]`."""
    name = name or f"value_range:{column}"
    if column not in frame.columns:
        return CheckResult(name=name, ok=False, message=f"Column '{column}' not present.")
    series = frame[column].dropna()
    if series.empty:
        return CheckResult(name=name, ok=False, message=f"Column '{column}' has no non-null values.")
    observed_min = float(series.min())
    observed_max = float(series.max())
    ok = observed_min >= lo and observed_max <= hi
    return CheckResult(
        name=name,
        ok=ok,
        message=f"Observed range [{observed_min}, {observed_max}], expected [{lo}, {hi}].",
    )


def check_file_nonempty(path: Path, *, min_size_bytes: int = 1, name: str = "file_nonempty") -> CheckResult:
    """Fail if `path` is missing or smaller than `min_size_bytes`."""
    try:
        size = path.stat().st_size
    except OSError as exc:
        return CheckResult(name=name, ok=False, message=f"Could not stat file: {exc}")
    ok = size >= min_size_bytes
    return CheckResult(name=name, ok=ok, message=f"{size} bytes (min {min_size_bytes}).")


def check_zip_integrity(path: Path, *, name: str = "zip_integrity") -> CheckResult:
    """Fail if `path` isn't a structurally intact ZIP archive."""
    from src.data.sources.sensor_data.fetch.data.download import _is_parseable_zip

    ok = _is_parseable_zip(path)
    return CheckResult(name=name, ok=ok, message="Valid ZIP archive." if ok else "Not a parseable ZIP archive.")


def check_gpkg_layer_readable(path: Path, layer: str, *, name: str | None = None) -> CheckResult:
    """Fail if `layer` can't be opened from the GeoPackage at `path` (metadata-only, no full geometry load)."""
    name = name or f"gpkg_layer:{layer}"
    try:
        try:
            import pyogrio

            info = pyogrio.read_info(str(path), layer=layer)
            feature_count = info["features"]
        except ImportError:
            import geopandas as gpd

            frame = gpd.read_file(path, layer=layer, rows=1)
            feature_count = len(frame)
    except Exception as exc:
        return CheckResult(name=name, ok=False, message=f"Could not read layer '{layer}': {exc}")
    ok = feature_count is None or feature_count != 0
    return CheckResult(name=name, ok=ok, message=f"Layer '{layer}' readable ({feature_count} features).")


def check_raster_header_readable(path: Path, *, name: str = "raster_header") -> CheckResult:
    """Fail if `path`'s raster header can't be opened, or has no bands/pixels (header-only, never reads pixels)."""
    try:
        import rasterio

        with rasterio.open(path) as dataset:
            ok = dataset.count >= 1 and dataset.width > 0 and dataset.height > 0
            message = f"{dataset.count} band(s), {dataset.width}x{dataset.height}."
    except Exception as exc:
        return CheckResult(name=name, ok=False, message=f"Could not read raster header: {exc}")
    return CheckResult(name=name, ok=ok, message=message)


def _sampled_files_cache_key(path: Path) -> str:
    return path.name


def _load_sampled_files_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_sampled_files_cache(cache_path: Path, cache: dict) -> None:
    from src.data.shared.batches import atomic_write_text

    atomic_write_text(str(cache_path), json.dumps(cache))


def check_sampled_files(
    paths: list[Path],
    check_fn: Callable[[Path], CheckResult],
    *,
    cache_path: Path | None = None,
    sample_limit: int | None = None,
    name: str = "sampled_files",
) -> CheckResult:
    """Run `check_fn` over the `sample_limit` most-recently-modified `paths`.

    Files unchanged (same size/mtime) since the last run are skipped via an
    optional on-disk cache at `cache_path`, so repeated runs against a large,
    mostly-static raw archive only re-check what's new or changed.
    """
    if not paths:
        return CheckResult(name=name, ok=False, message="No files to sample.")

    try:
        sortable = sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError as exc:
        return CheckResult(name=name, ok=False, message=f"Could not stat files: {exc}")

    sample = sortable if sample_limit is None else sortable[:sample_limit]
    cache = _load_sampled_files_cache(cache_path) if cache_path is not None else {}
    cache_dirty = False

    failures: list[str] = []
    checked = 0
    for path in sample:
        try:
            stat_result = path.stat()
        except OSError as exc:
            failures.append(f"{path.name} ({exc})")
            continue

        key = _sampled_files_cache_key(path)
        cached = cache.get(key)
        if (
            cached is not None
            and cached.get("size") == stat_result.st_size
            and cached.get("mtime_ns") == stat_result.st_mtime_ns
        ):
            ok = cached["ok"]
        else:
            result = check_fn(path)
            ok = result.ok
            cache[key] = {"size": stat_result.st_size, "mtime_ns": stat_result.st_mtime_ns, "ok": ok}
            cache_dirty = True

        checked += 1
        if not ok:
            failures.append(path.name)

    if cache_dirty and cache_path is not None:
        _save_sampled_files_cache(cache_path, cache)

    ok = not failures
    message = f"{checked - len(failures)}/{checked} sampled files ok."
    if failures:
        message += f" Failed: {failures[:10]}."
    return CheckResult(name=name, ok=ok, message=message)


__all__ = [
    "CheckResult",
    "VerificationResult",
    "check_file_nonempty",
    "check_gpkg_layer_readable",
    "check_null_fraction",
    "check_raster_header_readable",
    "check_required_columns",
    "check_sampled_files",
    "check_value_range",
    "check_zip_integrity",
]
