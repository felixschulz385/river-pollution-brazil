"""Atomic shard checkpoints and canonical sensor-analysis output merging."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import pandas as pd

from .results import SensorAnalysisRun
from ..settings import SensorAnalysisSettings


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def input_fingerprint(settings: SensorAnalysisSettings) -> str:
    """Fingerprint settings and input metadata without reading input contents."""
    inputs = [
        settings.sensor_data_path,
        settings.land_cover_path,
        settings.climate_data_path,
        settings.transformations_path,
        settings.trenches_path,
    ]
    input_metadata = []
    for path in inputs:
        if path is None:
            input_metadata.append(None)
            continue
        try:
            stat = path.stat()
            input_metadata.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        except OSError:
            input_metadata.append({"path": str(path), "missing": True})
    payload = json.dumps(
        {"settings": _jsonable(asdict(settings)), "inputs": input_metadata},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def work_root(run_dir: Path, fingerprint: str) -> Path:
    return run_dir / "_work" / fingerprint


def shard_dir(run_dir: Path, fingerprint: str, shard_index: int) -> Path:
    return work_root(run_dir, fingerprint) / f"shard-{shard_index:05d}"


def _atomic_directory(target: Path, writer) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    temporary.mkdir()
    try:
        writer(temporary)
        os.replace(temporary, target)
    except Exception:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()
        raise


def write_chunk(
    run_dir: Path,
    fingerprint: str,
    shard_index: int,
    chunk_index: int,
    results: pd.DataFrame,
    manifest: pd.DataFrame,
) -> Path:
    """Persist one immutable results/manifest chunk atomically."""
    target = shard_dir(run_dir, fingerprint, shard_index) / f"chunk-{chunk_index:05d}"
    if target.exists():
        raise FileExistsError(f"Checkpoint chunk already exists: {target}")

    def writer(directory: Path) -> None:
        if not results.empty:
            results.to_parquet(directory / "results.parquet", index=False)
        manifest.to_parquet(directory / "manifest.parquet", index=False)

    _atomic_directory(target, writer)
    return target


def write_shard_metadata(
    run_dir: Path,
    fingerprint: str,
    shard_index: int,
    shard_count: int,
    expected_spec_ids: list[str],
    settings: SensorAnalysisSettings,
) -> None:
    directory = shard_dir(run_dir, fingerprint, shard_index)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "metadata.json"
    payload = {
        "fingerprint": fingerprint,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "expected_spec_ids": sorted(expected_spec_ids),
        "settings": _jsonable(asdict(settings)),
    }
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != payload:
            raise ValueError(f"Existing shard metadata does not match this run: {directory}")
        return
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True))
    os.replace(temporary, path)


def completed_spec_ids(run_dir: Path, fingerprint: str, shard_index: int) -> set[str]:
    directory = shard_dir(run_dir, fingerprint, shard_index)
    completed: set[str] = set()
    for chunk in sorted(directory.glob("chunk-*")):
        manifest_path = chunk / "manifest.parquet"
        if manifest_path.exists():
            completed.update(pd.read_parquet(manifest_path, columns=["spec_id"])["spec_id"].astype(str))
    return completed


def mark_shard_complete(run_dir: Path, fingerprint: str, shard_index: int) -> None:
    path = shard_dir(run_dir, fingerprint, shard_index) / "_SUCCESS"
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text("ok\n")
    os.replace(temporary, path)


def latest_fingerprint(run_dir: Path) -> str | None:
    """Return the most-recently-modified checkpoint fingerprint, if any."""
    candidates = [path for path in (run_dir / "_work").glob("*") if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime_ns).name


def shard_progress(run_dir: Path, fingerprint: str) -> list[dict]:
    """Summarize per-shard completion for an in-progress checkpoint run."""
    progress = []
    for directory in sorted(work_root(run_dir, fingerprint).glob("shard-*")):
        metadata_path = directory / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text())
        progress.append(
            {
                "shard_index": metadata["shard_index"],
                "shard_count": metadata["shard_count"],
                "complete": (directory / "_SUCCESS").exists(),
                "specs_expected": len(metadata["expected_spec_ids"]),
                "specs_done": len(completed_spec_ids(run_dir, fingerprint, metadata["shard_index"])),
            }
        )
    return progress


def load_partial_results(run_dir: Path, fingerprint: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate checkpointed chunks for a fingerprint without requiring completion."""
    manifests: list[pd.DataFrame] = []
    results: list[pd.DataFrame] = []
    for directory in sorted(work_root(run_dir, fingerprint).glob("shard-*")):
        for chunk in sorted(directory.glob("chunk-*")):
            manifest_path = chunk / "manifest.parquet"
            if manifest_path.exists():
                manifests.append(pd.read_parquet(manifest_path))
            result_path = chunk / "results.parquet"
            if result_path.exists():
                results.append(pd.read_parquet(result_path))
    manifest = pd.concat(manifests, ignore_index=True) if manifests else pd.DataFrame()
    result_frame = (
        pd.concat(results, ignore_index=True)
        if results
        else pd.DataFrame({"spec_id": pd.Series(dtype="string")})
    )
    return result_frame, manifest


def merge_shards(
    run_dir: Path,
    fingerprint: str | None,
    expected_shards: int,
    settings: SensorAnalysisSettings,
) -> SensorAnalysisRun:
    """Validate all completed shards and atomically publish canonical outputs."""
    if fingerprint is None:
        candidates = [path for path in (run_dir / "_work").glob("*") if path.is_dir()]
        if not candidates:
            raise ValueError("No shard checkpoint runs were found.")
        fingerprint = max(candidates, key=lambda path: path.stat().st_mtime_ns).name
    manifests: list[pd.DataFrame] = []
    results: list[pd.DataFrame] = []
    expected_ids: set[str] = set()
    settings_payload: dict[str, object] | None = None
    for shard_index in range(expected_shards):
        directory = shard_dir(run_dir, fingerprint, shard_index)
        metadata_path = directory / "metadata.json"
        if not (directory / "_SUCCESS").exists() or not metadata_path.exists():
            raise ValueError(f"Shard {shard_index} is incomplete.")
        metadata = json.loads(metadata_path.read_text())
        if metadata["fingerprint"] != fingerprint or metadata["shard_count"] != expected_shards:
            raise ValueError(f"Shard {shard_index} belongs to a different run.")
        if settings_payload is None:
            settings_payload = metadata["settings"]
        elif settings_payload != metadata["settings"]:
            raise ValueError("Shard settings differ within the same run.")
        expected_ids.update(metadata["expected_spec_ids"])
        for chunk in sorted(directory.glob("chunk-*")):
            manifests.append(pd.read_parquet(chunk / "manifest.parquet"))
            result_path = chunk / "results.parquet"
            if result_path.exists():
                results.append(pd.read_parquet(result_path))
    manifest = pd.concat(manifests, ignore_index=True) if manifests else pd.DataFrame()
    if manifest.empty or "spec_id" not in manifest:
        raise ValueError("No checkpoint manifest records were found.")
    actual_ids = set(manifest["spec_id"].astype(str))
    if actual_ids != expected_ids or manifest["spec_id"].duplicated().any():
        raise ValueError("Shard checkpoints are missing or duplicate model specifications.")
    result_frame = (
        pd.concat(results, ignore_index=True)
        if results
        else pd.DataFrame({"spec_id": pd.Series(dtype="string")})
    )
    summary = {
        "models_total": int(len(manifest)),
        "models_succeeded": int(manifest["status"].eq("ok").sum()),
        "models_failed": int(manifest["status"].eq("failed").sum()),
        "run_fingerprint": fingerprint,
        "shards": expected_shards,
    }
    run = SensorAnalysisRun(result_frame, manifest, summary, run_dir)
    temporary = run_dir / f".publish-{uuid4().hex}"
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        result_frame.to_parquet(temporary / "results.parquet", index=False)
        manifest.to_parquet(temporary / "manifest.parquet", index=False)
        pd.DataFrame({"key": list(summary), "value": [json.dumps(value) for value in summary.values()]}).to_parquet(
            temporary / "summary.parquet", index=False
        )
        payload = settings_payload or _jsonable(asdict(settings))
        pd.DataFrame({"key": list(payload), "value": [json.dumps(value, default=str) for value in payload.values()]}).to_parquet(
            temporary / "settings.parquet", index=False
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        for name in ("results.parquet", "manifest.parquet", "summary.parquet", "settings.parquet"):
            os.replace(temporary / name, run_dir / name)
    finally:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()
    return run


__all__ = [
    "completed_spec_ids",
    "input_fingerprint",
    "latest_fingerprint",
    "load_partial_results",
    "mark_shard_complete",
    "merge_shards",
    "shard_dir",
    "shard_progress",
    "write_chunk",
    "write_shard_metadata",
]
