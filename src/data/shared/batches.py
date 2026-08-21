"""Shared batch-manifest storage helpers for resumable data downloads."""

from __future__ import annotations

import json
import logging
import os
from typing import Iterable

logger = logging.getLogger(__name__)


def table_raw_dir(root_dir: str, *parts: str) -> str:
    return os.path.join(root_dir, "data", *parts)


def batch_table_dir(root_dir: str, dataset_name: str, table_name: str) -> str:
    return table_raw_dir(root_dir, dataset_name, "raw", table_name)


def batch_output_dir(root_dir: str, dataset_name: str, table_name: str) -> str:
    return os.path.join(batch_table_dir(root_dir, dataset_name, table_name), "batches")


def manifest_path(root_dir: str, dataset_name: str, table_name: str) -> str:
    return os.path.join(batch_table_dir(root_dir, dataset_name, table_name), "manifest.jsonl")


def batch_output_path(
    root_dir: str,
    dataset_name: str,
    table_name: str,
    batch_id: str,
    suffix: str = ".parquet",
) -> str:
    return os.path.join(batch_output_dir(root_dir, dataset_name, table_name), f"{batch_id}{suffix}")


def atomic_write_text(path: str, text: str, encoding: str = "utf-8") -> None:
    """Write `text` to `path` via temp-file + `os.replace`, so a crash or
    SIGKILL mid-write can never leave a truncated/corrupt file behind --
    readers always see either the old complete file or the new one."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temp_path = f"{path}.tmp-{os.getpid()}"
    with open(temp_path, "w", encoding=encoding) as handle:
        handle.write(text)
    os.replace(temp_path, path)


def atomic_write_bytes(path: str, data: bytes) -> None:
    """Binary counterpart to `atomic_write_text`."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temp_path = f"{path}.tmp-{os.getpid()}"
    with open(temp_path, "wb") as handle:
        handle.write(data)
    os.replace(temp_path, path)


def load_manifest(root_dir: str, dataset_name: str, table_name: str) -> list[dict]:
    path = manifest_path(root_dir, dataset_name, table_name)
    if not os.path.exists(path):
        return []

    entries = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                # A truncated/corrupt line (e.g. the process was killed mid-write
                # by an old, non-atomic `write_manifest`) shouldn't crash every
                # future resumed run -- skip it and let the batch it belonged to
                # be re-planned instead.
                logger.warning(
                    "Skipping malformed manifest line %s in %s", line_number, path
                )
    return entries


def write_manifest(root_dir: str, dataset_name: str, table_name: str, entries: Iterable[dict]) -> str:
    path = manifest_path(root_dir, dataset_name, table_name)
    text = "".join(json.dumps(entry, ensure_ascii=True, sort_keys=True) + "\n" for entry in entries)
    atomic_write_text(path, text)
    return path


def initialize_manifest(
    root_dir: str,
    dataset_name: str,
    table_name: str,
    planned_entries: list[dict],
) -> list[dict]:
    existing_entries = {
        entry["batch_id"]: entry for entry in load_manifest(root_dir, dataset_name, table_name)
    }
    merged_entries = []
    for planned_entry in planned_entries:
        existing_entry = existing_entries.get(planned_entry["batch_id"], {})
        merged_entry = planned_entry.copy()
        existing_status = existing_entry.get("status")
        existing_raw_path = existing_entry.get("raw_path")
        if existing_status == "completed" and existing_raw_path and os.path.exists(existing_raw_path):
            merged_entry["status"] = "completed"
            merged_entry["raw_path"] = existing_raw_path
            merged_entry["error"] = None
        elif existing_status == "skipped":
            merged_entry["status"] = "skipped"
            merged_entry["raw_path"] = existing_entry.get("raw_path", planned_entry["raw_path"])
            merged_entry["error"] = existing_entry.get("error")
        else:
            merged_entry["status"] = "pending"
            merged_entry["raw_path"] = planned_entry["raw_path"]
            merged_entry["error"] = existing_entry.get("error")
        merged_entries.append(merged_entry)
    write_manifest(root_dir, dataset_name, table_name, merged_entries)
    return merged_entries


def update_manifest_entry(
    root_dir: str,
    dataset_name: str,
    table_name: str,
    entries: list[dict],
    batch_id: str,
    **updates,
) -> None:
    for entry in entries:
        if entry["batch_id"] == batch_id:
            entry.update(updates)
            break
    else:
        raise ValueError(
            f"No manifest entry with batch_id={batch_id!r} in {dataset_name}/{table_name} "
            "(stale reference after a manifest re-plan?)."
        )
    write_manifest(root_dir, dataset_name, table_name, entries)


def completed_batch_paths(root_dir: str, dataset_name: str, table_name: str) -> list[str]:
    return [
        entry["raw_path"]
        for entry in load_manifest(root_dir, dataset_name, table_name)
        if entry.get("status") == "completed"
        and entry.get("raw_path")
        and os.path.exists(entry["raw_path"])
    ]
