"""Shared batch-manifest storage helpers for resumable data downloads."""

from __future__ import annotations

import json
import os
from typing import Iterable


def table_raw_dir(root_dir: str, *parts: str) -> str:
    path = os.path.join(root_dir, "data", *parts)
    os.makedirs(path, exist_ok=True)
    return path


def batch_table_dir(root_dir: str, dataset_name: str, table_name: str) -> str:
    return table_raw_dir(root_dir, dataset_name, "raw", table_name)


def batch_output_dir(root_dir: str, dataset_name: str, table_name: str) -> str:
    path = os.path.join(batch_table_dir(root_dir, dataset_name, table_name), "batches")
    os.makedirs(path, exist_ok=True)
    return path


def manifest_path(root_dir: str, dataset_name: str, table_name: str) -> str:
    return os.path.join(batch_table_dir(root_dir, dataset_name, table_name), "manifest.jsonl")


def batch_output_path(root_dir: str, dataset_name: str, table_name: str, batch_id: str) -> str:
    return os.path.join(batch_output_dir(root_dir, dataset_name, table_name), f"{batch_id}.parquet")


def load_manifest(root_dir: str, dataset_name: str, table_name: str) -> list[dict]:
    path = manifest_path(root_dir, dataset_name, table_name)
    if not os.path.exists(path):
        return []

    entries = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def write_manifest(root_dir: str, dataset_name: str, table_name: str, entries: Iterable[dict]) -> str:
    path = manifest_path(root_dir, dataset_name, table_name)
    with open(path, "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=True, sort_keys=True))
            handle.write("\n")
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
    write_manifest(root_dir, dataset_name, table_name, entries)


def completed_batch_paths(root_dir: str, dataset_name: str, table_name: str) -> list[str]:
    return [
        entry["raw_path"]
        for entry in load_manifest(root_dir, dataset_name, table_name)
        if entry.get("status") == "completed"
        and entry.get("raw_path")
        and os.path.exists(entry["raw_path"])
    ]
