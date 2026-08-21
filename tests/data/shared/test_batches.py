from __future__ import annotations

import os

import pytest

from src.data.shared.batches import (
    load_manifest,
    manifest_path,
    update_manifest_entry,
    write_manifest,
)


def test_write_manifest_is_atomic_and_leaves_no_temp_file(tmp_path) -> None:
    entries = [{"batch_id": "b1", "status": "done"}, {"batch_id": "b2", "status": "pending"}]

    path = write_manifest(str(tmp_path), "dataset", "table", entries)

    assert os.path.exists(path)
    assert not any(name.endswith(".tmp") or ".tmp-" in name for name in os.listdir(os.path.dirname(path)))
    assert load_manifest(str(tmp_path), "dataset", "table") == entries


def test_load_manifest_skips_malformed_line_instead_of_raising(tmp_path) -> None:
    path = manifest_path(str(tmp_path), "dataset", "table")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write('{"batch_id": "b1", "status": "done"}\n')
        handle.write('{"batch_id": "b2", "stat')  # truncated, e.g. crash mid-write
        handle.write("\n")
        handle.write('{"batch_id": "b3", "status": "pending"}\n')

    entries = load_manifest(str(tmp_path), "dataset", "table")

    assert entries == [
        {"batch_id": "b1", "status": "done"},
        {"batch_id": "b3", "status": "pending"},
    ]


def test_update_manifest_entry_raises_for_unknown_batch_id(tmp_path) -> None:
    entries = [{"batch_id": "b1", "status": "pending"}]

    with pytest.raises(ValueError, match="b-does-not-exist"):
        update_manifest_entry(
            str(tmp_path), "dataset", "table", entries, "b-does-not-exist", status="completed"
        )
