"""Cheap metadata-only fingerprinting for the verification cache.

Generalizes `input_fingerprint()` from `src/analysis/sensor_data/checkpoints.py`:
hashes `(path, size, mtime_ns)` triples rather than file contents so it stays
fast against large intermediates (climate's zarr store, hundreds of GRIB
files) while still changing whenever the underlying data actually changes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def compute_fingerprint(paths: Iterable[Path]) -> str:
    """Hash `(path, size, mtime_ns)` metadata for `paths` without reading contents."""
    metadata = []
    for path in paths:
        path = Path(path)
        try:
            stat = path.stat()
            metadata.append(
                {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            )
        except OSError:
            metadata.append({"path": str(path), "missing": True})
    payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = ["compute_fingerprint"]
