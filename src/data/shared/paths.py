"""Shared helpers for building per-source data paths.

Every source under ``src/data/sources/`` stores its files under
``data/<source>/{raw,processed[/<stage>],auxiliary}``. These helpers centralize
that convention so sources don't each re-derive it independently.
"""

from __future__ import annotations

import os
from pathlib import Path


# Bulky, disposable working files (DuckDB spill, Parquet part files, ...) go
# here rather than the system temp dir. On the cluster a job's cwd is the
# project directory and ``scratch_nobackup`` there is the large, fast,
# un-backed-up scratch area -- ``/tmp`` (where ``tempfile`` defaults) is small
# and would OOM/ENOSPC on these jobs.
SCRATCH_DIR_NAME = "scratch_nobackup"
SCRATCH_DIR_ENV_VAR = "RIVER_POLLUTION_SCRATCH_DIR"


def scratch_root(root_dir: str | Path = ".") -> Path:
    """Return (creating it if needed) the base directory for scratch working files.

    Defaults to ``<root_dir>/scratch_nobackup``; override with the
    ``RIVER_POLLUTION_SCRATCH_DIR`` environment variable (e.g. to point at a
    node-local ``$TMPDIR``). Pass the resulting path as ``dir=`` to
    ``tempfile.mkdtemp`` / ``tempfile.TemporaryDirectory``.

    Always returns an absolute path: callers hand it to DuckDB
    (``PRAGMA temp_directory``) and to ``read_parquet`` globs, which resolve
    against the process CWD, and long-running jobs may ``os.chdir`` mid-run.
    """
    override = os.environ.get(SCRATCH_DIR_ENV_VAR)
    base = Path(override).expanduser() if override else Path(root_dir) / SCRATCH_DIR_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base.resolve()


def source_root(root_dir: str | Path, source: str) -> Path:
    """Root directory for a source's data, e.g. ``data/climate``."""
    return Path(root_dir) / "data" / source


def raw_dir(root_dir: str | Path, source: str) -> Path:
    """Raw/unprocessed input directory for a source."""
    return source_root(root_dir, source) / "raw"


def processed_dir(root_dir: str | Path, source: str, stage: str | None = None) -> Path:
    """Processed-output directory for a source, optionally scoped to a stage.

    ``stage`` should match the source's own processing phase names (e.g.
    ``"extract"``/``"aggregate"`` for sources with `phases` in
    ``src.cli.SOURCE_REGISTRY``); omit it for single-stage sources.
    """
    base = source_root(root_dir, source) / "processed"
    return base / stage if stage else base


def auxiliary_dir(root_dir: str | Path, source: str) -> Path:
    """Static/reference-data directory for a source (e.g. lookup tables)."""
    return source_root(root_dir, source) / "auxiliary"
