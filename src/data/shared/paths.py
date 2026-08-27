"""Shared helpers for building per-source data paths.

Every source under ``src/data/sources/`` stores its files under
``data/<source>/{raw,processed[/<stage>],auxiliary}``. These helpers centralize
that convention so sources don't each re-derive it independently.
"""

from __future__ import annotations

from pathlib import Path


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
