#!/usr/bin/env python3

"""Compatibility wrapper for legacy `src/data/cli.py` invocations."""

from __future__ import annotations

import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cli import main as repository_main  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Forward legacy data CLI invocations to the repository CLI."""
    forwarded_argv = ["data", *(argv or sys.argv[1:])]
    return repository_main(forwarded_argv)


if __name__ == "__main__":
    raise SystemExit(main())
