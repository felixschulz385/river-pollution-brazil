"""Repository-level CLI entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from code.analysis.cli import main as analysis_main  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""
    parser = argparse.ArgumentParser(description="Project CLI")
    subparsers = parser.add_subparsers(dest="module", required=True)

    analysis_parser = subparsers.add_parser("analysis", help="Run analysis workflows")
    analysis_parser.add_argument(
        "analysis_module",
        choices=["sensor-data"],
        help="Analysis workflow to run.",
    )
    analysis_parser.add_argument(
        "analysis_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the selected analysis module.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the repository CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.module == "analysis":
        if args.analysis_module != "sensor-data":
            parser.error(f"Unsupported analysis module: {args.analysis_module}")
        return analysis_main(args.analysis_args)
    parser.error(f"Unknown module: {args.module}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
