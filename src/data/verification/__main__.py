import argparse
import logging

from .constants import SOURCES
from .core import Verification


logger = logging.getLogger(__name__)

STATUS_STYLES = {
    "verified": "green",
    "outstanding": "yellow",
    "failed": "red",
    "not_present_locally": "dim",
    "not_applicable": "dim",
}


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging for standalone verification execution."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def configure_parser(parser, include_action=True):
    """Add verification CLI arguments to ``parser``."""
    if include_action:
        parser.add_argument("action", choices=["verify", "summary"])
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--source", default=None, choices=list(SOURCES))
    parser.add_argument("--force", action="store_true")
    return parser


def _truncate_timestamp(value: str) -> str:
    """Drop microseconds/timezone noise from an ISO8601 `verified_at` value."""
    from datetime import datetime

    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return value
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def _fetched_display(report) -> str:
    completeness = report.fetch_completeness or {}
    present = completeness.get("present")
    expected = completeness.get("expected")
    if expected is not None and present == expected:
        return "complete"
    if expected is not None:
        return f"{present}/{expected}"
    return f"{present}/?"


def _latest_stage(name, report) -> str:
    """Which processing phase this source has most recently reached.

    Sources with `phases` in `src.cli.SOURCE_REGISTRY` (climate, land_cover,
    sensor_data) split preprocessing into extract -> aggregate; everything
    else is a single preprocess step.
    """
    from src.cli import SOURCE_REGISTRY

    spec = SOURCE_REGISTRY.get(name)
    phases = spec["phases"] if spec else None
    fetch_present = (report.fetch_completeness or {}).get("present") or 0

    if phases is None:
        return "preprocess" if (report.outputs_present or fetch_present) else "-"
    extract_stage, aggregate_stage = phases
    if report.outputs_present:
        return aggregate_stage
    if fetch_present:
        return extract_stage
    return "-"


def _build_table(title, reports, *, source_column, fetched_column, show_stage, caption=None):
    from rich.table import Table

    from .sources import SOURCE_ADAPTERS

    table = Table(title=title, caption=caption)
    # no_wrap on the identifying columns: when the table is wider than the
    # console, rich shrinks wrap-able columns first -- without this,
    # source/status names themselves got truncated instead of wrapping.
    table.add_column(source_column, no_wrap=True)
    table.add_column("Fetch method", no_wrap=True)
    table.add_column(fetched_column)
    table.add_column("Fetch Status", no_wrap=True)
    table.add_column("Fetched checks passed")
    if show_stage:
        table.add_column("Latest Preprocess Stage", no_wrap=True)
    table.add_column("Preprocess Status", no_wrap=True)
    table.add_column("Checks passed")
    table.add_column("Last verified")

    for name, report in reports.items():
        style = STATUS_STYLES.get(report.status, "")
        checks = report.checks or []
        passed = sum(1 for check in checks if check.get("ok"))
        checks_display = f"{passed}/{len(checks)}" if checks else "-"
        status_display = f"[{style}]{report.status}[/{style}]" if style else report.status

        fetch_style = STATUS_STYLES.get(report.fetch_status, "")
        fetched_checks = report.fetched_checks or []
        fetched_passed = sum(1 for check in fetched_checks if check.get("ok"))
        fetched_checks_display = f"{fetched_passed}/{len(fetched_checks)}" if fetched_checks else "-"
        fetch_status_display = f"[{fetch_style}]{report.fetch_status}[/{fetch_style}]" if fetch_style else report.fetch_status

        fetch_method = SOURCE_ADAPTERS[name].fetch_method
        row = [name, fetch_method, _fetched_display(report), fetch_status_display, fetched_checks_display]
        if show_stage:
            row.append(_latest_stage(name, report))
        row += [status_display, checks_display, _truncate_timestamp(report.verified_at)]
        table.add_row(*row)

    return table


def _terminal_width() -> int:
    """Real terminal width when attached to one, else a stable fallback.

    Only trust the detected width when stdout is an actual TTY: a `COLUMNS`
    env var can be (and in some CI/test harnesses is) inherited by piped or
    captured processes with no real terminal behind them, which would
    otherwise squeeze the table down to an unreadable width for no reason.
    """
    import shutil
    import sys

    if sys.stdout.isatty():
        return shutil.get_terminal_size(fallback=(140, 24)).columns
    return 140


def _render_table(reports):
    from rich.console import Console

    # assembly isn't a fetch source -- it's the final join step consuming the
    # other 7, and its "fetched" number means "declared upstream source files
    # present" (setup/assembly_datasets.yaml), not raw ingestion. Keeping it
    # in the same table as e.g. climate's GRIB-file completeness conflates
    # two different things, so it gets its own table below.
    assembly_reports = {name: report for name, report in reports.items() if name == "assembly"}
    other_reports = {name: report for name, report in reports.items() if name != "assembly"}

    console = Console(width=_terminal_width())
    if other_reports:
        console.print(
            _build_table(
                "Data Verification Summary",
                other_reports,
                source_column="Source",
                fetched_column="Fetched",
                show_stage=True,
            )
        )
    if assembly_reports:
        console.print(
            _build_table(
                "Assembly (final join step)",
                assembly_reports,
                source_column="Dataset group",
                fetched_column="Upstream sources present/expected",
                show_stage=False,
                caption="Rolls up completeness of setup/assembly_datasets.yaml's declared upstream source paths, not raw fetched files.",
            )
        )


def run(args):
    """Execute the requested verification action for parsed ``args``."""
    agent = Verification(root_dir=args.root_dir)
    if args.action == "verify":
        reports = agent.verify(source=args.source, force=args.force)
        for name, report in reports.items():
            logger.info("%s: %s (fingerprint=%s)", name, report.status, report.fingerprint)
        return reports

    reports = agent.summary(source=args.source, force=args.force)
    _render_table(reports)
    return reports


def main():
    parser = argparse.ArgumentParser(description="Run data-verification workflows")
    configure_parser(parser)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for standalone execution",
    )

    args = parser.parse_args()
    configure_logging(args.log_level)
    logger.info("Starting standalone verification %s", args.action)
    run(args)
    logger.info("Completed standalone verification %s", args.action)


if __name__ == "__main__":
    main()
