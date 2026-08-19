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


def _build_table(title, reports, *, source_column, fetched_column, caption=None):
    from rich.table import Table

    from .sources import SOURCE_ADAPTERS

    table = Table(title=title, caption=caption)
    # no_wrap on the identifying columns: when the table is wider than the
    # console (common for non-interactive/piped output, where rich falls
    # back to an 80-column default), rich shrinks wrap-able columns first --
    # without this, source/status names themselves got truncated instead.
    table.add_column(source_column, no_wrap=True)
    table.add_column("Fetch method", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column(fetched_column)
    table.add_column("Checks passed")
    table.add_column("Last verified")

    for name, report in reports.items():
        style = STATUS_STYLES.get(report.status, "")
        completeness = report.fetch_completeness or {}
        present = completeness.get("present")
        expected = completeness.get("expected")
        fetched_display = f"{present}/{expected}" if expected is not None else f"{present}/?"
        checks = report.checks or []
        passed = sum(1 for check in checks if check.get("ok"))
        checks_display = f"{passed}/{len(checks)}" if checks else "-"
        status_display = f"[{style}]{report.status}[/{style}]" if style else report.status
        fetch_method = SOURCE_ADAPTERS[name].fetch_method
        table.add_row(name, fetch_method, status_display, fetched_display, checks_display, report.verified_at)

    return table


def _render_table(reports):
    from rich.console import Console

    # assembly isn't a fetch source -- it's the final join step consuming the
    # other 7, and its "fetched" number means "declared upstream source files
    # present" (setup/assembly_datasets.yaml), not raw ingestion. Keeping it
    # in the same table as e.g. climate's GRIB-file completeness conflates
    # two different things, so it gets its own table below.
    assembly_reports = {name: report for name, report in reports.items() if name == "assembly"}
    other_reports = {name: report for name, report in reports.items() if name != "assembly"}

    # rich falls back to an 80-column default when stdout isn't a TTY (e.g.
    # piped/captured output); explicit min width keeps six columns readable.
    console = Console(width=max(Console().size.width, 140))
    if other_reports:
        console.print(
            _build_table(
                "Data Verification Summary",
                other_reports,
                source_column="Source",
                fetched_column="Fetched (present/expected)",
            )
        )
    if assembly_reports:
        console.print(
            _build_table(
                "Assembly (final join step)",
                assembly_reports,
                source_column="Dataset group",
                fetched_column="Upstream sources present/expected",
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
