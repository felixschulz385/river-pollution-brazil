"""Repository-level CLI entrypoint."""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


logger = logging.getLogger(__name__)

# Each entry describes one data source's package and which verbs it
# supports. `phases` is None for sources with a single preprocessing step,
# or a tuple of ("extract", "aggregate") for sources whose preprocessing is
# split into raw extraction and a roll-up into sensor/ADM2 upstream panels
# (mapped onto that source's own `fetch`/`preprocess`/`assemble` actions).
SOURCE_REGISTRY = {
    "gadm": dict(package="src.data.sources.gadm", fetch=False, phases=None),
    "health": dict(package="src.data.sources.health", fetch=True, phases=None),
    "climate": dict(package="src.data.sources.climate", fetch=True, phases=("extract", "aggregate")),
    "sensor_data": dict(package="src.data.sources.sensor_data", fetch=True, phases=("extract", "aggregate")),
    "land_cover": dict(package="src.data.sources.land_cover", fetch=True, phases=("extract", "aggregate")),
    "population": dict(package="src.data.sources.population", fetch=True, phases=None),
    "river_network": dict(package="src.data.sources.river_network", fetch=False, phases=None),
    "biomes": dict(package="src.data.sources.biomes", fetch=True, phases=None),
}


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once for CLI and batch execution."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""
    from src.data.assembly.__main__ import configure_parser as configure_assembly_parser
    from src.data.verification.__main__ import configure_parser as configure_verification_parser

    parser = argparse.ArgumentParser(
        description="Project CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for CLI execution (default: INFO).",
    )
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

    data_parser = subparsers.add_parser("data", help="Run data workflows")
    data_subparsers = data_parser.add_subparsers(dest="data_verb", required=True)

    summary_parser = data_subparsers.add_parser("summary", help="Summarize pipeline completeness")
    summary_parser.set_defaults(data_verb="summary")
    configure_verification_parser(summary_parser, include_action=False)

    verify_parser = data_subparsers.add_parser("verify", help="Verify preprocessed outputs")
    verify_parser.set_defaults(data_verb="verify")
    configure_verification_parser(verify_parser, include_action=False)

    fetch_parser = data_subparsers.add_parser(
        "fetch",
        help="Fetch raw data for a source",
        epilog="--source and --slurm must precede source-specific flags.",
    )
    fetch_parser.set_defaults(data_verb="fetch")
    fetch_parser.add_argument("--source", required=True, choices=sorted(SOURCE_REGISTRY))
    fetch_parser.add_argument("--slurm", action="store_true", help="Submit as a Slurm job instead of running locally.")
    _add_dependency_flags(fetch_parser)

    preprocess_parser = data_subparsers.add_parser(
        "preprocess",
        help="Preprocess a source's raw data",
        epilog="--source, --phase, and --slurm must precede source-specific flags.",
    )
    preprocess_parser.set_defaults(data_verb="preprocess")
    preprocess_parser.add_argument("--source", required=True, choices=sorted(SOURCE_REGISTRY))
    preprocess_parser.add_argument(
        "--phase",
        choices=["extract", "aggregate"],
        default=None,
        help="For sources with a two-stage preprocess: raw extraction vs. roll-up into upstream panels.",
    )
    preprocess_parser.add_argument("--slurm", action="store_true", help="Submit as a Slurm job instead of running locally.")
    _add_dependency_flags(preprocess_parser)

    assemble_parser = data_subparsers.add_parser(
        "assemble", help="Join preprocessed sources into an analysis-ready dataset"
    )
    assemble_parser.set_defaults(data_verb="assemble")
    configure_assembly_parser(assemble_parser, include_action=False)
    assemble_parser.add_argument("--slurm", action="store_true", help="Submit as a Slurm job instead of running locally.")
    _add_dependency_flags(assemble_parser)

    return parser


def _add_dependency_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--chain",
        action="store_true",
        help=(
            "Auto-run any unmet prerequisite fetch/preprocess steps first, in "
            "dependency order, before this one. Prerequisite steps always run "
            "locally, even when combined with --slurm (only the originally "
            "requested step is submitted to Slurm)."
        ),
    )
    parser.add_argument(
        "--skip-dependency-check",
        action="store_true",
        help="Bypass the prerequisite check and run even if prerequisites aren't satisfied.",
    )


def _resolve_action(source: str, verb: str, phase: str | None) -> str:
    """Map a (source, verb, phase) combo onto that source module's `action` value."""
    spec = SOURCE_REGISTRY[source]

    if verb == "fetch":
        if not spec["fetch"]:
            raise ValueError(f"'{source}' has no automated fetch step (raw data is placed manually).")
        if phase is not None:
            raise ValueError("--phase is only valid for 'preprocess'.")
        return "fetch"

    # verb == "preprocess"
    phases = spec["phases"]
    if phases is None:
        if phase is not None:
            raise ValueError(f"'{source}' has a single preprocessing step; --phase is not supported.")
        return "preprocess"

    if phase is None:
        raise ValueError(f"'{source}' requires --phase, one of: {', '.join(phases)}.")
    return "preprocess" if phase == "extract" else "assemble"


def _slurm_job_key(source: str | None, verb: str, phase: str | None) -> str:
    if verb == "assemble" and source is None:
        return "assemble"
    key = f"{source}.{verb}"
    return f"{key}.{phase}" if phase else key


def _submit_slurm(job_key: str, log_dir_name: str) -> int:
    from src.data.shared.slurm import SlurmJobSpecError, load_job_spec, render_sbatch_script, submit_job

    command_argv = [arg for arg in sys.argv[1:] if arg != "--slurm"]
    try:
        spec = load_job_spec(job_key)
    except SlurmJobSpecError as exc:
        logger.error("%s", exc)
        return 1

    script = render_sbatch_script(spec, command_argv, log_dir=f"log/{log_dir_name}")
    job_id = submit_job(script)
    logger.info("Submitted Slurm job %s for '%s' (job-name=%s)", job_id, job_key, spec["job_name"])
    print(job_id)
    return 0


def _dispatch_module(source: str, action: str, module_args: list[str]) -> None:
    """Shared "import the source's __main__, parse its own flags, run()" core,
    reused both for the directly-requested step (with the user's own
    remainder flags) and for chained prerequisite steps (with just
    --root-dir)."""
    module = importlib.import_module(f"{SOURCE_REGISTRY[source]['package']}.__main__")
    source_parser = argparse.ArgumentParser()
    module.configure_parser(source_parser, include_action=False)
    source_args = source_parser.parse_args(module_args)
    source_args.action = action
    module.run(source_args)


def _dispatch_source_step(source: str, verb: str, root_dir: str) -> None:
    """Run one chained prerequisite step locally with default source-module
    flags (only --root-dir carried through). For a phased source's
    "preprocess" step, runs both phases in sequence: "preprocessed" (for
    gating purposes) means the source's final output artifact exists, which
    for a phased source only appears once its aggregate phase has run."""
    phases = SOURCE_REGISTRY[source]["phases"]
    step_phases = phases if (verb == "preprocess" and phases is not None) else (None,)
    for phase in step_phases:
        action = _resolve_action(source, verb, phase)
        _dispatch_module(source, action, ["--root-dir", root_dir])
        logger.info("Chain: completed %s %s (%s) successfully", source, verb, action)


def _extract_root_dir(remainder: list[str]) -> str:
    """Pull --root-dir out of `data fetch`/`data preprocess`'s forwarded
    remainder args (each source's own parser defines it, e.g.
    `land_cover/__main__.py`), defaulting to "." like every source module
    already does, so the dependency check looks at the same root the actual
    run will use."""
    for index, token in enumerate(remainder):
        if token == "--root-dir" and index + 1 < len(remainder):
            return remainder[index + 1]
        if token.startswith("--root-dir="):
            return token.split("=", 1)[1]
    return "."


def _enforce_dependencies(source: str, verb: str, root_dir: str, *, chain: bool) -> bool:
    """Check `(source, verb)`'s prerequisites. Returns True if the caller
    should stop (hard-blocked). When `chain` is True, runs every unmet
    prerequisite step first (locally, regardless of --slurm) instead of
    blocking. Raises ValueError if the chain can't be resolved, e.g. a
    manual-placement source's (river_network's) raw data is missing."""
    from src.data.pipeline_dependencies import build_chain, unmet_prerequisites

    unmet = unmet_prerequisites(source, verb, root_dir=root_dir)
    if not unmet:
        return False

    if not chain:
        logger.error("Cannot %s '%s': prerequisites not satisfied.", verb, source)
        for problem in unmet:
            logger.error("  - %s", problem)
        logger.error(
            "Pass --chain to run these automatically first, or --skip-dependency-check to bypass this check."
        )
        return True

    chain_steps = build_chain(source, verb, root_dir=root_dir)
    for chain_source, chain_verb in chain_steps[:-1]:
        logger.info("Chain: running %s %s (prerequisite of %s %s)", chain_source, chain_verb, source, verb)
        _dispatch_source_step(chain_source, chain_verb, root_dir)
    return False


def _run_source_verb(args: argparse.Namespace) -> int:
    """Dispatch `data fetch`/`data preprocess` to the resolved source module."""
    source = args.source
    verb = args.data_verb
    phase = getattr(args, "phase", None)

    try:
        action = _resolve_action(source, verb, phase)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    if not args.skip_dependency_check:
        root_dir = _extract_root_dir(args.remainder)
        try:
            if _enforce_dependencies(source, verb, root_dir, chain=args.chain):
                return 1
        except ValueError as exc:
            logger.error("%s", exc)
            return 1

    if args.slurm:
        return _submit_slurm(_slurm_job_key(source, verb, phase), log_dir_name=f"{source}_{verb}")

    _dispatch_module(source, action, args.remainder)
    logger.info("Completed %s %s (%s) successfully", source, verb, action)
    return 0


def _run_assemble(args: argparse.Namespace) -> int:
    from src.data.assembly.__main__ import run as run_assembly
    from src.data.pipeline_dependencies import ASSEMBLY_NODE

    if not args.skip_dependency_check:
        root_dir = getattr(args, "root_dir", ".")
        try:
            if _enforce_dependencies(ASSEMBLY_NODE, "assemble", root_dir, chain=args.chain):
                return 1
        except ValueError as exc:
            logger.error("%s", exc)
            return 1

    if args.slurm:
        return _submit_slurm(_slurm_job_key(None, "assemble", None), log_dir_name="assemble")

    run_assembly(args)
    logger.info("Completed assemble successfully")
    return 0


def _run_verification(args: argparse.Namespace) -> int:
    from src.data.verification.__main__ import run as run_verification

    args.action = args.data_verb  # "summary" or "verify"
    run_verification(args)
    return 0


def _run_data_cli(args: argparse.Namespace) -> int:
    """Dispatch a parsed `data` command to the right module."""
    try:
        if args.data_verb in ("summary", "verify"):
            return _run_verification(args)
        if args.data_verb in ("fetch", "preprocess"):
            return _run_source_verb(args)
        if args.data_verb == "assemble":
            return _run_assemble(args)
        raise ValueError(f"Unknown data verb: {args.data_verb}")
    except Exception as exc:
        logger.error("Error running data %s: %s", args.data_verb, exc)
        logger.debug("Traceback:", exc_info=True)
        return 1


def _maybe_show_source_help(argv: list[str]) -> None:
    """If this is `data fetch|preprocess --source X --help`, show X's own
    flags (not just --source/--phase/--slurm) and exit. `--help` on the
    generic `fetch`/`preprocess` subparser can't see them since they're only
    known once --source is resolved.
    """
    if len(argv) < 2 or argv[0] != "data" or argv[1] not in ("fetch", "preprocess"):
        return
    if "--help" not in argv and "-h" not in argv:
        return
    if "--source" not in argv:
        return
    source = argv[argv.index("--source") + 1]
    if source not in SOURCE_REGISTRY:
        return

    module = importlib.import_module(f"{SOURCE_REGISTRY[source]['package']}.__main__")
    source_parser = argparse.ArgumentParser(
        prog=f"python -m src.cli data {argv[1]} --source {source}",
        description=f"'{argv[1]}' flags for source '{source}' (in addition to --source/--phase/--slurm).",
    )
    module.configure_parser(source_parser, include_action=False)
    source_parser.parse_args(["--help"])  # always exits


def main(argv: list[str] | None = None) -> int:
    """Run the repository CLI."""
    _maybe_show_source_help(list(argv if argv is not None else sys.argv[1:]))
    parser = build_parser()
    # `data fetch`/`data preprocess` forward unrecognized flags to their
    # source module's own parser; argparse.REMAINDER doesn't reliably capture
    # leading "--flag" tokens when it follows other optionals in the same
    # subparser, so we use parse_known_args instead and forward the leftovers
    # ourselves. Every other subcommand still rejects unknown arguments.
    args, extra = parser.parse_known_args(argv)
    if args.module == "data" and getattr(args, "data_verb", None) in ("fetch", "preprocess"):
        args.remainder = extra
    elif extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    configure_logging(args.log_level)

    if args.module == "analysis":
        if args.analysis_module != "sensor-data":
            parser.error(f"Unsupported analysis module: {args.analysis_module}")
        from src.analysis.cli import main as analysis_main

        return analysis_main(["--log-level", args.log_level, *args.analysis_args])

    if args.module == "data":
        return _run_data_cli(args)

    parser.error(f"Unknown module: {args.module}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
