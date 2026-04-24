"""CLI for analysis workflows."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from code.analysis.sensor_data import list_groups, run_suite  # noqa: E402
from code.analysis.settings import DEFAULT_SETTINGS, SensorAnalysisSettings  # noqa: E402


def configure_logging(level: str) -> None:
    """Configure CLI logging."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    values = [item.strip() for item in value.split(",") if item.strip()]
    return values or None


def _build_settings(args: argparse.Namespace) -> SensorAnalysisSettings:
    return SensorAnalysisSettings(
        project_root=DEFAULT_SETTINGS.project_root,
        sensor_data_path=Path(args.sensor_data_path or DEFAULT_SETTINGS.sensor_data_path),
        land_cover_path=Path(args.land_cover_path or DEFAULT_SETTINGS.land_cover_path),
        transformations_path=Path(
            args.transformations_path or DEFAULT_SETTINGS.transformations_path
        ),
        trenches_path=Path(args.trenches_path or DEFAULT_SETTINGS.trenches_path),
        output_dir=Path(args.output_dir or DEFAULT_SETTINGS.output_dir),
        distance_buckets=DEFAULT_SETTINGS.distance_buckets,
        land_cover_subclasses=DEFAULT_SETTINGS.land_cover_subclasses,
        land_cover_statistic=DEFAULT_SETTINGS.land_cover_statistic,
        land_cover_transform=DEFAULT_SETTINGS.land_cover_transform,
        fixed_effects=DEFAULT_SETTINGS.fixed_effects,
        fixed_effect_variables=DEFAULT_SETTINGS.fixed_effect_variables,
        cluster_variable=DEFAULT_SETTINGS.cluster_variable,
        vcov_type=DEFAULT_SETTINGS.vcov_type,
        minimum_observations=args.min_observations,
        importance_tiers=DEFAULT_SETTINGS.importance_tiers,
        controls=DEFAULT_SETTINGS.controls,
        excluded_pollutant_columns=DEFAULT_SETTINGS.excluded_pollutant_columns,
        type_group_names=DEFAULT_SETTINGS.type_group_names,
        subclass_labels=DEFAULT_SETTINGS.subclass_labels,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the analysis CLI parser."""
    parser = argparse.ArgumentParser(description="Analysis CLI")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for CLI execution.",
    )
    parser.add_argument(
        "--sensor-data-path",
        default=None,
        help="Override the assembled sensor data parquet path.",
    )
    parser.add_argument(
        "--land-cover-path",
        default=None,
        help="Override the assembled land-cover parquet path.",
    )
    parser.add_argument(
        "--transformations-path",
        default=None,
        help="Override the water-quality transformation metadata path.",
    )
    parser.add_argument(
        "--trenches-path",
        default=None,
        help="Override the river-network trenches parquet path.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the output directory for analysis artifacts.",
    )
    parser.add_argument(
        "--min-observations",
        type=int,
        default=DEFAULT_SETTINGS.minimum_observations,
        help="Minimum non-null observations required to include a pollutant.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the sensor analysis suite")
    run_parser.add_argument(
        "--pollutant-group-kind",
        default="all",
        choices=["all", "type", "importance"],
        help="Pollutant group kind to select.",
    )
    run_parser.add_argument(
        "--pollutant-group",
        default="all",
        help="Pollutant group name within the selected group kind.",
    )
    run_parser.add_argument(
        "--pollutants",
        default=None,
        help="Comma-separated explicit pollutant list.",
    )
    run_parser.add_argument(
        "--land-cover-subclasses",
        default=None,
        help="Comma-separated land-cover subclass ids such as c41,c42.",
    )
    run_parser.add_argument(
        "--max-distance-step",
        type=int,
        default=None,
        help="Maximum cumulative distance step to run.",
    )

    groups_parser = subparsers.add_parser("list-groups", help="List pollutant groups")
    groups_parser.add_argument(
        "--as-json",
        action="store_true",
        help="Print the group listing as JSON.",
    )
    return parser


def _print_group_listing(group_listing: dict[str, dict[str, list[str]]], as_json: bool) -> None:
    if as_json:
        print(json.dumps(group_listing, indent=2))
        return
    for group_kind, groups in group_listing.items():
        print(f"[{group_kind}]")
        for name, pollutants in sorted(groups.items()):
            print(f"{name}: {', '.join(pollutants)}")


def main(argv: list[str] | None = None) -> int:
    """Run the analysis CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    settings = _build_settings(args)

    if args.command == "list-groups":
        groups = list_groups(settings, minimum_observations=args.min_observations)
        _print_group_listing(groups, args.as_json)
        return 0

    if args.command == "run":
        run = run_suite(
            settings,
            pollutant_group_kind=args.pollutant_group_kind,
            pollutant_group=args.pollutant_group,
            pollutants=_csv_list(args.pollutants),
            land_cover_subclasses=_csv_list(args.land_cover_subclasses),
            max_distance_step=args.max_distance_step,
            output_dir=args.output_dir,
            min_observations=args.min_observations,
            save_outputs=True,
        )
        print(run.output_dir)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
