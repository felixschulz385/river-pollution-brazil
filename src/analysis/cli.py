"""CLI for analysis workflows."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.sensor_data import list_groups, run_plotly_app, run_suite  # noqa: E402
from src.analysis.sensor_data.runner import merge_suite  # noqa: E402
from src.analysis.settings import DEFAULT_SETTINGS, SensorAnalysisSettings  # noqa: E402


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


def _tuple_list(value: str | None) -> tuple[str, ...] | None:
    values = _csv_list(value)
    return tuple(values) if values is not None else None


def _build_settings(args: argparse.Namespace) -> SensorAnalysisSettings:
    sensor_id_column = args.sensor_id_column or DEFAULT_SETTINGS.sensor_id_column
    climate_data_path = (
        Path(args.climate_data_path)
        if args.climate_data_path
        else DEFAULT_SETTINGS.climate_data_path
    )
    default_climate_join_keys = DEFAULT_SETTINGS.climate_join_keys
    if sensor_id_column != DEFAULT_SETTINGS.sensor_id_column:
        default_climate_join_keys = (sensor_id_column, DEFAULT_SETTINGS.climate_join_keys[1])
    climate_join_keys = _tuple_list(args.climate_join_keys) or default_climate_join_keys
    if len(climate_join_keys) != 2:
        raise ValueError(
            "--climate-join-keys must provide exactly two comma-separated columns."
        )
    sensor_id_aliases = _tuple_list(args.sensor_id_aliases) or DEFAULT_SETTINGS.sensor_id_aliases
    distance_buckets = _tuple_list(args.distance_buckets) or DEFAULT_SETTINGS.distance_buckets
    land_cover_subclasses = (
        _tuple_list(args.available_land_cover_subclasses)
        or DEFAULT_SETTINGS.land_cover_subclasses
    )
    fixed_effects = tuple(
        sensor_id_column if effect == DEFAULT_SETTINGS.sensor_id_column else effect
        for effect in DEFAULT_SETTINGS.fixed_effects
    )
    cluster_variable = args.cluster_variable or sensor_id_column
    return SensorAnalysisSettings(
        project_root=DEFAULT_SETTINGS.project_root,
        sensor_data_path=Path(args.sensor_data_path or DEFAULT_SETTINGS.sensor_data_path),
        land_cover_path=Path(args.land_cover_path or DEFAULT_SETTINGS.land_cover_path),
        climate_data_path=climate_data_path,
        transformations_path=Path(
            args.transformations_path or DEFAULT_SETTINGS.transformations_path
        ),
        trenches_path=Path(args.trenches_path or DEFAULT_SETTINGS.trenches_path),
        output_dir=Path(args.output_dir or DEFAULT_SETTINGS.output_dir),
        sensor_id_column=sensor_id_column,
        sensor_id_aliases=sensor_id_aliases,
        datetime_column=args.datetime_column or DEFAULT_SETTINGS.datetime_column,
        date_column=args.date_column or DEFAULT_SETTINGS.date_column,
        climate_join_keys=climate_join_keys,
        climate_column_prefix=args.climate_column_prefix or DEFAULT_SETTINGS.climate_column_prefix,
        climate_count_suffix=args.climate_count_suffix or DEFAULT_SETTINGS.climate_count_suffix,
        climate_interaction_mode=args.climate_interaction_mode or DEFAULT_SETTINGS.climate_interaction_mode,
        distance_buckets=distance_buckets,
        land_cover_subclasses=land_cover_subclasses,
        land_cover_statistic=args.land_cover_statistic or DEFAULT_SETTINGS.land_cover_statistic,
        land_cover_transform=DEFAULT_SETTINGS.land_cover_transform,
        fixed_effects=fixed_effects,
        fixed_effect_variables=DEFAULT_SETTINGS.fixed_effect_variables,
        cluster_variable=cluster_variable,
        vcov_type=DEFAULT_SETTINGS.vcov_type,
        minimum_observations=args.min_observations,
        map_tolerance=args.map_tolerance,
        map_max_iterations=args.map_max_iterations,
        importance_tiers=DEFAULT_SETTINGS.importance_tiers,
        controls=DEFAULT_SETTINGS.controls,
        climate_variables=DEFAULT_SETTINGS.climate_variables,
        model_families=tuple(
            _csv_list(getattr(args, "model_families", None))
            or DEFAULT_SETTINGS.model_families
        ),
        lasso_settings=DEFAULT_SETTINGS.lasso_settings,
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
        "--sensor-id-column",
        default=None,
        help="Panel id column used for sensors/stations. Default: station_code.",
    )
    parser.add_argument(
        "--sensor-id-aliases",
        default=None,
        help="Comma-separated fallback id aliases to normalize onto the sensor id column.",
    )
    parser.add_argument(
        "--datetime-column",
        default=None,
        help="Datetime source column used when a date column must be materialized.",
    )
    parser.add_argument(
        "--date-column",
        default=None,
        help="Date column used for panel joins and FE construction.",
    )
    parser.add_argument(
        "--land-cover-path",
        default=None,
        help="Override the assembled land-cover parquet path.",
    )
    parser.add_argument(
        "--climate-data-path",
        default=None,
        help="Override the assembled upstream-climate parquet path.",
    )
    parser.add_argument(
        "--climate-join-keys",
        default=None,
        help="Comma-separated join keys for the climate data. Default: station_code,date.",
    )
    parser.add_argument(
        "--climate-column-prefix",
        default=None,
        help="Prefix used to auto-discover climate variables, e.g. cl_.",
    )
    parser.add_argument(
        "--climate-count-suffix",
        default=None,
        help="Suffix identifying non-regression climate count columns, e.g. _n.",
    )
    parser.add_argument(
        "--climate-interaction-mode",
        default=DEFAULT_SETTINGS.climate_interaction_mode,
        choices=["same_bucket", "cumulative", "all"],
        help="How climate variables can interact with land-cover buckets.",
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
    parser.add_argument(
        "--distance-buckets",
        default=None,
        help="Comma-separated ordered distance buckets, e.g. 0_10km,10_50km,50_100km.",
    )
    parser.add_argument(
        "--available-land-cover-subclasses",
        default=None,
        help="Comma-separated available land-cover subclasses in the input design.",
    )
    parser.add_argument(
        "--land-cover-statistic",
        default=None,
        help="Land-cover statistic suffix to use from the input, e.g. cnt or shr.",
    )
    parser.add_argument(
        "--cluster-variable",
        default=None,
        help="Override the clustering variable used for inference.",
    )
    parser.add_argument(
        "--map-tolerance",
        type=float,
        default=DEFAULT_SETTINGS.map_tolerance,
        help="Convergence tolerance for MAP demeaning.",
    )
    parser.add_argument(
        "--map-max-iterations",
        type=int,
        default=DEFAULT_SETTINGS.map_max_iterations,
        help="Maximum MAP iterations for fixed-effect absorption.",
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
    run_parser.add_argument(
        "--model-families",
        default=None,
        help="Comma-separated model families such as crude_twfe,post_lasso.",
    )
    run_parser.add_argument("--lasso-jobs", type=int, default=None, help="Workers for LASSO CV; defaults to SLURM_CPUS_PER_TASK.")
    run_parser.add_argument("--shard-count", type=int, default=1, help="Total deterministic execution shards.")
    run_parser.add_argument("--shard-index", type=int, default=0, help="Zero-based shard index to execute.")
    run_parser.add_argument("--resume", action="store_true", help="Skip completed checkpointed specifications.")
    run_parser.add_argument("--checkpoint-models", type=int, default=25, help="Completed models per immutable checkpoint chunk.")

    merge_parser = subparsers.add_parser("merge", help="Merge completed sensor-analysis shard checkpoints")
    merge_parser.add_argument("--run-dir", required=True, help="Canonical run directory containing _work checkpoints.")
    merge_parser.add_argument("--run-fingerprint", default=None, help="Fingerprint printed by shard runs; defaults to the newest checkpoint run.")
    merge_parser.add_argument("--expected-shards", type=int, required=True, help="Expected number of completed shards.")

    groups_parser = subparsers.add_parser("list-groups", help="List pollutant groups")
    groups_parser.add_argument(
        "--as-json",
        action="store_true",
        help="Print the group listing as JSON.",
    )
    plotly_parser = subparsers.add_parser(
        "plotly-app",
        help="Serve an interactive Plotly app for saved regression outputs",
    )
    plotly_parser.add_argument(
        "--results-dir",
        default=None,
        help="Base directory containing saved analysis run subdirectories.",
    )
    plotly_parser.add_argument(
        "--run-name",
        default=None,
        help="Optional saved run name to select on app startup.",
    )
    plotly_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface for the Dash server.",
    )
    plotly_parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port for the Dash server.",
    )
    plotly_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Dash debug mode.",
    )
    plotly_parser.add_argument(
        "--max-facets",
        type=int,
        default=12,
        help="Maximum number of pollutant-subclass facets shown in land-cover panels.",
    )
    plotly_parser.add_argument(
        "--top-terms",
        type=int,
        default=20,
        help="Number of non-land-cover terms to show in ranking and forest plots.",
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
            model_families=_csv_list(args.model_families),
            output_dir=args.output_dir,
            min_observations=args.min_observations,
            save_outputs=True,
            lasso_jobs=args.lasso_jobs,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
            resume=args.resume,
            checkpoint_models=args.checkpoint_models,
        )
        print(run.output_dir)
        return 0

    if args.command == "merge":
        run = merge_suite(
            settings,
            run_dir=args.run_dir,
            run_fingerprint=args.run_fingerprint,
            expected_shards=args.expected_shards,
        )
        print(run.output_dir)
        return 0

    if args.command == "plotly-app":
        run_plotly_app(
            results_dir=Path(args.results_dir or settings.output_dir),
            run_name=args.run_name,
            host=args.host,
            port=args.port,
            debug=args.debug,
            settings=settings,
            max_facets=args.max_facets,
            top_terms=args.top_terms,
        )
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
