# Output File Formats

This document describes the persisted files written by
`code/data/river_network.py` and `code/data/land_cover.py`.

## River Network Storage

### `river_trenches.parquet`

Main persisted trench table and single source of truth for matrix indexing.

Core columns:

- `trench_id`
- `upstream_node`
- `downstream_node`
- `distance`
- `estuary_distance`
- `system_id`
- `upstream_node_index`
- `downstream_node_index`
- `trench_index`
- `geometry`

### `drainage_areas.parquet`

One retained drainage polygon per `trench_id`.

Core columns:

- `trench_id`
- `drainage_area`
- `within_brazil`
- `geometry`

### `trench_adm2_matches.parquet`

Exploded trench-to-ADM2 relation table.

Core columns:

- `trench_id`
- `adm2`

Semantics:

- One row per intersecting trench/ADM2 match.
- This file is only the relation table produced from the spatial join subset.

### `river_system_matrices.pkl`

Python pickle storing sparse graph products keyed by `system_id`.

Top-level keys:

- `node_reachability_matrices`
- `node_distance_matrices`
- `trench_reachability_matrices`
- `trench_distance_matrices`

Matrix order is derived from `river_trenches.parquet`.

## RiverNetwork Methods

### `compute_distance_matrices()`

Builds all sparse matrices and writes matrix-index columns back onto
`self.trenches`.

### `build_trench_adm2_table(...)`

Builds `self.trench_adm2_table` as an exploded relation table with columns:

- `trench_id`
- `adm2`

### `annotate_drainage_areas_with_country_membership(...)`

Adds the `within_brazil` flag to `self.drainage_areas`.

## Land Cover Outputs

### `land_cover_results.feather`

One row per (`trench_id`, `year`) with `land_cover_class_*` columns.

### `land_cover_river_aggregated.feather`

One row per (`adm2_id`, `year`) produced by upstream aggregation.

Semantics:

- Seed trenches come from `trench_adm2_matches.parquet`.
- If a trench intersects multiple ADM2 regions, it contributes to each of
  them.
- Matrix lookup uses `system_id` and `trench_index` from
  `river_trenches.parquet`.
