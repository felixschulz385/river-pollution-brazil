# Output File Formats

This document describes the persisted files written by
`src/data/sources/river_network/`, `src/data/sources/land_cover/`, and climate assembly
outputs.

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

Semantics:

- Node matrices use the per-system node ordering implied by
  `upstream_node_index` and `downstream_node_index`.
- `node_reachability_matrices[system_id][i, j] == 1` means upstream node `i`
  can reach downstream node `j`.
- `node_distance_matrices[system_id][i, j]` is the along-network distance from
  node `i` to node `j`.
- Trench matrices use `trench_index` within each `system_id`.
- `trench_reachability_matrices[system_id][i, j] == 1` means upstream trench
  `j` drains into downstream trench `i`.
- `trench_distance_matrices[system_id][i, j]` is the along-network distance
  from upstream trench `j` to downstream trench `i`.
- Consumers that want all upstream trenches for a target trench should read the
  row corresponding to that target trench's `trench_index` and map nonzero
  column positions back to trench ids using `river_trenches.parquet`.

## RiverNetwork Methods

### `compute_distance_matrices()`

Builds all sparse matrices and writes matrix-index columns back onto
`self.trenches`.

### `build_trench_adm2_table(...)`

Builds `self.trench_adm2_table` as an exploded relation table with columns:

- `trench_id`
- `adm2`

### `build_adm2_dominant_system_table()`

Builds `self.adm2_dominant_system_table` with columns:

- `adm2`
- `system_id`

`system_id` is chosen per ADM2 as the river system with the largest summed
intersecting trench distance.

### `annotate_drainage_areas_with_country_membership(...)`

Adds the `within_brazil` flag to `self.drainage_areas`.

## Land Cover Outputs

### `land_cover_results.feather`

One row per (`trench_id`, `year`) with `land_cover_class_*` columns.

### `land_cover_sensor_upstream.parquet`

One row per (`station_code`, `year`) produced by the `land-cover assemble --variant sensor`
pipeline.

### `land_cover_adm2_upstream.parquet`

One row per (`adm2_id`, `year`) produced by the `land-cover assemble --variant adm2`
pipeline.

Semantics:

- Seed trenches come from `trench_adm2_matches.parquet`.
- If a trench intersects multiple ADM2 regions, it contributes to each of
  them.
- Matrix lookup uses `system_id` and `trench_index` from
  `river_trenches.parquet`.
- Both assembled outputs use the same bucketed upstream columns, such as
  `lc_0_10km_tot`, `lc_0_10km_n`, `lc_0_10km_c41_cnt`, and
  `lc_0_10km_c41_shr`.

## Population Outputs

### `data/population/raw/population_raw.parquet`

Raw BigQuery extract from `basedosdados.br_ms_populacao.municipio` joined to
the municipality directory for names.

### `data/population/population.parquet`

Cleaned municipality population table with columns:

- `mun_id`
- `year`
- `sex`
- `age_group`
- `population`

## Climate Outputs

### `data/climate/processed/era5_land.parquet`

One row per (`trench_id`, `date`) produced from the processed ERA5-Land store.

Semantics:

- The Zarr store remains the internal daily raster intermediate.
- This parquet is the analysis-facing preprocess output.
- Values are polygon means over drainage areas for each trench and day.

### `data/climate/processed/era5_land/climate_sensor_upstream.parquet`

One row per station-day sensor observation, indexed by (`station_code`, `datetime`).

Semantics:

- Same-day duplicate water-quality rows are collapsed before assembly.
- Includes a single matched `trench_id` per station plus upstream distance-bucket
  climate features.
- Daily and trailing 7/30/90/180/365 day means are computed from the trench/day
  climate table.

### `data/climate/processed/era5_land/climate_adm2_upstream_yearly.parquet`

One row per (`adm2_id`, `year`) produced by upstream aggregation.

Semantics:

- Daily trench climate is first rolled up to trench/year using variable-specific
  annual summaries.
- Upstream aggregation then uses the saved river-network trench matrices.

## Biome Outputs

### `data/biomes/biome_adm2.parquet`

One row per `mun_id` with a `biome` label, produced by `src.data.sources.biomes`
intersecting IBGE's terrestrial biome polygons with GADM ADM2 boundaries.

Semantics:

- `biome` is the biome with the largest intersecting area within the
  municipality (dominant biome by area, not a fractional composition).

### `data/biomes/biome_sensor.parquet`

One row per `station_code` with a `biome` label, produced by a point-in-polygon
join between monitoring-station coordinates and the same biome polygons.

Semantics:

- Stations that fall just outside every polygon (e.g. coastline
  simplification) are assigned the nearest biome instead of a missing value.

