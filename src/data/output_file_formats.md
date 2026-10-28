# Output File Formats

This document describes the persisted files written by
`src/data/sources/river_network/`, `src/data/sources/land_cover/`, and climate assembly
outputs.

## River Network Storage

### `river_network_trenches.parquet`

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

### `river_network_drainage_areas.parquet`

One retained drainage polygon per `trench_id`.

Core columns:

- `trench_id`
- `drainage_area`
- `within_brazil`
- `geometry`

### `river_network_trench_adm2_matches.parquet`

Exploded trench-to-ADM2 relation table.

Core columns:

- `trench_id`
- `adm2`

Semantics:

- One row per intersecting trench/ADM2 match.
- This file is only the relation table produced from the spatial join subset.

### `river_network_adm2_dominant_systems.parquet`

One row per ADM2 region, giving its dominant river system.

Core columns:

- `adm2`
- `system_id`

Semantics:

- `system_id` is chosen per ADM2 as the river system with the largest summed
  intersecting trench distance (see `build_adm2_dominant_system_table()`
  below).
- Derived from `river_network_trench_adm2_matches.parquet` joined against
  `river_network_trenches.parquet`'s `system_id`/`distance` columns.

### `river_network_otto_basins.parquet`

One row per `trench_id` with its Otto Pfafstetter basin codes
`otto_basin_l1` ... `otto_basin_l6` (strings), read from the BHO drainage-area
layer's attributes (see `otto_basins.py`).

### `river_network_adm2_otto_basins.parquet`

One row per ADM2 region (`adm2`) with its dominant Otto basin code at each
level (`otto_basin_l1` ... `otto_basin_l6`, strings).

Semantics:

- Per level, the basin with the largest summed `drainage_area` of the ADM2's
  intersecting trenches (from `river_network_trench_adm2_matches.parquet`);
  ties go to the smaller code.
- Levels are resolved independently, so a coarser code need not prefix a finer
  one for regions straddling a basin boundary (0.3% of ADM2 at levels 2/3).
- Written by `write_otto_basin_table()` whenever the trench-ADM2 matches and
  drainage areas exist in the output directory.

### `river_network_system_matrices.pkl`

Python pickle storing sparse graph products keyed by `system_id`.

Top-level keys:

- `node_reachability_matrices`
- `node_distance_matrices`
- `trench_reachability_matrices`
- `trench_distance_matrices`

Matrix order is derived from `river_network_trenches.parquet`.

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
  column positions back to trench ids using `river_network_trenches.parquet`.

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

### `data/land_cover/processed/extract/land_cover.parquet`

One row per (`trench_id`, `year`) with `land_cover_class_*` columns.

Both aggregate-stage outputs below are produced together by the default
`data preprocess --source land_cover --phase aggregate` run (`--variant all`);
pass `--variant sensor` or `--variant adm2` to build just one.

### `data/land_cover/processed/aggregate/land_cover_sensor_upstream.parquet`

Long-format, one row per (`station_code`, `year`, `bucket`, `land_cover_class`).
`bucket` is the 25 km-wide upstream distance ring; `n`/`cnt`/`share` are the
reachable-trench count, raw class count, and fractional class share within that
bucket. `land_cover_class` is a row value and `share` is a single fractional
column, not per-class `{class}_shr` columns.

### `data/land_cover/processed/aggregate/land_cover_adm2_upstream.parquet`

Long-format, one row per (`mun_id`, `year`, `bucket`, `land_cover_class`),
produced by `aggregate_along_rivers()`. Same `bucket`/`n`/`cnt`/`share` columns
as the sensor output, plus `bucket_intersects_adm2`.

Semantics:

- Seed trenches come from `river_network_trench_adm2_matches.parquet`.
- If a trench intersects multiple ADM2 regions, it contributes to each of
  them.
- Matrix lookup uses `system_id` and `trench_index` from
  `river_network_trenches.parquet`.

### `data/land_cover/processed/extract/land_cover_mining.parquet`

The MapBiomas Collection 11 mining-substances layer per (`trench_id`, `year`),
1985–2025, built with `data preprocess --source land_cover --phase extract
--product mining` (rasters in `data/land_cover/raw/mapbiomas11_mining`, legend
`data/land_cover/auxiliary/mapbiomas_mining_legend.xlsx`). Same layout as
`land_cover.parquet`, with pixel counts:

- `land_cover_total`: all pixels of the drainage polygon; `land_cover_class_0`
  the pixels without mining;
- `land_cover_class_1` … `_5`: the mining groups of
  `src.data.sources.land_cover.mining.MINING_GROUPS` (1 garimpo gold,
  2 garimpo of unknown substance, 3 other garimpo, 4 industrial gold,
  5 other industrial);
- `land_cover_class_<group><phase>`: each group split into phase 1 (active,
  exposed soil) and 2 (revegetation).

Class 0 and the groups add up to the total, and the phases to their group.

### `data/land_cover/processed/aggregate/land_cover_mining_adm2_upstream.parquet`

The ADM2 aggregate of the mining extract (`--phase aggregate --product
mining`), in the layout of `land_cover_adm2_upstream.parquet`:
`land_cover_class` is -1 (bucket total), 0, the group ids and the
group-phase ids. `src.analysis.panels.adm2_upstream_mining` and the
`mining_bucketed` assembly sources collapse it into `mining_*` area shares.

### `data/land_cover/processed/extract/mining_substances_area_by_class_year.csv`

National mined pixels and area (km²) by raw ABCC class code and year, written
by `scripts/prepare_mapbiomas_mining.py` when it merges the Earth Engine tiles.

## Population Outputs

### `data/population/raw/population_raw.parquet`

Raw BigQuery extract from `basedosdados.br_ms_populacao.municipio` joined to
the municipality directory for names.

### `data/population/processed/population.parquet`

Cleaned municipality population table with columns:

- `mun_id`
- `year`
- `sex`
- `age_group`
- `population`

## Climate Outputs

### `data/climate/processed/extract/climate_era5_land.parquet`

One row per (`trench_id`, `date`) produced from the processed ERA5-Land store.

Semantics:

- The Zarr store (`data/climate/raw/era5_land.zarr_nobackup`) remains the
  internal daily raster intermediate.
- This parquet is the analysis-facing preprocess output.
- Values are polygon means over drainage areas for each trench and day.

### `data/climate/processed/aggregate/climate_sensor_upstream.parquet`

One row per station-day sensor observation, indexed by (`station_code`, `datetime`).

Semantics:

- Same-day duplicate water-quality rows are collapsed before assembly.
- Includes a single matched `trench_id` per station plus upstream distance-bucket
  climate features.
- Daily and trailing 7/30/90/180/365 day means are computed from the trench/day
  climate table.

### `data/climate/processed/aggregate/climate_adm2_upstream_yearly.parquet`

One row per (`adm2_id`, `year`) produced by upstream aggregation.

Semantics:

- Daily trench climate is first rolled up to trench/year using variable-specific
  annual summaries.
- Upstream aggregation then uses the saved river-network trench matrices.

## Biome Outputs

### `data/biomes/processed/biomes_adm2.parquet`

One row per `mun_id` with a `biome` label, produced by `src.data.sources.biomes`
intersecting IBGE's terrestrial biome polygons with GADM ADM2 boundaries.

Semantics:

- `biome` is the biome with the largest intersecting area within the
  municipality (dominant biome by area, not a fractional composition).

### `data/biomes/processed/biomes_sensor.parquet`

One row per `station_code` with a `biome` label, produced by a point-in-polygon
join between monitoring-station coordinates and the same biome polygons.

Semantics:

- Stations that fall just outside every polygon (e.g. coastline
  simplification) are assigned the nearest biome instead of a missing value.

## Geology Outputs

### `data/geology/processed/geology_adm2.parquet`

One row per `mun_id`, produced by `src.data.sources.geology` from the
Geological Survey of Brazil's (SGB) Geobank mineral occurrences and its gold
province and district polygons (`data/geology/raw/sgb_gold.gpkg`).

Columns:

- `area_km2`: municipality area (SIRGAS 2000 / Brazil Polyconic).
- `gold_occurrences`: all gold occurrences, split by economic status into
  `gold_occurrences_unexploited`, `_garimpo`, `_mine` and `_undetermined`.
  `gold_occurrences_alluvial` counts those with an alluvial host (alluvium,
  gravel, eluvium, colluvium, detrital sediment).
- `gold_district_share`, `gold_province_share`: share of the municipality's
  area inside any gold district or province.

Semantics:

- Only the never-exploited occurrences are predetermined geology. Garimpo and
  mine occurrences, and the provinces, were largely recorded or drawn where
  mining already was.
- Occurrence positions are accurate to a few km. Points outside every
  municipality are assigned to the nearest one within 5 km, else dropped.
- Municipalities without occurrences have zero counts.

### `data/geology/processed/geology_adm2_upstream.parquet`

One row per `mun_id` (the 5,564 municipalities matched to river trenches),
produced by `python -m src.data.sources.geology preprocess --product upstream`
from the same occurrences and the river network. Each occurrence is placed in
the trench whose drainage polygon contains it (the nearest within 5 km for
points just outside every polygon).

Columns:

- `<group>_upstream_<reach>`: occurrences in the trenches upstream of the
  municipality, for groups `gold`, `gold_unexploited`, `gold_garimpo`,
  `gold_mine`, `gold_alluvial` (as in `geology_adm2`) and `tin` (substance
  "Estanho"), and reaches `50` and `200` km and `all` (unlimited).
- `<group>_upstream_otto4`, `<group>_downstream_otto4`: 0/1, any occurrence
  in an Otto level-4 sub-basin upstream (downstream) of a level-4 sub-basin
  the municipality's trenches lie in, for groups `gold`, `gold_unexploited`
  and `tin`. Moreno-Louzada's exposure.

Semantics:

- Reaches are measured along the river network on the shifted distance scale
  of the land-cover ADM2 aggregation: 0 is the upstream end of the trenches
  touching the municipality, and occurrences in those trenches themselves
  (the negative buckets) are not upstream. `geology_adm2` counts the
  occurrences inside the municipality.
- "Upstream" in the Otto binaries follows Pfafstetter coding within the
  level-3 parent: a sub-basin is upstream of a main-stem interbasin (odd last
  digit) with a lower last digit, and nothing is upstream of a tributary (even
  last digit). Deposits in the municipality's own sub-basins do not count, and
  nothing upstream across level-3 boundaries does.
- Each trench is counted once per municipality, at its smallest distance.

## Malaria Outputs

Produced by `src.data.sources.malaria` from the record-level SIVEP-Malária
notifications 2004–2024 (Zenodo 20706957, `data/malaria/raw/`). A case is a
positive exam (`res_exam` filled and not "Negativo") that is not a
cure-verification slide (`id_lvc` or `tipo_lam` = LVC). A garimpo case has
`cod_ocup` = "Garimpagem" (activity in the prior 15 days; "Mineracao" is not
included). Municipality codes are 6-digit IBGE; SIVEP's 5399xx placeholders
and codes without an IBGE match are treated as missing. `year` is the
notification year.

### `data/malaria/processed/malaria_adm2.parquet`

One row per (`mun_id`, `year`) with any record (13,792 rows, 1,419
municipalities); absent rows have zero cases. `adm2_panel` carries these
columns zero-filled for 2004–2024 and missing outside those years.
The same cases are counted from three sites:

- by infection municipality (Brazil-infected only): `inf_cases`,
  `inf_garimpo`, and its split into `inf_garimpo_local` (living in the
  infection municipality) and `inf_garimpo_incoming` (living in another
  Brazilian municipality), with `inf_garimpo_incoming_other_state`;
- by residence municipality: `res_cases`, `res_garimpo`,
  `res_garimpo_elsewhere` (infected in another Brazilian municipality) and
  `res_garimpo_abroad`;
- by notification municipality: `noti_exams` (all exams, LVC dropped) and
  `noti_positives`. Negatives carry almost no infection or residence site from
  2007, so testing can only be measured here.

Semantics:

- Local + incoming is less than `inf_garimpo` by the cases without a valid
  residence (under 0.1%).
- "Local" cannot be verified without a patient identifier: for local cases
  residence, infection and notification site usually coincide.

### `data/malaria/processed/malaria_garimpo_flows.parquet`

Garimpo cases by `year`, residence municipality `res_mun_id`, infection
municipality `inf_mun_id` (missing when infected abroad) and `inf_country`
("Brasil" or the country of infection), with `cases` ≥ 1. Includes cases
infected where they live, so it can serve as a destination-choice table.

Semantics:

- Only patients with a valid Brazilian residence are included. Garimpo cases
  of foreign residents, mostly Venezuelans infected in Venezuela (5,391 in
  2004–24), have no residence code and are left out, as they are from the
  `res_` counts.
- Brazil-infected cases without a valid infection code are left out (14 in
  2004–24), so per-year `cases` can fall slightly short of `res_garimpo`.


## Commodity Price Outputs

Produced by `src.data.sources.commodity_prices` from the World Bank's monthly
Commodity Price Data (the Pink Sheet, nominal gold price in USD per troy
ounce, `data/commodity_prices/raw/cmo_historical_data_monthly.xlsx`) and the
Banco Central do Brasil's SGS series 1 (daily PTAX sell rate, BRL per USD,
from 1994-07-01, `bcb_sgs_1_brl_usd_daily.parquet`). Both are re-downloaded
on every fetch, as the Pink Sheet is revised monthly.

### `data/commodity_prices/processed/gold_price_monthly.parquet`

One row per (`year`, `month`) with both series: `gold_usd_oz`, `brl_per_usd`
(mean of the month's daily rates), `gold_brl_oz` (their product),
`log_gold_usd` and `log_gold_brl`.

### `data/commodity_prices/processed/gold_price_annual.parquet`

One row per `year` with all twelve months (1995 onward): the means of the
monthly `gold_usd_oz`, `brl_per_usd` and `gold_brl_oz`, the logs of the
annual means `log_gold_usd` and `log_gold_brl`, and `months`.
