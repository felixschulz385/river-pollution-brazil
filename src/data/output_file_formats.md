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

## Health Outputs

Produced by `src.data.sources.health` from DATASUS TabNet tables by residence
municipality (`mun_id`, 6-digit IBGE) and year. Municipalities a year's table
does not list stay absent.

### `data/health/processed/health_births.parquet`

SINASC births summary, one row per (`mun_id`, `year`), 1994 onward:

- `births`; `births_low_weight` (< 2,500 g) over `births_weight_known`;
  `births_preterm` (< 37 weeks) over `births_gestation_known`. The `*_known`
  denominators exclude "Ignorado".
- `births_anomaly`: congenital anomaly detected at birth ("Sim"), with
  `births_anomaly_ignored`.
- `births_apgar{1,5}_below_6`, `births_apgar{1,5}_below_8`: APGAR score at 1
  and 5 minutes below 6 and below 8, with `births_apgar{1,5}_ignored`.

Semantics:

- TabNet bins APGAR as 0–2, 3–5, 6–7 and 8–10, so the usual "below 7" cut is
  not available.
- The anomaly and APGAR columns are present only when their tables
  (`health_congenital_anomaly.parquet`, `health_apgar_1.parquet`,
  `health_apgar_5.parquet`) are processed.
- Preterm counts are comparable from 2012: SINASC's gestation coding changes in
  2011.
- Coverage (national, TabNet as of 2026-10-28): the anomaly field starts in 1999,
  with no "Ignorado" category in 1999–2000; its known share is 89–92% in
  2001–05 and at least 95% from 2006. APGAR is known for 58–86% of births in
  1994–98, 89–93% in 1999–2004 and at least 95% from 2009. The anomaly rate
  rises from 6.1 per 1,000 births (2004) to 10.2 (2023), so it carries a
  detection trend.

### `data/health/processed/health_infant_deaths.parquet`

SIM infant deaths (under 1 year, form `inf10br`) by ICD-10 chapter, one row
per (`mun_id`, `year`), 1996 onward: `infant_deaths` (all causes),
`infant_deaths_congenital` (chapter XVII), `infant_deaths_ill_defined`
(XVIII), `infant_deaths_external` (XX, V01–Y98) and
`infant_deaths_nonexternal` (all but XX). A chapter absent from a year's table
counts as zero. TabNet lists only municipality-years with an infant death;
`adm2_panel` carries these columns zero-filled for 1996–2024, and the births
summary's columns as they are.

## Protected Area Outputs

Produced by `src.data.sources.protected_areas` from two inputs:

- **Indigenous lands:** FUNAI's `tis_poligonais_portarias` layer, placed by
  hand in `data/protected_areas/raw/funai_<date>/` (the newest snapshot holding
  `tis_poligonais_portarias.zip`; provenance in its `PROVENANCE.md`). One
  polygon per land with its current stage, modality and the date and act of
  each stage.
- **Conservation units:** the CNUC polygons from INPE TerraBrasilis
  (`terrabrasilis.dpi.inpe.br/geoserver`, WFS), clipped to the Legal Amazon and
  to each biome (`data/protected_areas/raw/terrabrasilis_protected_areas.gpkg`,
  one layer per WFS layer), with each unit's creation year. The layers include
  units created up to about 2020.

### `data/protected_areas/processed/protected_areas.parquet`

GeoParquet (EPSG:4326), one row per area:

- `area_id`: `ti:<FUNAI code>` or `uc:<n>` (units sorted by name).
- `kind`: `indigenous_land` or `conservation_unit`.
- `name`, `category` (units: the CNUC category; Indigenous lands: the
  modality, e.g. Tradicionalmente ocupada, Reserva Indígena), `group` (`PI`
  integral protection, `US` sustainable use), `sphere` (federal, state,
  municipal), `stage` (Indigenous lands: Em Estudo, Delimitada, Declarada,
  Homologada, Encaminhada RI, Regularizada), `year_created` (conservation
  units: the first year in `ano_cria`).
- `date_declared`, `date_homologated`: Indigenous lands' declaratory ordinance
  and homologation decree, where FUNAI records them.
- `year_barred`: the year mining became barred. Units: the creation year.
  Indigenous lands: the year of the first declaration, homologation or registry
  date (many older lands were homologated without a declaratory ordinance).
  Missing when the area is not barred or the land has none of these dates.
- `mining_barred`: mining has no legal option. Indigenous lands from the
  declaration onward (by stage, or by a declaration, homologation or registry
  date where FUNAI's stage lags its dates); integral-protection units and extractive reserves (SNUC,
  art. 18 §6). APAs, national forests (where mining is licensed), RDS, ARIEs
  and RPPNs do not count.
- `in_force_2012`: in force by the end of 2012, before the 2013 good-faith
  rule. Units created by 2012; barred Indigenous lands with `year_barred` up to
  2012. Barred lands without dates count when homologated or regularised (old
  reserves and three old Mato Grosso lands), not when a reserve is still being
  acquired (Encaminhada RI).

Semantics:

- A unit's pieces (one per region layer) are merged by name, category, sphere
  and year. Stray lines and degenerate rings are dropped and the polygons
  repaired.
- FUNAI's polygons overlap where a land under revision covers its original;
  the shares below take the union, so overlaps count once.

### `data/protected_areas/processed/protected_areas_adm2.parquet`

One row per `mun_id`, the share of the municipality's area in areas that bar
mining:

- `indigenous_land_share`: Indigenous lands.
- `conservation_barred_share`: integral-protection units and extractive reserves.
- `protected_share`: either (the union, so overlaps count once).
- `protected_2012_share`: those in force by 2012.

`adm2_panel` carries every column, constant across years. The deposits inside
these areas (lead 1's IGD) are counted in the geology outputs.

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
- `tin_occurrences`, `tin_occurrences_unexploited`: tin (substance "Estanho")
  occurrences, all and never exploited; the other-minerals placebo.
- `gold_occurrences_protected`, `gold_occurrences_unexploited_protected`,
  `tin_occurrences_protected`: occurrences inside protected areas that bar
  mining and were in force by 2012 (`protected_areas`: `mining_barred` and
  `in_force_2012`). The never-exploited gold count is lead 1's IGD.
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
  `gold_mine`, `gold_alluvial` (as in `geology_adm2`), `tin` (substance
  "Estanho"), and `gold_protected`, `gold_unexploited_protected` and
  `tin_protected` (inside protected areas in force by 2012, as in
  `geology_adm2`), and reaches `50` and `200` km and `all` (unlimited).
- `<group>_upstream_otto4`, `<group>_downstream_otto4`: 0/1, any occurrence
  in an Otto level-4 sub-basin upstream (downstream) of a level-4 sub-basin
  the municipality's trenches lie in, for groups `gold`, `gold_unexploited`,
  `tin`, `gold_unexploited_protected` and `tin_protected`. Moreno-Louzada's
  exposure.

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
- `adm2_panel` carries every column, constant across years.

## Malaria Outputs

Produced by `src.data.sources.malaria` from the record-level SIVEP-Malária
notifications 2004–2024 (Zenodo 20706957, `data/malaria/raw/`) and the
preliminary years from 2025 (Zenodo 20706949, from LAI requests, updated
monthly; dated snapshots `data/malaria/raw/preliminary_<YYYY-MM-DD>/`, the
newest read). Outputs stop at the last complete month of notifications: the
yearly tables at the last complete year (2025 in the 2026-10-30 snapshot), the
monthly table at the last complete month (2026-04). Preliminary counts can
still rise with late notifications. A case is a
positive exam (`res_exam` filled and not "Negativo") that is not a
cure-verification slide (`id_lvc` or `tipo_lam` = LVC). A garimpo case has
`cod_ocup` = "Garimpagem" (activity in the prior 15 days; "Mineracao" is not
included). Municipality codes are 6-digit IBGE; SIVEP's 5399xx placeholders
and codes without an IBGE match are treated as missing. `year` is the
notification year.

### `data/malaria/processed/malaria_adm2.parquet`

One row per (`mun_id`, `year`) with any record (14,349 rows through 2025);
absent rows have zero cases. `adm2_panel` carries these columns zero-filled for
2004–2024 and missing outside those years; the panel's spine stops in 2024, so
the preliminary 2025 is only in this table.
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

### `data/malaria/processed/malaria_adm2_monthly.parquet`

Positives by infection municipality (`mun_id`) and `month` (the first day of
the notification month, from `dt_notif`), for timing around dated events such
as IBAMA raids. Only municipality-months with a case are listed.

- `inf_cases`, `inf_garimpo`, `inf_garimpo_local`, `inf_garimpo_incoming`: as
  in `malaria_adm2.parquet` (incoming: living in another Brazilian
  municipality).
- Notifications are dated when notified, typically days after symptoms start
  and up to about two weeks after infection.

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
annual means `log_gold_usd` and `log_gold_brl`, and `months`. `adm2_panel`
joins `gold_usd_oz`, `gold_brl_oz`, `log_gold_usd` and `log_gold_brl` by year.

## Mining Permit Outputs

Produced by `src.data.sources.mining` from the ANM Cadastro Mineiro (SCM)
microdata (`data/mining/raw/anm_scm_microdados_<date>/microdados-scm.zip`, the
newest snapshot; placed by hand, as the ANM firewall blocks scripted
downloads). Tables read: `Processo`, `ProcessoTitulo`, `ProcessoMunicipio`,
`ProcessoSubstancia` and `Municipio`.

### `data/mining/processed/mining_permits_adm2.parquet`

One row per (`mun_id`, `year`) for every municipality in the SCM list and
1985–2026, zero-filled. For each group `<g>` (`all`; `gold`: ouro, minério de
ouro, ouro nativo; `tin`: estanho, minério de estanho, cassiterita, pirita de
estanho):

- `plg_requests_<g>`: requests for a PLG (permissão de lavra garimpeira) by
  protocol year: request types PLG and change of regime to PLG. ANM's
  availability procedures and public offers of free areas are left out.
- `plg_request_area_ha_<g>`: their area in hectares.
- `plg_grants_<g>`: PLGs first granted in the year (publication of the
  process's earliest PLG title).
- `plg_active_<g>`: PLGs in force at some point in the year, from the grant
  year to the latest PLG expiry.
- `plg_active_area_ha_<g>`: their area.

Semantics:

- A process counts in every municipality it lists; its area is split equally
  across them (the microdata have no polygons).
- A process belongs to a substance group if it lists any of the group's
  substances at any time. "Ouro pigmento" (orpiment) is not gold.
- Renewals and extensions are further titles of the same process: they extend
  the span in force but are not grants. A process whose PLG titles carry no
  expiry is in force for five years from the grant (Lei 7.805/1989, art. 7).
- Cancellations and forfeitures are not dated in the title table, so a PLG
  revoked before its expiry still counts as in force.
- Zero or negative process areas are treated as missing (area 0).
- `adm2_panel` joins every column.

### `data/mining/processed/mining_requests_monthly.parquet`

One row per (`mun_id`, `month`) with at least one PLG request, 1985–2026;
months without a request are absent. `month` is the first day of the protocol
month (timestamp).

- `plg_requests_all`, `plg_requests_gold`, `plg_requests_tin`: the requests of
  `mining_permits_adm2`, by protocol month. Summed over a year's months they
  give the yearly table's `plg_requests_<g>`.

Not joined into `adm2_panel`; the quarterly timing analysis of lead 1 reads it
directly (`src/analysis/garimpo_laundering.py`).

## Enforcement Outputs

Produced by `src.data.sources.enforcement` from IBAMA's Fiscalização open data
(`dadosabertos.ibama.gov.br`), placed by hand in a dated snapshot folder
`data/enforcement/raw/ibama_<YYYY-MM-DD>/` (the portal blocks requests from
outside Brazil). The folder's `PROVENANCE.md` and `SCHEMA.md` document the
download, the formats, the keys and the coverage; the newest complete snapshot
is read. Tables read: fines (`auto_infracao`, one CSV per year), embargoes
(`termo_de_embargo`), seizures (`apreensao`) with their seized items (`bem`),
and destruction terms (`termo_destruicao_inutilizacao`) with their destroyed
items (`bem_destruido`).

### `data/enforcement/processed/ibama_records.parquet`

One row per fine, embargo, seizure or destruction term that is not cancelled,
deleted or replaced (`SIT_CANCELADO`/`ST_CANCELADO` = "N"). Names, CPF/CNPJ,
plates and free text are dropped.

- `action`: `fines`, `embargoes`, `seizures` or `destructions`.
- `record_id`: `seq:<system ID>`, or `code:<printed code>-<series>` for
  records from the AIe app without a system ID. Unique within an action.
- `date`, `year`: the fine, embargo, seizure or destruction date. Dates before
  1977 or after the snapshot (typos such as 2925) are missing.
- `mun_id` (6-digit IBGE), `uf`: where the action took place.
- `lon`, `lat`: the record's point. Pairs outside Brazil's bounding box are
  swapped back if swapped, and missing otherwise (0/0, typos).
- `value_brl`: the fine as set by the agent (fines), or the value of the seized
  or destroyed goods; nominal reais.
- `area_ha`: embargoed area (embargoes). `lifted_date`: date the embargo was
  lifted, if it was.
- `items`: number of seized or destroyed items on the term (seizures and
  destructions).
- `operation`: operation name; `operation_code`, `operation_seq`: the
  enforcement action's code and system ID, which link records of one operation
  (destruction terms rarely reference a fine, as equipment is often destroyed
  with no offender identified).
- `fine_seq`, `fine_code`: the fine a term references (for fines, their own
  IDs). IDs are trimmed, upper-cased and stripped of float formatting.
- `mining`, `garimpo`: mining terms in the description fields (fines: the
  description and the offence text; terms: their description, complementary
  information and justification) or in the type and description of the term's
  items. `garimpo` matches garimp*, ouro, mercúrio or cassiterita; `mining`
  adds lavra, mineração, minério and extraction of minerals. Text is
  lower-cased and accent-folded, and names are removed first: places (Ouro
  Preto, São José do Ouro), farms (Fazenda Estrela de Ouro), the colour, and
  mercury-vapour lamps. Location fields are not searched, as they carry place
  names (Lavras, the Transgarimpeira road). A few watercourse names (Igarapé do
  Ouro) still match. Every garimpo record is a mining record.

### `data/enforcement/processed/enforcement_adm2.parquet`

One row per (`mun_id`, `year`) with at least one located record, up to the
last full year before the snapshot. For each class `<c>` (`all`, `mining`,
`garimpo`):

- `ibama_fines_<c>`, `ibama_fine_value_brl_<c>`: fines and their summed value.
- `ibama_embargoes_<c>`, `ibama_embargo_area_ha_<c>`: embargoes and their
  summed area.
- `ibama_seizures_<c>`: seizure terms.
- `ibama_destructions_<c>`: destruction terms.

Semantics:

- Each action's columns are 0 from the year the action is recorded (fines
  1996, embargoes 2000, seizures 2005, destruction terms 2014) and missing
  before. `adm2_panel` joins every column and fills the same years with 0.
- Seizure records stop when IBAMA's seizure export was last refreshed (2026-05
  in the 2026-10-29 snapshot); the table's last year is complete for every
  action.
- Records count where they took place, by their own municipality code.

## Night-Light Outputs

Produced by `src.data.sources.nightlights` from the Earth Observation Group's
annual VIIRS night-light composites (VNL, Colorado School of Mines), downloaded
by hand behind EOG's login and kept as served (`.dat.tif.gz`) in folders under
`data/nightlights/raw/`:

- `eog_vnl_v21/`: v2.1, 2012 (April–December) to 2021, Suomi NPP; a link to the
  copies of the growth-and-temperature project.
- `eog_vnl_<YYYY-MM-DD>/`: v2.2, 2022 onward, with `PROVENANCE.md`, `SCHEMA.md`
  and `SHA256SUMS`. 2022 combines Suomi NPP with three months of NOAA-20; later
  years are Suomi NPP only. v2.2 does not reprocess earlier years.

Every file covers the globe on one grid of 15″ pixels (86401 × 33601, EPSG:4326,
pixel centres from (−180, 75)). The preprocess reads the `average_masked`
(background, fires and aurora masked to 0) and `cf_cvg` (cloud-free
observations) layers over the window of Brazil's municipalities. Each file is
identified by the original name EOG wrote into its TIFF document-name tag: some
v2.1 copies were saved under another file's name, and a file whose tag
disagrees with its name is skipped (the 2013 `cf_cvg` copy holds the 2012
median, so 2013 has no coverage).

### `data/nightlights/processed/nightlights_adm2.parquet`

One row per (`mun_id`, `year`), every municipality in every year from 2012.

- `viirs_version` (`v2.1`, `v2.2`), `viirs_satellite` (`npp`, or `npp-j01` for
  2022).
- For each zone `<z>`:
  - `viirs_pixels<z>`: pixels whose centre lies in the zone;
  - `viirs_lit_pixels<z>`: those with masked radiance above 0;
  - `viirs_radiance<z>`: summed masked radiance (nW/cm²/sr, unweighted by pixel
    area).

  The zones are:
  - `` (no suffix): the whole municipality;
  - `_protected`: areas that bar mining and were in force by 2012
    (`protected_areas`, lead 1's IGD areas);
  - `_gold5km`: within 5 km of an SGB gold occurrence (`geology`, any economic
    status);
  - `_gold5km_protected`: both.
- `viirs_cf_cvg_mean`: mean cloud-free observations over the municipality's
  pixels (missing in 2013).

Semantics:

- Negative masked radiance (a few cells; −1.5 marks a missing median in v2) is
  set to 0.
- v2.2 changes the method at the 2021/2022 boundary: a month needs only one
  cloud-free observation for the median (v2.1: three), and cells without a
  median are left out of the background threshold. NOAA-20 enters in 2022.
- The composites jump in some years, and most in dim municipalities. In 2023,
  the first NPP-only v2.2 year, log radiance rises 0.27 in the dimmest fifth of
  Legal Amazon municipalities and 0.13 in the brightest. Compare across years
  with year effects by brightness class, not levels or common year effects.
- 2012 and 2013 are not corrected for stray light (`vcmcfg`); later years are
  (`vcmslcfg`).
