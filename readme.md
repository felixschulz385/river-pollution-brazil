# Land Use Change, River Pollution and Health Externalities in Brazil

Research code for land use, river pollution and health in Brazil, and for the supply of
illegal gold mining (*garimpo*) in the Amazon. The repository builds a river network for
Brazil, processes land-cover, water-quality, health, malaria, mining, enforcement and
related data along it, assembles them into panels, and estimates the analysis in
notebooks.

![Drainage polygon processing](/output/figures/weights_example.png)

## Project Scope

The project has three linked arms:

| Arm | Question | Panel | Write-up |
|---|---|---|---|
| Pollution | Does upstream land-cover change degrade river water quality? | `data/assembly/sensor_panel.parquet` (station × sample date) | `output/notebooks/pollution/memo.md` |
| Health | Does upstream land-cover change affect health in downstream municipalities? | `data/assembly/adm2_panel.parquet` (municipality × year) | `output/notebooks/health/memo.md` |
| Garimpo | What drives garimpo supply, who responds, and which policy lever reduces it? | `adm2_panel` and monthly tables of the sources | `output/research/mining_economics_leads.md` |

## Repository Layout

- `src/cli.py`: the command-line entry point.
- `src/data/`: data pipelines.
  - `sources/<source>/`: one package per data source (fetch, preprocess).
  - `assembly/`: joins processed sources into the panels defined in
    `setup/assembly_datasets.yaml`.
  - `verification/`: checks of each source's outputs.
  - `shared/`: paths, Slurm submission, upstream aggregation, web drivers.
  - `output_file_formats.md`: the format of every processed table.
- `src/analysis/`: estimators used by the notebooks, one module per topic, and the
  sensor-data regression pipeline (`sensor_data/`).
- `output/notebooks/`: analysis notebooks in `pollution/`, `health/` and `garimpo/`.
- `output/research/`: the garimpo research design and evidence, and the literature
  search.
- `output/figures/`: figures used in this readme.
- `scripts/`: the manual staging steps of the MapBiomas mining layer.
- `setup/`: Slurm job specs and scripts, the panel definitions, a Docker image for
  Selenium.
- `tests/`: tests mirroring `src/`.
- `data/`, `log/`, `output/analysis/`: data, job logs and analysis tables; not tracked.

The master's-thesis material (replication files in `submission/`, Quarto documents,
presentations and thesis figures) is preserved at the `thesis-archive` tag.

## Documentation

| Document | Content |
|---|---|
| `readme.md` | Setup, the CLI, how to fetch and preprocess each source |
| `src/data/output_file_formats.md` | Columns, keys and construction of every processed table |
| `src/data/sources/climate/README.md` | The ERA5-Land variants and how to run them |
| `setup/slurm_scripts/README.md` | Slurm submission and the remaining scripts |
| `output/notebooks/README.md` | Index of the notebooks and the code behind them |
| `output/notebooks/pollution/memo.md`, `output/notebooks/health/memo.md` | Model, results and caveats of each arm |
| `output/research/mining_economics_leads.md` | The garimpo research design: summary, leads, work plan, Evidence E1–E17, data inventory |

## Environment

Python ≥ 3.11. `pyproject.toml` lists the dependencies; install the package in editable
mode:

```bash
pip install -e .
```

The fetch pipelines that scrape web forms (DATASUS TabNet, the water-quality portal)
need Chrome; `webdriver-manager` fetches the matching chromedriver on first use. A
minimal Selenium image is in `setup/docker/Dockerfile`. The population fetch reads from
Google BigQuery and needs Google Cloud credentials.

## Running on the HPC (scicore)

- **Compute nodes have no internet access.** Every `data fetch` and every scraper runs
  on a node with internet: an interactive session with an HTTP proxy
  (`HTTP_PROXY`/`HTTPS_PROXY`) or the transfer node. Fetches resume from their per-batch
  manifests (`data/<source>/raw/<table>/manifest.jsonl`), so long fetches can run in
  the background and be restarted.
- **Everything heavy goes to Slurm:** add `--slurm` to `data preprocess` and
  `data assemble` (resource specs in `setup/slurm_jobs.yaml`, keyed
  `<source>.<verb>[.<phase>][.<product>]`), and execute notebooks with
  `sbatch setup/slurm_scripts/execute_notebook.sh <notebook.ipynb>`.

## Command-Line Interface

```bash
python -m src.cli --help
```

`data` exposes a verb-first API, with `--source` picking which pipeline source a verb
applies to:

```bash
python -m src.cli data summary                       # pipeline completeness overview
python -m src.cli data verify     --source <source>  # verify one source's outputs
python -m src.cli data fetch      --source <source> [source flags...]
python -m src.cli data preprocess --source <source> [--phase extract|aggregate] [source flags...]
python -m src.cli data assemble   --dataset <id>     # join preprocessed sources
```

`--source` and `--phase` must precede any source-specific flags. `--source` is one of
`gadm`, `health`, `climate`, `sensor_data`, `land_cover`, `population`,
`river_network`, `biomes`, `geology`, `malaria`, `commodity_prices`, `mining`,
`enforcement`, `protected_areas`, `nightlights`, `cloud_cover`. `--phase` only applies
to sources whose preprocessing has two stages (`climate`, `land_cover`): `extract`
(raw → per-source processed output) and `aggregate` (roll-up into sensor/ADM2 upstream
panels).

`fetch`/`preprocess`/`assemble` refuse to run if a prerequisite source hasn't been
fetched/preprocessed yet (e.g. `land_cover preprocess` requires `river_network` to be
preprocessed first), printing which prerequisites are missing and how to fix them. Add
`--chain` to run whatever is missing first, in the correct order, or
`--skip-dependency-check` to bypass the check.

Add `--slurm` to `fetch`/`preprocess`/`assemble` to submit the same invocation as a
Slurm job instead of running it locally (see `setup/slurm_scripts/README.md`).
Prerequisite steps run via `--chain` always run locally; only the requested step is
submitted as a job.

Each source package also has its own entry point, e.g.
`python -m src.data.sources.climate --help`; `src/cli.py` delegates to these.

The panels are rebuilt with `data assemble --dataset adm2_panel` and
`--dataset sensor_panel`, and checked with `data verify --source assembly`.

## Data Workflows

The formats of all outputs are in `src/data/output_file_formats.md`.

### GADM and Biomes

GADM 4.1 boundaries for Brazil are placed by hand as `data/gadm/raw/gadm41_BRA.gpkg`;
the preprocess simplifies them into the geopackage that the biomes, river-network and
sensor-data sources share. The biomes source downloads IBGE's biome polygons and maps
each municipality and monitoring station to a biome.

```bash
python -m src.cli data preprocess --source gadm
python -m src.cli data fetch      --source biomes
python -m src.cli data preprocess --source biomes
```

Outputs: `data/gadm/processed/gadm_simplified.gpkg`,
`data/biomes/processed/biomes_adm2.parquet` and `biomes_sensor.parquet`.

### Health

DATASUS data through TabNet: SIH hospitalizations, SINASC births and SIM deaths, by
municipality and year.

```bash
python -m src.cli data fetch      --source health --subtype all
python -m src.cli data preprocess --source health --subtype all
```

Supported subtypes are `hospitalization`, `birth` and `mortality`. The fetch scrapes
TabNet with Selenium (add `--headless` to hide the browser) and needs internet: run it
in an interactive session with the HTTP proxy, or on a local machine and copy
`data/health/raw/` back with `rsync` (the raw folders carry the manifests). TabNet is
slow (15–25 s per form page) and flaky:

- A rerun skips every batch marked `completed` in its manifest and present on disk, and
  retries only failed or missing ones. The SINASC and SIM fetchers read the available
  years from the form, so a rerun adds new years.
- `hospitalization` also plans the SIH totals and ICD-chapter requests; a newly complete
  SIH year is fetched for all three.
- If the fetch stops with `DATASUS morbidity-list form ... is missing N of M configured
  values`, the form loaded only partially; rerun it.
- A failed year listing falls back to a fixed year range and rewrites the manifest, so
  check the manifests afterwards:

```bash
grep -c '"pending"\|"failed"' data/health/raw/SIH_*/manifest.jsonl   # expect 0
grep -c '"completed"' data/health/raw/birth_outcome_*/manifest.jsonl data/health/raw/mortality_age_*/manifest.jsonl data/health/raw/infant_deaths_icd10/manifest.jsonl
```

Then preprocess, rebuild `adm2_panel` and re-run `output/notebooks/health/01_descriptives.ipynb`.

Derived health outputs, in `data/health/processed/`:

- `health_birth_weight.parquet`, `health_gestational_duration.parquet`,
  `health_congenital_anomaly.parquet`, `health_apgar_1.parquet`,
  `health_apgar_5.parquet`: SINASC births by category
- `health_births.parquet`: the births summary (`--subtype birth_summary`)
- `health_mortality.parquet`, `health_infant_deaths.parquet`: SIM deaths by age
  band, and infant deaths by ICD-10 chapter
- `health_hospitalizations.parquet`, `health_hospitalizations_icd10_chapter.parquet`,
  `health_hospitalizations_selected_morbidity_list.parquet`: SIH hospitalizations

### Climate

ERA5-Land variables (temperature, precipitation, soil moisture, runoff, evaporation),
fetched into a local Zarr store, extracted per river trench and aggregated upstream of
each station and municipality. The fetch variants and their caveats are in
`src/data/sources/climate/README.md`.

```bash
python -m src.cli data fetch      --source climate --subtype <subtype>
python -m src.cli data preprocess --source climate --phase extract --slurm
python -m src.cli data preprocess --source climate --phase aggregate --slurm
```

Outputs: `data/climate/processed/extract/climate_era5_land.parquet`,
`data/climate/processed/aggregate/climate_sensor_upstream.parquet` and
`climate_adm2_upstream_yearly.parquet`.

### Water Quality

Water-quality processing is organized into a fetch stage (scrape raw data,
export it to parquet) and a single preprocess stage (clean the raw parquet,
then join it with GADM/river_network into the final panel):

```bash
python -m src.cli data fetch      --source sensor_data --root-dir .
python -m src.cli data preprocess --source sensor_data --root-dir .
```

`fetch` has no GADM/river_network dependency; `preprocess` does (it needs
both preprocessed first -- pass `--chain` to run them automatically).

The water-quality fetch pipeline supports additional options for browser execution and partial reruns, including:

- `--headless`
- `--keep-browser-on-error`
- `--single-station`
- `--fetch-mode`
- `--preprocess-workers`
- `--preprocess-backend`

Outputs: `data/sensor_data/processed/aggregate/sensor_data.parquet`, and the
transformation metadata used by the sensor-data regression pipeline,
`data/sensor_data/processed/extract/water_quality_cleaning/sensor_data_water_quality_transformations.json`.

### Land Cover

Land-cover processing is also organized into fetch and preprocess (extract, then aggregate) stages:

```bash
python -m src.cli data fetch      --source land_cover
python -m src.cli data preprocess --source land_cover --phase extract --river-network-path data/river_network/processed
python -m src.cli data preprocess --source land_cover --phase aggregate --river-network-path data/river_network/processed
```

`fetch` downloads the MapBiomas Collection 10 annual coverage GeoTIFFs
(1985–2024) from MapBiomas's public storage into
`data/land_cover/raw/lc_mapbiomas10_30`, skipping years already present. It
needs internet access, so run it on the transfer node.

The aggregate stage builds two output variants, and runs both by default
(`--variant all`); pass `--variant sensor` or `--variant adm2` for just one:

- `sensor`: upstream land cover by monitoring station and year
- `adm2`: upstream land cover by municipality-level administrative unit and year

Outputs:

- `data/land_cover/processed/aggregate/land_cover_sensor_upstream.parquet`
- `data/land_cover/processed/aggregate/land_cover_adm2_upstream.parquet`

`--product mining` builds the MapBiomas Collection 11 mining-substances layer
(mining by garimpo/industrial, substance and phase) through the same two
phases, ADM2 only:

```bash
python -m src.cli data preprocess --source land_cover --phase extract   --product mining --slurm
python -m src.cli data preprocess --source land_cover --phase aggregate --product mining --slurm
```

Its rasters are not downloadable directly. They are staged in two manual steps:

1. `scripts/gee_export_mapbiomas_mining.js`, pasted into the Earth Engine Code
   Editor, exports the asset as tiles (and a national area table) to Google
   Drive; copy the tiles to `data/land_cover/raw/mapbiomas11_mining_tiles`.
2. `python scripts/prepare_mapbiomas_mining.py` merges the tiles into one
   GeoTIFF per year in `data/land_cover/raw/mapbiomas11_mining`, writes the
   pipeline legend and the national area by class and year.

Outputs: `data/land_cover/processed/extract/land_cover_mining.parquet` and
`data/land_cover/processed/aggregate/land_cover_mining_adm2_upstream.parquet`.

### Population

Population data is processed through:

```bash
python -m src.cli data fetch      --source population --root-dir .
python -m src.cli data preprocess --source population --root-dir .
```

The population fetch step pulls raw data from BigQuery and therefore requires appropriate Google Cloud credentials and access to the configured billing project.

The cleaned output is:

- `data/population/processed/population.parquet`

### River Network

The river-network workflow has no automated fetch (raw hydrography/GADM files
are placed manually) and is exposed through a single preprocess step:

```bash
python -m src.cli data preprocess --source river_network \
  --output-dir data/river_network/processed
```

`--gpkg-path` defaults to `data/river_network/raw/bho_2017_v_01_05_5k.gpkg`
(ANA's "Base Hidrográfica Ourinhos" 2017 v.01.05, 1:5k scale); pass it
explicitly to use a different hydrography geopackage.

Optional arguments allow spatial subsetting and administrative annotation, including:

- bounding box arguments `--min-lon`, `--min-lat`, `--max-lon`, `--max-lat`
- `--gadm-path`
- `--gadm-layer`
- `--gadm-adm2-layer`

Outputs in `data/river_network/processed/` include:

- `river_network_trenches.parquet`
- `river_network_drainage_areas.parquet`
- `river_network_otto_basins.parquet`: Otto Pfafstetter basin codes (levels 1–6) per trench. Rebuild it alone, without the full pipeline, with `python -m src.data.sources.river_network otto-basins`.

### Protected Areas

Indigenous lands from FUNAI, with the date of each stage, and conservation units
(CNUC) from INPE TerraBrasilis, one polygon per area, with whether mining is
barred there, since when, and whether the area was in force by 2012, and their
area shares per municipality. Geology needs the merged areas, so this source
runs first. The fetch downloads the conservation units. FUNAI's GeoServer does
not answer from the HPC, so its `tis_poligonais_portarias` layer is downloaded
by hand over a Brazilian VPN into `data/protected_areas/raw/funai_<date>/`.

```bash
python -m src.cli data fetch      --source protected_areas
python -m src.cli data preprocess --source protected_areas --slurm
python -m src.cli data preprocess --source protected_areas --slurm --product trenches
```

Outputs: `data/protected_areas/processed/protected_areas.parquet`,
`data/protected_areas/processed/protected_areas_adm2.parquet` and, with
`--product trenches`, the share of each river trench inside each area
(`protected_areas_trenches.parquet`).

### Geology

The Geological Survey of Brazil's mineral occurrences and gold provinces and
districts (`data/geology/raw/sgb_gold.gpkg`), counted per municipality
(`--product adm2`, the default) or upstream along the river network
(`--product upstream`, which needs the river network). Occurrences inside
protected areas that bar mining and were in force by 2012 (lead 1's IGD) are
counted separately, from the `protected_areas` output:

```bash
python -m src.cli data fetch      --source geology
python -m src.cli data preprocess --source geology --slurm
python -m src.cli data preprocess --source geology --product upstream --slurm
```

Outputs: `data/geology/processed/geology_adm2.parquet` and
`data/geology/processed/geology_adm2_upstream.parquet`.

### Malaria

Record-level SIVEP-Malária notifications 2004–2024 (Zenodo 20706957) and the
preliminary years from 2025 (Zenodo 20706949, updated monthly; each fetch that
finds the record changed adds a dated snapshot), counted by infection,
residence and notification municipality, with garimpo cases split into local
and incoming workers and infections abroad:

```bash
python -m src.cli data fetch      --source malaria
python -m src.cli data preprocess --source malaria
```

Outputs: `data/malaria/processed/malaria_adm2.parquet`, cases by infection
municipality and notification month `data/malaria/processed/malaria_adm2_monthly.parquet`,
and the garimpo flow table `data/malaria/processed/malaria_garimpo_flows.parquet`.

### Commodity Prices

The World Bank Pink Sheet gold price and the Banco Central do Brasil's
BRL/USD rate (SGS series 1), both downloaded on every fetch:

```bash
python -m src.cli data fetch      --source commodity_prices
python -m src.cli data preprocess --source commodity_prices
```

Outputs: monthly and annual gold prices in USD and BRL, with logs,
`data/commodity_prices/processed/gold_price_{monthly,annual}.parquet`.

### Mining Permits

The ANM Cadastro Mineiro (SCM) microdata, counted per municipality and year:
PLG (garimpo permit) requests, grants and permits in force, for all
substances, gold and tin. The ANM firewall blocks scripted downloads, so
`microdados-scm.zip` (from `dadosabertos.anm.gov.br/SCM/microdados/`) is
downloaded in a browser and placed in
`data/mining/raw/anm_scm_microdados_<date>/`; the newest snapshot is used.

```bash
python -m src.cli data preprocess --source mining --slurm
```

Outputs: `data/mining/processed/mining_permits_adm2.parquet` (municipality × year) and
`data/mining/processed/mining_requests_monthly.parquet` (requests by municipality and protocol month).

### Enforcement

IBAMA's Fiscalização open data: fines, embargoes, seizures and destruction
terms, cleaned into one record table (no personal data) with mining and garimpo
flags from the descriptions, and counted per municipality and year. The portal
(`dadosabertos.ibama.gov.br`) blocks requests from outside Brazil, so the files
are downloaded over a Brazilian connection and placed, with their provenance,
schema and checksum files, in `data/enforcement/raw/ibama_<YYYY-MM-DD>/`; the
newest complete snapshot is used.

```bash
python -m src.cli data preprocess --source enforcement --slurm
```

Outputs: `data/enforcement/processed/ibama_records.parquet` and
`data/enforcement/processed/enforcement_adm2.parquet`.

The same source carries the Ministry of the Environment's Priority List of
Amazon municipalities for deforestation control, and the monitored list of
those that left it, for 2008–2022 (Zenodo record 8412144, transcribed from the
Diário Oficial), recoded to municipalities against GADM:

```bash
python -m src.cli data fetch --source enforcement
python -m src.cli data preprocess --source enforcement --product priority_list
```

Output: `data/enforcement/processed/priority_list_adm2.parquet`.

### Cloud Cover

INPE's DETER cloud masks, as the share of each Legal Amazon municipality under
cloud per month, 2005–2016: the instrument of Assunção, Gandour & Rocha (2023)
for IBAMA's enforcement. TerraBrasilis does not publish the masks, so they are
read from the authors' openICPSR replication package (project 132281, V1),
placed by hand as `data/cloud_cover/raw/132281-V1.zip`. The same source counts
TerraBrasilis's DETER-B alerts (from August 2016) per municipality and month,
for all classes, clear-cut deforestation and mining, from the shapefile
download placed as `data/cloud_cover/raw/deter-amz-<DD>-<MM>-<YYYY>-*.zip`.

```bash
python -m src.cli data preprocess --source cloud_cover --slurm
python -m src.cli data preprocess --source cloud_cover --product deter_alerts --slurm
```

Outputs: `data/cloud_cover/processed/deter_clouds_adm2_month.parquet` and
`deter_clouds_adm2.parquet` (calendar- and PRODES-year means), and
`deter_alerts_adm2_month.parquet` and `deter_alerts_adm2.parquet` (calendar years).

### Night Lights

EOG's annual VIIRS night-light composites (VNL v2.1 for 2012–2021, v2.2 from
2022), as a measure of activity: radiance and lit pixels per municipality and
year, over the whole municipality, inside protected areas that bar mining, and
within 5 km of SGB gold occurrences. EOG requires a login, so the files are
downloaded by hand and placed, as served, in `data/nightlights/raw/`. The
preprocess reads the `protected_areas` output and the `geology` occurrences.

```bash
python -m src.cli data preprocess --source nightlights --slurm
```

Output: `data/nightlights/processed/nightlights_adm2.parquet`.


## Analysis

The analysis runs in notebooks under `output/notebooks/` (index in
`output/notebooks/README.md`). Each notebook imports its estimator from
`src/analysis/<topic>.py`, which is covered by tests in `tests/analysis/`. Notebooks
are executed in place on Slurm:

```bash
sbatch setup/slurm_scripts/execute_notebook.sh output/notebooks/garimpo/06_raids.ipynb
```

Tables go to `output/analysis/<arm>/` (not tracked); the executed notebooks carry them.

`src/analysis/sensor_data/` is a separate regression pipeline for the sensor panel,
run through the CLI (`python -m src.cli analysis sensor-data list-groups`,
`... analysis sensor-data run`), with pollutant-group selection, land-cover subclass
filters and distance-step truncation. It writes one directory per model to
`output/analysis/sensor_data/`. The pollution notebooks do not use it.

![Project workflow](/output/figures/data_workflow.png)

## Tests

```bash
python -m pytest -q -p no:cacheprovider
```

Tests mirror `src/`: `tests/data/sources/<source>/`, `tests/data/{assembly,shared,verification}/`
and `tests/analysis/`.

## Data Availability

Data are not tracked. Depending on the source, raw data are fetched from web sources
or APIs, read from Google BigQuery, or downloaded by hand where a portal blocks scripted
or foreign access (the source sections above say which). Raw data under `data/*/raw/`
are never modified; processed outputs are rebuilt only through `data preprocess` and
`data assemble`.

## License

Unless stated otherwise in individual files or downstream materials, the written project materials are licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License:

https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode
