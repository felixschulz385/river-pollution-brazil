# Land Use Change, River Pollution and Health Externalities in Brazil

This repository contains the code, intermediate data products, analysis outputs, and supporting material for a research project on deforestation, river pollution, and health externalities in Brazil. The project combines geospatial river-network construction, land-cover aggregation, water-quality data processing, health and population data preparation, and downstream econometric analysis.

The main maintained workflow is organized around a repository-level Python CLI under `src/cli.py`.

![Drainage polygon processing](/output/figures/weights_example.png)

## Project Scope

At a high level, the repository supports five linked tasks:

1. Build a river-network representation for Brazil, including drainage areas and upstream reachability relationships.
2. Process land-cover data and aggregate upstream exposure measures by distance band.
3. Fetch, clean, and assemble water-quality sensor observations.
4. Fetch and preprocess health and population data.
5. Estimate analysis models linking upstream land cover and water quality outcomes.

The repository also contains exploratory notebooks, figures, and Quarto documents used during the project.

## Repository Layout

The main directories are:

- `src/`: Python and R source code.
- `src/data/`: data ingestion, preprocessing, and assembly workflows.
- `src/analysis/`: analysis pipelines, especially the sensor-data regression workflow.
- `src/assemble/`: standalone R assembly scripts.
- `data/`: local data products used by the codebase.
- `output/`: analysis outputs, figures, and documents.
- `output/notebooks/`: analysis notebooks, split into `pollution/` (sensor panel), `health/` (ADM2 panel) and `garimpo/` (measurement of garimpo); see `output/notebooks/README.md`.
- `output/research/`: research design notes, including `mining_economics_leads.md` (the garimpo-supply design and its evidence).
- `setup/`: Docker and Slurm/HPC support files.
- `tests/`: automated tests for the maintained Python components.

## Current Entry Points

The repository’s current top-level Python entry point is:

```bash
python3 -m src.cli --help
```

`data` exposes a verb-first API, with `--source` picking which pipeline
source a verb applies to:

```bash
python3 -m src.cli data summary                       # pipeline completeness overview
python3 -m src.cli data verify  --source <source>      # verify one source's outputs
python3 -m src.cli data fetch      --source <source> [source flags...]
python3 -m src.cli data preprocess --source <source> [--phase extract|aggregate] [source flags...]
python3 -m src.cli data assemble   --dataset <id>       # join preprocessed sources
```

`--source` and `--phase` must precede any source-specific flags. `--source`
is one of `health`, `climate`, `sensor_data`, `land_cover`, `population`,
`river_network`, `biomes`, `protected_areas`, `geology`, `malaria`, `commodity_prices`, `mining`, `enforcement`, `nightlights`. `--phase` only applies to sources whose
preprocessing has two stages (`climate`, `land_cover`): `extract` (raw →
per-source processed output) and `aggregate` (roll-up into sensor/ADM2
upstream panels). `sensor_data` has a single preprocess step that does both
in one call.

`fetch`/`preprocess`/`assemble` refuse to run if a prerequisite source
hasn't been fetched/preprocessed yet (e.g. `land_cover preprocess` requires
`river_network` to be preprocessed first), printing exactly which
prerequisites are missing and how to fix them. Add `--chain` to
auto-run whatever's missing first, in the correct order, or
`--skip-dependency-check` to bypass the check entirely.

Add `--slurm` to `fetch`/`preprocess`/`assemble` to submit the same
invocation as a Slurm job instead of running it locally — see
`setup/slurm_scripts/README.md` and `setup/slurm_jobs.yaml`. Prerequisite
steps run via `--chain` always run locally, even when combined with
`--slurm`; only the originally requested step is submitted as a job.

Each data submodule also owns its own standalone entry point, e.g.
`python3 -m src.data.sources.climate --help`; `src/cli.py` delegates to these rather
than duplicating their argument parsing.

## Data Workflows

### Health

Health data can be fetched and preprocessed through the CLI:

```bash
python3 -m src.cli data fetch      --source health --subtype all
python3 -m src.cli data preprocess --source health --subtype all
```

Supported health subtypes are `mortality`, `hospitalization`, and `birth`.
The fetch scrapes DATASUS TabNet with Selenium (headless Chrome), so it needs
internet access.

Derived health outputs, in `data/health/processed/` (formats in
`src/data/output_file_formats.md`):

- `health_birth_weight.parquet`, `health_gestational_duration.parquet`,
  `health_congenital_anomaly.parquet`, `health_apgar_1.parquet`,
  `health_apgar_5.parquet`: SINASC births by category
- `health_births.parquet`: the births summary (`--subtype birth_summary`)
- `health_mortality.parquet`, `health_infant_deaths.parquet`: SIM deaths by age
  band, and infant deaths by ICD-10 chapter
- `health_hospitalizations.parquet`, `health_hospitalizations_icd10_chapter.parquet`,
  `health_hospitalizations_selected_morbidity_list.parquet`: SIH hospitalizations

### Water Quality

Water-quality processing is organized into a fetch stage (scrape raw data,
export it to parquet) and a single preprocess stage (clean the raw parquet,
then join it with GADM/river_network into the final panel):

```bash
python3 -m src.cli data fetch      --source sensor_data --root-dir .
python3 -m src.cli data preprocess --source sensor_data --root-dir .
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

The main assembled output currently tracked in the repository is:

- `data/sensor_data/processed/aggregate/sensor_data.parquet`

Transformation metadata used by the analysis pipeline is stored in:

- `data/sensor_data/processed/extract/water_quality_cleaning/sensor_data_water_quality_transformations.json`

### Land Cover

Land-cover processing is also organized into fetch and preprocess (extract, then aggregate) stages:

```bash
python3 -m src.cli data fetch      --source land_cover
python3 -m src.cli data preprocess --source land_cover --phase extract --river-network-path data/river_network/processed
python3 -m src.cli data preprocess --source land_cover --phase aggregate --river-network-path data/river_network/processed
```

`fetch` downloads the MapBiomas Collection 10 annual coverage GeoTIFFs
(1985–2024) from MapBiomas's public storage into
`data/land_cover/raw/lc_mapbiomas10_30`, skipping years already present. It
needs internet access, so run it on the transfer node.

The aggregate stage builds two output variants, and runs both by default
(`--variant all`); pass `--variant sensor` or `--variant adm2` for just one:

- `sensor`: upstream land cover by monitoring station and year
- `adm2`: upstream land cover by municipality-level administrative unit and year

Current derived land-cover outputs include:

- `data/land_cover/processed/aggregate/land_cover_sensor_upstream.parquet`
- `data/land_cover/processed/aggregate/land_cover_adm2_upstream.parquet`

`--product mining` builds the MapBiomas Collection 11 mining-substances layer
(mining by garimpo/industrial, substance and phase) through the same two
phases, ADM2 only:

```bash
python3 -m src.cli data preprocess --source land_cover --phase extract   --product mining --slurm
python3 -m src.cli data preprocess --source land_cover --phase aggregate --product mining --slurm
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
python3 -m src.cli data fetch      --source population --root-dir .
python3 -m src.cli data preprocess --source population --root-dir .
```

The population fetch step pulls raw data from BigQuery and therefore requires appropriate Google Cloud credentials and access to the configured billing project.

The cleaned output is:

- `data/population/processed/population.parquet`

### River Network

The river-network workflow has no automated fetch (raw hydrography/GADM files
are placed manually) and is exposed through a single preprocess step:

```bash
python3 -m src.cli data preprocess --source river_network \
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

Current river-network outputs in `data/river_network/processed/` include:

- `river_network_trenches.parquet`
- `river_network_drainage_areas.parquet`
- `river_network_otto_basins.parquet`: Otto Pfafstetter basin codes (levels 1–6) per trench. Rebuild it alone, without the full pipeline, with `python3 -m src.data.sources.river_network otto-basins`.

The expected persisted formats are described in `src/data/output_file_formats.md`.

### Protected Areas

Indigenous lands from FUNAI, with the date of each stage, and conservation units
(CNUC) from INPE TerraBrasilis, one polygon per area, with whether mining is
barred there, since when, and whether the area was in force by 2012, and their
area shares per municipality. Geology needs the merged areas, so this source
runs first. The fetch downloads the conservation units. FUNAI's GeoServer does
not answer from the HPC, so its `tis_poligonais_portarias` layer is downloaded
by hand over a Brazilian VPN into `data/protected_areas/raw/funai_<date>/`.

```bash
python3 -m src.cli data fetch      --source protected_areas
python3 -m src.cli data preprocess --source protected_areas --slurm
python3 -m src.cli data preprocess --source protected_areas --slurm --product trenches
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
python3 -m src.cli data fetch      --source geology
python3 -m src.cli data preprocess --source geology --slurm
python3 -m src.cli data preprocess --source geology --product upstream --slurm
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
python3 -m src.cli data fetch      --source malaria
python3 -m src.cli data preprocess --source malaria
```

Outputs: `data/malaria/processed/malaria_adm2.parquet`, cases by infection
municipality and notification month `data/malaria/processed/malaria_adm2_monthly.parquet`,
and the garimpo flow table `data/malaria/processed/malaria_garimpo_flows.parquet`.

### Commodity Prices

The World Bank Pink Sheet gold price and the Banco Central do Brasil's
BRL/USD rate (SGS series 1), both downloaded on every fetch:

```bash
python3 -m src.cli data fetch      --source commodity_prices
python3 -m src.cli data preprocess --source commodity_prices
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
python3 -m src.cli data preprocess --source mining --slurm
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
python3 -m src.cli data preprocess --source enforcement --slurm
```

Outputs: `data/enforcement/processed/ibama_records.parquet` and
`data/enforcement/processed/enforcement_adm2.parquet`.

### Night Lights

EOG's annual VIIRS night-light composites (VNL v2.1 for 2012–2021, v2.2 from
2022), as a measure of activity: radiance and lit pixels per municipality and
year, over the whole municipality, inside protected areas that bar mining, and
within 5 km of SGB gold occurrences. EOG requires a login, so the files are
downloaded by hand and placed, as served, in `data/nightlights/raw/`. The
preprocess reads the `protected_areas` output and the `geology` occurrences.

```bash
python3 -m src.cli data preprocess --source nightlights --slurm
```

Output: `data/nightlights/processed/nightlights_adm2.parquet`.

## Analysis Workflow

The maintained analysis pipeline is the sensor-data workflow under `src/analysis/sensor_data/`.

At the repository level it can be invoked as:

```bash
python3 -m src.cli analysis sensor-data list-groups
python3 -m src.cli analysis sensor-data run
```

Direct invocation is also possible:

```bash
python3 -m src.analysis.cli --help
```

This second command currently requires the analysis dependencies to be installed locally; in this environment it fails if packages such as `pandas` are missing.

The analysis CLI supports:

- pollutant-group selection by type or importance tier
- explicit pollutant lists
- land-cover subclass filtering
- distance-step truncation
- custom input and output paths

Example:

```bash
python3 -m src.cli analysis sensor-data run \
  --pollutant-group-kind type \
  --pollutant-group core_physicochemical \
  --land-cover-subclasses c41,c42 \
  --max-distance-step 3
```

By default, analysis settings point to:

- `data/sensor_data/processed/aggregate/sensor_data.parquet`
- `data/land_cover/processed/aggregate/land_cover_sensor_upstream.parquet`
- `data/sensor_data/processed/extract/water_quality_cleaning/sensor_data_water_quality_transformations.json`
- `data/river_network/processed/river_network_trenches.parquet`
- `output/analysis/sensor_data/`

Each run writes a dedicated model subdirectory with:

- `results.parquet`
- `manifest.parquet`
- `summary.json`
- `settings.json`

## Documents And Figures

The repository includes project outputs beyond code:

- figures under `output/figures/`
- research notes under `output/research/`

The master's-thesis material (replication files in `submission/`, Quarto documents,
presentations and thesis figures) was removed from the tree; it is preserved at the
`thesis-archive` tag.

![Project workflow](/output/figures/data_workflow.png)

## Environment And Dependencies

This repository does not currently include a fully specified Python environment file such as `pyproject.toml`, `requirements.txt`, or `environment.yml`. Setup therefore has to be reconstructed from the codebase and, where relevant, the execution environment used during the project.

The maintained Python code uses packages from the following areas:

- data handling: `pandas`, `duckdb`, `numpy`
- geospatial processing: `geopandas`, `shapely`, `rasterio`, `rioxarray`, `xarray`, `odc-geo`, `pysheds`, `geocube`
- scientific computing: `scipy`, `sparse`, `joblib`
- modeling: `pyfixest`, `statsmodels`, `scikit-learn`
- browser automation and downloading: `selenium`, `webdriver-manager`, `requests`
- testing: `pytest`

There are also R scripts and notebooks that depend on an R environment not declared in a machine-readable way in this repository.

For browser-based scraping, the repository includes a minimal Selenium Docker image at `setup/docker/Dockerfile`.

## Suggested Local Setup

A practical starting point for local work is:

1. Create a Python virtual environment.
2. Install core scientific, geospatial, testing, and CLI dependencies used by the maintained modules.
3. Install any additional packages required by the specific workflow you intend to run.
4. Ensure browser automation dependencies are available if you need the fetch pipelines.

For example:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scipy duckdb geopandas shapely rasterio rioxarray xarray odc-geo geocube pysheds sparse joblib pyfixest statsmodels scikit-learn selenium webdriver-manager requests pytest
```

This is a reasonable baseline, not a guaranteed complete environment for every script in the repository.

## Tests

The repository contains automated tests for several of the maintained Python components, including:

- land-cover preprocessing and assembly
- population preprocessing
- sensor-data analysis logic

Run the test suite with:

```bash
pytest
```

The tests are concentrated in `tests/data/<submodule>/` (mirroring `src/data/`),
plus `tests/analysis/sensor_data/` for the analysis layer.

## Status And Conventions

The codebase reflects a research project with multiple generations of tooling. In practice:

- `src/cli.py`, `src/data/`, `src/analysis/`, and the corresponding tests are the best starting points for current work.

## Data Availability

Some derived data products are present in the repository, but raw source data access varies by workflow. Depending on the module, data may need to be:

- downloaded manually
- fetched from web sources
- pulled from Google BigQuery
- regenerated from local or remote geospatial inputs

If full replication data is not included, the relevant scripts and output schemas are still present and can be used to reconstruct the pipeline where source access permits.

## License

Unless stated otherwise in individual files or downstream materials, the written project materials are licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License:

https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode
