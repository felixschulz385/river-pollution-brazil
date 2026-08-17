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
- `src/experiments/`: notebooks and exploratory analysis.
- `data/`: local data products used by the codebase.
- `output/`: analysis outputs, figures, and documents.
- `setup/`: Docker and Slurm/HPC support files.
- `tests/`: automated tests for the maintained Python components.
- `submission/`: archived submission-related files.

## Current Entry Points

The repository’s current top-level Python entry point is:

```bash
python3 -m src.cli --help
```

This exposes two styles of commands:

- grouped commands under `data` and `analysis`
- flat compatibility aliases such as `health`, `water-quality`, `land-cover`, `population`, and `river-network`

Each data submodule also owns its own standalone entry point, e.g.
`python3 -m src.data.climate --help`; `src/cli.py` delegates to these rather
than duplicating their argument parsing.

## Data Workflows

### Health

Health data can be fetched and preprocessed through the CLI:

```bash
python3 -m src.cli data health fetch --subtype all
python3 -m src.cli data health preprocess --subtype all
```

Supported health subtypes are `mortality`, `hospitalization`, and `birth`.

The repository currently contains derived health outputs such as:

- `data/health/birth_weight.parquet`
- `data/health/gestational_duration.parquet`
- `data/health/hospitalizations.parquet`
- `data/health/hospitalizations_icd10_chapter.parquet`
- `data/health/hospitalizations_selected_morbidity_list.parquet`

### Water Quality

Water-quality processing is organized into fetch, preprocess, and assemble stages:

```bash
python3 -m src.cli data water-quality fetch --root-dir .
python3 -m src.cli data water-quality preprocess --root-dir .
python3 -m src.cli data water-quality assemble --root-dir .
```

The water-quality fetch pipeline supports additional options for browser execution and partial reruns, including:

- `--headless`
- `--keep-browser-on-error`
- `--single-station`
- `--fetch-mode`
- `--preprocess-workers`
- `--preprocess-backend`

The main assembled output currently tracked in the repository is:

- `data/sensor_data/water_quality_streamflow.parquet`

Transformation metadata used by the analysis pipeline is stored in:

- `data/sensor_data/water_quality_transformations.json`

### Land Cover

Land-cover processing is also organized into fetch, preprocess, and assemble stages:

```bash
python3 -m src.cli data land-cover fetch
python3 -m src.cli data land-cover preprocess --river-network-path data/river_network
python3 -m src.cli data land-cover assemble --variant sensor --river-network-path data/river_network
```

In the current code, `fetch` is only a placeholder and expects raw MapBiomas files to be obtained manually and placed in the configured data directory.

The assembly step supports at least two output variants:

- `sensor`: upstream land cover by monitoring station and year
- `adm2`: upstream land cover by municipality-level administrative unit and year

Current derived land-cover outputs include:

- `data/land_cover/land_cover_sensor_upstream.parquet`

### Population

Population data is processed through:

```bash
python3 -m src.cli data population fetch --root-dir .
python3 -m src.cli data population preprocess --root-dir .
```

The population fetch step pulls raw data from BigQuery and therefore requires appropriate Google Cloud credentials and access to the configured billing project.

The cleaned output is:

- `data/population/population.parquet`

### River Network

The river-network workflow is exposed through:

```bash
python3 -m src.cli data river-network generate \
  --gpkg-path path/to/source.gpkg \
  --output-dir data/river_network
```

Optional arguments allow spatial subsetting and administrative annotation, including:

- bounding box arguments `--min-lon`, `--min-lat`, `--max-lon`, `--max-lat`
- `--gadm-path`
- `--gadm-layer`
- `--gadm-adm2-layer`

Current river-network outputs in `data/river_network/` include:

- `trenches.parquet`
- `river_trenches.parquet`
- `drainage_areas.parquet`

The expected persisted formats are described in `src/data/output_file_formats.md`.

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

- `data/sensor_data/water_quality_streamflow.parquet`
- `data/land_cover/land_cover_sensor_upstream.parquet`
- `data/sensor_data/water_quality_transformations.json`
- `data/river_network/trenches.parquet`
- `output/analysis/sensor_data/`

Each run writes a dedicated model subdirectory with:

- `results.parquet`
- `manifest.parquet`
- `summary.json`
- `settings.json`

## Documents And Figures

The repository includes project outputs beyond code:

- figures under `output/figures/`
- Quarto documents under `output/documents/`
- a paper source at `output/documents/thesis_paper/paper.qmd`

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
- `src/experiments/` contains exploratory notebooks and is useful for context, diagnostics, and replication of intermediate development steps.

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
