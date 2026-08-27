# Handoff: data path migration on scicore

Two sessions' worth of output-path renames landed in this repo. Nothing changed
in *content* — every step below is a plain `mv`, no reprocessing needed,
except the one flagged fix at the bottom.

## 0. Sync code first

```bash
git pull   # or however you normally sync -- just make sure you're on/past commit 89454d7
```

The sensor_data `archives/` and `water_quality_cleaning/` subfolder moves
(step 4 below) are **not committed yet** on my end — hold off on those two
`mv` blocks until that lands, or check with me first.

## 1. Already-committed renames (commit 89454d7) — run now

```bash
cd /scicore/home/meiera/schulz0022/projects/river-pollution-brazil

# gadm: raw file moves under raw/
mkdir -p data/gadm/raw
[ -f data/gadm/gadm41_BRA.gpkg ] && mv data/gadm/gadm41_BRA.gpkg data/gadm/raw/gadm41_BRA.gpkg

# river_network: 5 processed outputs get the river_network_ prefix
cd data/river_network/processed 2>/dev/null && {
  [ -f river_trenches.parquet ]        && mv river_trenches.parquet        river_network_trenches.parquet
  [ -f drainage_areas.parquet ]        && mv drainage_areas.parquet        river_network_drainage_areas.parquet
  [ -f river_system_matrices.pkl ]     && mv river_system_matrices.pkl     river_network_system_matrices.pkl
  [ -f trench_adm2_matches.parquet ]   && mv trench_adm2_matches.parquet   river_network_trench_adm2_matches.parquet
  [ -f adm2_dominant_systems.parquet ] && mv adm2_dominant_systems.parquet river_network_adm2_dominant_systems.parquet
  cd -
}

# biomes: biome_ -> biomes_
cd data/biomes/processed 2>/dev/null && {
  [ -f biome_adm2.parquet ]   && mv biome_adm2.parquet   biomes_adm2.parquet
  [ -f biome_sensor.parquet ] && mv biome_sensor.parquet biomes_sensor.parquet
  cd -
}

# sensor_data: extract + aggregate outputs get the sensor_data_ prefix (still flat at this point)
cd data/sensor_data/processed/extract 2>/dev/null && {
  [ -f water_quality.parquet ]                    && mv water_quality.parquet                    sensor_data_water_quality.parquet
  [ -f streamflow.parquet ]                       && mv streamflow.parquet                       sensor_data_streamflow.parquet
  [ -f stations_rivers.parquet ]                  && mv stations_rivers.parquet                  sensor_data_stations_rivers.parquet
  [ -f water_quality_transformations.json ]       && mv water_quality_transformations.json       sensor_data_water_quality_transformations.json
  [ -f water_quality_cleaning_flags.parquet ]     && mv water_quality_cleaning_flags.parquet     sensor_data_water_quality_cleaning_flags.parquet
  [ -f water_quality_cleaning_summary.parquet ]   && mv water_quality_cleaning_summary.parquet   sensor_data_water_quality_cleaning_summary.parquet
  cd -
}
cd data/sensor_data/processed/aggregate 2>/dev/null && {
  [ -f water_quality_streamflow.parquet ] && mv water_quality_streamflow.parquet sensor_data_water_quality_streamflow.parquet
  cd -
}

# climate: extract-stage output only (aggregate outputs already correctly named)
cd data/climate/processed/extract 2>/dev/null && {
  [ -f era5_land.parquet ] && mv era5_land.parquet climate_era5_land.parquet
  cd -
}

# health: everything gets the health_ prefix
cd data/health/processed 2>/dev/null && {
  [ -f hospitalizations.parquet ]                     && mv hospitalizations.parquet                     health_hospitalizations.parquet
  [ -f hospitalizations_icd10_chapter.parquet ]       && mv hospitalizations_icd10_chapter.parquet       health_hospitalizations_icd10_chapter.parquet
  [ -f hospitalizations_selected_morbidity_list.parquet ] && mv hospitalizations_selected_morbidity_list.parquet health_hospitalizations_selected_morbidity_list.parquet
  [ -f birth_weight.parquet ]                         && mv birth_weight.parquet                         health_birth_weight.parquet
  [ -f gestational_duration.parquet ]                 && mv gestational_duration.parquet                 health_gestational_duration.parquet
  [ -f scraping_pre_1996.csv ]                        && mv scraping_pre_1996.csv                        health_scraping_pre_1996.csv
  [ -f scraping_post_1996.csv ]                       && mv scraping_post_1996.csv                       health_scraping_post_1996.csv
  cd -
}
```

land_cover and population need no data movement (already correctly named).

## 2. Not yet committed — hold until it lands, then run

```bash
# sensor_data raw archives move into their own subfolder (was flat in raw/,
# mixed in with sensor_data.duckdb/sensor_downloads.duckdb)
cd data/sensor_data/raw 2>/dev/null && {
  mkdir -p archives
  find . -maxdepth 1 -name '*.zip' -exec mv {} archives/ \;
  cd -
}

# 3 water-quality-cleaning QA byproducts move into their own extract subfolder
cd data/sensor_data/processed/extract 2>/dev/null && {
  mkdir -p water_quality_cleaning
  for f in sensor_data_water_quality_transformations.json \
           sensor_data_water_quality_cleaning_flags.parquet \
           sensor_data_water_quality_cleaning_summary.parquet; do
    [ -f "$f" ] && mv "$f" "water_quality_cleaning/$f"
  done
  cd -
}
```

## 3. Actual fix needed (not a rename) — sensor_data's `stations` table

Your `data verify --source sensor_data` was crashing on `KeyError: 'geometry'`
reading the `stations` table — it's stale/malformed (missing the geometry
column its own metadata expects), unrelated to any rename. The crash is now
handled gracefully (reports a failed check instead), but the table itself is
still bad. Cheapest fix, no re-scraping:

```python
from src.data.sources.sensor_data.fetch.stations.inventory import fetch_station_inventory, preprocess_station_inventory
fetch_station_inventory(root_dir=".")
preprocess_station_inventory(root_dir=".")
```

## 4. Verify

```bash
python -m src.cli data summary
```

Each source's `Preprocess Status` should go back to reflecting real state
instead of `outstanding` (paths not found) — if a source shows `outstanding`
after migrating, check the exact filename it wrote vs. what `data summary`
expects.
