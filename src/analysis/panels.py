"""Loaders and variable groups for the assembled analysis panels.

Shared by the notebooks under `output/notebooks/` so that the pollution
(sensor-indexed) and health (ADM2-indexed) workflows read the same columns
the same way. Variable groups are derived from `setup/assembly_datasets.yaml`
where possible rather than hard-coded, so they track the assembly config.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.sources.land_cover.composition import compute_kernel_weighted_composition
from src.data.sources.land_cover.constants import LAND_COVER_COMPOSITION_BUCKET_MAP
from src.data.sources.land_cover.mining import MINING_ADM2_UPSTREAM_PATH, compute_mining_composition

from .sensor_data.catalog import _classify_pollutant_type
from .settings import PROJECT_ROOT


ASSEMBLY_CONFIG_PATH = PROJECT_ROOT / "setup" / "assembly_datasets.yaml"
SENSOR_PANEL_PATH = PROJECT_ROOT / "data" / "assembly" / "sensor_panel.parquet"
ADM2_PANEL_PATH = PROJECT_ROOT / "data" / "assembly" / "adm2_panel.parquet"
HEALTH_PROCESSED_DIR = PROJECT_ROOT / "data" / "health" / "processed"
LAND_COVER_AGGREGATE_DIR = PROJECT_ROOT / "data" / "land_cover" / "processed" / "aggregate"
LAND_COVER_SENSOR_BUCKETED_PATH = LAND_COVER_AGGREGATE_DIR / "land_cover_sensor_upstream.parquet"
LAND_COVER_ADM2_BUCKETED_PATH = LAND_COVER_AGGREGATE_DIR / "land_cover_adm2_upstream.parquet"
MINING_ADM2_BUCKETED_PATH = PROJECT_ROOT / MINING_ADM2_UPSTREAM_PATH

# Years covered by both the land-cover (MapBiomas) and climate (ERA5-Land)
# upstream aggregates; outside this window the exposure/controls are missing.
EXPOSURE_YEARS = (1985, 2024)

LAND_COVER_SHARE_COLUMNS = [
    "lc_forest",
    "lc_nonforest_nat",
    "lc_pasture",
    "lc_agriculture",
    "lc_farming_unclassified",
    "lc_urban",
    "lc_mining",
    "lc_other",
    "lc_water",
]
LAND_COVER_ALR_COLUMNS = [
    "alr_pasture",
    "alr_agriculture",
    "alr_farming_unclassified",
    "alr_urban",
    "alr_mining",
    "alr_other",
]

CLIMATE_VARIABLES = ["2t", "2d", "tp", "sro", "ssro", "pev", "swvl1", "swvl2", "2t_daily_min", "2t_daily_max"]
SENSOR_CLIMATE_WINDOWS = ["day", "7d", "30d", "90d", "180d", "365d"]

STREAMFLOW_COLUMNS = [
    "streamflow_discharge_day",
    "streamflow_discharge_mean_7d",
    "streamflow_discharge_mean_31d",
]

# Columns of the water_quality_streamflow source that are sampling metadata
# or covariates rather than water-quality analytes.
SENSOR_NON_ANALYTE_COLUMNS = {
    "rained",
    "depth",
    "depth_parameter",
    "air_temperature",
    "trench_id",
    "streamflow_match_count",
    "streamflow_nonnull_day_count",
    "streamflow_total_weight",
    "streamflow_nearest_distance_m",
    *STREAMFLOW_COLUMNS,
}


def sensor_climate_columns(variables=None, windows=None):
    """Return `{variable}_mean_{window}` sensor-panel climate column names."""
    return [
        f"{variable}_mean_{window}"
        for variable in (variables or CLIMATE_VARIABLES)
        for window in (windows or SENSOR_CLIMATE_WINDOWS)
    ]


def _assembly_source_variables(dataset_id, source_name):
    with open(ASSEMBLY_CONFIG_PATH) as config_file:
        config = yaml.safe_load(config_file)
    for dataset in config["datasets"]:
        if dataset["id"] != dataset_id:
            continue
        for source in dataset["sources"]:
            if source["name"] == source_name:
                return list(source["variables"])
    raise KeyError(f"No source '{source_name}' in assembly dataset '{dataset_id}'.")


def sensor_analyte_columns():
    """Return the water-quality analyte columns declared for the sensor panel."""
    variables = _assembly_source_variables("sensor_panel", "water_quality_streamflow")
    return [variable for variable in variables if variable not in SENSOR_NON_ANALYTE_COLUMNS]


def analyte_type_groups(analytes):
    """Map each analyte to the pollutant type group used by the sensor_data pipeline."""
    return {analyte: _classify_pollutant_type(analyte) for analyte in analytes}


def _restrict_years(frame, years):
    if years is None:
        return frame
    first_year, last_year = years
    return frame[frame["year"].between(first_year, last_year)]


def load_sensor_panel(path=SENSOR_PANEL_PATH, years=EXPOSURE_YEARS):
    """Load the sensor x sample-datetime panel, restricted to `years` (None keeps all)."""
    frame = pd.read_parquet(path)
    frame = frame.dropna(subset=["datetime"])
    frame["year"] = frame["year"].astype(int)
    frame = _restrict_years(frame, years)
    frame["month"] = frame["datetime"].dt.month
    return frame.reset_index(drop=True)


def load_adm2_panel(path=ADM2_PANEL_PATH, years=EXPOSURE_YEARS):
    """Load the municipality x year panel, restricted to `years` (None keeps all).

    `has_upstream` is False for headwater municipalities, which have local
    (`local_*`) but no upstream (`lc_*`/`alr_*`, upstream climate) exposure.
    """
    frame = pd.read_parquet(path)
    frame["has_upstream"] = frame["lc_forest"].notna()
    return _restrict_years(frame, years).reset_index(drop=True)


def ring_land_cover(path, entity_column, min_km=0, max_km=np.inf, years=EXPOSURE_YEARS):
    """Recompute land-cover composition from the 25 km rings in [min_km, max_km).

    Same inv-sqrt kernel and ALR transform as assembly, restricted to a subset
    of the rings of a bucketed land-cover file (`LAND_COVER_*_BUCKETED_PATH`);
    weights are renormalized over the rings each entity has. ADM2 trench-body
    rings (negative buckets) are never included, as in assembly's upstream block.
    """
    buckets, bucket_map = _read_rings(path, min_km, max_km, years)
    return compute_kernel_weighted_composition(buckets, entity_columns=[entity_column, "year"], bucket_map=bucket_map)


def _read_rings(path, min_km, max_km, years):
    """Read the upstream rings in [min_km, max_km) and `years` of a bucketed file, with their bucket map."""
    bucket_map = {
        bucket: spec
        for bucket, spec in LAND_COVER_COMPOSITION_BUCKET_MAP.items()
        if min_km <= bucket < max_km
    }
    first_year, last_year = years
    buckets = pd.read_parquet(
        path,
        filters=[
            ("bucket", ">=", max(min_km, 0)),
            ("bucket", "<", max_km),
            ("year", ">=", first_year),
            ("year", "<=", last_year),
        ],
    )
    return buckets, bucket_map


def adm2_upstream_land_cover(min_km=0, max_km=np.inf, years=EXPOSURE_YEARS):
    """Municipality upstream land cover from the rings in [min_km, max_km) (see `ring_land_cover`)."""
    return ring_land_cover(LAND_COVER_ADM2_BUCKETED_PATH, "mun_id", min_km, max_km, years)


def adm2_upstream_mining(min_km=0, max_km=np.inf, years=EXPOSURE_YEARS, path=MINING_ADM2_BUCKETED_PATH):
    """Municipality upstream mining shares by type (MapBiomas mining substances) from the rings in [min_km, max_km).

    Same rings and kernel as `ring_land_cover`; columns are
    `land_cover.mining.mining_column_names` (e.g. `mining_garimpo_gold`).
    """
    buckets, bucket_map = _read_rings(path, min_km, max_km, years)
    return compute_mining_composition(buckets, entity_columns=["mun_id", "year"], bucket_map=bucket_map)


def adm2_local_mining(years=EXPOSURE_YEARS, path=MINING_ADM2_BUCKETED_PATH):
    """Municipality local (own-trench) mining shares by type, as `local_mining_*` columns.

    The rings of the trenches that touch the municipality
    (`bucket_intersects_adm2`, the negative buckets), equally weighted: the
    definition of the `local_` land-cover block and of `adm2_panel`'s
    `local_c11_*` columns, which cover the panel's years only.
    """
    first_year, last_year = years
    buckets = pd.read_parquet(
        path,
        filters=[("bucket_intersects_adm2", "==", True), ("year", ">=", first_year), ("year", "<=", last_year)],
    )
    bucket_map = {int(bucket): (f"local_{int(bucket)}", float(bucket)) for bucket in sorted(buckets["bucket"].unique())}
    local = compute_mining_composition(
        buckets, entity_columns=["mun_id", "year"], bucket_map=bucket_map, kernel="uniform", bandwidth=np.inf
    )
    return local.rename(columns={column: f"local_{column}" for column in local.columns if column.startswith("mining_")})


# --- Health outcomes ---------------------------------------------------------
# SIH hospitalizations and SINASC births are joined into `adm2_panel` by
# assembly; these helpers name their column groups from the config.

# SIH legacy (<= 2007) and current (>= 2008) series never overlap in a year.
SIH_LAST_LEGACY_YEAR = 2007


def hospitalization_columns(breakdown=None):
    """Return the `adm2_panel` hospitalization columns for one breakdown.

    `None` gives the all-cause metrics (`hosp_*_count`), `"icd10_chapter"`
    the `hosp_icd_*` counts and `"morbidity_channel"` the channel counts.
    """
    sources = {
        None: "hospitalizations",
        "icd10_chapter": "hospitalizations_icd10_chapter",
        "morbidity_channel": "hospitalizations_morbidity_channel",
    }
    if breakdown not in sources:
        raise ValueError(f"Unknown breakdown {breakdown!r}.")
    return _assembly_source_variables("adm2_panel", sources[breakdown])


def birth_columns():
    """Return the `adm2_panel` SINASC birth-count columns."""
    return _assembly_source_variables("adm2_panel", "births")


def mortality_columns():
    """Return the `adm2_panel` SIM death-count columns (total and by age band)."""
    return _assembly_source_variables("adm2_panel", "mortality")


def sih_source_system(year):
    """Label each year's SIH series: `"legacy"` up to 2007, `"current"` from 2008."""
    return pd.Series(year).le(SIH_LAST_LEGACY_YEAR).map({True: "legacy", False: "current"}).to_numpy()
