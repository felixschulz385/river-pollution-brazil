# Climate data pipeline

Fetches and preprocesses climate variables for the project's Colombia/Brazil
area of interest (`ERA5_AREA = [5.27, -73.99, -33.75, -34.78]`, i.e. N/W/S/E)
into a single local Zarr store:

```
data/climate/raw/era5_land.zarr_nobackup
```

All commands below go through the shared CLI. `--subtype` selects a *fetch*
variant only -- `preprocess` always drains every GRIB-origin variant
(`era5_land_hourly`, `era5_land_daily`) together into the shared store, since
neither is a choice worth splitting on downstream of fetch:

```
python code/data/cli.py climate fetch --subtype <subtype> [--root-dir .]
python code/data/cli.py climate preprocess [--root-dir .]
```

`--root-dir` defaults to the current working directory and should point at the
repo root (the folder containing `data/`, `setup/`, `code/`).

## Prerequisites

- **CDS account + API key** at `setup/secrets/.cdsapi`:
  ```
  url: https://cds.climate.copernicus.eu/api
  key: <your-personal-access-token>
  ```
  This one key is used for both the classic job-submission API and the ARCO
  Zarr Bearer-token access below - no separate credential needed.
- **Python environment** with, at minimum: `numpy`, `pandas`, `xarray`, `dask`,
  `zarr`, `cdsapi`, `ecmwf-datastores-client`, `earthkit-data`, `eccodes`,
  `cfgrib`, `odc-geo`, plus `aiohttp`/`requests` (listed in
  [`requirements.txt`](../../../requirements.txt), required by `xarray.open_zarr`
  for the ARCO `https://` store). On this machine, the validated environment is the
  `311` conda env (`C:\Users\schulz0022\conda-envs\311`); activate it in
  PowerShell with `setup/codex_311_env.ps1`, or point any Python 3.11 env with
  the packages above at the same secrets file.

## Data sources

| Subtype | CDS dataset | Variables | Mechanism |
|---|---|---|---|
| `era5_land_hourly` | `reanalysis-era5-land` (GRIB) | `surface_runoff`, `sub_surface_runoff`, `potential_evaporation` | async job submission, polled, downloaded, then aggregated hourly->daily locally |
| `era5_land_daily` | `derived-era5-land-daily-statistics` | `2t`, `2d`, `swvl1`, `swvl2` | async job submission (CDS computes the daily mean itself) - kept temporarily as a cross-check against `era5_land_arco`, see note below |
| `era5_land_arco` | ARCO Zarr (`arco.datastores.ecmwf.int`) | `2t`, `2d`, `swvl1`, `swvl2`, `tp` | reads a live, always-available Zarr store directly - no job queue |

`surface_runoff`/`sub_surface_runoff`/`potential_evaporation` are the only
variables **not** offered by CDS's ARCO product, so they're the only ones
still going through the GRIB path.

## Running each subtype

### ARCO (`era5_land_arco`) - recommended for 2t/2d/swvl1/swvl2/tp

Unlike the GRIB path, there's no separate raw-download step - opening the
live ARCO store, slicing to our area, aggregating hourly -> daily, and
writing into the local store all happen in one place, under `fetch`:

```
python code/data/cli.py climate fetch --subtype era5_land_arco --root-dir .
```

`era5_land_arco` has no separate `preprocess` step - opening the live store,
aggregating, and writing all happen under `fetch` above.

Notes:
- **Long-running.** It iterates the full 1985-2024 range across 3 ARCO
  groups on every call. It has no CLI flags for a narrower range yet - if you
  need one, call `climate.preprocess.era5_land_arco.preprocess_era5_land_arco(root_dir=..., start=..., end=...)`
  directly from Python instead of the CLI.
- **Safely resumable.** Progress is tracked per (ARCO group, year-month) in
  `data/climate/raw/era5_land_arco_progress.json`. Re-running
  the same command skips everything already marked `"processed"`, so it's
  fine to Ctrl-C and restart.
- **ARCO is a beta service.** CDS explicitly reserves the right to throttle or
  close access at any time; if a run starts failing with connection/auth
  errors, check https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=analysis_ready_data
  before assuming it's a bug here.

### GRIB hourly (`era5_land_hourly`) - for sro/ssro/pev only

Two steps, since this path still goes through CDS's async job queue:

```
python code/data/cli.py climate fetch --subtype era5_land_hourly --root-dir .
python code/data/cli.py climate preprocess --root-dir .
```

`fetch` submits one job per (year, month) batch (480 total), polls CDS, and
downloads each GRIB file as it completes; it loops internally until every
batch is downloaded or no jobs remain active, so it can take a long time on a
first run - it's safe to re-run if interrupted (per-batch manifests in
`data/climate/raw/era5_land_hourly/*.manifest.json` track status). `preprocess`
then processes any downloaded-but-not-yet-processed GRIB files, aggregates
hourly -> daily, writes to the store, and deletes the raw GRIB file.

The 480 monthly manifests under `data/climate/raw/era5_land_hourly/` were
reset (deleted) after the reshape-bug fix below landed, since they previously
all had `"preprocess_status": "processed"` from a full historical run under
the old buggy code - which would otherwise have blocked `fetch` from
resubmitting anything (`_can_submit_from_manifest` skips already-`"processed"`
batches). Running `fetch` now will submit all 480 jobs fresh (only for
`sro`/`ssro`/`pev`, the trimmed variable list) - that's real CDS quota and
likely a very long wall-clock time, since `reanalysis-era5-land` only allows
1 concurrently *running* remote job at a time (see
`DATASET_RUNNING_REMOTE_REQUEST_LIMITS` in `fetch/common.py`). It's safe to
run in the background and resume if interrupted.

### CDS daily-mean (`era5_land_daily`) - temporary cross-check, not required

```
python code/data/cli.py climate fetch --subtype era5_land_daily --root-dir .
python code/data/cli.py climate preprocess --root-dir .
```

This duplicates `2t`/`2d`/`swvl1`/`swvl2` via a second CDS product that
computes the daily mean server-side, kept temporarily so its output can be
diffed against `era5_land_arco`'s as an independent sanity check. Once that's
been done for a few months, this subtype is expected to be retired. Note that
`preprocess` is the same command as for `era5_land_hourly` above - it always
processes whatever GRIB input is ready across both subtypes in one pass.

## Known caveat: historical data needs reprocessing after the reshape-bug fix

`era5_land.zarr_nobackup` currently contains data written by an older version of the
GRIB reader that had a field-ordering bug (cross-contaminated variables -
e.g. `2t_daily_min` pinned near 0 K, `tp`/`sro`/`ssro`/`pev` inflated to
~10^6 mm/day). That bug is fixed now; getting corrected historical values
differs by variable:

- **`2t`, `2d`, `swvl1`, `swvl2`, `tp`:** `era5_land_arco` has never been run
  against the real store (its progress ledger doesn't exist yet), so a normal
  `fetch --subtype era5_land_arco` run will process the full 1985-2024 range
  fresh from ARCO and **overwrite** the old corrupted values for these 5
  variables in place. Just run it.

- **`sro`, `ssro`, `pev`:** their 480 monthly manifests under
  `data/climate/raw/era5_land_hourly/` have already been reset (deleted) so
  that `fetch --subtype era5_land_hourly` will resubmit all 480 CDS jobs
  fresh and reprocess them with the fixed code. This is real CDS quota and
  wall-clock cost (see the note in "Running each subtype" above) - it hasn't
  been run yet as of this writing, so `sro`/`ssro`/`pev` in the store are
  still the old corrupted values until someone runs it.

`era5_land_daily`'s 480 manifests were **not** reset - that subtype only ever
fed `2t`/`2d`/`swvl1`/`swvl2`, already superseded by `era5_land_arco` above,
so there's nothing to gain from re-running it.
