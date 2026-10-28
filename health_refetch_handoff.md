# Handoff: DATASUS health re-download on a local machine

> **Done 2026-10-22.** All three requests below were fetched, synced back and processed.
> Keep these steps for adding new years: a rerun fetches only what the forms add.

The scicore nodes have no internet access, so the remaining DATASUS TABNET
downloads (Selenium + Chrome) run locally. The raw files then go back to scicore,
where preprocessing and assembly run. Everything below is resumable: a rerun
skips every batch already marked `completed` in its manifest *and* present on
disk.

| What | Why | Command (`--subtype`) |
|---|---|---|
| 48 SIH morbidity-channel batches: most channels 2021–2024, pregnancy 2009, renal 2013 | health blocker 2 | `hospitalization` |
| SINASC births (birth weight, gestational duration), 1994 to latest | blocker 6: the raw SINASC files are not on scicore; the bin harmonization needs them | `birth` |
| SIM deaths by municipality × age band, 1979 to latest | blocker 7: no mortality data yet | `mortality` |

The SINASC and SIM fetchers now read the available years from the form
(previously fixed at 1994–2023 and 1979–2021) and select exactly one period file
per query. A plain click used to add the year to the form's preselected latest
period.

## 1. Copy code and raw data to the local machine

```bash
# local machine, in the repo
git fetch && git checkout pollution-analysis && git pull

# The SIH raw folders (≈230 MB) carry the manifests and the batches already
# downloaded. Without them the fetch starts from scratch.
rsync -av <user>@<scicore-login>:/scicore/home/meiera/schulz0022/projects/river-pollution-brazil/data/health/raw/ data/health/raw/
```

## 2. Environment

Python ≥ 3.11 with the package dependencies (`pip install -e .`, which pulls in
`selenium` and `webdriver-manager`) and a local Chrome. `webdriver-manager`
fetches the matching chromedriver on first use.

## 3. Download

```bash
python -m src.data.sources.health fetch --subtype hospitalization   # add --headless to hide the browser
python -m src.data.sources.health fetch --subtype birth
python -m src.data.sources.health fetch --subtype mortality
```

- `hospitalization` also re-plans the SIH totals and ICD-chapter requests. Their
  completed batches are skipped; any newly complete SIH year (e.g. 2025) is fetched
  for all three requests.
- If it stops with `DATASUS morbidity-list form ... is missing N of M configured
  values`, the morbidity form loaded only partially. Rerun it. Planning used to
  continue silently with the partial list, which is what broke 2008+.
- On any other failure, rerun the same command; only failed or missing batches are retried.

Check afterwards:

```bash
grep -c '"pending"\|"failed"' data/health/raw/SIH_*/manifest.jsonl   # expect 0
grep -c '"completed"' data/health/raw/birth_outcome_*/manifest.jsonl data/health/raw/mortality_age_*/manifest.jsonl data/health/raw/infant_deaths_icd10/manifest.jsonl
```

## 4. Copy the raw data back to scicore

```bash
rsync -av data/health/raw/ <user>@<scicore-login>:/scicore/home/meiera/schulz0022/projects/river-pollution-brazil/data/health/raw/
```

## 5. On scicore

```bash
python -m src.data.sources.health preprocess --subtype all
python -m src.data.assembly assemble --dataset adm2_panel
jupyter nbconvert --to notebook --execute output/notebooks/health/01_descriptives.ipynb --inplace
```
