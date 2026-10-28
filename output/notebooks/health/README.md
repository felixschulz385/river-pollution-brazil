# Health stage: status

**Status (2026-10-28): estimated.** The decision table in `01_descriptives.ipynb` §9 is
decided and implemented in `src/analysis/health.py`; `02_analysis.ipynb` runs the
baseline, the placebo comparison, robustness, distance bands, biome heterogeneity and
wild-cluster score-bootstrap inference, and a mining arm (results in
`output/analysis/health/`).

**Main result:** upstream land-cover change has no detectable effect on health in
downstream municipalities. No upstream pasture or agriculture coefficient is significant
for any primary outcome, the 95% CIs exclude increases above about 0.6% (pasture) and 1.1%
(agriculture) per within-municipality SD, and no upstream effect differs from the injury
placebo. See the summary at the top of `02_analysis.ipynb`.

**Mining arm (§9):** pooled over the 1,190 municipalities with mining within 50 km
upstream, mining has no robust health effect. In the Amazon, mining appearing upstream
raises infant deaths by about 23% (bootstrap p = 0.002; flat injury placebo; the rise
follows onset in the event study). This is suggestive only: 36 onsets in 18 basins, a
small pre-onset bump, and it survives multiple-testing correction narrowly. It is not
gold: it is unchanged after dropping every municipality with garimpo gold upstream
(`03_garimpo_gold.ipynb`).

**Gold (`03_garimpo_gold`, `04_moreno_louzada`):** no credible downstream health effect.
- The upstream MapBiomas Collection 11 garimpo-gold share raises infant deaths only as
  much as it raises the injury placebo, and at 50 km the effect rests on one municipality.
- Moreno-Louzada's design (Otto-4 gold upstream × log gold price in BRL, Legal Amazon,
  2007–2021) does not reproduce his anomaly result (+0.84 per 1,000 births, Otto-3 SE 0.98,
  against his +2.28). Congenital infant deaths rise (+0.31, bootstrap p 0.016), but not
  with state×year FE, while the injury placebo falls significantly and there is no first
  stage on upstream garimpo.

## Design

- **Question:** does upstream land-cover change affect health in downstream municipalities?
- **Unit:** municipality (6-digit IBGE `mun_id`) × year, from `data/assembly/adm2_panel.parquet`.
- **Window:** 2000–2024. Population (the denominator) starts in 2000; land cover ends in 2024.
- **Candidate outcomes:**
  - SIH hospitalization rates per 1,000 by ICD-10 chapter: 01 infectious/parasitic,
    11 digestive, 14 genitourinary, 16 perinatal, 17 congenital; 19 injury as placebo.
  - Low-birth-weight share and preterm share (SINASC; preterm only from 2012, see below);
    congenital anomalies at birth (from 1999) and APGAR below 6 or 8 at 1 and 5 minutes.
  - Infant and under-5 mortality per 1,000 live births, all-cause mortality (SIM); infant
    deaths by ICD-10 chapter (congenital, external, all but external).
  - All-cause hospitalization and in-hospital death rates.
- **Exposure:** land cover in 25 km rings upstream of the municipality (`lc_*` / `alr_*`).

## Blockers

None open. The DATASUS fetch scrapes TabNet with Selenium and needs internet access: run
it locally following [`health_refetch_handoff.md`](../../../health_refetch_handoff.md), or
on scicore through an HTTP proxy (`HTTP_PROXY`/`HTTPS_PROXY`). The anomaly, APGAR and
infant-death tables were fetched on scicore that way. The fetch resumes from its per-year
manifests.

Resolved:
- **1. Health outcomes in `adm2_panel`.** `adm2_panel` now carries SIH hospitalizations
  (`hosp_*_count` all-cause, `hosp_icd_01` … `hosp_icd_99` by ICD-10 chapter, and
  `hosp_<channel>` morbidity channels, via `long_pivot` with `rename`/`column_template`)
  and SINASC births (`births`, `births_low_weight`, `births_preterm` plus their `*_known`
  denominators, from `health_births.parquet`, built by
  `python -m src.data.sources.health preprocess --subtype birth_summary`). The values
  match the former temporary loaders exactly. `panels.hospitalization_columns()` and
  `panels.birth_columns()` name the column groups.
- **5. Distance kernel.** Both ADM2 sources (land cover, climate) use the sensor side's
  inv-sqrt kernel, which is also `DEFAULT_ADM2_DISTANCE_KERNEL` now. The former Gaussian
  with bandwidth 1,000,000 km weighted every ring equally; within-municipality
  land-cover changes under the two kernels correlate at 0.986 or more. The pollution
  arm's 200 km reach cap is applied in analysis code and is still a health model-setup
  decision.
- **2. Morbidity channels after 2008.** They were (near) zero from 2008 on
  because a fetch run whose current-SIH form loaded only morbidity-list values 1–2
  planned gastrointestinal with 2 of its 6 values, silently dropped the other 14
  channels, and rewrote the manifest; preprocessing then zero-filled the unlisted
  batches. The complete batches of an earlier run were still on disk.
  A one-off repair script (commit 7aa1444) rebuilt the manifest from them (backup
  `manifest.jsonl.bak-20261021-162620`); the fetch now refuses to plan partial channels,
  and `_complete_panel` zero-fills only within fetched batches, so unfetched batches stay
  missing (this also turned ICD chapter 99 into NaN in years where it does not exist, and
  filled legacy 1999 injuries and 2001 toxic poisoning, which were zero). Channels are now
  continuous across 2007/2008 (2010/2006 ratios 0.87–1.35, except vector-borne at 2.3,
  the 2010 dengue epidemic). The 48 batches missing from the old run (most channels
  2021–2024, pregnancy 2009, renal 2013) were then fetched, so every channel is complete
  for 1998–2024. Caveat: legacy-2000 injuries lists fewer causes than other years.
- **6. Birth bins.** The re-downloaded SINASC tables sum to `Total` in every year, so the
  missing modal bins came from the old download, not from the bin labels; no
  harmonization was needed. The 28–36 week bin is split into 28–31/32–36 from 2000
  (partly from 1996), which the preterm count already handles. The preterm share has a real break with the 2011 SINASC form (7% in
  2010, 12.5% in 2012). 2025–26 are in the processed table but outside the panel window
  (2026 is partial).
- **7. Mortality.** SIM deaths by municipality of residence and age band, 1979–2024
  (ICD-9 to 1995, ICD-10 from 1996), are in `health_mortality.parquet`
  (`preprocess --subtype mortality`) and in `adm2_panel` as `deaths_total` and
  `deaths_<age band>` (`panels.mortality_columns()`). 98–99.9% of each year's national
  deaths fall in an identified municipality, and registered infant mortality falls from
  21 (2000) to 12 (2015–2024) per 1,000 live births. Municipality-years SIM does not list
  stay missing: before 2000 these are mostly municipalities not yet created.
- **3, 4, 9. Local exposure and headwater municipalities.** Bucketed ADM2 sources take
  `scope: local`, which averages (equal weight) the rings of the trenches touching the
  municipality (`bucket_intersects_adm2`) into `local_lc_*`, `local_alr_*` and
  `local_<climate>` columns. The local land-cover source is the panel's spine, so the
  217 headwater municipalities (3.8% of hospitalizations) stay in with NaN upstream
  exposure; `panels.load_adm2_panel()` adds `has_upstream`. Local and upstream land-cover
  changes correlate at 0.78–0.86 for forest, pasture and agriculture, and the
  within-FE ALR terms at 0.20–0.52 (2000–2024). Local and upstream climate correlate at 0.79–0.94
  within municipality. Two coastal municipalities have no local climate.
- **8. Basin id for clustering.** `adm2_panel` carries `otto_basin_l2` … `otto_basin_l5`,
  the municipality's dominant basin by summed drainage area of its intersecting trenches
  (`river_network_adm2_otto_basins.parquet`, written by `python -m
  src.data.sources.river_network otto-basins`). Otto 3: 217 basins over the panel's
  municipalities, effective about 39.

## Descriptive findings so far

- **Outcomes:**
  - ICD-chapter rates are complete and have few zeros, except congenital (14% zeros)
    and perinatal (6%). Infant mortality is zero in 22% of municipality-years.
  - Dispersion falls with population, which argues for population weights or PPML with
    population as exposure.
- **Identifying variation:** after municipality + state×year FE, about 0.4–2% of the
  variance in land-cover shares is left (ALR terms: 0.6–1.9%).
  - The long differences are substantial, though: 55% of municipalities saw upstream
    pasture shift by more than 5 percentage points between 2000 and 2024.
- **Placebo:** the raw binscatter suggests injury hospitalizations rise with upstream
  pasture, but under the proposed specification the placebo is not significant (upstream
  pasture +0.024, SE 0.029 with the local block, vs +0.040 without it). Its estimates are
  still as large as any water outcome's.
- **Reach:** 63% of municipalities have no upstream rings beyond 200 km, so the 200 km cap
  changes little (capped vs full ALR correlate 0.96–0.99 within municipality).
- **Inference:** 27 states, effective number about 14. Errors are likely correlated along
  rivers, which state clustering does not capture; Otto-3 basins (217, effective about 39)
  do.
