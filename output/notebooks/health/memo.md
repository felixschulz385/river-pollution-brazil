# Memo: upstream land cover and health in downstream municipalities (health arm)

*Status 2026-10-27. Source: [`02_analysis.ipynb`](02_analysis.ipynb), run of 2026-10-22, and the
mining checks in the leads note's Evidence section;
model decisions in [`01_descriptives.ipynb`](01_descriptives.ipynb) §9.*

## Question

Does land-cover change upstream of a municipality, along its rivers, affect the health of
its residents? This is the second link of the chain land cover → water quality →
health.

## Data

- 138,912 municipality-years for 5,562 municipalities, 2000–2024.
- **SIH:** hospitalizations by municipality of residence, by ICD-10 chapter and by
  morbidity group.
- **SINASC:** births, with birth weight and gestation.
- **SIM:** deaths by age.
- **IBGE:** population.
- **Exposure and climate:** MapBiomas land cover and ERA5-Land climate, both along the
  river network.
- **Headwater municipalities:** the 215 municipalities without upstream rivers stay in
  the sample.

## Data notes

- **Health columns in `adm2_panel`:**
  - SIH hospitalizations: `hosp_*_count` all-cause, `hosp_icd_01` … `hosp_icd_99` by
    ICD-10 chapter, and `hosp_<channel>` morbidity channels.
  - SINASC births: `births`, `births_low_weight`, `births_preterm` and their `*_known`
    denominators, from `health_births.parquet`.
  - SIM deaths: `deaths_total` and `deaths_<age band>`.
  - `panels.hospitalization_columns()`, `panels.birth_columns()` and
    `panels.mortality_columns()` name the groups.
- **Morbidity channels** are complete for 1998–2024 and continuous across the
  2007/2008 SIH switch: 2010/2006 ratios are 0.87–1.35, except vector-borne at 2.3 (the
  2010 dengue epidemic).
  - Batches that were never fetched stay missing; zero-filling applies only within
    fetched batches.
  - Injuries in legacy 2000 list fewer causes than in other years.
- **Births:** the SINASC tables sum to `Total` in every year.
  - The 28–36 week bin splits into 28–31/32–36 from 2000 (partly from 1996), which the
    preterm count handles.
  - The preterm share breaks with the 2011 SINASC form (7% in 2010, 12.5% in 2012).
  - 2025–26 are in the processed table but outside the panel window.
- **Mortality:** SIM deaths by municipality of residence and age band, 1979–2024
  (ICD-9 to 1995, ICD-10 from 1996).
  - 98–99.9% of each year's national deaths fall in an identified municipality.
  - Registered infant mortality falls from 21 (2000) to 12 (2015–2024) per 1,000 live
    births.
  - Municipality-years that SIM does not list stay missing. Before 2000 these are
    mostly municipalities not yet created.
- **Local exposure:** bucketed ADM2 sources with `scope: local` average the rings of the
  trenches touching the municipality, with equal weights, into `local_lc_*`,
  `local_alr_*` and `local_<climate>`.
  - The local land-cover source is the panel's spine, so headwater municipalities stay
    in with missing upstream exposure. `panels.load_adm2_panel()` adds `has_upstream`.
  - Local and upstream land-cover changes correlate at 0.78–0.86 for forest, pasture and
    agriculture, and the within-FE ALR terms at 0.20–0.52.
  - Local and upstream climate correlate at 0.79–0.94 within municipality. Two coastal
    municipalities have no local climate.
- **Distance kernel:** land cover and climate use the sensor side's inverse-square-root
  kernel (`DEFAULT_ADM2_DISTANCE_KERNEL`). The 200 km reach cap is applied in
  `src/analysis/health.py`.
- **Clustering:** `adm2_panel` carries `otto_basin_l2` … `otto_basin_l5`, the
  municipality's dominant basin by summed drainage area of its intersecting trenches
  (`river_network_adm2_otto_basins.parquet`).

## Descriptive evidence

From `01_descriptives.ipynb`, which motivates the model choices:

- **Outcomes:** ICD-chapter rates have few zeros, except congenital (14%) and perinatal
  (6%). Infant mortality is zero in 22% of municipality-years. Dispersion falls with
  population, which argues for PPML with population as exposure.
- **Identifying variation:** after municipality + state×year FE, about 0.4–2% of the
  variance in land-cover shares is left (ALR terms: 0.6–1.9%). The long differences are
  substantial: 55% of municipalities saw upstream pasture shift by more than
  5 percentage points between 2000 and 2024.
- **Placebo:** the raw binscatter suggests injury hospitalizations rise with upstream
  pasture. Under the chosen specification the placebo is not significant (+0.024, SE
  0.029 with the local block, vs +0.040 without it).
- **Reach:** 63% of municipalities have no upstream rings beyond 200 km, so the 200 km
  cap changes little (capped and full ALR correlate at 0.96–0.99 within municipality).
- **Inference:** 27 states, effective number about 14. Errors are likely correlated
  along rivers, which state clustering does not capture. Otto-3 basins (217, effective
  about 39) do.

## Model

One PPML regression per outcome (`pyfixest.fepois`):

```
count ~ upstream ALR(pasture, agriculture, unclassified farming, urban, other)   [within 200 km]
      + local ALR(same classes)                                                  [municipality's own area]
      + local temperature, precipitation, soil moisture
      + offset(log exposure)
  | municipality + state×year
```

- **Exposure offset:** population for hospitalizations, live births for infant and
  under-5 deaths, births with known weight or gestation for low birth weight and
  preterm.
- **Upstream exposure:** same definition as the pollution arm (200 km reach,
  inverse-square-root kernel, ALR against natural land).
- **Local block:** absorbs the local economic effects of land use (income, development,
  access to care). The upstream coefficients are therefore net of local land use.
- **Outcomes:**
  - *Primary:* infectious/parasitic (ICD 01), digestive (ICD 11) and gastrointestinal
    hospitalizations; infant deaths; low birth weight.
  - *Secondary (10):* genitourinary (ICD 14), perinatal (16) and congenital (17)
    hospitalizations; the renal/urinary, skin-contact, leptospirosis, toxic-poisoning
    and vector-borne morbidity channels; under-5 deaths; the preterm share (from 2012).
  - *Placebos:* injury/poisoning (ICD 19), injuries and mental-health hospitalizations.
- **Inference:** clustered by Otto level-3 basin (217 basins), with a restricted
  wild-cluster score bootstrap for PPML (9,999 draws).
- **Placebo comparison:** each primary outcome is stacked with the injury placebo, so
  the difference from the placebo is tested in one model.

## Results

- **No detectable effect of upstream pasture or agriculture.** No coefficient is
  significant for any primary outcome. The estimates are precise: per within-municipality
  SD of exposure, the 95% CIs exclude increases above about 0.6% (pasture) and 1.1%
  (agriculture). Most point estimates for water-related outcomes are negative.
- **Nothing beats the placebo.** No upstream effect differs from the injury placebo.
  Where the gap comes close (infectious hospitalizations, p ≈ 0.05), the water-related
  outcome responds *less* than injuries.
- **Other land-cover classes.** 4 of 40 upstream coefficients pass the bootstrap, where
  about 2 are expected by chance. All are small (under 1% per SD) and form no pattern.
- **Local land use carries the confounding.** Local urbanization has the largest
  coefficients, and they are largest for the injury placebos (+0.23 to +0.26). That
  points to local development or access to care, not water. Local pasture →
  gastrointestinal hospitalizations (+0.16, p = 0.02) is the one water-plausible
  association, and it beats the placebo only marginally (p = 0.06).
- **Robustness.** The result holds with OLS on log rates, without 2020–21, with upstream
  climate or a hospital-use control, for 2008–2024 only, and for municipalities with
  upstream rivers only. Under biome×year FE the injury placebo becomes significant
  (+0.09), which supports state×year FE.
- **Heterogeneity.** Distance bands show no pattern. Biome slopes show no water pattern;
  the clearest heterogeneity is in the placebo, where frontier pasture (Amazon +0.25,
  Cerrado +0.20) goes with more injury hospitalizations.
- **Mining** (separate arm: 1,190 municipalities with mining within 50 km upstream, 216
  switching; upstream and local mining presence and share).
  - *Pooled:* no robust health effect; 2 of 44 coefficients pass the bootstrap, about
    chance.
  - *Amazon:* mining appearing upstream raises infant deaths by about 23% (+0.21,
    bootstrap p = 0.002), against −0.02 in the other biomes, with no placebo response.
  - *Event study:* infant deaths are flat before onset, rise from the year after onset,
    and stay up.

## Caveats

- **Little identifying variation.** About 0.4–2% of the exposure variance is
  identifying; the nulls are informative because the outcome counts are large.
- **The Amazon mining result is suggestive, not established.**
  - It rests on 36 mining onsets in 18 basins.
  - There is a small pre-onset bump two years before onset.
  - It survives a multiple-testing correction across the 22 regional tests only
    narrowly.
  - MapBiomas does not separate gold mining from other mining.
- **Measurement.** The preterm share has a recording break in 2011 (used from 2012
  only). Hospitalizations partly measure access to care, which the local block and
  placebos are there to catch.
- **Aggregation.** Municipality-level outcomes average over residents with very
  different exposure to river water, and water treatment may break the link.

## Bottom line

Read together with the pollution arm, the broad land-cover changes that move river water
quality do not translate into measurable health effects at municipality level. Mining is
the possible exception, but the Amazon infant-mortality result has no measured water link.

- **No mercury data.** The pollution arm's mercury effect does not survive a unit-coding
  correction, and it was identified outside the Amazon. The stations have almost no
  mercury readings in the Amazon.
- **Robust but no dose-response:**
  - 11 of the 36 onsets fall in 2020, a national MapBiomas mapping step.
  - Dropping the 2020 onsets and the Maranhão/coastal-Pará switchers leaves the effect at
    +0.25 to +0.26 (bootstrap p ≤ 0.014).
  - Keeping only switchers with a mined share of at least 0.01 pp gives +0.06 (p = 0.18).
- **Not gold.**
  - The MapBiomas mining-substances layer (Collection 11) separates garimpo gold from
    other mining. Only 9 of the 216 switchers ever have garimpo gold within 50 km
    upstream, so at least 29 of the 36 Amazon onsets have none.
  - Dropping every municipality with gold upstream leaves the effect unchanged (+0.21 to
    +0.22, bootstrap p ≤ 0.0005).
- **The upstream garimpo-gold share gives no credible effect** (municipality and
  state×year FE, 2000–2019, 50 and 200 km).
  - Infant deaths rise by 1.5–3% per SD of gold share, but injury hospitalizations rise
    as much. At 200 km they have the strongest bootstrap p-values (0.004–0.005).
  - At 50 km the infant-death coefficient rests on one municipality, Centro do Guilherme
    (MA). Without it the coefficient falls from 0.35 to 0.16 per pp (boot p 0.62).
  - Adding 2020–24 cuts it to a third.
  - Adding the municipality's own C11 gold share leaves the upstream coefficients
    unchanged, and local gold itself is never significant (boot p ≥ 0.11).
- **Moreno-Louzada's price-driven design does not hold up either**
  (`04_moreno_louzada.ipynb`). The design takes gold in an upstream Otto-4 sub-basin ×
  the log gold price in BRL, in the Legal Amazon without the municipalities holding a
  deposit, 2007–2021.
  - Anomalies at birth: +0.84 per 1,000 births per log point (Otto-3 SE 0.98, bootstrap
    p 0.43), against his +2.28. Birth weight, preterm, APGAR and non-external infant
    deaths are null.
  - Congenital infant deaths rise (+0.31 per 1,000 births, bootstrap p 0.016), but the
    coefficient halves and loses significance with state×year FE or without weights.
  - The injury placebo falls significantly, and downstream gold does as much for
    anomalies as upstream gold.
  - There is no first stage: the interaction does not move upstream C11 garimpo gold or
    garimpo malaria cases.
  - The gold price in BRL is persistent and jumps in 2020. The price-exposure SE of
    arXiv 2512.10076 is close to clustering by year and rejects where cluster and
    randomization inference do not.

So the Amazon result is a robust association with non-gold mining frontiers, not a
mercury dose. Neither mapped gold mining nor the gold price × upstream deposits design
shows a credible downstream health effect. Details and code locations are in
`output/research/mining_economics_leads.md` (Evidence, E2, E4 and E9).

## Files

- Model setup: `src/analysis/health.py`
- Inference: `src/analysis/inference.py` (`poisson_score_bootstrap`)
- Coefficients and bootstrap: `output/analysis/health/`
