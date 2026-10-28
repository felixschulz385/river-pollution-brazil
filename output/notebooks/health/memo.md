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
  - *Secondary:* 10 further outcomes.
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
