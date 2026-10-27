# Memo: upstream land cover and river water quality (pollution arm)

*Status 2026-10-26. Source: [`02_analysis.ipynb`](02_analysis.ipynb), run of
2026-10-26; model decisions in
[`01_descriptives.ipynb`](01_descriptives.ipynb) §11.*

## Question

Does land-cover change upstream of a monitoring station (pasture, agriculture,
urbanization, mining) change the river's water quality?

## Data

- 183,608 water samples from 5,773 monitoring stations, 1985–2024.
- Upstream land cover from MapBiomas, aggregated along the river network in 25 km rings.
- Climate from ERA5-Land.
- Each outcome is estimated on its identifying stations: at least 8 samples of the
  outcome in at least 3 years.

## Model

One OLS regression per outcome (`pyfixest.feols`):

```
y ~ ALR(pasture, agriculture, unclassified farming, urban, other)
    + (pasture, agriculture) × 7-day surface runoff
    + climate controls (precipitation, runoff, soil moisture, evaporation, temperature at 1–365 day windows)
  | station + biome×year + biome×month
```

- **Exposure:** land cover up to 200 km upstream, weighted by an inverse-square-root
  distance kernel. Each class enters as its additive log ratio (ALR) to natural land, so
  a 10% higher pasture-to-natural ratio changes a logged outcome by about 0.1·β·100 %.
  The within-station SD of pasture ALR is about 0.2.
- **Outcomes, continuous (11):** turbidity, TSS, BOD, COD, total phosphorus, ammoniacal
  N, nitrates, fecal indicator and conductivity, all in logs; dissolved oxygen and pH in
  levels.
- **Outcomes, violations (19):** 0/1 indicators of exceeding the CONAMA 357 class-2
  limits, in percentage points. Station-periods whose lab reporting limit sits at or
  above the legal limit are excluded. For trace metals (limits ≤ 0.05 mg/L), values at
  least 100× the limit in a station-period whose lowest positive value is at or below
  the limit are treated as µg/L entries and set to missing (`drop_unit_errors`).
- **Inference:** clustered by Otto level-3 sub-basin (207 basins), with restricted
  wild-cluster bootstrap p-values (9,999 draws).

## Results

Identification rests on about 1% of the land-cover variation, so most coefficients are
imprecise nulls. There are modestly more significant results than chance would give:
6 of 55 continuous and 8 of 95 violation tests at 5%, against about 3 and 5 expected.

- **Agriculture → organic load.** One within-station SD more upstream agriculture raises
  COD by about 4% (bootstrap p = 0.004). It lowers the fecal indicator by about 10%
  (p = 0.03) and conductivity by about 3% (p = 0.02). All three are stable across
  station-year means, PPML on levels and a discharge control.
- **Pasture → metals.** One within-station SD more pasture raises the probability of a
  lead violation by about 5 pp (base rate 28%, p = 0.004) and of a cadmium violation by
  about 4 pp (base 23%, p = 0.012), consistent with erosion from degraded pasture.
  Pasture has no robust effect on the continuous core outcomes.
- **Mining** (separate arm: 1,205 stations with mining within 50 km upstream, 158 of
  which switch).
  - *Lead:* mining appearing upstream raises lead violations by about 4 pp (p = 0.045).
  - *Mercury:* no robust effect. The +14 pp estimate before the unit-error rule
    (p = 0.004) does not survive removing suspected µg/L entries: +5.7 pp, SE 4.4,
    bootstrap p = 0.34, 11 basins.
  - *Where mercury is identified:* the mercury coefficient rests on Mata Atlântica and
    Cerrado stations in MG, SP, RJ and SC, sampled mostly 1985–99: 15 stations in 13
    basins before the recode, about 14 in 11 basins after it. There are almost no mercury readings in the Amazon
    (4 in the ANA Amazon basin, none in the Madeira), so the arm says nothing about
    garimpo.
  - The size of the mined area has no positive effect, and leaving mining out of the main
    model does not move the other coefficients.
  - The pasture → lead and cadmium results are unchanged by the same recode.
- **Distance and heterogeneity.**
  - Distance bands (0–50, 50–150, 150–500 km) are significant at about the chance rate,
    and their signs reverse, so there is no credible distance gradient.
  - Biome slopes show some heterogeneity (8 of 48 significant) without a consistent
    pattern.
  - The Mata Atlântica, which dominates the sample, gives precise nulls.

## Caveats

- **Little identifying variation.** About 1% of the land-cover variance is identifying;
  expect wide intervals.
- **Multiple testing.** Read individual significant results as suggestive.
- **Metal violations.** These samples cover only 19–34 Otto-3 basins (11–20 in the
  mining arm), so only the bootstrap p-values are reliable.
- **PPML on levels.** It gives larger, positive nutrient effects (total phosphorus,
  nitrates), so nutrient results depend on the functional form.

## Files

- Model setup: `src/analysis/pollution.py`
- Inference: `src/analysis/inference.py`
- Coefficients and bootstrap: `output/analysis/pollution/`
