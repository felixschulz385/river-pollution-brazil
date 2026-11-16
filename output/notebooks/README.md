# Analysis notebooks

Three arms, each with its write-up; results and status are recorded there, not here.

| Arm | Question | Panel (index) | Write-up |
|---|---|---|---|
| `pollution/` | Does upstream land-cover change degrade water quality? | `data/assembly/sensor_panel.parquet` (`station_code`, `datetime`) | [`pollution/memo.md`](pollution/memo.md) |
| `health/` | Does upstream land-cover change affect health downstream? | `data/assembly/adm2_panel.parquet` (`mun_id`, `year`) | [`health/memo.md`](health/memo.md) |
| `garimpo/` | What drives garimpo supply, who responds, and which lever reduces it? | `adm2_panel` and monthly source tables | [`mining_economics_leads.md`](../research/mining_economics_leads.md) (Summary, Evidence) |

In the pollution and health arms, land cover enters in 25 km rings upstream of the
station or municipality.

## Workflow

The pollution and health arms have the same two steps:

1. **`01_descriptives.ipynb`**: the descriptive evidence needed to *choose* the model
   (sample, outcomes, transforms, FE, exposure definition, kernel, controls, clustering).
   Every section ends with an **→ Informs** note, and the final section is a decision
   table to fill in.
2. **`02_analysis.ipynb`**: estimation, implementing the decisions from step 1.
   `pollution/02_analysis` is built on `src/analysis/pollution/model.py`, `health/02_analysis`
   on `src/analysis/health/model.py`.

Re-run a descriptives notebook after any re-assembly, and record changed decisions in
its table before touching `02_analysis`.

Other notebooks:
- `health/03_garimpo_gold.ipynb`: the garimpo-gold gate, health outcomes on the upstream
  MapBiomas Collection 11 garimpo-gold share (50 and 200 km, with and without the local
  share), the Centro do Guilherme leave-one-out, and the Amazon presence result without
  gold (`src/analysis/health/garimpo_gold.py`).
- `health/04_moreno_louzada.ipynb`: Moreno-Louzada's design, gold deposits upstream
  (Otto-4) × the log gold price in BRL on birth outcomes and infant deaths in the Legal
  Amazon, with robustness, placebos, a first stage and four inferences
  (`src/analysis/health/moreno_louzada.py`).
- `garimpo/01_measurement.ipynb`: how well the C11 layer, SIVEP-Malária and the SGB
  geology measure garimpo (Evidence E3, E5, E6 of
  `output/research/mining_economics_leads.md`).
- `garimpo/02_price_supply.ipynb`: does garimpo respond to the gold price where there is
  gold geology? Own never-exploited gold occurrences × the log gold price on C11 garimpo
  and SIVEP garimpo cases, with an event study (Evidence E10;
  `src/analysis/garimpo/supply.py`).
- `garimpo/03_laundering.ipynb`: do garimpo and permits respond to the 2013 good-faith
  rule and its 2023 suspension where gold lies inside protected areas (IGD)? IGD ×
  post-2013 and × post-2023 on C11, SIVEP garimpo cases and gold PLG requests and
  grants, with an event study and permutation inference, and the timing of the
  requests by quarter (Evidence E11; `src/analysis/garimpo/laundering.py`).
- `garimpo/04_shift_2019.ipynb`: is the 2019–20 shift in garimpo the Roraima rush or the
  collapse of IBAMA enforcement? Totals and the IGD event study by state, and IBAMA
  presence before 2015 × post-2019 and × post-2023 on fines, garimpo cases, gold PLG
  requests and non-garimpo malaria, with deforestation-frontier and Priority List controls,
  forest loss as an outcome, permutation inference, presence quintiles and the outcomes
  through 2025 (Evidence E12; `src/analysis/garimpo/shift.py`).
- `garimpo/05_nightlights.ipynb`: do VIIRS night lights measure garimpo activity? Light
  in the whole municipality, near gold and inside barred areas against garimpo cases and
  C11 area, the composites' brightness-specific jumps, and E12's IGD and presence dates
  on light with brightness class × year FE (Evidence E13;
  `src/analysis/garimpo/nightlights.py`).
- `garimpo/06_raids.ipynb`: do IBAMA's garimpo raids reduce garimpo malaria where they
  happen and move it next door? A stacked monthly event study of raid episodes from 2019
  to April 2025
  for the raided and the bordering municipalities, with a triple difference against
  non-garimpo malaria, and bordering municipalities split by garimpo before the raid
  (Evidence E14; `src/analysis/garimpo/raids.py`).
- `garimpo/07_groups.ipynb`: who responds to what? Local and incoming garimpo cases, and
  their difference, on drought, upstream garimpo, the gold price × deposits and
  enforcement presence (lead 2's prediction 5; Evidence E15; `src/analysis/garimpo/groups.py`).
- `garimpo/08_demarcation.ipynb`: does pending protection attract garimpo into
  Indigenous lands? C11 garimpo inside each Legal Amazon land from the river trenches, by
  stage, and stacked event studies around declaration and homologation (lead 4; Evidence
  E16; `src/analysis/garimpo/demarcation.py`).
- `garimpo/09_raid_intensity.ipynb`: does sustained raiding lower garimpo malaria? Raid
  terms as a continuous treatment by month, in the municipality, its bordering
  municipalities and the others within 150 km, with leads, lags and a triple difference
  against non-garimpo malaria (Evidence E17; `src/analysis/garimpo/intensity.py`).
- `health/upstream_mining_municipality_map.ipynb`: municipality upstream-deforestation
  map (its `land_cover.feather` input is stale).
- `pollution/maps.ipynb`: upstream drainage-area maps for four example sensors
  (mining-, pasture-, urban-dominated and low-disturbance), traced from each sensor's trench.

## Shared code

- `src/analysis/core/panels.py`: panel loaders (restricted to the 1985–2024 exposure window)
  and variable groups derived from `setup/assembly_datasets.yaml`, including the
  health column groups; `ring_land_cover()` recomputes land cover from a subset of the
  25 km rings (sensor or ADM2).
- `src/analysis/pollution/model.py`: the pollution model setup (sample rules, outcome cleaning,
  combined fecal indicator, CONAMA 357 exceedances, land cover recomputed within 200 km
  and per distance band). `build_pollution_panel()` returns the estimation frame.
- `src/analysis/health/model.py`: the health model setup (window, outcome counts and exposure
  offsets, upstream land cover within 200 km and per band next to the local block,
  controls, FE keys). `build_health_panel()` returns the estimation frame.
- `src/analysis/core/inference.py`: wild-cluster bootstrap for high-dimensional FE models
  (OLS), and a wild-cluster score bootstrap for PPML fits.
- `src/analysis/health/moreno_louzada.py`: the estimator of `health/04`, with the
  price-exposure variance of Moreno-Louzada, Figueira and Picchetti and randomization
  inference over random-walk price paths. The garimpo modules reuse its estimation
  helpers, the Legal Amazon sample and its FE and clustering defaults.
- `src/analysis/core/descriptives.py`: FE demeaning (pyfixest), variation tables, distribution
  tables, binned scatters, effective cluster counts.
- Each other notebook's estimator is named in its entry above.
