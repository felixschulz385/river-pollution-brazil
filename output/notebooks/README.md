# Analysis notebooks

Upstream land-cover change → river pollution → health, estimated in two linked arms:

| Arm | Question | Panel (index) | Exposure |
|---|---|---|---|
| `pollution/` | Does upstream land-cover change degrade water quality? | `data/assembly/sensor_panel.parquet` (`station_code`, `datetime`) | land cover in 25 km rings upstream of the station |
| `health/` | Does upstream land-cover change affect health downstream? | `data/assembly/adm2_panel.parquet` (`mun_id`, `year`) | land cover in 25 km rings upstream of the municipality |

Each arm has a one-page memo with the model and results:
[`pollution/memo.md`](pollution/memo.md), [`health/memo.md`](health/memo.md).

## Workflow

Each arm has the same two steps:

1. **`01_descriptives.ipynb`**: the descriptive evidence needed to *choose* the model
   (sample, outcomes, transforms, FE, exposure definition, kernel, controls, clustering).
   Every section ends with an **→ Informs** note, and the final section is a decision
   table to fill in.
2. **`02_analysis.ipynb`**: estimation, implementing the decisions from step 1.
   `pollution/02_analysis` is built on `src/analysis/pollution.py`, `health/02_analysis`
   on `src/analysis/health.py`.

Re-run a descriptives notebook after any re-assembly, and record changed decisions in
its table before touching `02_analysis`.

Other notebooks:
- `health/03_garimpo_gold.ipynb`: the garimpo-gold gate, health outcomes on the upstream
  MapBiomas Collection 11 garimpo-gold share (50 and 200 km, with and without the local
  share), the Centro do Guilherme leave-one-out, and the Amazon presence result without
  gold (`src/analysis/garimpo_gold.py`).
- `health/04_moreno_louzada.ipynb`: Moreno-Louzada's design, gold deposits upstream
  (Otto-4) × the log gold price in BRL on birth outcomes and infant deaths in the Legal
  Amazon, with robustness, placebos, a first stage and four inferences
  (`src/analysis/moreno_louzada.py`).
- `garimpo/01_measurement.ipynb`: how well the C11 layer, SIVEP-Malária and the SGB
  geology measure garimpo (Evidence E3, E5, E6 of
  `output/research/mining_economics_leads.md`).
- `garimpo/02_price_supply.ipynb`: does garimpo respond to the gold price where there is
  gold geology? Own never-exploited gold occurrences × the log gold price on C11 garimpo
  and SIVEP garimpo cases, with an event study (Evidence E10;
  `src/analysis/garimpo_supply.py`).
- `garimpo/03_laundering.ipynb`: do garimpo and permits respond to the 2013 good-faith
  rule and its 2023 suspension where gold lies inside protected areas (IGD)? IGD ×
  post-2013 and × post-2023 on C11, SIVEP garimpo cases and gold PLG requests and
  grants, with an event study and permutation inference, and the timing of the
  requests by quarter (Evidence E11; `src/analysis/garimpo_laundering.py`).
- `garimpo/04_shift_2019.ipynb`: is the 2019–20 shift in garimpo the Roraima rush or the
  collapse of IBAMA enforcement? Totals and the IGD event study by state, and IBAMA
  presence before 2015 × post-2019 and × post-2023 on fines, garimpo cases, gold PLG
  requests and non-garimpo malaria, with deforestation-frontier controls, forest loss as an
  outcome, permutation inference and presence quintiles (Evidence E12;
  `src/analysis/garimpo_shift.py`).
- `garimpo/05_nightlights.ipynb`: do VIIRS night lights measure garimpo activity? Light
  in the whole municipality, near gold and inside barred areas against garimpo cases and
  C11 area, the composites' brightness-specific jumps, and E12's IGD and presence dates
  on light with brightness class × year FE (Evidence E13;
  `src/analysis/garimpo_nightlights.py`).
- `garimpo/06_raids.ipynb`: do IBAMA's garimpo raids reduce garimpo malaria where they
  happen and move it next door? A stacked monthly event study of raid episodes from 2019
  to April 2025
  for the raided and the bordering municipalities, with a triple difference against
  non-garimpo malaria (Evidence E14; `src/analysis/garimpo_raids.py`).
- `garimpo/07_groups.ipynb`: who responds to what? Local and incoming garimpo cases, and
  their difference, on drought, upstream garimpo, the gold price × deposits and
  enforcement presence (lead 2's prediction 5; Evidence E15; `src/analysis/garimpo_groups.py`).
- `health/upstream_mining_municipality_map.ipynb`: municipality upstream-deforestation
  map (moved from `src/analysis/notebooks/`; its `land_cover.feather` input is stale).
- `pollution/maps.ipynb`: upstream drainage-area maps for four example sensors
  (mining-, pasture-, urban-dominated and low-disturbance), traced from each sensor's trench.

## Shared code

- `src/analysis/panels.py`: panel loaders (restricted to the 1985–2024 exposure window)
  and variable groups derived from `setup/assembly_datasets.yaml`, including the
  health column groups; `ring_land_cover()` recomputes land cover from a subset of the
  25 km rings (sensor or ADM2).
- `src/analysis/pollution.py`: the pollution model setup (sample rules, outcome cleaning,
  combined fecal indicator, CONAMA 357 exceedances, land cover recomputed within 200 km
  and per distance band). `build_pollution_panel()` returns the estimation frame.
- `src/analysis/health.py`: the health model setup (window, outcome counts and exposure
  offsets, upstream land cover within 200 km and per band next to the local block,
  controls, FE keys). `build_health_panel()` returns the estimation frame.
- `src/analysis/inference.py`: wild-cluster bootstrap for high-dimensional FE models
  (OLS), and a wild-cluster score bootstrap for PPML fits.
- `src/analysis/garimpo_gold.py`, `src/analysis/moreno_louzada.py`: the estimators of
  `health/03` and `health/04`; the latter adds the price-exposure variance of
  Moreno-Louzada, Figueira and Picchetti and randomization inference over random-walk
  price paths.
- `src/analysis/garimpo_supply.py`: the estimator of `garimpo/02`, reusing the
  Moreno-Louzada helpers (randomization with control terms, e.g. deposit trends).
- `src/analysis/garimpo_laundering.py`: the estimator of `garimpo/03`, with a permutation
  of the exposure across municipalities that rebuilds every exposure term.
- `src/analysis/descriptives.py`: FE demeaning (pyfixest), variation tables, distribution
  tables, binned scatters, effective cluster counts.

The maintained CLI regression pipeline (`python -m src.cli analysis sensor-data`) lives
in `src/analysis/sensor_data/` and is separate from these notebooks.

## Status (as of 2026-10-30)

**Pollution: active.** The model setup is decided (§11 of `pollution/01_descriptives.ipynb`)
and implemented in `pollution/02_analysis.ipynb`, with standard errors clustered on Otto
level-3 sub-basins (below). Coefficients are exported to
`output/analysis/pollution/coefficients.csv`.

**Health: estimated.** `health/02_analysis` implements the decisions of
`health/01_descriptives` §9 (`src/analysis/health.py`): PPML with exposure offsets,
municipality + state×year FE, upstream land cover capped at 200 km next to the local
block, Otto-3 clustering with a wild-cluster score bootstrap. Upstream land-cover change
has no detectable effect on hospitalizations, infant deaths or low birth weight, and no
effect differs from the injury placebo. The mining arm (§9) finds no pooled effect but a
suggestive rise in infant deaths (about +23%) where mining appears upstream in the
Amazon, which is not gold mining. Neither the upstream garimpo-gold share (`health/03`)
nor Moreno-Louzada's gold price × upstream deposits design (`health/04`) gives a credible
health effect: both fail the injury placebo. Coefficients are exported to
`output/analysis/health/`; see [`health/README.md`](health/README.md).

**Garimpo: initial evidence for leads 1–3.** Results are recorded in
`output/research/mining_economics_leads.md` (Summary and Evidence E3–E15).
- `garimpo/01_measurement`: C11, SIVEP-Malária and the SGB geology as measures of
  garimpo.
- `garimpo/02_price_supply`: garimpo in gold-geology municipalities does not follow the
  gold price. C11 garimpo grows on a trend, and garimpo cases break upward in 2019, mostly
  in Roraima.
- `garimpo/03_laundering`: where gold lies inside protected areas, gold PLG requests are
  higher after the 2013 good-faith rule and lower after its 2023 suspension by year.
  - By quarter they do not move when the rule starts (the best-fitting break is 2019Q3).
  - C11 garimpo and garimpo cases follow neither date.
- `garimpo/04_shift_2019`: the 2019 case break in IGD municipalities is mostly the
  Roraima rush. Across the Legal Amazon, garimpo cases rose from 2019 where IBAMA had been
  present and its fines fell. This survives the deforestation-frontier checks.
- `garimpo/05_nightlights`: VIIRS annual composites do not see garimpo.
- `garimpo/06_raids`: a single raid (2019 to April 2025) neither reduces garimpo malaria
  measurably within a year nor moves it to bordering municipalities.
- `garimpo/07_groups`: drought and upstream garimpo do not move local workers more than
  incoming ones. Incoming workers gain on local ones in deposit municipalities as the
  price rises, which cannot be told from a trend.

## Sub-basins (Otto Pfafstetter)

`river_network_otto_basins.parquet` (in `data/river_network/processed/`) maps every
`trench_id` to its Otto basin code at levels 1–6 (`otto_basin_l1` … `otto_basin_l6`,
string codes). The codes come from the attributes of the BHO drainage-area layer, so the
table needs no geometry and no rerun of the network/matrix pipeline:

```bash
python -m src.data.sources.river_network otto-basins   # ~10 s; also written by a full preprocess
```

`sensor_panel` carries `otto_basin_l2` … `otto_basin_l5` via the `otto_basins` assembly
source, and the pollution analysis clusters on `otto_basin_l3`. Cluster counts over
sensor samples:

| Level | Basins | Effective (inverse HHI) | Largest share |
|---|---|---|---|
| `system_id` | 261 | 5.0 | 42% |
| Otto 2 | 42 | 5.3 | 33% |
| **Otto 3** (used) | 207 | 27.2 | 9% |
| Otto 4 | 843 | 98.2 | 4% |
| Otto 5 | 2,276 | 347.2 | 2% |

Because some outcome samples have only a few dozen basins, §5.7 of
`pollution/02_analysis.ipynb` adds restricted wild-cluster bootstrap p-values (9,999
reps, Otto 3) for every land-cover coefficient (`src/analysis/inference.py`). pyfixest's
built-in `wildboottest` is infeasible here: it builds a dense one-hot matrix of all
~5,700 station FE. The module therefore bootstraps the FE-demeaned design, which is
equivalent by Frisch–Waugh–Lovell. Results are in
`output/analysis/pollution/wild_cluster_bootstrap.csv`.

`adm2_panel` carries the same levels for each municipality's dominant basin (largest
summed drainage area of its intersecting trenches, from
`river_network_adm2_otto_basins.parquet`).

## Mining arm

Mining is not in the baseline: 56% of station-years have no mining upstream, and only
2.6% of stations switch. It is estimated separately in §6 of
`pollution/02_analysis.ipynb`:
- **Sample:** the 1,205 stations with any mining within 50 km upstream (158 switch
  presence).
- **Regressors:** mining presence (`mining_present_50`) and the mined share of the
  50 km reach (`mining_pp_50`), on top of the baseline specification.
- **Outcomes:** turbidity, TSS, conductivity, pH, DO and the CONAMA metal exceedances.
- **Inference:** wild-cluster bootstrap, since the metal samples have only 12–20
  Otto-3 basins.

**Headline result:** mining appearing upstream raises mercury violations by about
14 pp (bootstrap p = 0.004). Adding 200 km `alr_mining` to the full-sample baseline
moves the other land-cover coefficients by at most 0.33 SE, so leaving mining out of the
baseline does not bias them.
