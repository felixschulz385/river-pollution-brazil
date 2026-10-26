# Gold, enforcement and downstream health: research leads

*Draft, 2026-10-22. Builds on the pollution and health arms of this project
(`output/notebooks/pollution/memo.md`, `output/notebooks/health/memo.md`).*

## Summary

The project's empirical arms show that upstream land-cover change moves river water
quality in a few specific ways, but that this mostly does not reach health:

- **Pollution.** Pasture raises lead and cadmium violations, agriculture raises COD, and
  mining raises mercury violations.
- **Health.** There is no detectable effect of upstream pasture or agriculture on
  hospitalizations, infant deaths or birth outcomes.
- **The exception is mining in the Amazon.** Where mining appears within 50 km upstream,
  infant deaths rise by about 23% (bootstrap p = 0.002), and the injury placebo does not
  move.
- **That result is thin.** It rests on 36 mining onsets in 18 basins, and there is a
  small bump two years before onset.
- **Many of those onsets are probably not garimpo** (HPC check, 2026-10-26):
  - 13 of the 36 are in Maranhão and 5 on the northeast Pará coast.
  - 11 onsets fall in 2020 alone. This is a national step in MapBiomas Collection 10
    rather than an Amazon event:
    - 41 first-time upstream onsets in Brazil in 2020, against about 7 a year before;
      30 of them are outside the Amazon.
    - Mapped mining area rose 29% (Amazon 37%) in that one year, against 4–8% a year
      before.
  - The median mined share at onset is 0.0009 pp of the 50 km reach, a pixel or two.
  - *Robustness:* the infant-death effect survives dropping the 2020 onsets and the
    Maranhão/coastal-Pará switchers (+0.25 to +0.26, bootstrap p ≤ 0.014; injury placebo
    flat).
  - *No dose-response:* keeping only switchers with a mined share of at least 0.01 pp at
    onset+1 gives +0.06 (p = 0.18). The effect comes from switchers with a few pixels of
    mining, which cannot be a mercury dose. The presence switch is a marker of something
    (frontier activity, unmapped dredge mining, or chance), not a measured exposure.
- **No measured mercury link in the Amazon.** The station data have no mercury readings
  in the Madeira basin and 4 in the whole Amazon basin. The pollution arm's mercury
  result comes from 15 stations in MG, SP, RJ and SC, sampled mostly 1985–99, and it
  does not survive removing suspected µg/L entries (+5.7 to +6.9 pp, bootstrap p ≥ 0.28).
  The lead and cadmium results are unaffected by the same recode.

Mining is therefore where an economic story can carry the health evidence. This report
develops four connected leads:

1. **Gold-laundering policy:** laundering policy drives garimpo, and garimpo drives
   downstream contamination and health.
2. **A two-group model of garimpo:** outside crews move to where enforcement is weak, and
   local residents turn to mining as their traditional economy erodes.
3. **Enforcement displacement:** how much mining enforcement relocates rather than
   removes, and where the health burden goes.
4. **Indigenous-land demarcation:** whether the gap between legal protection and actual
   control attracts mining into protected territory.

Leads 1, 3 and 4 each move upstream mining for a different reason (price, enforcement
cost, legal status), and lead 2 predicts who responds to each. Together they give three
independent sources of variation for one downstream health effect, and a test of
mechanism. The proposed core paper combines leads 1–3; lead 4 is a section or a separate
paper.

## Closest literature and positioning

Two papers sit directly on these leads. Both build on the river-direction design of
Dias, Rocha & Soares (2023, *ReStud*), which this project also uses.

| | Pereira & Pucci (2026, *JPubE*) | Moreno-Louzada (2024, WP) | This project |
|---|---|---|---|
| Shock | Law 12,844/2013 (good-faith presumption), 2006–2019 | log gold price × gold deposits, 2007–2021 | mining onsets; proposed: 2013/2023 laundering policy, enforcement, demarcation |
| Exposure | gold deposit inside a protected area, in the municipality itself (IGD) | gold deposit upstream on Otto level-4 sub-basins, binary; municipalities with deposits dropped | directed river network, distance-weighted |
| Outcome | homicides (+9.4 per 100k, ~30%); garimpo inside protected areas (imprecise spike 2013–14) | birth anomalies (+2.2 per 1,000 per log point), birthweight (−11 g), preterm; LBW rate null | mercury violations → infant deaths, LBW, hospitalizations |
| Water link | none | none (assumed) | mercury measured at stations, but none in the Amazon (see caveats) |

**What is taken.**

- **Pereira & Pucci** already estimate the 2013 law's effect on illegal mining (as a
  first stage, imprecisely) and on local violence. Their Fig. 1 also computes the
  laundering measure of lead 1 in aggregate: declared garimpo sales per km² of mined
  permit area jump from about 80 to 170 kg/km² in 2013.
- **Moreno-Louzada** already claims the headline that births downstream of gold deposits
  worsen, with a downstream-deposit placebo and an other-minerals placebo.

**What remains ours.**

- **The measured water link: no longer available for mercury.** Neither paper observes
  contamination, but neither do we in the Amazon. The station data have 4 mercury readings
  in the whole ANA Amazon basin, none in the Madeira. The pollution arm's mercury result is
  identified from 15 Mata Atlântica/Cerrado stations, and it is fragile to unit coding.

  What the Amazon stations do measure (24,854 samples at 466 stations, mostly physical
  and chemical parameters) may still carry a garimpo signature through turbidity and
  suspended solids. That is the candidate replacement, and it is untested.
- **Distance along a directed network.** Moreno-Louzada lists distance-based exposure and
  a link to observed mining as future work; that is this project's pipeline.
- **Few-cluster inference.** Moreno-Louzada clusters on a handful of level-3 basins
  without a bootstrap. His own later work (Moreno-Louzada, Figueira & Picchetti 2025,
  arXiv 2512.10076) shows that conventional standard errors over-reject in single-price
  exposure designs. In a gold × homicides application for the Amazon, the corrected
  standard errors are about twice as large. Any gold-price × geology term here should
  report them next to the wild-cluster bootstrap.
- **Separating local from downstream channels.** Pereira & Pucci's channel (violence,
  income, migration) is local; the downstream channel is water. Carrying their exposure
  along the network separates the two (lead 1).
- **Leads 2 and 3** are untouched by both papers.

**Weak points in Moreno-Louzada worth noting.**

- **Timing.** The raw treated–control divergence (their Fig. 6) opens in 2014–16, while
  gold prices were falling from the 2012 peak. That timing fits the 2013 law better than
  a price channel.
- **Zika.** Congenital-anomaly reporting surged in 2015–16 with the Zika epidemic; the
  coefficient halves under state×year fixed effects (2.24 → 1.08).
- **Weights.** Weighting by births gives Manaus and Belém, downstream of nearly every
  deposit, very large weight.

**Consistency with the health arm.** Moreno-Louzada's LBW-rate null matches the health
arm's LBW null; infant deaths and anomalies are the margins that move.

**Further close work from the systematic search**
(closed 2026-10-26; `output/research/literature/gold_mining_health/screening.md`):

- **Parfitt (2024, SSRN; essay 2 of his FGV PhD thesis, open access)** is the closest
  competitor for the health arm. **Read.**
  - *Design:* the 2004 garimpo surge in Madre de Dios (Peru/Bolivia) as an upstream
    shock. The 11 municipalities on the Madeira are compared with 110 others in Amazonas
    and Rondônia, 1997–2015. Exposure is "on the river" plus distance bands along it;
    there is no water data.
  - *Results:* infant mortality +7.5 per 1,000 (+44%). By cause (per 1,000): fetal
    +0.14, congenital +1.1, neonatal +2.4. Birthweight −63 g; preterm null. Driven by
    Rondônia within about 600 km of the border; without state capitals, 4.3 (p<0.10).
    Wild-bootstrap p = 0.015 in the preferred column, 0.065–0.10 elsewhere.
  - *Weak points:*
    - Mining in Madre de Dios grew mostly in 2008–10, not in 2004.
    - The Santo Antônio and Jirau dams (from about 2008) and Brazil's own dredge mining
      on the Madeira fall in the treated group.
    - Brazil's shoot-down rule for drug flights (Lei 9.614/1998) took effect with
      Decreto 5.144 of 16 July 2004, the same year. It pushed cocaine routes onto rivers
      out of Bolivia, including the Madeira (Pereira, Pucci & Soares 2024, "Landing on
      Water", IZA DP 17425).
  - *Status:* apparently unpublished. The AEJ: Policy claim from a search snippet has no
    support: no journal version is indexed, and his IDEAS list has no mining paper.
  - *For this project:* it is the health arm's Amazon infant-death result, with a
    foreign source of mining. The step it assumes cannot be tested with the project's
    stations: there is no mercury reading at any Madeira-basin station in any year. Any
    test would need external series, such as fish or hair mercury studies on the Madeira,
    or the hydropower monitoring programmes (from about 2009 only). Suspended solids at
    4 Madeira stations are flat between 1999–2003 and 2005–10 at three of them. That is
    uninformative, since Madeira sediment is dominated by the Andes.
- **Romero & Saavedra (2015 WP; ch. 2 of Saavedra's 2017 Stanford PhD thesis), "The
  effects of gold mining on newborns' health"** (read; never published standalone).
  - *Design:* Colombian births 1999/2001–2012 near legal gold titles. Exposure is
    "near" a title (within 20 km) or "upstream" (a title up to 25 km upstream along the
    river).
  - *Results:* low APGAR −0.51 pp near mines (income) and +0.45 pp downstream (+9.8%,
    p<0.01); LBW null.
  - *For this project:* the closest precedent for splitting β_loc from β_up, over a much
    shorter river distance and with no water data.
  - Chapter 1 of the thesis carries the Colombian royalty reform through illegal mining
    to downstream newborns. That uses a weak IV (F = 3.23) and includes a flow-inversion
    placebo. The published EER paper (below) drops this section, so the full policy →
    mining → newborns chain is unpublished.
- **Amaral, Pantaleão & Rio Tinto (2026, SocArXiv), "The Cost of Gold: Commodity Shocks
  and Violence Against Peasants in Brazil"** (read).
  - *Design:* municipality Bartik, 2000–2019: baseline 2000 MapBiomas mining share ×
    domestic gold price instruments MapBiomas garimpo and gold area. About 1,000
    municipalities, with a Legal Amazon subsample.
  - *Results:* a log point of garimpo area raises homicides by about 10 per 100,000;
    broader gold area raises CPT land conflicts. Total mortality also rises in the
    reduced form, while recorded intoxications do not.
  - *Weak points:* the "undetermined cause" estimates flip sign between their two
    instruments.
  - *What it takes:* a local price-based first stage next to Moreno-Louzada's, and a
    violence channel next to P&P's. No downstream outcome.
  - *For this project:* local mining raises all-cause and external-cause deaths
    (He et al. find the same for Indigenous infant deaths from external causes).
    Downstream infant-death outcomes should therefore exclude external causes (ICD-10
    V–Y) and keep the local term β_loc.
- **Saavedra & Romero (2021, EER)** — illegal mining's response to a Colombian royalty
  reform (read). The closest analog to lead 1's laundering-cost-as-a-tax.
  - *Results:* cutting the municipal royalty share from about 55% to 10% raised the
    illegal share of mined area by 4.21 pp relative to Peru, and by 1.18 pp per point of
    royalty rate.
  - *Where the rise happens:* outside national parks, where a permit is possible. A
    change in the cost of being legal only moves mining that has a legal option. The 2013
    presumption instead lets gold from places with no legal option be sold, which is why
    lead 1 uses deposits inside protected areas (IGD).
  - *For lead 2:* fewer than 20% of mine owners live in the municipality of their mine
    (fn. 8).
- **Vashold et al. (2025, JDE)** — mines' downstream effects on agriculture in Africa
  using directed river networks. A method precedent for the asymmetry tests.
- **He, Pede, de Aguiar Barros & Heilmayr (2026, SSRN 6914819), "Gold Mining Increases
  Malaria Burden and Compounds Indigenous Health Inequality in the Brazilian Amazon"**
  (read).
  - *Design:* shift-share IV, with LASSO-selected rock formations × two-year gold price
    instrumenting MapBiomas gold-mining area. The sample is 86 gold municipalities and
    174 neighbors (2003–2019) plus 106 Indigenous health sub-districts. Malaria comes
    from SIVEP microdata by municipality of infection, obtained through an
    access-to-information request.
  - *Results:* +0.45 API per km² of mining. Spillovers reach 250 km in 50 km
    centroid-distance bands, and they account for most of the 1.35 million attributable
    cases.
  - *Weak points:*
    - Itaituba and Jacareacanga carry most of the identifying variation; leave-one-out
      estimates range from 0.25 to 4.6.
    - The effect disappears in the 2003–2024 panel.
    - Falsification outcomes move: HIV and diabetes hospitalizations.
    - Indigenous infant mortality, including congenital, is null.
  - *What it takes from lead 2:* the causal mining → malaria link and distance
    spillovers.
  - *What remains for lead 2:* spillovers are measured by distance only, not by river
    direction. Infections are not split into residents and non-residents. Those are
    lead 2's two tests.

**Added from the Google Scholar citer lists (2026-10-23, both read):**

- **Oliveira (2025, SSRN 5435977; current title "Deregulating gold, deforesting the
  Amazon: supply-chain liability and illegal mining in Brazil's Indigenous Lands")**.
  - *Design:* difference-in-differences on 8,911 grid cells of 5 km in 15 Indigenous
    lands, 2009–2018. Treated cells lie within 10 km of a pre-2013 gold deposit, with
    Indigenous land × year FE and a gold price × proximity control.
  - *Results:* deforestation rises after 2013 (PPML +126%; 56% of the loss in treated
    cells), through mining area, forest-to-mining and forest-to-pasture conversion, not
    agriculture. Placebos with diamond and tin deposits are null. Kayapó, Munduruku and
    Yanomami drive the effect.
  - *Timing:* the event study rises gradually, small in 2013–15 and largest in 2016–18.
    It stops in 2018 to avoid the Bolsonaro years.
  - *What it takes from lead 1:* the first stage of the 2013 law inside Indigenous lands,
    with the same data (Geological Survey deposits, MapBiomas).
  - *What it confirms:* the first stage exists and is measurable in MapBiomas.
  - *What remains for lead 1:* downstream water and health, the 2023 reversal, and
    exposure outside Indigenous lands.
  - *Ghost permits:* the conclusion proposes cross-checking each permit's declared output
    against its area, geology and satellite footprint as a policy fix, but does not
    estimate it. That is lead 1's ghost-permit test, still open.
- **Hemsley, Libera & Rocha (2026, SSRN 7634888; submitted to Economics Letters)**.
  - *Design:* triple difference, municipality contains an Indigenous territory × overlaps
    a Geological Survey gold province × post-2016, 1,874 municipalities in the Amazon and
    Cerrado, 2010–2022.
  - *Results:* homicides +7.2 per 100,000 (about 0.3 SD). The event study is flat
    through 2017 and jumps in 2018, matching the enforcement collapse in Nunes et al.
    (2024, *Sci. Rep.*): IBAMA field agents fell from 965 to 691.
  - *2013 vs. 2016:* in their two-break model, the 2013 component is +3.2 to +4.0 and
    mostly insignificant; the post-2016 increment is significant.
  - *Implications:* nearly lead 1's exposure, with a second, enforcement shock. It also
    gives lead 3 a documented weakening episode.
  - *Data:* replication data on Zenodo (10.5281/zenodo.22213133).
- **Araujo, Possebom & Setti (2026, arXiv 2509.06076)** — DETER enforcement, instrumented
  by cloud cover, cuts Amazon homicides by ~15%. A cloud-cover instrument is available
  for lead 3.
- **De Salazar et al. (2024, Lancet Reg. Health Americas)** — malaria spillover into
  Indigenous communities in Guyana after Brazil's 2023 crackdown. Cross-border
  displacement evidence for lead 3.
- **Dethier et al. (2023, Conservation Letters), "Operation Mercury"** — Peru's 2019
  armed intervention cut mining in targeted zones by 70–90% while new mining
  accelerated nearby. The closest displacement analog for lead 3.
- **Moretz-Sohn & Costa (2026, OSF), "Protected Area Erasure Accelerates Deforestation"**
  — relevant to lead 4's protected-area-stability threat.

**Implication.** The health arm alone is no longer a contribution. It is needed as the
downstream outcome of the economic shocks in leads 1–3, which none of these papers use.

## Why the project is well placed

- **Directed river network.** The network links every 25 km stretch of river to the
  monitoring stations and municipalities downstream of it. Few land-use papers can
  connect a mining site to the specific populations that drink and fish its water.
- **Pipeline already built.** Water quality (CONAMA violations including mercury),
  municipal health outcomes (SIH, SINASC, SIM) and inference with few clusters (Otto-3
  basins, wild-cluster bootstrap for OLS and PPML) are already in place.
- **Plugging in a design.** A new source of variation in upstream mining only needs to
  enter at the exposure stage; everything downstream is reused.

## Lead 1: gold-laundering policy → garimpo → downstream health

**Question.** Does making illegal gold easier or harder to sell change where garimpo
happens, and does that show up downstream in mercury violations and infant health?

**Mechanism.**

- Illegal gold enters the formal market through small-scale mining permits
  (*permissões de lavra garimpeira*) that declare sales, often without matching
  extraction.
- Law 12,844/2013 (approved 19 July 2013) let gold buyers (*PCOs*) rely on the seller's
  own declaration of legal origin (the "good-faith" presumption), which lowered the cost
  of laundering. Pereira & Pucci (2026) use this shock for violence.
- The reversal came in 2023, which raised the cost again:
  - **Electronic invoice:** IN RFB 2.138 (published 30 March 2023) made an electronic
    invoice for gold mandatory from 3 July 2023. One industry source says IN RFB 2.150
    postponed this to 1 August 2023 (unconfirmed).
  - **Supreme Court:** Justice Gilmar Mendes suspended the good-faith presumption (art.
    39 §4) in an injunction in early April 2023 (STF news of 5 April; ADI 7273 and 7345).
    The plenary confirmed it unanimously in a virtual session ending 2 May 2023, and
    declared the provision unconstitutional on the merits in March 2025.

  With annual data, 2023 is a partial year and 2024 the first full year after the
  reversal. It falls outside Pereira & Pucci's 2006–2019 window.
- Laundering cost acts as a tax on the gold price miners receive. Laundering is done on
  paper, so it should matter most where gold can only be mined illegally, i.e. deposits
  inside protected areas, not necessarily near the permits that absorb the output.

**Design.** Take Pereira & Pucci's exposure and carry it along the river network:

```
Y_mt = β_up·(upstream IGD_m × post-2013_t) + β_loc·(local IGD_m × post-2013_t)
     + β_up'·(upstream IGD_m × post-2023_t) + β_loc'·(local IGD_m × post-2023_t)
     + gold price_t × gold geology_m + municipality FE + state×year FE
```

- **Exposure.** IGD: gold deposits inside protected areas, upstream of *m* (distance-
  weighted on the network) and in *m* itself, defined from deposits and protected areas
  registered before 2013.
- **Local vs. upstream.** β_loc absorbs Pereira & Pucci's local channel (violence,
  income, migration); β_up is the water channel. The asymmetry against IGD *downstream*
  of *m* is the placebo.
- **Two shocks of opposite sign.** 2013 should raise and 2023 lower downstream harm.
- **Ghost permits as heterogeneity.** Permit area that declares gold sales but shows
  little mining footprint (pre-2013) becomes a mechanism test, not the main exposure.
  The Monitor da Mineração's MEND flag (mining expected but not detected) is a ready-made
  version: 769 of 15,461 gold garimpo processes in the 2026 snapshot. It is
  cross-sectional, so it can only define heterogeneity groups. The ANM grant dates
  (event 513) supply the timing.
- **Outcomes.** Mercury and lead violations at downstream stations, though there are
  none in the Amazon (see caveats), so turbidity and suspended solids are the feasible
  water outcomes there; infant deaths and
  congenital anomalies (SINASC) in downstream municipalities. Count infant deaths
  excluding external causes (ICD-10 V–Y), which local mining raises through violence
  (Amaral et al. 2026; He et al. 2026, Table S23).
- **Estimation.** First the reduced form, then the laws as instruments for upstream
  mining.

**Data.**

- **In the pipeline:** the river network, water-quality stations, health outcomes, and a
  mining exposure.
- **To add:**
  - mining-permit polygons with type, mineral and dates (National Mining Agency,
    SIGMINE);
  - royalty payments by permit holder (CFEM). *Pereira & Pucci (App. A.1) report that
    CFEM identifies the permit only for industrial mines, not for garimpo; Manzolli et
    al. (2021) nonetheless match declared output to permits. Check before relying on
    it;*
  - deposits inside protected areas (Geological Survey deposits × FUNAI/MMA
    boundaries), as in Pereira & Pucci;
  - the MapBiomas mining module, which separates garimpo from industrial mining and gold
    from other ores;
  - a gold-geology layer (Geological Survey of Brazil).

**Threats and checks.**

- **Deposits proxy for mining in general.** Control for gold price × geology and run a
  placebo with deposits of other garimpo minerals inside protected areas (as Pereira &
  Pucci do).
- **2023 coincides with Lula's enforcement surge.** The surge targeted Indigenous lands
  and specific rivers. It overlaps with IGD exposure more than the 2013 law did, so the
  2023 coefficient is a joint laundering-plus-enforcement shock unless lead 3's
  operation geography separates them.
- **2016–18 enforcement collapse hits the same places.**
  - Hemsley, Libera & Rocha (2026) find a homicide break in 2018 in Indigenous
    territories × gold areas, nearly the same exposure as IGD.
  - Oliveira's deforestation event study is also largest in 2016–18, and his Indigenous
    land × year FE do not absorb enforcement weakening that differs by distance to
    deposits within a territory. Neither paper separates the law from the enforcement
    collapse for mining itself.
  - Here: report the event study by year, and read the law from 2013–15, before the
    break. Add IGD × post-2016, or Nunes et al.'s IBAMA field-agent series × IGD.
  - The 2023 reversal is the cleaner shock of the two, because the law and enforcement
    both tighten.
- **The injury placebo can legitimately move.** Pereira & Pucci show homicides rising in
  IGD municipalities, partly among non-residents. The trading hubs (Itaituba, Peixoto
  Azevedo, Poconé: 67% of garimpo royalties) lie downstream of major mining. A moving
  injury placebo in downstream hubs is not a falsification; the upstream-vs-downstream
  asymmetry and the local term carry the identification instead.
- **Pre-trends.** Check for trends before 2013.
- **Timing of the existing onsets (checked 2026-10-26).**
  - 11 of the 36 Amazon onsets are in 2012 or earlier, 1 in 2013, and 24 in 2014 or
    later (11 in 2020).
  - Infant deaths split by onset: ≤2012 +0.22 (bootstrap p = 0.076, 11 switchers,
    7 basins); ≥2013 +0.19 (p = 0.001, 25 switchers, 14 basins).
  - The effect is similar before and after the law, so it is not specific to the 2013
    shock. The late group's significance leans on the suspect 2020 cluster.
- **MapBiomas 2020 step.** Collection 10's single mining class (raw ID 30) jumps
  nationally in 2020. Any onset or event-study design on MapBiomas mining must treat
  2020 as a mapping break: end event windows in 2019, or drop 2020 onsets. Check
  whether the mining module shows the same step.

**Contribution.** Pereira & Pucci establish the 2013 law's effect on illegal mining and
local violence, and Oliveira (2025) its effect on mining deforestation inside
Indigenous lands (grid cells near pre-2013 deposits, 2009–2018). This lead takes their shock downstream: it estimates the law's effect
on river mercury and downstream infant health, separates that water channel from their
local channel, and adds the 2023 reversal as a second shock of opposite sign. It also
adds many more mining onsets than the current 36.

## Lead 2: a two-group model of garimpo supply

**Question.** Is garimpo driven by outside crews moving to where enforcement is weak, or
by locals turning to mining as their traditional economy erodes, and how do the two
interact?

**Model.**

- **Outside crews.** Mechanized crews (dredges, excavators) and their financiers are
  mobile and choose sites by expected profit:

  ```
  (gold price − laundering cost) × deposit quality − access cost − P(equipment destroyed) × equipment value
  ```

  Enforcement enters as a *relative* cost: concentrating it on one territory moves crews
  to the next-best deposit rather than out of mining.
- **Local residents.** Riverine, Indigenous and smallholder households choose between
  mining (working for crews, or deals with them) and the traditional economy: fishing,
  hunting, forest products, small-plot farming, transfers. They move into mining when
  that economy erodes, through fewer fish, deforestation, drought or lost cash income.
- **Feedback.** Crews contaminate and silt rivers, fishing declines downstream, and
  downstream residents move into mining. Mining can spread down a river without new
  deposits being found.
- **Policy implication.** Removing crews does not restore residents' livelihoods, which
  predicts quick relapse after enforcement.

**Predictions.**

1. **Displacement.** Outside crews respond to laundering and enforcement shocks, and are
   displaced rather than removed (tested in lead 3).
2. **Downstream asymmetry.** A mining onset raises later onsets downstream more than
   upstream at the same river distance. Deposits and crew movement are roughly
   symmetric along a river, while contamination flows only downstream.
3. **Fishing dependence.** Residents respond more where fishing matters and after
   droughts (e.g. the record-low rivers of 2023–24).
4. **Relapse.** Mining returns faster after enforcement where the traditional economy has
   already degraded.

**Data.**

- **Separating the two groups.** Malaria surveillance records (SIVEP-Malária) give the
  patient's occupation (including mining), likely place of infection and residence.
  Mining-related cases infected at a site but resident elsewhere measure outside crews;
  resident cases measure local participation. *Microdata are obtainable: He et al.
  (2026) received SIVEP records by municipality of infection through an
  access-to-information (LAI) request. Saldanha has since deposited record-level SIVEP
  2004–2024 on Zenodo (20706957), with occupation (`cod_ocup`, 5 = garimpagem),
  residence and probable-infection municipality. It is now on the HPC.*
- **Fishing dependence.** Registered artisanal fishers and closed-season payments to
  fishers (*seguro-defeso*), both public by municipality.
- **Drought.** ERA5 precipitation and soil moisture, already in the pipeline.
- **River direction.** The asymmetry test uses the project's directed reachability and
  distance matrices directly.

**Threats.**

- **Geology.** Alluvial gold concentrates in lower reaches, which confounds the asymmetry
  test; control for geology.
- **Malaria programmes.** Malaria cases also respond to control programmes; use shares
  and state×year fixed effects.
- **Dredges.** Dredge mining on rivers leaves little land footprint and is missed by
  MapBiomas.

**Contribution.** The most novel lead. The downstream-asymmetry test is only possible
with a directed river network linked to outcomes. The model also says which policy
works: if residents drive mining, enforcement alone relapses. It relaxes the one
assumption of Pereira & Pucci's contest model (App. B) that matters here, no migration
of miners between municipalities, and their results already point to outside crews:
effects are smaller by victim residence than by place of death (their Table C.6),
Indigenous victims are a small share, and effects are larger for deposits in
Conservation Areas than in Indigenous lands. Colombian evidence points the same way:
fewer than 20% of mine owners live in the municipality of their mine (Saavedra & Romero
2021, fn. 8), and gold miners migrate and recruit through social networks (Fletcher &
Saavedra 2019).

## Lead 3: enforcement displacement

**Question.** When enforcement shuts down mining in one territory, how much moves
elsewhere, where does it go, and does the health burden move with it?

**Shocks.**

- the Yanomami health emergency and enforcement operation (2023);
- large dredge operations on the Madeira River (2021, 2024, Operation Boiuna 2025);
- operations in Munduruku and Kayapó lands;
- in the other direction, the 2016–18 collapse in federal field enforcement (IBAMA
  field agents 965 → 691; Nunes et al. 2024; used for homicides by Hemsley, Libera &
  Rocha 2026), and the 2019 drop in enforcement credibility, including political
  statements against destroying mining equipment.

**Design.**

- **Substitute sites.** Defined in advance as gold-bearing areas with river access that
  were not targeted, ranked by travel cost from the targeted territory along rivers and
  roads.
- **Event study.** Mining onsets at substitute sites by distance band, around each
  shock.
- **Displacement ratio.** The increase at substitute sites divided by the decrease in
  the targeted area; a ratio near 1 means pure relocation.
- **Health.** Whether mercury violations and infant deaths rise below the substitute
  sites as they fall below the targeted one.

**Data.**

- **To add:** IBAMA fines, embargoes and seizures (geocoded, public), with operation
  dates from official reporting.
- **In the pipeline or planned:** the MapBiomas mining module and the river network for
  travel distances and downstream outcomes.

**Threats.**

- **Few operations.** Few large operations means few independent units; use
  randomization inference across placebo operation dates.
- **Coinciding national changes.** Several shocks coincide with national changes; use
  state×year fixed effects and placebo substitute sites without gold.
- **Dredges.** Dredge mining is invisible to land-cover maps; IBAMA seizure records partly
  fill the gap.

**Contribution.** It quantifies what is so far anecdotal in Brazil (Yanomami → Kayapó,
dredges returning on the Madeira within months). Displacement itself is documented
elsewhere: in Peru, Operation Mercury (Dethier et al. 2023) was followed by new mining
nearby, and after the 2023 Yanomami crackdown malaria spilled over into Guyana (De
Salazar et al. 2024).

The one experimental benchmark is Saavedra (2022 WP, RePEc rie:riecdt:89; read).
Colombian municipalities were randomized to have five satellite-detected mine locations
disclosed to the mayor, the Air Force, or both. Illegal mining fell in the disclosed
1 km cells and the 8 cells around them, but rose elsewhere in the same municipality:
11% gross, 7% net, so about a third displaced. This came from information alone, not
raids. Three limits matter for comparing it with lead 3:
- The far-cell increase is in the illegal share of mined area, and is not significant
  in km².
- Moves into other municipalities are not measured.
- The outcome sample is about 110 municipalities.

Lead 3 improves on each: substitute sites across municipalities and along rivers, mined
area as the outcome, and downstream health. The paper also finds mayors 15–16 pp less
likely to confirm a true mine when it is illegal (the Air Force shows no gap). That is
direct evidence of local collusion, which supports why local and federal enforcement
should be told apart. Pereira,
Pucci & Soares (2024, "Landing on Water") show the same kind of response for drug
trafficking: air interdiction moved routes onto rivers.

What is new here is a displacement ratio for armed enforcement within Brazil, and the
downstream health burden moving with it. A finding that enforcement relocates the health
burden rather than removing it would be a strong policy result, and it directly tests
prediction 1 of lead 2.

## Lead 4: Indigenous-land demarcation and mining inside protected territory

**Question.** Does the gap between legal protection and actual control, especially while
demarcation is pending, attract garimpo into Indigenous lands and protected areas?

**Mechanism.**

- **Protection without control.** Protected status blocks legal mining but not demand.
  Lands with intact deposits and weak state presence become targets.
- **Exposure while pending.** Demarcation of Indigenous lands moves through declared,
  delimited, homologated and regularized stages, and the pending stages may be the most
  exposed.
- **Anticipation.** Bills to legalize mining on Indigenous lands (e.g. PL 191/2020), and
  the mining-permit requests filed over them, act as signals that miners anticipate.

**Design.**

- **Demarcation.** A before-after comparison of mining inside Indigenous lands around the
  homologation date, exploiting the staggered stages.
- **Protected areas.** Strict-protection versus sustainable-use categories.
- **Anticipation.** An event study of permit requests and mining onsets around the
  introduction of PL 191/2020.
- **Downstream.** Effects on the municipalities below these lands.

**Data.**

- **To add:**
  - Indigenous-land boundaries with stage and decree dates (FUNAI);
  - protected-area boundaries with creation dates (national register of conservation
    units);
  - permit requests over Indigenous lands (SIGMINE).
- **In the pipeline:** mining and downstream outcomes.

**Threats.**

- **Political timing.** Demarcation timing is political and may itself respond to mining
  pressure; check for trends before demarcation.
- **Related literature.** Work on property rights in Indigenous lands and deforestation
  exists (e.g. Baragwanath & Bayi 2020), though not on mining. Two Colombian design
  precedents:
  - Romero & Saavedra (2019 WP) compare areas just inside and outside communal titles,
    before and after titling.
  - Ladino, Saavedra & Wiesner (2021, *JPubE*) show that a pre-announced
    coca-substitution programme triggered anticipatory planting that halved its effect.
    This is the template for the PL 191/2020 anticipation test.
- **Little variation in protection.** Pereira & Pucci rely on protected areas being
  stable: 90% of Indigenous lands and 80% of Conservation Areas existed by 2006. That
  supports protected status as exposure in lead 1 but leaves little staggered variation
  here, which is why this lead stays a section unless the stages within Indigenous lands
  vary enough.
- **Indigenous health data.** Indigenous health is poorly captured in municipal DATASUS
  data; it is recorded in a separate system run by the indigenous health secretariat
  (SESAI).

**Contribution.** It explains where displaced crews go, connecting the legal map to the
substitute sites of lead 3.

## How the leads complement each other

| Lead | What it shifts | Which group responds (lead 2) | What it identifies |
|---|---|---|---|
| 1 | Gold price received (laundering cost) | Both, strongest near permits | Supply response to price; first instrument for upstream mining |
| 3 | Relative enforcement cost at a site | Outside crews | Mobility and displacement elasticity; second instrument |
| 4 | Legal status and pending protection | Crews (where they go); residents (participation deals) | Location choice; third instrument |
| 2 | The model linking them | Both | Which group dominates; downstream spread; relapse |

1. **Three independent shocks for one health effect.**
   - The current Amazon result rests on 36 onsets of unknown origin.
   - Leads 1, 3 and 4 each move upstream mining for a different reason, with different
     confounds.
   - A similar health effect per unit of upstream mining across all three is much
     stronger evidence than any single design.
   - If the effects differ, lead 2 suggests why: mechanized crew mining uses more mercury
     than local mining.
2. **Lead 2 turns the shocks into a test of mechanism.** Crews respond to laundering and
   enforcement and are displaced. Residents respond to lost fishing and drought and
   relapse. Only contamination produces the downstream asymmetry. Finding the predicted
   pattern across shocks is what makes this an economics paper rather than three reduced
   forms.
3. **Leads 3 and 4 complete each other's geography.** Lead 3 shows crews leaving
   targeted areas; lead 4 shows where weak protection lets them land. Together with the
   river network, they map who downstream bears the health cost.
4. **One shared data layer serves all four.** The new layer links to the existing
   river-stretch and municipality IDs:
   - the MapBiomas mining module;
   - mining permits and royalty payments;
   - IBAMA enforcement records;
   - Indigenous-land and protected-area boundaries;
   - malaria surveillance records;
   - fisher counts.
5. **One policy question.** Which lever reduces downstream harm per unit of effort:
   controlling gold flows (lead 1), destroying equipment on site (lead 3), or completing
   demarcation (lead 4)? Lead 2 explains why on-site enforcement alone relapses where
   livelihoods have collapsed.

## Proposed packaging and next steps

**Core paper (combining the pollution and health evidence with an economic driver).**
Lead 1 is the main design, lead 2 provides the model and mechanism tests, and lead 3 is
a section on displacement and on where the health burden goes.

**Lead 4.** Either a section on where displaced mining lands, or a separate paper if the
demarcation timing holds up.

**Order of work.**

0. **Cheap checks first.**
   - Done: CFEM does not identify garimpo sales by permit. Only 289 of 15,461 gold garimpo
     processes in the permit layer carry accumulated CFEM. The ghost-permit test has to use
     the satellite side (mined area inside the permit, `transborda`) and sales from other
     sources.
   - Done: timing of the 36 Amazon onsets relative to July 2013, and the 2020 cluster,
     which is a national MapBiomas step (see the summary and lead 1 threats). Gold areas
     could not be checked: no garimpo, gold-deposit or SIGMINE layer exists on the HPC.
   - Add the SINASC congenital-anomaly flag as a birth outcome, so the health arm is
     directly comparable with Moreno-Louzada (only an SIH hospitalization group exists
     now, in `src/data/sources/health/fetch/datasus.py`).
   - Add low APGAR (SINASC APGAR1/APGAR5 < 7) as a birth outcome, for comparability
     with Romero & Saavedra. Parfitt & Goulart (2026, *JDE*) use the field for about 25
     million births.
   - Madeira check: not possible, because no Madeira station has mercury data.
   - Coverage of turbidity and suspended solids in the Amazon:
     - Turbidity at basin-1 stations is thin before 2008 (≤ 11 stations a year), then
       30–80 Madeira stations a year from 2009. Electrical conductivity is the most
       continuous series (1997–2016).
     - In the pollution mining sample, only 10 Amazon stations switch mining presence
       with turbidity or suspended solids on both sides, and 7 of those switch in
       2020–21, the MapBiomas step.
     - So an Amazon water stage from stations is feasible only after 2008, with a better
       mining exposure. Satellite-derived river turbidity is the alternative.
   - Literature search: closed 2026-10-26. It covered:
     - forward and backward citations (OpenAlex, Semantic Scholar, Google Scholar);
     - thesis repositories;
     - RePEc author lists;
     - an author re-screen of the corpus;
     - full reads of Parfitt and Romero & Saavedra.

     Broad keyword queries were skipped. See `literature/gold_mining_health/screening.md`.
1. **Build the shared data layer.** Start with the MapBiomas mining module (garimpo vs.
   industrial, gold vs. other ores; export running), then deposits inside protected
   areas and the permit layer. Replace the project's single mining class with
   garimpo-gold mining, and re-run the health mining arm as the first check, with event
   windows ending in 2019 or the 2020 onsets dropped. That decides whether there is an
   Amazon health result to build on.
2. **Malaria microdata: obtained** (SIVEP 2004–2024 from Zenodo, see the data
   acquisition notes). Lead 2 can start once the mining layer is in.
3. **Decide the water stage.** The options are station turbidity from 2008, satellite
   river turbidity, or the reduced form only.
4. **Estimate lead 1.** Run the reduced form on garimpo-gold onsets, then on the
   downstream water outcome chosen in step 3, and on infant deaths.
5. **Add lead 3.** Shocks, substitute sites, and the displacement ratio.
6. **Add lead 4.** Demarcation stages and anticipation shocks.

## Data inventory

| Data | Source | Use | Status |
|---|---|---|---|
| River network, reachability and distance matrices | project (BHO/ANA) | downstream links, asymmetry test | in pipeline |
| Water quality incl. mercury, lead, cadmium violations | project (monitoring-station panel) | contamination outcome | in pipeline |
| Hospitalizations, births, deaths by municipality | DATASUS (SIH, SINASC, SIM) | health outcomes | in pipeline |
| Land cover incl. a single mining class | MapBiomas Collection 10 (class 30) | current mining exposure | in pipeline; national step in 2020 |
| Mining module (garimpo vs. industrial, by ore) | MapBiomas Collection 11 mining substances (GEE asset) | mining exposure by type | exporting to Drive (see below); the gate for the health arm and lead 1 |
| Mining permits with type, ore, dates | MapBiomas Monitor da Mineração (ANM processes + flags) | laundering exposure (lead 1), anticipation (lead 4) | on the HPC (snapshot 2026-10-26); grant dates need the ANM event history |
| Royalty payments (CFEM) | National Mining Agency | declared output, laundering measure | by permit for only 289 of 15,461 gold garimpo processes; not usable for garimpo |
| Gold geology and deposit points | Geological Survey of Brazil | IGD exposure (lead 1), deposit quality, controls | to add |
| Congenital-anomaly flag on births | DATASUS (SINASC) | birth outcome comparable with Moreno-Louzada | to add |
| APGAR at 1 and 5 minutes | DATASUS (SINASC) | low-APGAR outcome comparable with Romero & Saavedra | to add |
| Fines, embargoes, seizures | IBAMA | enforcement shocks (lead 3) | to add |
| Indigenous lands with stage and dates | FUNAI | lead 4 | to add |
| Protected areas with creation dates | National register of conservation units | lead 4 | to add |
| Malaria cases with occupation, infection site, residence | SIVEP-Malária (Saldanha, Zenodo 20706957, CC-BY) | separating crews and residents (lead 2) | 2004–2024 on the HPC; 2025–26 preliminary in Zenodo 20706949 |
| Registered fishers, closed-season payments | Fisheries registry, federal transparency portal | fishing dependence (lead 2) | to add |
| Gold price | world market series | price shock | to add |

**Data acquisition (status 2026-10-26).**

- **Getting data onto the HPC.** sciCORE compute nodes have no internet, but the transfer
  node `transfer12.scicore.unibas.ch` does. Anything with a direct URL is fetched there
  with `curl`; manual downloads go from the laptop with `rsync` to transfer12.
- **MapBiomas mining substances** (GEE asset
  `projects/mapbiomas-public/assets/brazil/lulc/collection11/mapbiomas_brazil_collection11_mining_substances_v1`).
  - *Bands:* `classification_1985` … `classification_2025` (41 bands), EPSG:4326 on the
    MapBiomas grid.
  - *Class codes ABCC:* A = 1 garimpo, 2 industrial; B = 1 exposed soil, 2 revegetation;
    CC = substance (15 gold, 14 tin, 09 copper, 24 precious stones, 58 atypical, 59 no
    substance).
  - *Legend:* CSV in `data/land_cover/auxiliary/` (from MapBiomas's legend-code page).
  - *Sample check:* around Itaituba, garimpo gold dominates in 2025 (1115 and 1215).
- **Export route.** Platform downloads failed, and Google blocks the Earth Engine Python
  login for this account, so the export runs from the Code Editor
  (`scripts/gee_export_mapbiomas_mining.js`):
  - uint16, 12,288-pixel tiles to Drive folder `mapbiomas_mining_c11`;
  - a national area-by-class CSV, to check for the 2020 step.
  
  Next: rsync the tiles to the HPC, merge them to one GeoTIFF per year, and add a
  mining-legend extraction to the land-cover pipeline.
- **Permits.** The Monitor da Mineração process layer (270,877 ANM processes; 21,836
  garimpo, 15,461 of them gold) is at
  `data/mining/raw/mapbiomas_monitor_processos_2026-10-26/geoserver.zip` on the HPC.
  - *Caveats:* it is a current snapshot, so expired processes are probably missing. `ano`
    is the process-number year. The municipality, Indigenous-land and protected-area ID
    fields are broken by the export (recompute them spatially).
  - *Flags,* per MapBiomas's descriptions: `transborda` = mining overflowing the permit
    polygon; `em_area_re` = in a restricted area. The inconsistency flags cover mining in
    the wrong phase and mining that predates the authorization. The mapping of `mend` and
    `permissao_` is unconfirmed.
- **ANM event history** (grant dates; browser check 2026-10-26). The ANM's Cadastro
  Mineiro microdata (`dadosabertos.anm.gov.br/SCM/microdados/microdados-scm.zip`,
  181 MB, updated daily, 1934 onward) have the full history, joined to SIGMINE on
  `DSProcesso`. The firewall returns 403 to `curl`, including from transfer12, so it has
  to be downloaded in a browser and rsynced. The extract of 2026-10-25, with its
  dictionary and entity-relationship diagram, is at
  `data/mining/raw/anm_scm_microdados_2026-10-26/` on the HPC (checksums verified).
  - `ProcessoEvento.txt` (about 1.1 GB, `;`-separated, CP-1252): process, event ID,
    event date, note, and the Diário Oficial extract. `Evento.txt` describes about 2,900
    event codes:
    - PLG grant published = 513; PLG renewal = 523;
    - PLG filed = 333; rejections = 334, 335, 337, 388; codes from 521 onward not yet
      read.
  - `ProcessoTitulo.txt`: title number, legal-document type (2 = PLG), situation
    (outorgado, renovado, caduco, cancelado, cassado, …), publication and expiry dates.
  - `Processo.txt`: active flag `BTAtivo`. About 71% of the first 88,000 rows are
    inactive, so extinct processes are included, unlike the Monitor and SIGMINE active
    layers.
  - *Dictionary:* `metadados-microdados-scm.ods`; entity-relationship diagrams:
    `mer-microdados-scm.pdf`.
  - *Still unchecked:* the start year of the events table and whether it covers extinct
    processes.
  - *Polygons including inactive processes and garimpo reserves:* SIGMINE on dados.gov.br.
- **Monitor da Mineração flags.** MapBiomas publishes no attribute dictionary or ATBD for
  the process layer. What its factsheet does define:
  - transbordo (mining beyond the permit limits);
  - temporal inconsistency (mining before the authorization's validity);
  - CFEM "quitado" (any CFEM payment in the last six months);
  - restricted areas (including all strict-protection conservation units);
  - MEND = "Mineração Esperada Não Detectada" (mining expected but not detected).

  The rest are unconfirmed; the contact is contato@mapbiomas.org.
- **SIVEP-Malária 2004–2024** (Saldanha, Zenodo 20706957) is at
  `data/health/raw/sivep_malaria/` on the HPC. It has activity in the prior 15 days
  (`cod_ocup`, 5 = garimpagem), municipality of residence, and probable municipality and
  locality of infection. No access request is needed.

## Other leads considered

Ranked below leads 1–4, mainly because they are crowded or do not use the project's
river and health data:

5. **Bolsonaro and Lula election shocks.** Best used as shocks within leads 1–3 rather
   than on their own: the variation is over time only, and the topic is heavily studied
   for deforestation.
6. **Rural credit rule (Resolution 3,545) and substitution into mining.** An open question
   with a known design, but a likely precise null given that gold prices dominate. Useful
   as a test of the residents' side within lead 2.
7. **Random federal corruption audits.** A clean randomization, but an unclear link to
   garimpo, since mining permits are federal. A quick side test.
8. **The CAR registry, land titling and pasture as a possession claim.** Important and
   data-rich, but crowded, and the health arm found no pasture effect to carry a health
   story.
9. **Fragmenting clearings below satellite detection thresholds.** Already documented by
   a recent study of the priority-municipality blacklist.
10. **Supply-chain rules pushing activity into mining.** Weak priors (cattle returns are
    far below soy) and no data yet. The 2026 soy-moratorium collapse becomes testable
    once MapBiomas covers 2026–27.

## Caveats

- **Unverified sources.** Several sources behind the original brainstorm are from
  2025–26 and still need checking before a paper relies on them: the blacklisting
  fragmentation study, the 2026 reviews, and the Greenpeace and Grist permit figures.
- **Policy dates.** The 2013 law was approved on 19 July 2013 (Pereira & Pucci, §2.3).
  The 2023 dates are in lead 1, from secondary legal sources: injunction early April
  2023, plenary 2 May 2023, invoice 3 July (or 1 August) 2023. Check them against the
  STF case record and the Diário Oficial before citing.
- **Literature search.** Closed on 2026-10-26 without broad keyword queries. Citation
  channels missed several close papers that were found only on re-screening, so a
  paper with a new title or outside these networks could still be missing.
- **Dredges.** River-dredge mining is largely invisible to land-cover data, so any design
  based on MapBiomas undercounts it. A UnB geography thesis develops satellite
  detection of dredges (D. H. C. Pereira 2023, repositorio.unb.br/handle/10482/48214;
  unread).
- **Mercury data (HPC check, 2026-10-26).**
  - *Coverage:* the project's stations have essentially no mercury readings in the
    Amazon. The ANA Amazon and Tocantins basins have 4 readings in total, at two
    Teles Pires stations in 2019–20, and the Madeira has none. No state-network stations
    with mercury were found in the Amazon.
  - *Pollution arm's mining → mercury result (+13.7 pp):* identified from 15 stations in
    MG, SP, RJ and SC, sampled mostly 1985–99.
  - *Unit coding:* 58 station-periods mix values of 0.0002 or below with values of 0.1
    or above, which looks like µg/L entries in a mg/L column. Setting those high values
    to missing gives +6.9 pp (SE 4.5, bootstrap p = 0.28).
  - *Consequence:* the "measured water link" framing in this note needed revising.
  - *Done:* the rule is now in the pipeline (`drop_unit_errors`), and the pollution
    notebooks were re-run on 2026-10-26 (commit `bd8b356`).
    - Mercury: +5.7 pp (SE 4.35, bootstrap p = 0.34, 11 basins).
    - Lead and cadmium: unchanged.
    - Values set to missing: 209 (177 mercury, 24 lead, 7 arsenic, 1 cadmium).
