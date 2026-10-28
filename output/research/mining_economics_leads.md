# Garimpo supply in the Brazilian Amazon: research design

*Living document; status as of 2026-10-28. It builds on the project's pollution and
health arms (`output/notebooks/pollution/memo.md`, `output/notebooks/health/memo.md`).
The evidence behind each claim is collected in the [Evidence](#evidence) section, with
the location of its code.*

## Summary

**The paper.** What drives the supply of illegal gold mining (*garimpo*) in the
Brazilian Amazon, who responds, and which policy lever reduces it once displacement and
relapse are counted?

- **A model of two groups** (lead 2):
  - Outside crews are mobile and choose sites by expected profit.
  - Local residents choose between mining and a traditional economy that mining itself
    erodes.
- **Three policy shocks**, each identifying one parameter of that model:
  - laundering policy and the gold price received (lead 1, the main design);
  - enforcement (lead 3, a section on displacement);
  - Indigenous-land demarcation (lead 4, a section or a separate paper).
- **Outcomes:**
  - garimpo area from MapBiomas Collection 11 (C11);
  - malaria cases from SIVEP among garimpo workers, split into local and incoming
    workers and infections abroad;
  - mining permits from the National Mining Agency (ANM).
- **Health as a consequence, not the headline.** Three pieces:
  - a downstream null for mapped gold mining;
  - Moreno-Louzada's price × geology design, which does not hold up in this data;
  - malaria carried home by mining crews.

**What the project's evidence shows so far.**

- **Pollution.** Upstream pasture raises lead and cadmium violations and agriculture
  raises COD. Mining raises lead violations. The mercury result does not survive a
  unit-coding correction, and the stations have almost no mercury readings in the Amazon
  ([E1](#e1-pollution-arm-and-mercury-data)).
- **Health.** There is no effect of upstream pasture or agriculture.
  - *Mapped garimpo gold:* no credible downstream health effect. The upstream gold
    share fails the injury placebo, and at 50 km it rests on one municipality
    ([E4](#e4-gate-upstream-garimpo-gold-share-and-health)).
  - *The earlier Amazon result:* infant deaths rise where mining appears within 50 km
    upstream (+23%). The result is robust but not about gold: it survives dropping every
    municipality with gold upstream, and C11 assigns its onsets to non-gold mining
    ([E2](#e2-the-amazon-mining-presence-result), [E4](#e4-gate-upstream-garimpo-gold-share-and-health)).
  - *Moreno-Louzada's design:* gold upstream × the gold price does not reproduce his
    anomaly result (+0.84 per 1,000 births per log point, SE 0.98, against +2.28). Only
    congenital infant deaths respond, and not with state×year FE. The injury placebo
    fails, and the interaction does not move garimpo upstream
    ([E9](#e9-moreno-louzadas-design)).
- **Garimpo is measurable in SIVEP.**
  - The split between local and incoming workers is feasible ([E5](#e5-sivep-as-a-measure-of-garimpo-workers)).
  - One in five garimpo cases was infected abroad.
  - SIVEP sees mining that MapBiomas misses, but only where malaria is transmitted.
    C11 garimpo gold fits garimpo cases no better than C10 mining (Spearman 0.36 vs.
    0.37), and 80% of case municipalities have none.
- **Gold geology.** The Geological Survey of Brazil's (SGB) never-exploited gold
  occurrences predict where garimpo is about as well as all occurrences. They are,
  however, thin (74 Legal Amazon municipalities) ([E6](#e6-sgb-gold-geology)).

**Open before the design is final.**

- A novelty check on malaria importation by miners.
- The permit layer (grant dates), needed for lead 1's laundering exposure.

## The paper

### Question and policy angle

Brazil has three levers against garimpo:

- control of gold flows (the 2013 good-faith rule and its 2023 suspension);
- destruction of equipment on site (federal enforcement operations);
- completing Indigenous-land demarcation.

The paper asks which one reduces garimpo per unit of effort, net of displacement to
other sites and of relapse. The model explains why the answer differs by group: crews
move, while residents whose livelihoods have collapsed relapse.

### Structure

| Part | Role | What it identifies | Main data |
|---|---|---|---|
| Lead 2: crews and residents | Model and mechanism tests | Which group responds to what; downstream spread; relapse | SIVEP local, incoming and abroad cases; flow table; C11 |
| Lead 1: laundering policy and the gold price | Main design | Supply response to the price received, by group | C11 garimpo area, SIVEP, permits, SGB deposits |
| Lead 3: enforcement | Section | Enforcement response; displacement ratio, including across the border | IBAMA, SIVEP flows, C11 |
| Lead 4: demarcation | Section or separate paper | Where crews go when protection is pending | FUNAI, CNUC, permits |
| Consequences: health | Section | Downstream null; price × geology test; malaria importation | DATASUS, SIVEP |

### How the parts fit

| Lead | What it shifts | Which group responds (lead 2) | Parameter |
|---|---|---|---|
| 1 | Gold price received (laundering cost) | Both, strongest near permits | Supply elasticity to price, by group |
| 3 | Relative enforcement cost at a site | Outside crews | Enforcement elasticity; displacement ratio |
| 4 | Legal status and pending protection | Crews (where they go); residents (participation deals) | Location choice |

1. **One supply system, three shocks.** Each shock moves garimpo for a different reason,
   with different confounds. The model predicts which group responds to which shock.
   Finding that pattern is what makes this an economics paper rather than three reduced
   forms.
2. **Lead 2 supplies the mechanism tests.**
   - Crews respond to laundering and enforcement, and are displaced.
   - Residents respond to lost fishing and drought, and relapse.
   - Only contamination produces a downstream asymmetry in the spread of mining.
3. **Leads 3 and 4 complete each other's geography.** Lead 3 shows crews leaving
   targeted areas; lead 4 shows where weak protection lets them land.
4. **One data layer serves all parts.** It links to the project's river-stretch and
   municipality IDs:
   - C11 mining substances;
   - SIVEP;
   - SGB geology;
   - permits;
   - IBAMA enforcement;
   - Indigenous lands and protected areas;
   - fishers.
5. **One policy comparison.** Lever effects in common units of garimpo, with the
   displacement ratio from lead 3 and relapse from lead 2.

### Health in the paper

- **Downstream null for mapped gold.** Upstream garimpo-gold share has no credible
  effect on infant deaths or birth outcomes ([E4](#e4-gate-upstream-garimpo-gold-share-and-health)).
  Reported as a finding, with its limits:
  - mapped area misses dredges;
  - mapped expansion is not random.
- **Moreno-Louzada's design does not hold up** ([E9](#e9-moreno-louzadas-design)).
  Upstream gold deposits × the log gold price in reais, with anomalies at birth, birth
  outcomes and infant deaths. It was the test of a downstream effect through
  price-induced activity rather than mapped area.
  - The anomaly coefficient is a third of his and insignificant in every specification
    under cluster, bootstrap and randomization inference.
  - Congenital infant deaths respond, but not with state×year FE or unweighted.
  - The injury placebo fails, and there is no first stage on upstream garimpo.
  - So the null generalizes. The price in reais trends up and jumps in 2020, and the
    interaction picks up differential trends between exposed and other municipalities.
    This is consistent with his result being a trend rather than Zika: dropping 2015–17
    changes little.
- **Malaria importation** (planned, after a novelty check). Crews infected at mining
  sites carry malaria to where they live.
  - Test: non-garimpo cases rise in crews' home municipalities after garimpo shocks
    elsewhere.
  - It answers lead 3's question of where the health burden goes, using the SIVEP flow
    table.
- **The earlier presence result** is a robust association with non-gold mining
  frontiers. It is documented in the health memo and not built on.

## Closest literature and positioning

Two papers sit directly on the paper. Both build on the river-direction design of Dias,
Rocha & Soares (2023, *ReStud*), which this project also uses.

| | Pereira & Pucci (2026, *JPubE*) | Moreno-Louzada (2024, WP) | This project |
|---|---|---|---|
| Shock | Law 12,844/2013 (good-faith presumption), 2006–2019 | log gold price × gold deposits, 2007–2021 | 2013 rule and 2023 suspension, gold price in reais, enforcement, demarcation |
| Exposure | gold deposit inside a protected area, in the municipality itself (IGD) | gold deposit upstream on Otto level-4 sub-basins, binary; municipalities with deposits dropped | deposits (never-exploited SGB occurrences), IGD, along a directed river network |
| Outcome | homicides (+9.4 per 100k, ~30%); garimpo inside protected areas (imprecise spike 2013–14) | birth anomalies (+2.2 per 1,000 per log point), birthweight (−11 g), preterm; LBW rate null | garimpo area; SIVEP garimpo workers by group; permits; downstream health as a consequence |
| Who mines | not observed (contest model without migration) | not observed | local vs. incoming workers, infections abroad |

**What is taken.**

- **Pereira & Pucci** estimate the 2013 law's effect on illegal mining (as a first stage,
  imprecisely) and on local violence. Their Fig. 1 computes lead 1's laundering measure in
  aggregate: declared garimpo sales per km² of mined permit area jump from about 80 to
  170 kg/km² in 2013.
- **Moreno-Louzada** claims that births downstream of gold deposits worsen, with a
  downstream-deposit placebo and an other-minerals placebo.

**What remains ours.**

- **Who supplies garimpo.** No paper splits miners into local residents and incoming
  crews, or observes where they come from. SIVEP's residence and infection sites do both
  (lead 2).
- **A precise supply response.** C11 garimpo area and SIVEP give the first stage that
  Pereira & Pucci estimate imprecisely, by group, and through the 2023 reversal.
- **Displacement with a measured outside option.** Infections abroad and the flow table
  measure where crews go, including across the border (lead 3).
- **Distance along a directed network.** Moreno-Louzada lists distance-based exposure and
  a link to observed mining as future work; that is this project's pipeline. It also
  makes the downstream-spread test of lead 2 possible.
- **Few-cluster inference.** Moreno-Louzada clusters on a handful of level-3 basins
  without a bootstrap. His later work (Moreno-Louzada, Figueira & Picchetti 2025, arXiv
  2512.10076) shows that conventional standard errors over-reject in single-price
  exposure designs. In a gold × homicides application for the Amazon, the corrected
  standard errors are about twice as large. Every gold-price × geology term here reports
  them next to the wild-cluster bootstrap and randomization over random-walk price paths.
  In this project's health application they come out *smaller* than the Otto-3 SEs and
  close to clustering by year ([E9](#e9-moreno-louzadas-design)).

**Moreno-Louzada and this project's health null.**

- *The comparable test* ([E9](#e9-moreno-louzadas-design)): his design in this project's
  data (Otto-4 gold upstream × log gold price in BRL, SINASC anomalies at birth, Legal
  Amazon, 2007–2021) gives +0.84 anomalies per 1,000 births per log point (Otto-3 SE
  0.98), against his +2.28 (SE 1.16).
- *Where they agree:* the LBW rate is null in both.
- *His weak points, as tested:*
  - **Timing.** The raw treated–control divergence (his Fig. 6) opens in 2014–16, while
    gold prices were falling from the 2012 peak. That fits the 2013 law better than a
    price channel.
  - **Zika.** Congenital-anomaly reporting surged in 2015–16 with the Zika epidemic. His
    coefficient halves under state×year fixed effects (2.24 → 1.08). Here it falls from
    0.84 to 0.44 under state×year FE; dropping 2015–17 leaves it at 0.96.
  - **Weights.** Weighting by births gives Manaus and Belém, downstream of nearly every
    deposit, very large weight. Unweighted, the anomaly coefficient is 0.41; without the
    two capitals, 0.63.
  - **Placebos.** Downstream gold moves anomalies as much as upstream gold (1.79, p 0.11),
    and the injury placebo falls significantly with the interaction.
  - **Mechanism.** The interaction does not move upstream C11 garimpo gold or garimpo
    malaria cases.
- *Our own congenital measures:* SIH chapter-17 hospitalizations fall with the
  interaction (−2.6 per 1,000 births), while congenital infant deaths rise (+0.31,
  bootstrap p 0.016). With the mapped-area design, the congenital coefficient has his
  sign (+7–14% per SD of gold share) but fails the injury placebo
  ([E4](#e4-gate-upstream-garimpo-gold-share-and-health)).

**Further close work** (systematic search; `output/research/literature/gold_mining_health/screening.md`):

- **Parfitt (2024, SSRN; essay 2 of his FGV PhD thesis)**, read.
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
  - *Status:* apparently unpublished; no journal version is indexed.
  - *For this project:* a downstream health result whose mercury step cannot be checked.
    No Madeira-basin station has a mercury reading in any year.
- **Romero & Saavedra (2015 WP; ch. 2 of Saavedra's 2017 Stanford PhD thesis), "The
  effects of gold mining on newborns' health"**, read; never published standalone.
  - *Design:* Colombian births 1999/2001–2012 near legal gold titles. Exposure is "near"
    a title (within 20 km) or "upstream" (a title up to 25 km upstream along the river).
  - *Results:* low APGAR −0.51 pp near mines (income) and +0.45 pp downstream (+9.8%,
    p<0.01); LBW null.
  - *For this project:* the closest precedent for splitting local from upstream effects,
    over a much shorter river distance and with no water data. Chapter 1 carries the
    Colombian royalty reform through illegal mining to downstream newborns, with a weak IV
    (F = 3.23) and a flow-inversion placebo; the published EER paper drops it.
- **Amaral, Pantaleão & Rio Tinto (2026, SocArXiv), "The Cost of Gold: Commodity Shocks
  and Violence Against Peasants in Brazil"**, read.
  - *Design:* municipality Bartik, 2000–2019: baseline 2000 MapBiomas mining share ×
    domestic gold price instruments MapBiomas garimpo and gold area, about 1,000
    municipalities, with a Legal Amazon subsample.
  - *Results:* a log point of garimpo area raises homicides by about 10 per 100,000;
    broader gold area raises CPT land conflicts. Total mortality also rises in the
    reduced form, while recorded intoxications do not.
  - *Weak points:* the "undetermined cause" estimates flip sign between their two
    instruments.
  - *For this project:* a local price-based first stage, close to lead 1's. Local mining
    raises external-cause deaths, so downstream infant-death outcomes should exclude
    ICD-10 V–Y (He et al. find the same for Indigenous infant deaths).
- **Saavedra & Romero (2021, EER)**, read: illegal mining's response to a Colombian
  royalty reform, the closest analog to lead 1's laundering cost as a tax.
  - *Results:* cutting the municipal royalty share from about 55% to 10% raised the
    illegal share of mined area by 4.21 pp relative to Peru, and by 1.18 pp per point of
    royalty rate.
  - *Where the rise happens:* outside national parks, where a permit is possible. A
    change in the cost of being legal only moves mining that has a legal option. The 2013
    presumption instead lets gold from places with no legal option be sold, which is why
    lead 1 uses deposits inside protected areas (IGD).
  - *For lead 2:* fewer than 20% of mine owners live in the municipality of their mine
    (fn. 8).
- **He, Pede, de Aguiar Barros & Heilmayr (2026, SSRN 6914819), "Gold Mining Increases
  Malaria Burden and Compounds Indigenous Health Inequality in the Brazilian Amazon"**,
  read.
  - *Design:* shift-share IV, with LASSO-selected rock formations × two-year gold price
    instrumenting MapBiomas gold-mining area. The sample is 86 gold municipalities and
    174 neighbors (2003–2019) plus 106 Indigenous health sub-districts. Malaria comes
    from SIVEP microdata by municipality of infection.
  - *Results:* +0.45 API per km² of mining. Spillovers reach 250 km in 50 km
    centroid-distance bands and account for most of the 1.35 million attributable cases.
  - *Weak points:*
    - Itaituba and Jacareacanga carry most of the identifying variation; leave-one-out
      estimates range from 0.25 to 4.6.
    - The effect disappears in the 2003–2024 panel.
    - Falsification outcomes move (HIV and diabetes hospitalizations).
    - Indigenous infant mortality, including congenital, is null.
  - *What it takes:* the causal mining → malaria link, and distance spillovers.
  - *What remains:*
    - spillovers by river direction rather than distance;
    - the split of infections into local and incoming workers;
    - malaria carried to crews' home municipalities. Their spillovers are measured
      around the mine, not at the workers' residences.
- **Oliveira (2025, SSRN 5435977), "Deregulating gold, deforesting the Amazon:
  supply-chain liability and illegal mining in Brazil's Indigenous Lands"**, read.
  - *Design:* difference-in-differences on 8,911 grid cells of 5 km in 15 Indigenous
    lands, 2009–2018. Treated cells lie within 10 km of a pre-2013 gold deposit, with
    Indigenous land × year FE and a gold price × proximity control.
  - *Results:* deforestation rises after 2013 (PPML +126%; 56% of the loss in treated
    cells), through mining area, forest-to-mining and forest-to-pasture conversion, not
    agriculture. Placebos with diamond and tin deposits are null. Kayapó, Munduruku and
    Yanomami drive the effect. The event study rises gradually, largest in 2016–18, and
    stops in 2018.
  - *What it takes:* lead 1's first stage inside Indigenous lands, with SGB deposits and
    MapBiomas.
  - *What remains:* exposure outside Indigenous lands, the 2023 reversal, the split by
    group, and the ghost-permit test. His conclusion proposes cross-checking each
    permit's declared output against its area, geology and satellite footprint, but does
    not estimate it.
- **Hemsley, Libera & Rocha (2026, SSRN 7634888; submitted to Economics Letters).**
  - *Design:* triple difference: the municipality contains an Indigenous territory ×
    overlaps an SGB gold province × post-2016. 1,874 municipalities in the Amazon and
    Cerrado, 2010–2022.
  - *Results:* homicides +7.2 per 100,000 (about 0.3 SD). The event study is flat through
    2017 and jumps in 2018, matching the enforcement collapse in Nunes et al. (2024,
    *Sci. Rep.*): IBAMA field agents fell from 965 to 691. In their two-break model, the
    2013 component is +3.2 to +4.0 and mostly insignificant.
  - *For this project:* nearly lead 1's exposure, with an enforcement shock that lead 3
    can use. Replication data on Zenodo (10.5281/zenodo.22213133).
- **Araujo, Possebom & Setti (2026, arXiv 2509.06076):** DETER enforcement, instrumented
  by cloud cover, cuts Amazon homicides by ~15%. The same cloud-cover instrument is
  available for lead 3.
- **De Salazar et al. (2024, Lancet Reg. Health Americas):** malaria spillover into
  Indigenous communities in Guyana after Brazil's 2023 crackdown. Cross-border
  displacement evidence for lead 3.
- **Dethier et al. (2023, Conservation Letters), "Operation Mercury":** Peru's 2019 armed
  intervention cut mining in targeted zones by 70–90% while new mining accelerated
  nearby. The closest displacement analog for lead 3.
- **Saavedra (2022 WP, RePEc rie:riecdt:89)**, read: the one experimental benchmark for
  lead 3 (see there).
- **Vashold et al. (2025, JDE):** mines' downstream effects on agriculture in Africa
  using directed river networks. A method precedent for the asymmetry test.
- **Moretz-Sohn & Costa (2026, OSF), "Protected Area Erasure Accelerates
  Deforestation":** relevant to lead 4's protected-area-stability threat.

## Why the project is well placed

- **Directed river network.** Every 25 km river stretch is linked to the municipalities
  and stations downstream of it. That makes the downstream-spread test (lead 2),
  substitute sites along rivers (lead 3) and downstream consequences possible.
- **Record-level malaria data.** SIVEP 2004–2024 with occupation, residence and
  infection site, processed into local, incoming and abroad counts and a flow table.
- **Pipeline already built.** Every dataset is a module of the data pipeline with fetch,
  preprocess and verification steps. Health outcomes (SIH, SINASC, SIM) and few-cluster
  inference (Otto-3 basins, wild-cluster bootstrap for OLS and PPML) are in place.

## Lead 1: laundering policy, the gold price and garimpo supply

**Question.** Does making illegal gold easier or harder to sell change how much garimpo
happens, where, and who does it?

**Mechanism.**

- Illegal gold enters the formal market through small-scale mining permits
  (*permissões de lavra garimpeira*, PLG) that declare sales, often without matching
  extraction.
- Law 12,844/2013 (approved 19 July 2013) let gold buyers (*PCOs*) rely on the seller's
  own declaration of legal origin (the "good-faith" presumption). That lowered the cost of
  laundering. Pereira & Pucci (2026) use this shock for violence.
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

**Design.** A supply equation with intensity outcomes; C11 garimpo gold barely has
onsets ([E3](#e3-the-c11-mining-substances-layer)):

```
X_mt = β_13·(IGD_m × post-2013_t) + β_23·(IGD_m × post-2023_t)
     + γ·(log gold price in reais_t × deposits_m) + municipality FE + state×year FE
```

- **Outcomes X_mt:**
  - C11 garimpo-gold area or share, on 2000–2019;
  - SIVEP local and incoming garimpo cases, through 2024. The 2023 reversal is only
    observable here, because C11 steps in 2020.
  - new PLG grants (ANM event 513).
- **Exposure:**
  - *IGD:* gold deposits inside protected areas, registered before 2013, in the
    municipality and upstream along the network.
  - *deposits_m:* never-exploited SGB gold occurrences, restricted to the Legal Amazon.
    Occurrences recorded at garimpos or mines are not predetermined ([E6](#e6-sgb-gold-geology)).
- **By group.** The same equation on local and on incoming cases gives the model's
  prediction 5.
- **Ghost permits as heterogeneity.** Permit area that declares gold sales but shows
  little mining footprint before 2013 is a mechanism test.
  - The Monitor da Mineração's MEND flag (mining expected but not detected) is a
    ready-made version: 769 of 15,461 gold garimpo processes in the snapshot.
  - It is cross-sectional, so it defines groups only. ANM grant dates supply the timing.
  - CFEM royalties cannot measure declared sales by permit ([E7](#e7-cfem-royalties-by-permit)).
- **Inference.** Otto-3 clusters with the wild-cluster bootstrap. For the price term,
  add the price-exposure SEs of Moreno-Louzada, Figueira & Picchetti (2025) and
  randomization over random-walk price paths (`src/analysis/moreno_louzada.py`).
  - The price-exposure SE treats yearly prices as independent and comes out close to
    clustering by year.
  - With a persistent price, that understates the uncertainty, and the random-walk
    randomization is the binding check ([E9](#e9-moreno-louzadas-design)).

**Threats and checks.**

- **Deposits proxy for mining in general.** Run a placebo with deposits of other garimpo
  minerals inside protected areas, as Pereira & Pucci do. Keep the price × deposits
  control.
- **2023 coincides with Lula's enforcement surge.** The surge targeted Indigenous lands
  and specific rivers. It overlaps with IGD exposure more than the 2013 law did, so the
  2023 coefficient is a joint laundering-plus-enforcement shock unless lead 3's
  operation geography separates them.
- **The 2016–18 enforcement collapse hits the same places.**
  - Hemsley, Libera & Rocha (2026) find a homicide break in 2018 in Indigenous
    territories × gold areas, nearly the same exposure as IGD.
  - Oliveira's deforestation event study is also largest in 2016–18, and his Indigenous
    land × year FE do not absorb enforcement weakening that differs by distance to
    deposits within a territory.
  - Here: report the event study by year, read the law from 2013–15, and add IGD ×
    post-2016 or Nunes et al.'s IBAMA field-agent series × IGD.
  - The 2023 reversal is the cleaner shock, because the law and enforcement both tighten.
- **The 2020 step.** C11 garimpo gold rises 15.1% in 2020, against 4.0–9.7% a year in
  2016–2023, and industrial gold rises 12.8%, against 1.7–2.9% in 2016–19. That is hard
  to read as real activity ([E3](#e3-the-c11-mining-substances-layer)).
  - Part of it is real: the gold price in reais rose by about two-thirds in 2020, and
    SIVEP garimpo cases rose 55%.
  - Area-based designs stop in 2019. SIVEP-based designs must hold up under both
    readings.
- **C11 growth accelerates before 2020.** National garimpo-gold growth runs at about 1%
  a year in 2000–11, 2–3% in 2012–15 and 4–5% in 2016–19. The event study has to
  separate this acceleration from the gold price in reais.
- **SIVEP is not gold-specific.** "Garimpagem" covers every substance. The Rondônia
  cluster around Ariquemes is probably cassiterite (tin). Use the gold geology to
  restrict or interact.
- **Pre-trends.** Check for trends before 2013.

**Contribution.** Pereira & Pucci establish the 2013 law's effect on local violence, with
an imprecise first stage. Oliveira (2025) establishes its effect on mining deforestation
inside Indigenous lands. This lead estimates the supply response itself, with a precise
measure of garimpo area and of who does the mining, and adds the 2023 reversal as a second
shock of opposite sign.

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

**Formal sketch.** Sites i are basins or municipalities on the directed river network,
with deposit quality q_i and access cost c_i. X_it is garimpo area.

1. **Crews choose a site (logit).** A mass M of crews picks the site with the highest
   value:

   ```
   V_it = (p_t − λ_t)·q_i − c_i − e_it·K − w_it·ℓ + ε
   s_it = exp(V_it) / (exp(V_0t) + Σ_j exp(V_jt))
   ```

   - λ_t is the laundering cost: lowered by the 2013 good-faith rule, raised by the 2023
     STF suspension (lead 1).
   - e_it is the chance that equipment is destroyed, K its value.
   - w_it·ℓ is the local labour a crew needs (guides, workers, boat operators).
   - s_0 is the outside option of leaving mining or the region; infections abroad
     measure part of it.

   Enforcement at i moves crews to other sites in proportion to their shares, and a
   fraction s_0/(1 − s_i) leaves mining. The displacement ratio
   (1 − s_i − s_0)/(1 − s_i) is the parameter lead 3 estimates.
2. **Residents choose an occupation.** Household h mines if w_it − κ_h > y^T_it, where
   y^T_it = f_i(Fish_it, Forest_it, Rain_it) + transfers. κ_h is the household's cost of
   mining (norms, risk, community rules). The resident mining share is
   R_it = F(w_it − y^T_it).
3. **A local labour market links the groups.** Crew labour demand D(s_it·M) and resident
   supply R_it set w_it. The groups are strategic complements:
   - crews raise the wage and pull residents in;
   - a pool of residents willing to work lowers crews' costs (and, in a richer version,
     e_it, because fewer people report crews).

   This can produce tipping: a site that starts mining stays mining.
4. **River feedback.** Upstream mining lowers fish stocks downstream:

   ```
   H_it    = Σ_{j upstream of i} k(d_ji) · X_jt
   Fish_it = Fish_i0 − φ·H_it − δ·Drought_it
   ```

   Lower fish stocks reduce y^T and raise R, so mining spreads downstream. Contamination
   flows only downstream, while q and crew information spread roughly the same way in
   both directions. That difference identifies the livelihood channel.
5. **Dynamics (relapse).**
   - Fish stocks recover slowly.
   - Skills and gear for the traditional economy wear out while a household mines.
   - After a raid sets s_i to 0, R_i falls only partly where y^T is degraded, and the
     remaining labour pool lowers the cost of crews returning.

   This block stays verbal unless the relapse test is strong.

The paper uses blocks 1–4 as a conceptual framework with closed-form comparative
statics. The only structural piece is the crew logit, estimated on the SIVEP flow table
as a conditional logit over infection municipalities for workers from each residence
municipality. It gives distance costs, the enforcement response and the displacement
ratio for lead 3 directly.

**Predictions and tests.**

| # | Prediction | Model result | Test |
|---|---|---|---|
| 1 | Crews are displaced, not removed | Displacement ratio (1 − s_i − s_0)/(1 − s_i) | Crew response to enforcement at i and at substitute sites (lead 3) |
| 2 | Mining spreads downstream more than upstream | ∂X_i/∂X_upstream > ∂X_i/∂X_downstream at equal distance | Growth in garimpo share, or SIVEP case onsets, at i on upstream vs. downstream mining by river-distance bin |
| 3 | Residents respond more where fishing matters and after drought | ∂R/∂H larger where ∂y^T/∂Fish is larger | H × fisher share (registered fishers, seguro-defeso); H × drought (ERA5-Land, e.g. 2023–24) |
| 4 | Relapse is faster where livelihoods degraded | Relapse hazard rises with degradation | Re-entry hazard after enforcement × prior loss of y^T |
| 5 | Crews and residents respond to different shocks | Crews to p − λ and e; residents to H and drought | SIVEP incoming vs. local garimpo cases; cross-responses give the sign of the complementarity |

**Measurement with SIVEP** ([E5](#e5-sivep-as-a-measure-of-garimpo-workers)).

- A garimpo case is a positive exam with activity "Garimpagem" in the prior 15 days. It is
  local if the patient lives in the infection municipality, and incoming if they live in
  another Brazilian municipality.
- The local share is mostly a fixed property of each mining hub. The tests therefore use
  local and incoming *counts* within a municipality over time, controlling for total
  positives or exams, not the local share.
- Local and incoming cases are different populations: local cases have less schooling
  and more Indigenous patients. Local cases cannot be verified, though. Without a patient
  identifier, a local case's infection site could be a default copy of the residence.

**Data by model block.** Status in brackets; see the [data inventory](#data-inventory).

- **Block 1 (crews):**
  - crew presence: incoming garimpo cases [in the pipeline, `malaria`];
  - p_t: gold price in reais [in the pipeline, `commodity_prices`];
  - λ_t: policy dates [to verify], plus gold-buying points (DTVM/PCO) for variation
    across sites [to add];
  - q_i: SGB gold occurrences [in the pipeline, `geology`];
  - e_it: IBAMA seizures and equipment destruction [to add];
  - c_i: river distances [in the pipeline], roads and airstrips [to add];
  - the destination logit: the SIVEP flow table [in the pipeline].
- **Block 2 (residents):**
  - R_it: local garimpo cases [in the pipeline] over population [in the pipeline];
  - fish: upstream mining H_it [C11, in the pipeline]; fishers and seguro-defeso [to add];
  - forest: MapBiomas C10 [in the pipeline];
  - transfers: Bolsa Família [to add];
  - heterogeneity in κ: FUNAI lands [to add] and the Census Indigenous and riverine
    population shares [to add].
- **Block 3 (wage):** no series exists for informal garimpo work. Census 2010/2022
  garimpeiro occupation counts and formal mining employment in RAIS are proxies;
  otherwise test only the sign of the complementarity, through cross-responses.
- **Block 4 (river feedback):**
  - directed matrices [in the pipeline];
  - drought: ERA5-Land precipitation and soil water [in the pipeline]; ANA river stage
    [optional];
  - geology control [`geology`].

  The spread test also needs a downstream aggregation of mining, next to the upstream
  one. That is a computation, not new data.
- **Block 5 (relapse):** IBAMA enforcement timing [to add]; optionally INPE DETER mining
  alerts for re-entry within a year.

**Threats.**

- **Geology.** Alluvial gold concentrates in lower reaches, which confounds the spread
  test; control for geology.
- **Malaria transmission and testing.** Cases measure mining only where malaria is
  transmitted and tested (Peixoto de Azevedo, a major hub, has 60 cases). Control
  programmes also move cases. Use state×year fixed effects and total positives or exams
  as controls. Garimpo cases move only weakly with testing ([E5](#e5-sivep-as-a-measure-of-garimpo-workers)).
- **Unverifiable local cases.** Repeat patients cannot be linked, so the residence field
  cannot be validated directly. Incoming cases are notified at their residence 85% of the
  time, so for them the infection site is not a default.
- **Dredges.** Dredge mining on rivers leaves little land footprint and is missed by
  MapBiomas. SIVEP partly captures it.

**Contribution.** The most novel part.

- The downstream-spread test is only possible with a directed river network linked to
  outcomes.
- The model says which policy works: if residents drive mining, enforcement alone
  relapses.
- It relaxes the one assumption of Pereira & Pucci's contest model (App. B) that matters
  here, that miners do not migrate between municipalities. Their results already point
  to outside crews:
  - effects are smaller by victim residence than by place of death (their Table C.6);
  - Indigenous victims are a small share;
  - effects are larger for deposits in Conservation Areas than in Indigenous lands.
- Colombian evidence points the same way: fewer than 20% of mine owners live in the
  municipality of their mine (Saavedra & Romero 2021, fn. 8), and gold miners migrate and
  recruit through social networks (Fletcher & Saavedra 2019).

## Lead 3: enforcement displacement

**Question.** When enforcement shuts down mining in one territory, how much moves
elsewhere, where does it go, and does the health burden move with it?

**Shocks.**

- the Yanomami health emergency and enforcement operation (2023);
- large dredge operations on the Madeira River (2021, 2024, Operation Boiuna 2025);
- operations in Munduruku and Kayapó lands;
- in the other direction:
  - the 2016–18 collapse in federal field enforcement (IBAMA field agents 965 → 691;
    Nunes et al. 2024; used for homicides by Hemsley, Libera & Rocha 2026);
  - the 2019 drop in enforcement credibility, including political statements against
    destroying mining equipment;
- across the border: French Guiana's garimpo cases fall from about 5,700 (2004) to about
  100–180 a year from 2020, after its military operations against gold mining.
  Venezuelan infections peak in 2016–18.

**Design.**

- **Substitute sites.** Defined in advance as gold-bearing areas with river access that
  were not targeted, ranked by travel cost from the targeted territory along rivers and
  roads.
- **Outcomes.** Garimpo area and SIVEP incoming cases at substitute sites, by distance
  band, around each shock. The flow table shows whether workers from the same residence
  municipalities reappear at the substitute sites.
- **Displacement ratio.** The increase at substitute sites divided by the decrease in
  the targeted area; a ratio near 1 means pure relocation. The crew logit (lead 2) gives
  the model-implied ratio for comparison.
- **Where the health burden goes.** Malaria at the substitute sites, and in crews' home
  municipalities (the importation channel).

**Data.**

- **To add:** IBAMA fines, embargoes and seizures (geocoded, public), with operation
  dates from official reporting.
- **In the pipeline:** C11, SIVEP (counts and flows), the river network for travel
  distances.

**Threats.**

- **Few operations.** Few large operations means few independent units; use
  randomization inference across placebo operation dates.
- **Coinciding national changes.** Several shocks coincide with national changes; use
  state×year fixed effects and placebo substitute sites without gold.
- **Dredges.** Dredge mining is invisible to land-cover maps; IBAMA seizure records and
  SIVEP partly fill the gap.

**Contribution.** It quantifies what is so far anecdotal in Brazil (Yanomami → Kayapó,
dredges returning on the Madeira within months). Displacement itself is documented
elsewhere:

- in Peru, Operation Mercury (Dethier et al. 2023) was followed by new mining nearby;
- after the 2023 Yanomami crackdown, malaria spilled over into Guyana (De Salazar et al.
  2024).

The one experimental benchmark is Saavedra (2022 WP).
- *Design:* Colombian municipalities were randomized to have five satellite-detected
  mine locations disclosed to the mayor, the Air Force, or both.
- *Results:* illegal mining fell in the disclosed 1 km cells and the 8 cells around them,
  but rose elsewhere in the same municipality: 11% gross, 7% net, so about a third was
  displaced. This came from information alone, not raids.
- *Limits relative to lead 3:*
  - The far-cell increase is in the illegal share of mined area, and is not significant
    in km².
  - Moves into other municipalities are not measured.
  - The outcome sample is about 110 municipalities.
- *Local collusion:* mayors are 15–16 pp less likely to confirm a true mine when it is
  illegal; the Air Force shows no gap. That supports telling local and federal
  enforcement apart.

Pereira, Pucci & Soares (2024, "Landing on Water") show the same kind of response for
drug trafficking: air interdiction moved routes onto rivers.

What is new here is a displacement ratio for armed enforcement within Brazil and across
its borders, with the workers' origins observed. It directly tests prediction 1 of lead
2.

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
- **Anticipation.** An event study of permit requests and garimpo growth around the
  introduction of PL 191/2020.

**Data.**

- **To add:**
  - Indigenous-land boundaries with stage and decree dates (FUNAI);
  - protected-area boundaries with creation dates (national register of conservation
    units, CNUC);
  - permit requests over Indigenous lands (ANM microdata and SIGMINE).
- **In the pipeline:** C11, SIVEP.

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
  data; it is recorded in a separate system run by the Indigenous health secretariat
  (SESAI).

**Contribution.** It explains where displaced crews go, connecting the legal map to the
substitute sites of lead 3.

## Work plan

The evidence behind E2–E6 and E9 is in the repo (`src/analysis`, `output/notebooks/`).
The ADM2 panel carries:

- the C11 shares, upstream and local;
- the SGB geology, local and upstream;
- the malaria cases;
- the SINASC anomaly and APGAR counts and the infant deaths by ICD-10 chapter;
- the gold price in reais.

1. **Mining permits.** A `mining` source from the ANM Cadastro Mineiro microdata and the
   Monitor da Mineração layer: PLG grant and expiry dates, polygons, flags, by
   municipality-year. Open questions:
   - whether `Evento.txt` holds the event table;
   - the start year of the events;
   - the PLG codes from 521 onward.
2. **Lead 1 supply estimates.** The gold price in reais (`commodity_prices`) and the
   upstream-reach geology (`geology --product upstream`) are in the panel. Still needed:
   - IGD (needs FUNAI and CNUC boundaries);
   - the supply equation on C11 and SIVEP, by group. E9's null first stage, upstream
     and in the municipality, is a warning for the price × deposits term.
3. **Lead 2 first pass.**
   - Prediction 5: local vs. incoming cases on price × deposits and on drought.
   - Prediction 2: a downstream aggregation of mining for the spread test.
4. **Malaria importation.** Novelty check first, then the test with the flow table.
5. **Lead 3.** IBAMA enforcement data, substitute sites, displacement ratio, the crew
   logit.
6. **Lead 4.** FUNAI stages, CNUC dates, anticipation around PL 191/2020.
7. **Optional water outcome.** Station turbidity from 2008, or satellite river
   turbidity, as a pollution outcome of upstream garimpo ([E8](#e8-amazon-station-coverage-for-turbidity)).

## Evidence

Each entry gives the question, the result and where its code lives. "HPC scratch" means
the code exists only outside the repo.

### E1. Pollution arm and mercury data

*Code: `src/analysis/pollution.py`, `output/notebooks/pollution/` (memo).*

- **Results.** Pasture raises lead (+5 pp) and cadmium (+4 pp) violations, and
  agriculture raises COD (+4% per SD). Upstream mining raises lead violations by about
  4 pp.
- **Mercury coverage.** The project's stations have essentially no mercury readings in
  the Amazon:
  - the ANA Amazon and Tocantins basins have 4 readings in total, at two Teles Pires
    stations in 2019–20;
  - the Madeira has none;
  - no state-network station with mercury was found in the Amazon.
- **Unit coding.**
  - 58 station-periods mix values of 0.0002 or below with values of 0.1 or above, which
    looks like µg/L entries in a mg/L column.
  - The pipeline sets such values to missing (`drop_unit_errors`): 209 values, 177 of
    them mercury, 24 lead, 7 arsenic and 1 cadmium.
  - The mining → mercury coefficient then falls from +13.7 pp to +5.7 pp (SE 4.35,
    bootstrap p = 0.34, 11 basins). Lead and cadmium are unchanged.
- **Where mercury is identified:** 15 stations in MG, SP, RJ and SC, sampled mostly
  1985–99.

### E2. The Amazon mining-presence result

*Code: `src/analysis/health.py`, `output/notebooks/health/02_analysis.ipynb`; the gold
and 2020 diagnostics in `src/analysis/garimpo_gold.py` and
`output/notebooks/health/03_garimpo_gold.ipynb`.*

- **Result.** Where mining (C10, class 30) appears within 50 km upstream, Amazon infant
  deaths rise by 0.21 log points, about 23% (bootstrap p = 0.0015). It rests on 36 onsets in 18
  basins, and the injury placebo is flat.
- **Where the onsets are.**
  - 13 of the 36 are in Maranhão and 5 on the northeast Pará coast.
  - 11 fall in 2020, the year of a national C10 mapping step: 41 first-time upstream
    onsets in Brazil in 2020, against about 7 a year before, and 30 of them outside the
    Amazon.
  - Mapped mining area rose 29% (Amazon 37%) that year, against 4–8% a year before.
- **Robust:** dropping the 2020 onsets gives +0.26 (boot p 0.006). Dropping onsets from
  2020 on gives +0.26 (boot p 0.009), on 2000–2019 only +0.27 (0.013), and dropping
  2020–21 +0.21 (0.005). Dropping the Maranhão and coastal-Pará switchers gives +0.25
  to +0.26 (boot p ≤ 0.014).
- **No dose-response.**
  - The median mined share at onset is 0.0009 pp of the 50 km reach, a pixel or two.
  - Keeping only switchers with at least 0.01 pp at onset+1 gives +0.06 (p = 0.18).
- **Not specific to 2013:** onsets in 2012 or earlier give +0.22 (boot p = 0.076, 11
  switchers); onsets in 2013 or later give +0.19 (p = 0.001, 25 switchers).
- **Not gold** (C11, see E4): the result is unchanged after dropping every municipality
  with gold upstream. C11 assigns the onsets mainly to non-gold mining.

### E3. The C11 mining-substances layer

*Code: `scripts/prepare_mapbiomas_mining.py`, `src/data/sources/land_cover/mining.py`
(`data preprocess --source land_cover --product mining`), `panels.adm2_upstream_mining`;
checks in `output/notebooks/garimpo/01_measurement.ipynb`.*

- **Build checks.**
  - Per-trench totals are constant across years, and groups and phases add up.
  - The extract counts 1.3–1.8% more mined pixels than the national area table, steadily
    across years. This is probably pixels on trench boundaries counted twice.
- **National garimpo gold:** 202 km² (1985), 1,251 (2000), 1,440 (2013), 1,804 (2019),
  2,076 (2020), 2,920 (2025).
  - The layer is not a cumulative scar map: 859 of 3,640 class-years fall. It changes
    slowly, though.
  - Growth: 20–48% a year in 1985–90 (the series ramping up), under 1% in 2000–11, 2.7%
    in 2012–15, 4.3% in 2016–19.
- **The 2020 step.** Garimpo gold rises 15.1% and all mining 13.1% in 2020, against
  4.0–9.7% and 3.3–8.4% in the other years of 2016–2023. Industrial gold rises 12.8%,
  against 1.7–2.9% in 2016–19. The step appears in every class, so part of it is a change
  in classification between years.
- **Exposure within 50 km upstream.**
  - Garimpo gold is upstream of 112–134 municipalities a year (137 ever: MG 51, PA 27,
    MT 22, MA 10, AP 7), and of 678 without the 50 km cap.
  - It switches on 39 times in 27 municipalities (22 in MG, 5 in the Amazon), with a
    median share at onset of 1.3 × 10⁻⁶. The 2020 step shows up as growth where gold
    already was.
  - An onset design on garimpo gold therefore has nothing to identify from.
- **The C10 onsets in C11.** Of the 216 municipalities whose upstream C10 mining switches
  on, only 9 ever have garimpo gold within 50 km (MA 6, MG 2, RO 1).
  - The largest C11 group one year after onset, among the 36 Amazon switchers:
    garimpo of unknown substance 14, other garimpo 7, non-gold industrial 7, garimpo gold
    6 (all MA), none 2.
- **C10 and C11 disagree in places.** Nova Guarita's C11 gold share alone (3.03 pp in
  2000) exceeds its C10 total mining share (1.97 pp).

### E4. Gate: upstream garimpo-gold share and health

*Code: `src/analysis/garimpo_gold.py`, `output/notebooks/health/03_garimpo_gold.ipynb`;
tables in `output/analysis/health/garimpo_gold/`.* Municipality and state×year FE,
clustered by Otto-3 basin (38–57 clusters), wild-cluster bootstrap. Controls: other upstream mining, the local C10 mining
terms, the health arm's controls. Sample: 2000–2019.

- **Identifying variation.** Among municipalities with gold upstream at some point:
  - 32 Amazon municipalities gain at least 0.01 pp of gold share at 50 km, and 39 at
    200 km;
  - the variation is concentrated in three basins (Teles Pires, Tapajós, Gurupi);
  - the within-municipality SD is 0.04–0.09 pp.
- **Coefficients** (per within-SD of gold share, bootstrap p):

  | Outcome | 50 km | 200 km |
  |---|---|---|
  | Infant deaths | +2.1% (0.11) | +1.5% (0.010) |
  | Congenital hospitalizations | +9.3% (0.06) | +7.0% (0.065) |
  | Toxic poisoning | +6.5% (0.08) | +8.7% (0.031) |
  | Injury and poisoning (placebo) | +3.5% (0.10) | +5.3% (0.005) |
  | Injuries (placebo) | +3.7% (0.15) | +5.4% (0.004) |

  The Amazon-only term gives 1.5–1.7 times these per-SD effects (with a larger SD) and
  the same pattern. LBW, preterm and perinatal are null.
- **The placebo fails.** Injury hospitalizations rise as much per SD as infant deaths, and
  at 200 km they have the strongest bootstrap p-values. Something trending in gold
  municipalities raises hospitalizations in general.
- **One municipality at 50 km.** Without Centro do Guilherme (MA), infant deaths fall
  from 0.35 to 0.16 per pp (boot p 0.62). At 200 km the coefficient rises, but so do the
  placebos (2.4 per pp, boot p 0.004).
- **Not robust to 2020.** On 2000–2024, infant deaths fall to a third (0.11 per pp,
  p = 0.24) and the placebos become significant.
- **Outside the Amazon** the gold term has almost no variation (within SD 0.005–0.014
  pp) and is unidentified.
- **The earlier presence result is not gold.** It is unchanged after dropping every
  municipality with gold upstream:
  - dropping gold within 50 km: +0.21 (boot p 0.0003), 29 switchers;
  - dropping gold within 200 km: +0.22 (boot p 0.0005), 27 switchers;
  - the injury placebo is flat in both.
- **Local gold does not explain it.** 65% of gold pixels in the 50 km sample (40% at
  200 km) lie in the municipality's own trenches, which the upstream arm does not use.
  Adding the local C11 gold share is a check, not a design, since local gold is
  confounded with the local economy.
  - Local gold moves within 58 municipalities of the 50 km sample (within SD 0.12 pp).
  - The upstream coefficients barely change: infant deaths 0.345 → 0.327 per pp at
    50 km and 0.383 → 0.365 at 200 km; injury placebos 0.56 → 0.65 and 1.32 → 1.36.
  - Local gold has no significant coefficient for any outcome (boot p ≥ 0.11), and the
    failed placebo stays.

### E5. SIVEP as a measure of garimpo workers

*Code: `src/data/sources/malaria` (the sample rules); checks in
`output/notebooks/garimpo/01_measurement.ipynb`, tables in
`output/analysis/garimpo/measurement/`.*

- **Sample.** 5.38 million positive exams, excluding cure-verification slides. 312,276
  are garimpo cases.
  - The garimpo share of positives is 3–8% until 2019, then 10.9% (2020), 14.5%, 17.7%
    (2022), 13.6% and 11.2%.
  - The rise is driven by Roraima: 12,010 cases in 2022, against fewer than 100 a year
    until 2017.
- **Completeness.** For Brazil-infected garimpo cases, the infection and residence
  municipality are filled and valid in over 99.9% of records.
- **The split.** 60.3% of Brazil-infected garimpo cases are local and 39.7% incoming;
  13.5% come from another state.
  - The main flows run between neighbours or along known garimpo routes: Boa Vista →
    Alto Alegre (19,676) and Mucajaí (6,238), Novo Progresso → Itaituba, Ariquemes →
    Itapuã do Oeste, Laranjal do Jari → Almeirim.
  - The local share is a property of each hub: Alto Alegre 2%, Maués 13%, Itaituba 78%,
    Jacareacanga 86%, Calçoene 92%.
  - The national dip to 30% local in 2022 is composition (Roraima).
- **Who the local cases are.**
  - They look like workers: 84% male and 4% under 15, against 58% and 34% for other
    malaria in the same municipalities.
  - They have less schooling than incoming cases (12% vs. 27% with secondary school or
    more) and more Indigenous patients (5.7% vs. 0.9%), so the two are different
    populations.
  - Incoming cases are notified at their residence 85% of the time. For local cases,
    residence, infection and notification coincide, so a default-filled infection site
    cannot be ruled out.
- **Infection abroad.** 20.5% of garimpo cases were infected abroad: French Guiana
  33,600, Venezuela 17,200, Guyana 11,400, Suriname 1,500.
  - 5,391 of them are foreign residents, mostly Venezuelans infected in Venezuela (4,945,
    peaking in 2016–18). They have no Brazilian residence and are not in the flow table.
- **Testing.** Year-on-year changes in garimpo cases correlate 0.17–0.28 with changes
  in exams (pooled, top 25 infection municipalities and the 61 with at least 100 cases),
  against 0.47–0.53 between other positives and negatives. So garimpo counts are not
  mainly testing-driven.
  - Negative exams carry almost no infection or residence site from 2007, so testing is
    measured by notification municipality.
- **Fit to mapped mining.** Spearman correlation with 2010–22 mean garimpo cases across
  the 808 municipalities of the Legal Amazon states:

  | Measure | Local | Upstream 50 km | Upstream 200 km |
  |---|---|---|---|
  | C10 mining | 0.37 | 0.26 | |
  | C11 garimpo gold | 0.36 | 0.26 | 0.26 |
  | C11 garimpo, all substances | 0.36 | 0.29 | 0.30 |

  - C11 gold does no better than C10 mining, and the local measures fit best.
  - C11 gold is the most restrictive: 80% of case municipalities have no local garimpo
    gold, and they hold a third of the cases. Under C10, 54% have no mapped mining and
    they hold 3.8% of the cases.
  - Large case municipalities with little or no mapped gold: Alto Alegre, Calçoene,
    Mucajaí, Aripuanã. Their garimpos lie in Indigenous territories or across the
    border, are mapped under another substance, or are too small for 30 m pixels.
  - SIVEP and C11 are therefore complementary measures, not substitutes.
- **Coding breaks.**
  - `id_lvc` is the consistent cure-verification flag; `tipo_lam`'s LVC code fades out
    after 2010.
  - Race is missing before 2011, schooling is not comparable before 2011, and recurrence
    flags exist from 2011.

### E6. SGB gold geology

*Code: `src/data/sources/geology`; checks in
`output/notebooks/garimpo/01_measurement.ipynb`.*

- **Fit to C11 garimpo gold in 2020.**
  - 96% of garimpo-gold area lies inside a gold province, 44% inside a gold district,
    and 92% within 10 km of a gold occurrence.
  - Much of that is mechanical: occurrences were recorded at garimpos, and provinces
    were drawn around known mining.
  - With only the 1,073 never-exploited occurrences, 16% of garimpo-gold area is within
    10 km and 48% within 25 km.
- **Relevance across the 808 Legal Amazon municipalities.** Spearman correlation with
  2004–24 garimpo cases by infection municipality:
  - all occurrences 0.41, never exploited 0.38, garimpo-status 0.34;
  - district share 0.33, province share 0.29.
  - The never-exploited count predicts about as well as all occurrences, but only 74
    Amazon municipalities have one. Case municipalities without one hold 14.7% of cases;
    Maués (9,054 cases) and Porto Velho (3,992) have none.
- **Weak spots.**
  - The district polygons miss much of the Tapajós and Roraima: Itaituba's district
    share is 0.23%, and Amajari's is 0 despite 71 never-exploited occurrences.
  - The Rondônia garimpo cluster has no gold geology (probably tin).
  - Four of the top ten municipalities by occurrences are in Minas Gerais, so the
    measures are used within the Legal Amazon only.

### E7. CFEM royalties by permit

*Code: HPC scratch.* Only 289 of 15,461 gold garimpo processes in the permit layer carry
accumulated CFEM. CFEM therefore cannot measure declared sales by permit. The ghost-permit
test has to use the satellite side (mined area inside the permit, `transborda`) and sales
from other sources.

### E8. Amazon station coverage for turbidity

*Code: HPC scratch.*

- Turbidity at basin-1 stations is thin before 2008 (≤ 11 stations a year), then
  30–80 Madeira stations a year from 2009.
- Electrical conductivity is the most continuous series (1997–2016).
- Only 10 Amazon stations in the pollution mining sample switch mining presence with
  turbidity or suspended solids on both sides, and 7 of those switch in 2020–21.
- A station-based water outcome is therefore feasible only after 2008. Satellite river
  turbidity is the alternative.

### E9. Moreno-Louzada's design

*Code: `src/analysis/moreno_louzada.py`, `output/notebooks/health/04_moreno_louzada.ipynb`;
tables in `output/analysis/health/moreno_louzada/`.*

- **Design.**
  - *Exposure:* any SGB gold occurrence in an Otto-4 sub-basin upstream of one the
    municipality intersects (`gold_upstream_otto4`).
  - *Price:* the log annual gold price in BRL (Pink Sheet × BCB).
  - *Sample:* the Legal Amazon (eight states and Maranhão west of 44°W), without the
    municipalities holding an occurrence, 2007–2021.
  - *Model:* municipality and year FE, weighted by mean births.
- **Sample.** 772 Legal Amazon municipalities. 154 hold an occurrence, leaving 618 (his
  616). 161 are exposed, in 117 Otto-3 basins.
- **The price.** The log price in BRL rises from 7.21 (2007) to 9.18 (2021), with one
  step of +0.51 in 2020. Over the window the interaction is close to exposure × trend.
- **Baseline** (per 1,000 births per log point; p-values Otto-3 / wild bootstrap /
  two-way basin×year / P-E / random walk):

  | Outcome | Estimate | SE (Otto-3) | p-values | Mean |
  |---|---|---|---|---|
  | Anomalies at birth | +0.84 | 0.98 | 0.39 / 0.43 / 0.40 / 0.001 / 0.79 | 5.3 |
  | Low birth weight | −1.24 | 1.36 | 0.37 / 0.46 / 0.38 / 0.03 / 0.68 | 75 |
  | Preterm (2012–21) | +5.5 | 4.2 | 0.19 / 0.30 / 0.18 / 0.04 / 0.72 | 122 |
  | APGAR 5 min < 6 | −0.26 | 0.88 | 0.77 / 0.80 / 0.80 / 0.72 / 0.84 | 7.6 |
  | Infant deaths, non-external | +0.12 | 0.39 | 0.76 / 0.77 / 0.71 / 0.47 / 0.72 | 15.4 |
  | Infant deaths, congenital | +0.31 | 0.10 | 0.003 / 0.016 / 0.001 / 0.000 / 0.78 | 2.8 |
  | Congenital hospitalizations | −2.58 | 0.98 | 0.009 / 0.053 / 0.023 / 0.002 / 0.84 | 16.7 |
  | Injury hospitalizations (placebo, per 1,000 pop.) | −0.75 | 0.31 | 0.018 / 0.070 / 0.034 / 0.000 / 0.82 | 5.1 |

  N = 9,264 municipality-years (6,178 for preterm). His anomaly estimate is +2.28
  (SE 1.16).
- **Robustness of the anomaly coefficient.** It stays between 0.41 and 1.03 and is
  insignificant under the clustered and bootstrap inferences in every specification:
  - state×year FE: 0.44;
  - dropping 2015–17: 0.96;
  - unweighted: 0.41;
  - without Manaus and Belém: 0.63;
  - 2004–2023: 0.99;
  - USD price: 1.03.

  Never-exploited occurrences give 2.36 (p 0.09) on 103 exposed municipalities. The
  downstream-gold placebo gives 1.79 (p 0.11), as large as upstream gold.
- **Congenital infant deaths** survive dropping 2015–17, the capitals, the downstream
  control and the USD price (0.31–0.48). They halve with state×year FE (0.15, p 0.15),
  fall to 0.07 unweighted and to 0.18 on 2004–2023 (p 0.06), and are insignificant for
  never-exploited deposits.
- **The placebo fails.** Injury hospitalizations fall significantly in most
  specifications (bootstrap p 0.03–0.07 in seven of twelve). Congenital hospitalizations fall too,
  opposite in sign to congenital deaths.
- **No first stage.**
  - Upstream C11 garimpo gold: +0.001 pp per log point (p 0.20–0.21; 2007–19, mean
    0.005–0.008 pp).
  - Upstream garimpo malaria cases: −0.01 log points (p 0.97).
  - Garimpo malaria cases in the municipality: +0.05 (p 0.73).
- **Inference.**
  - Two-way clustering gives SEs within 25% of the Otto-3 SEs.
  - The price-exposure SE (arXiv 2512.10076, Theorem 1, with the price demeaned over the
    sample years) is 26–86% of the Otto-3 SE and close to clustering by year. It treats
    the 15 yearly prices as independent, which a persistent price with serially
    correlated residuals violates, and it rejects where nothing else does.
  - Randomization over 1,999 driftless random-walk price paths with the observed
    volatility gives p = 0.68–0.84 for every outcome.
- **Reading.** The design does not reproduce his result in this data. What remains
  (congenital deaths) depends on regional trends, comes with a failed placebo and has
  no first stage, so it reads as differential trends between exposed and other
  municipalities rather than a price-driven health effect.

## Data inventory

| Data | Source | Use | Status |
|---|---|---|---|
| River network, reachability and distance matrices | project (BHO/ANA) | directed links, spread test, travel distances | in pipeline (`river_network`) |
| Water quality, CONAMA violations | ANA monitoring stations | pollution outcomes | in pipeline (`sensor_data`) |
| Hospitalizations, births, deaths by municipality | DATASUS (SIH, SINASC, SIM) | health outcomes | in pipeline (`health`), including SINASC anomalies and APGAR and SIM infant deaths by ICD-10 chapter |
| Land cover incl. a single mining class | MapBiomas Collection 10 | land-cover exposures | in pipeline (`land_cover`); mapping step in 2020 |
| Mining substances (garimpo vs. industrial, by ore) | MapBiomas Collection 11 | garimpo area | in pipeline (`land_cover` with the mining legend); step in 2020 in every group |
| Malaria cases with occupation, infection site, residence | SIVEP-Malária (Saldanha, Zenodo 20706957, CC BY) | garimpo workers by group; flows | in pipeline (`malaria`); 2025–26 preliminary in Zenodo 20706949 |
| Gold occurrences, gold provinces and districts | Geological Survey of Brazil (SGB) | deposit quality, IGD, geology control | in pipeline (`geology`); stream-sediment assays not fetched |
| Mining permits with type, ore, dates, flags | ANM Cadastro Mineiro microdata; MapBiomas Monitor da Mineração | laundering exposure, ghost permits, anticipation | raw on the HPC; needs a `mining` source |
| Royalty payments (CFEM) | ANM | declared output | not usable for garimpo (E7) |
| Fines, embargoes, seizures | IBAMA | enforcement shocks | to add |
| Indigenous lands with stage and dates | FUNAI | IGD, lead 4 | to add |
| Protected areas with creation dates | CNUC | IGD, lead 4 | to add |
| Gold price | World Bank Pink Sheet × BCB BRL/USD rate | price shock | in pipeline (`commodity_prices`) |
| Gold-buying points (DTVM/PCO) | ANM, Central Bank | laundering exposure by site | to add |
| Registered fishers, closed-season payments | fisheries registry, federal transparency portal | fishing dependence | to add |
| Bolsa Família payments | federal transparency portal | residents' transfer income | to add |
| Roads and airstrips | OSM/DNIT; published mappings of clandestine airstrips | access cost | to add |
| Garimpeiro occupation; formal mining employment | Census 2010/2022; RAIS | wage proxy | to add |
| River stage | ANA HidroWeb gauges | drought | optional |
| Mining alerts | INPE DETER | re-entry within a year | optional |

### Sources and access

- **HPC access.** sciCORE compute nodes have no internet, but the transfer node
  `transfer12.scicore.unibas.ch` does. Anything with a direct URL is fetched there;
  manual downloads go from the laptop with `rsync` to transfer12.
- **MapBiomas mining substances (C11).**
  - *Asset:*
    `projects/mapbiomas-public/assets/brazil/lulc/collection11/mapbiomas_brazil_collection11_mining_substances_v1`.
  - *Bands:* `classification_1985` … `classification_2025` (41 bands), EPSG:4326 on the
    MapBiomas grid.
  - *Codes ABCC:* A = 1 garimpo, 2 industrial; B = 1 exposed soil, 2 revegetation; CC =
    substance (15 gold, 14 tin, 09 copper, 24 precious stones, 58 atypical, 59 no
    substance).
  - *Legend:* the CSV in `data/land_cover/auxiliary/`, from MapBiomas's legend-code page.
  - *Export:* platform downloads fail and the Earth Engine Python login is blocked for
    this account. The export therefore runs from the Code Editor
    (`scripts/gee_export_mapbiomas_mining.js`) as uint16 tiles to Google Drive.
    - The tiles in `data/land_cover/raw/mapbiomas11_mining_tiles/` are 61 non-empty
      tiles (Drive zip md5 `1796088b…`).
    - Their 2025 band reproduces the Earth Engine Console's Itaituba class counts to
      within 0.2%.
  - *Pipeline:*
    - `scripts/prepare_mapbiomas_mining.py` merges the tiles into one GeoTIFF per year
      (`data/land_cover/raw/mapbiomas11_mining/`, 0 = no mining). It also writes the
      pipeline legend and the national area by class and year.
    - The legend (`src/data/sources/land_cover/mining.py`) groups codes into garimpo
      gold, garimpo of unknown substance (58/59), other garimpo, industrial gold and
      other industrial, each split into active and revegetating.
    - The land-cover extract and ADM2 aggregation run unchanged on it, producing
      `land_cover_mining.parquet` and `land_cover_mining_adm2_upstream.parquet`.
    - `panels.adm2_upstream_mining` gives kernel-weighted upstream shares per
      municipality-year.
- **SIVEP-Malária 2004–2024** (Saldanha, Zenodo 20706957, CC BY), in `data/malaria/raw/`.
  - *Contents:* record-level notifications with activity in the prior 15 days
    (`cod_ocup`, "Garimpagem"), municipality of residence, probable municipality and
    locality of infection, and country of infection.
  - *Coding:* coded fields are stored as labels, and municipality codes are 6-digit IBGE
    (5399xx are placeholders). There is no patient identifier (`id_pacie` is age).
    Negative exams are included.
  - *Pipeline:* the `malaria` source.
    - `data fetch` downloads and md5-verifies the yearly files.
    - `data preprocess` writes `malaria_adm2.parquet` (sparse: zero-fill against the
      panel) and `malaria_garimpo_flows.parquet` (Brazilian residents only).
    - Formats are in `src/data/output_file_formats.md`.
- **SGB gold geology** (fetched 2026-10-27), in `data/geology/raw/sgb_gold.gpkg`.
  - *Source:* the SGB geoportal's ArcGIS REST services:
    - all 36,484 Geobank mineral occurrences (4,478 gold);
    - 20 gold provinces and 94 gold districts, apparently from the Informe de Recursos
      Minerais *Províncias e Distritos Auríferos do Brasil* (Série Ouro 18, 2022).
  - *Gold occurrences by economic status:* 1,103 garimpo, 332 mine, 1,073 never
    exploited, 1,970 undetermined; 686 alluvial by host material.
  - *Accuracy:* positions are mostly digitized from 1:250,000 and 1:100,000 sheets, so
    they are accurate to a few km. `DATA_CADASTRO` is the database entry date (2000–06),
    not the discovery date.
  - *Pipeline:* the `geology` source writes `geology_adm2.parquet`: counts by status,
    alluvial counts, and district and province area shares. With `--product upstream`
    it writes `geology_adm2_upstream.parquet`: counts upstream along the river network
    within 50 km, 200 km and without limit, and the Otto-4 upstream and downstream
    binaries.
  - *Not fetched:* SGB stream-sediment gold assays (`geoquimica_ouro`): 71,621 samples
    since 2012 and 25,428 / 13,096 earlier samples in the ppm / ppb layers, which may
    overlap. The table also has Hg.
- **Monitor da Mineração process layer** (snapshot 2026-10-26), in
  `data/mining/raw/mapbiomas_monitor_processos_2026-10-26/geoserver.zip`.
  - *Contents:* 270,877 ANM processes; 21,836 garimpo, 15,461 of them gold.
  - *Caveats:*
    - It is a current snapshot, so expired processes are probably missing.
    - `ano` is the process-number year.
    - The municipality, Indigenous-land and protected-area ID fields are broken by the
      export; recompute them spatially.
  - *Flags defined in MapBiomas's factsheet:*
    - transbordo (`transborda`, mining beyond the permit limits);
    - temporal inconsistency (mining before the authorization's validity);
    - CFEM "quitado" (any CFEM payment in the last six months);
    - restricted areas (`em_area_re`, including all strict-protection conservation
      units);
    - MEND, "Mineração Esperada Não Detectada" (mining expected but not detected).

    The mapping of `mend` and `permissao_` to columns is unconfirmed, and there is no
    attribute dictionary. The contact is contato@mapbiomas.org.
- **ANM Cadastro Mineiro microdata** (extract of 2026-10-25), in
  `data/mining/raw/anm_scm_microdados_2026-10-26/` (checksums verified).
  - *Source:* `dadosabertos.anm.gov.br/SCM/microdados/microdados-scm.zip` (181 MB,
    updated daily, 1934 onward), joined to SIGMINE on `DSProcesso`.
  - *Access:* the firewall returns 403 to scripted downloads, including from transfer12.
    Download it in a browser and rsync it.
  - *`ProcessoEvento.txt`* (about 1.1 GB, `;`-separated, CP-1252): process, event ID,
    event date, note, and the Diário Oficial extract. `Evento.txt` describes about 2,900
    event codes: PLG filed = 333; rejections = 334, 335, 337, 388; PLG grant published =
    513; PLG renewal = 523.
  - *`ProcessoTitulo.txt`:* title number, legal-document type (2 = PLG), situation
    (outorgado, renovado, caduco, cancelado, cassado, …), publication and expiry dates.
  - *`Processo.txt`:* active flag `BTAtivo`. About 71% of the first 88,000 rows are
    inactive, so extinct processes are included, unlike the Monitor and SIGMINE active
    layers.
  - *Documentation:* dictionary `metadados-microdados-scm.ods`; entity-relationship
    diagrams `mer-microdados-scm.pdf`.
  - *Polygons including inactive processes and garimpo reserves:* SIGMINE on dados.gov.br.

## Other leads considered

Ranked below leads 1–4, mainly because they are crowded or do not use the project's
river and malaria data:

5. **Bolsonaro and Lula election shocks.** Best used as shocks within leads 1–3 rather
   than on their own: the variation is over time only, and the topic is heavily studied
   for deforestation.
6. **Rural credit rule (Resolution 3,545) and substitution into mining.** An open question
   with a known design, but a likely precise null given that gold prices dominate. Useful
   as a test of the residents' side within lead 2.
7. **Random federal corruption audits.** A clean randomization, but an unclear link to
   garimpo, since mining permits are federal. A quick side test.
8. **The CAR registry, land titling and pasture as a possession claim.** Important and
   data-rich, but crowded, and the health arm found no pasture effect.
9. **Fragmenting clearings below satellite detection thresholds.** Already documented by
   a recent study of the priority-municipality blacklist.
10. **Supply-chain rules pushing activity into mining.** Weak priors (cattle returns are
    far below soy) and no data yet. The 2026 soy-moratorium collapse becomes testable
    once MapBiomas covers 2026–27.

## Caveats

- **Unverified sources.** Several sources behind the original brainstorm are from
  2025–26 and still need checking before the paper relies on them: the blacklisting
  fragmentation study, the 2026 reviews, and the Greenpeace and Grist permit figures.
- **Policy dates.** The 2013 law was approved on 19 July 2013 (Pereira & Pucci, §2.3).
  The 2023 dates in lead 1 come from secondary legal sources: injunction early April
  2023, plenary 2 May 2023, invoice 3 July (or 1 August) 2023. Check them against the
  STF case record and the Diário Oficial before citing.
- **Literature search.** It covered forward and backward citations (OpenAlex, Semantic
  Scholar, Google Scholar), thesis repositories, RePEc author lists and an author
  re-screen, without broad keyword queries. Citation channels missed several close papers
  that were found only on re-screening, so a paper with a new title or outside these
  networks could still be missing. Malaria importation by miners was not part of the
  search.
- **Dredges.** River-dredge mining is largely invisible to land-cover data, so any design
  based on MapBiomas undercounts it. A UnB geography thesis develops satellite detection
  of dredges (D. H. C. Pereira 2023, repositorio.unb.br/handle/10482/48214; unread).
