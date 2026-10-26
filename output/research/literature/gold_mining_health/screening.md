# Literature screening: gold mining, enforcement and downstream health (Brazil)

*First pass 2026-10-22, Google Scholar imports and re-screen 2026-10-23, with the `literature-search` skill.
Full-text reads (Parfitt, Romero & Saavedra) and RePEc/IDEAS author lists 2026-10-26. Corpus: 3,633 records in
`corpus.jsonl`; ranking in `ranked.md`.*

## Novelty claims checked

1. Upstream gold mining raises downstream infant mortality / worsens births in Brazil, via
   rivers (health arm).
2. Gold-laundering policy (2013 good-faith presumption, 2023 reversal) moves mining and,
   through rivers, downstream health (lead 1).
3. Garimpo supply as outside crews vs. local residents; downstream spread (lead 2).
4. Enforcement displaces rather than removes mining (lead 3).
5. Indigenous-land demarcation and mining (lead 4).

## Search log

- **Seeds (OpenAlex, all versions):** Dias, Rocha & Soares 2023 (ReStud + SSRN); Pereira &
  Pucci 2026 (JPubE + RePEc WP; 0 citers yet); Saavedra & Romero 2021 (EER); Chimeli &
  Soares 2017; Idrobo et al. 2014; Cortinhas Ferreira Neto et al. 2024; von der Goltz &
  Barnwal 2019; Swenson et al. 2011; Castro & Peterka 2023.
- **Not indexed:** Moreno-Louzada 2024 (thesis WP) and Romero & Saavedra, "The effects of
  gold mining on newborns' health" (found 2026-10-26 as a thesis chapter; see below).
- **Forward citations** of all seeds; **10 boolean OpenAlex sweeps** (English and
  Portuguese) from 2000; **5 BDTD thesis queries**.
- **Second pass (same day, with an OpenAlex key):** backward references of all seeds
  and 3 Spanish-language sweeps; corpus 2,128 → 2,776. No new competitor: the references
  add the economics canon (Berman et al. 2017; Aragón & Rud 2013), and the Spanish
  results are mostly descriptive or legal.
- **Romero & Saavedra (newborns) not located** in OpenAlex or by web search at this
  stage (found later in Saavedra's thesis). Two
  Colombian theses on the same question surfaced instead: García (2015, U. de Chile),
  "Injusticia ambiental en Colombia: minería y salud al nacer"; and a 2023 CUNY economics
  master's thesis on Colombian gold booms and fetal deaths (author not listed).
- **Semantic Scholar (with key):** citations and recommendations for all 9 seeds; corpus
  2,776 → 3,567. No new competitor, and it does not index Moreno-Louzada either. It also
  does not list P&P as citing DRS (only Chimeli & Soares, Idrobo et al. and Cortinhas
  Ferreira Neto et al.), consistent with P&P's reference list. The recommendations are
  mostly noise. Worth keeping:
  - "Mapping total mercury in rivers of Mato Grosso: baselines and contamination hotspots
    by biome" (2025, *Ambiente e Água*) — external validation for the pollution arm;
  - "Legitimacy-Conversion Crimes: A Crime Script Analysis of Illicit Gold Laundering"
    (2026, *Trends in Organized Crime*) — laundering mechanics for lead 1;
  - "Global mining has undermined forest conservation within and beyond protected areas"
    (2026, *Nature Communications*) — background for lead 4;
  - "Gold-Mining Pollution Exposure, Health Effects and Private Healthcare Expenditure in
    Tanzania" (2021) — a health-cost analog.
- **Google Scholar (2026-10-23, Zotero-connector exports in `pop/`):** "cited by" for DRS
  (109, the ReStud cluster), P&P (11) and Parfitt (4); Moreno-Louzada has no citers yet.
  Publish or Perish found only a minor DRS version (2 citers), so the lists were saved
  from the browser instead. Corpus 3,567 → 3,633; 58 of the 104 DRS citers are found by
  Google Scholar alone, almost all pesticide, water or agriculture papers.
- **Re-screen of the whole corpus** with claim-specific regex filters (laundering/law
  terms; enforcement or Indigenous land × mining; mining × migration; river network ×
  health). It found records the API channels had already collected but the first
  screening missed: Oliveira (2025) and Hemsley et al. (2026), both via OpenAlex, plus
  De Salazar et al. (2024) and Dethier et al. (2023).
- **DRS working-paper cluster (40 citers)** added to the same export (149 rows): no
  mining paper, only pesticide, agriculture and illicit-crop work.
- **Full texts (2026-10-26):** Parfitt's FGV thesis (essay 2) and Saavedra's Stanford
  thesis (chapters 1–2), both downloaded by Felix. Summaries below.
- **RePEc/IDEAS (2026-10-26).** RePEc has no open API (the official one needs an access
  code), and the IDEAS and EconPapers search forms cannot be scripted, so Felix pasted
  IDEAS name-search results for Leila Pereira, Rafael Parfitt and Santiago Saavedra. The
  Pereira and Saavedra pastes were cut off at about 19 and 37 items. Name searches mix in
  namesakes; registered author pages would be cleaner. No new competitor. Results:
  - Not in the corpus: Ladino, Saavedra & Wiesner (2021); Fergusson, Saavedra & Vargas
    (2020); Jimenez, Molina & Saavedra (2026).
  - In the corpus but missed by screening: Saavedra (2022), the disclosure RCT; and the
    2004-confounder reading of Pereira, Pucci & Soares, "Landing on Water" (listed before
    only as a lead-3 analog).
  - No Madeira mining paper on Parfitt's list (4 items), and no newborns paper on
    Saavedra's.
  - A second Saavedra export (RIS, about 40 items, cut off within the journal articles)
    added the full RCT abstract (net 7% after displacement), Fletcher & Saavedra (2019),
    Romero & Saavedra (2019, communal titling), the formalization RCT and Coy et al.
    (2021). The rest is noise (air purifiers, PES, eco-tourism, lockdown, poverty,
    labour supply, slaughterhouses).
  - Published EER version of Saavedra & Romero read: no newborn section.
- **Author re-screen of the corpus (2026-10-26)** on Saavedra, Romero, Pereira, Pucci,
  Soares, Rocha, Parfitt and Moreno-Louzada (author field; Pereira also with a topic
  filter). Not previously screened:
  - Moreno-Louzada, Figueira & Picchetti (2025), price-exposure inference (see method
    cautions);
  - Borges, Menezes-Filho & Moreno-Louzada (2024), Starlink and crime;
  - D. H. C. Pereira (2023), satellite detection of dredge gold mining;
  - Soares, Pereira & Pucci (2021), "Ilegalidade e Violência na Amazônia";
  - Soares & Souza (2025, *JDE*), "Too much of a good thing": the oil boom and crime in
    Brazil. Background only; it does not cover the Madeira dams.
  
  Pereira & Pucci (2021), "Working and Blood", is an early version of P&P. No new
  competitor.
- **Skipped (Felix, 2026-10-26):** broad Google Scholar keyword queries.
- **Not done:** IDEAS author pages for Pucci, Moreno-Louzada, Romero and Rocha; CitEc
  citers of P&P and "Landing on Water".

**Finding on channels.** The 61 OpenAlex citers of DRS contain neither P&P (which does not
cite DRS; its "Dias" is C.C.N. Dias 2024) nor Moreno-Louzada (not indexed), but do contain
Parfitt, the closest competitor for claim 1.

## Direct competitors

- **Parfitt (2024), "There's Something in the Water: Gold Rush, Illegal Mining, Mercury
  Pollution, and Infant Health in the Amazon Rainforest"**, SSRN 10.2139/ssrn.4931249;
  essay 2 of his FGV PhD thesis (open access: https://hdl.handle.net/10438/35027).
  **Read (thesis, 2026-10-26).**
  - *Design:* the 2004 surge in garimpo in Madre de Dios (Peru/Bolivia) as an upstream
    shock. DiD of the 11 municipalities on the Madeira against 110 others in Amazonas and
    Rondônia, annual 1997–2015, treatment from 2004. Exposure is "on the river" plus
    distance bands along it; no water data.
  - *Results:* infant mortality +7.5 per 1,000 (+44%). By cause (per 1,000): fetal
    +0.14, congenital +1.1, neonatal +2.4. Birthweight −63 g; preterm null. Driven by
    Rondônia within about 600 km of the border; without state capitals 4.3 (p<0.10).
    Synthetic control about 4 per 1,000.
  - *Inference:* wild-bootstrap p = 0.015 in the preferred column, 0.065–0.10 elsewhere.
  - *Weak points:*
    - Timing: mining area in Madre de Dios grew mostly in 2008–10 (Swenson et al.), not
      in 2004.
    - Santo Antônio and Jirau dams (from about 2008, large migration) and Brazil's own
      dredge mining on the Madeira fall in the treated group.
    - Brazil's 2004 shoot-down rule moved cocaine routes onto rivers out of Bolivia
      (Pereira, Pucci & Soares, "Landing on Water"), the same year and river.
  - *What it takes:* claim 1 for the Madeira basin, with a foreign source.
  - *What remains:* the water step he assumes. Test whether mercury violations at Madeira
    stations rose after 2004, and below which stretches.
  - *Status:* the AEJ: Policy acceptance (web-search snippet) is **unverified and
    doubtful**. The thesis has no journal version, OpenAlex lists only SSRN 2024 and the
    thesis, and his IDEAS list (heatwaves *JDE* 2026, police reform *JDE* 2026, PRONAF
    2019, cannabis 2025) has no mining paper. Treat it as unpublished.
- **Moreno-Louzada (2024)** — gold deposits upstream × gold price → birth anomalies,
  birthweight, preterm (see the research note). Takes claim 1 at Legal-Amazon scale.
- **Pereira & Pucci (2026, JPubE)** — 2013 law × gold deposits in protected areas →
  homicides. Takes the 2013 shock of claim 2 for local violence.
- **Amaral, Pantaleão & Rio Tinto (2026), "The Cost of Gold: Mining Booms, Violence, and
  Agrarian Conflict in Brazil"**, SocArXiv 10.31235/osf.io/hc9w2_v1.
  Current title "The Cost of Gold: Commodity Shocks and Violence Against Peasants in
  Brazil" (11 June 2026). **Read.**
  - *Design:* Bartik, 2000 MapBiomas mining share × BRL gold price, instrumenting
    MapBiomas garimpo/gold area; municipality panel 2000–2019.
  - *Results:* garimpo raises homicides (about 10 per 100,000 per log point); gold area
    raises CPT land conflicts; total mortality rises, intoxications do not.
  - *What it takes:* a local price-based first stage and a violence channel. No
    downstream or water outcome.
  - *Implication:* exclude external-cause infant deaths in downstream health outcomes.
- **Oliveira, G. M. de (2025), SSRN 10.2139/ssrn.5435977**; current title "Deregulating
  gold, deforesting the Amazon: supply-chain liability and illegal mining in Brazil's
  Indigenous Lands" (U. Bonn). **Read.**
  - *Design:* DiD on 5 km grid cells in 15 Indigenous lands, 2009–2018. Treated cells lie
    within 10 km of a pre-2013 gold deposit; Indigenous land × year FE; gold price ×
    proximity control.
  - *Results:* deforestation PPML +126%. Mining area, forest-to-mining and
    forest-to-pasture all rise; agriculture does not. Placebos with diamond and tin are
    null. The event study grows over 2013–18, largest in 2016–18.
  - *What it takes:* lead 1's first stage inside Indigenous lands. Downstream outcomes,
    the 2023 reversal and the ghost-permit test (which he proposes as policy) remain.
- **Hemsley, Libera & Rocha (2026), SSRN 10.2139/ssrn.7634888**, submitted to Economics
  Letters. **Read.**
  - *Design:* triple difference, Indigenous territory × gold province × post-2016,
    1,874 municipalities, 2010–2022.
  - *Results:* homicides +7.2 per 100,000. The event study jumps in 2018. In their
    two-break model, the 2013 component is small and mostly insignificant.
  - *What it takes:* nearly lead 1's exposure, with a confounding enforcement shock; for
    lead 3, a documented weakening episode (Nunes et al. 2024).

## Same design or mechanism, different setting

- **Romero & Saavedra (2015 WP), "The effects of gold mining on newborns' health"**;
  chapter 2 of Saavedra (2017), Stanford PhD thesis (https://purl.stanford.edu/bm761tq8280).
  Never published standalone; no IDEAS entry. **Read.**
  - *Design:* all Colombian births 1999/2001–2012 in the 642 municipalities within 25 km
    of a gold title; Amazon and Orinoquía excluded. Only legal titles in the extraction
    phase; 63% of mines are untitled (2010 census). "Near mining" = within 20 km of a
    title; "upstream" = within 20 km of a river with a title up to 25 km upstream along
    the course. Municipality, year and week FE, state trends.
  - *Results:* low APGAR −0.51 pp near mines (base 4.5%, p<0.10) and +0.45 pp downstream
    (p<0.01, +9.8%); both decay over the 5/10/20 km bands. LBW null. Near mines, more
    prenatal visits and in-hospital births (income channel); no selection downstream.
    Holy Week × upstream +0.20 pp (p<0.10), first trimester; low-powered by their own
    account.
  - *Weak points:* near-mining pre-trend 2–3 years before titling; untitled mines missing.
  - *Relevance:* the closest precedent for separating β_loc (income) from β_up
    (pollution). Colombia only, 25 km windows, no measured water quality.
- **Saavedra & Romero (2021, EER)** — illegal mining's response to a Colombian royalty
  reform. The closest analog to lead 1's "laundering cost as a tax". The thesis version
  (ch. 1, §1.6, Tables 1.7–1.8, A.12) carries the chain to newborns:
  - illegal mine downstream: high APGAR −0.68 pp (OLS, p<0.10), −2.53 (IV, n.s.);
  - IV is After × weak upstream institutions, F = 3.23 (weak);
  - flow-inversion placebo (downstream municipality's institutions): first stage −0.027,
    null;
  - legal ≤ illegal: p = 0.035 / 0.07; adding illegal mines doubles the downstream
    coefficient.
  
  This is the only precedent for lead 1's full chain (policy → illegal mining →
  downstream newborns), and it is unpublished.

  **Published EER read (2026-10-26):** it drops the newborn section entirely and cites
  the newborns paper only as "Romero and Saavedra, 2015". What it keeps:
  - Royalty share to the mining municipality cut from about 55% to 10% (2012). Illegal
    share of mined area +4.21 pp vs Peru; +1.18 pp per point of royalty rate (gold vs
    coal); about 80% larger with weak central institutions; smaller (n.s.) where armed
    groups rule.
  - The increase is concentrated outside national parks, where no permit is possible.
    For lead 1: a change in the cost of being legal only moves mining where a legal
    option exists. The 2013 presumption works the other way (it lets gold from where no
    permit is possible be sold as legal), so IGD is the right exposure.
  - Fn. 8: fewer than 20% of mine owners live in the municipality of their mine
    (Fletcher & Saavedra 2019). Supports outside crews in lead 2.
  - No cross-border spillover into Peru (Table A.8, col. 4); capital relocation is
    named but not modelled. Weak for lead 3.
  - Brazil was rejected as a control because it is not in the Andean Community, so it
    did not adopt the 2012 machinery-destruction decree (fn. 39).
  - The mining classifier uses Hansen loss year as an input and excludes forested pixels;
    TPR 32% at the chosen cutoff.
- **Saavedra (2022 WP, RePEc rie:riecdt:89), "The response of illegal mining to
  revealing its existence"** — RCT in Colombia disclosing satellite-detected illegal
  mines to local, national, both or no authorities. Illegal mining falls about 11% in
  disclosed and surrounding areas, similar across recipients, but rises in areas the
  information did not cover. **Net reduction only 7%**, so about a third of the gross
  effect is displaced. The only experimental displacement estimate found; a direct
  benchmark for lead 3's ratio (information, not raids). In the corpus, previously
  unscreened. **Read (April 2022 draft; still has "xx%" placeholders and broken table
  references, so cite as a WP).**
  - *Design:*
    - 2×2 RCT over about 842 mining municipalities, stratified by department. For 400,
      the mayor was told five randomly chosen predicted mine locations; for an
      orthogonal 400, the Air Force was told. Letters went out in September 2017,
      February 2019 and August 2021.
    - The outcome is the UNODC expert-validated gold-mining map, with 2014 and 2016
      before treatment and 2018–19 after. It covers only about 110 municipalities, on
      1 km grid cells.
  - *Spillover rings (Table A.11):* "disclosed" is the 1 km cell itself; "neighbour" is
    the 8 surrounding cells (<1 km); "far" is every other cell in a treated municipality.
    The outcome is the illegal share of mined area (mean 36%).
    - Disclosed: −8 to −15 pp.
    - Neighbour: −12 to −16 pp.
    - Far: +6.2**, +2.7 and +5.3** (local, national, both).
    
    11% gross falls to 7% net once the far cells are counted.
  - *Caveats for using the ratio:*
    - The far-cell increase is in the illegal *share*. In km² it is +0.0023, not
      significant (Table A.12). With 2016 titles held fixed it shrinks to 4.7*, 1.1 and
      3.9.
    - Displacement is only measured within the treated municipality. Moves into
      neighbouring or control municipalities are unmeasured, and the latter would
      contaminate the controls.
    - Small outcome sample (about 110 municipalities, 4 years).
  - *Other results:*
    - Mayors are 15–16 pp less likely to confirm a true mine when it is illegal, with
      municipality FE; the Air Force shows no gap. The author reads this as collusion,
      in line with Saavedra & Romero (2021).
    - Complaints doubled. Coca fell; Hansen deforestation, homicides, poverty and child
      labour are null.
    - About 80% of mayors say workers from closed mines switch occupation or migrate,
      which bears on lead 2.
  - *For lead 3:* the ring design (target / <1 km / rest of municipality) maps onto
    substitute-site distance bands. Lead 3 should go further: rings across municipalities
    and along rivers, outcomes in area rather than illegal share, and armed raids rather
    than letters.
- **Ladino, Saavedra & Wiesner (2021, JPubE), "One step ahead of the law"** — a
  pre-announced Colombian coca-substitution programme triggered anticipatory planting,
  with spillovers to neighbouring cells. A design precedent for lead 4's anticipation of
  PL 191/2020. Anticipation halved the programme's effect; neighbouring untargeted cells
  fell by a similar amount as recipients (positive spillovers). Not in the corpus.
- **Romero & Saavedra (2019 WP, Rosario 17508), "Communal property rights and
  deforestation"** — difference-in-discontinuities at communal-title borders, before and
  after titling, Colombia. Deforestation falls inside after titling, with positive
  spillovers nearby. A method precedent for lead 4's demarcation design. Not in the
  corpus.
- **Fletcher & Saavedra (2019 WP), "Job migration in a rivalry setting"** — co-migration
  of Colombian gold miners through social networks. Productive miners invite fewer, and
  more productive, connections. Micro evidence on how outside crews form (lead 2). In the
  corpus, unscreened.
- **Saavedra (2023 WP, Rosario 20793), formalization-assistance RCT** with informal coal
  and gold mines: no rise in formalization after two years, high attrition. **Coy,
  Pacheco, Peralta, Saavedra & Llanes (2021, Alianza EFI 19417)**, survey of Colombian
  miners: formal mines get about 30% higher sale prices. Gives a size for the price
  premium of legal gold that the 2013 presumption removes (lead 1). Neither is in the
  corpus.
- **Vashold, Pirich, Heinze & Kuschnig (2025, JDE), "Downstream impacts of mines on
  agriculture in Africa"** — upstream/downstream discontinuity on directed river
  networks. A method precedent for the project's asymmetry tests.
- **Jones (2026), "Lookout Below!"** — Gold King Mine spill (US), downstream LBW and
  preterm births.
- **Bonilla-Mejía (2020, JDE)** — illegal gold mining and human capital, Colombia.
- **Rozo (2020, WD)** — illegal gold mining and malaria, Colombia. Relevant to lead 2's
  malaria data.
- **Mittelberg, Skidmore et al. (2026, PNAS)** — laundering cattle through compliant
  intermediaries after Dirty-List sanctions, Brazil. A laundering-response analog for
  lead 1.
- **Scuticchio (2026, JCR)** — gold price shocks redeploy criminal groups across resource
  areas, Mexico. A displacement analog for lead 3.
- **Dethier et al. (2023, Conservation Letters), "Operation Mercury"** — Peru's 2019
  armed intervention: mining ponds in targeted zones went from +33–90%/year to −4–5%/year,
  70–90% abandonment, while new mining accelerated nearby. The closest enforcement-
  displacement analog for lead 3 (descriptive, satellite-based).
- **Álvarez-Berríos, L'Roe & Naughton-Treves (2021, ERL)** — ASGM formalization and
  mining deforestation, Madre de Dios. An analog for lead 1's formal-channel mechanism.
- **Arıman & Baran (2026, SSRN), "Silent Exposure: DDT, River Networks, and Infant
  Health"**; **Chen (2026, UC Davis thesis), hydrological networks and agricultural
  externalities**; **Cui & Luo (2026)**, China — further river-network health designs
  citing DRS. Method precedents only.

## Same setting, different outcome or angle

- **He, Pede, de Aguiar Barros & Heilmayr (2026), "Gold Mining Increases Malaria Burden
  and Compounds Indigenous Health Inequality in the Brazilian Amazon"**, SSRN
  10.2139/ssrn.6914819. **Read.**
  - *Design:* LASSO-selected rock formations × gold price as the instrument; SIVEP
    microdata obtained through an access-to-information (LAI) request.
  - *Results:* +0.45 API per km² of mining; spillovers by distance band up to 250 km;
    Indigenous sub-district estimates.
  - *Fragility:* two municipalities carry the identification; the effect is null over
    2003–2024.
  - *What remains for lead 2:* river direction, and the resident/non-resident split.
- **Oliveira & Miranda (2024, JIE)** — priority-municipality enforcement reduces land
  conflict. Adjacent to lead 3 (deforestation enforcement, not mining).
- **Assunção, Gandour & Rocha (2023, AEJ: Applied)** — DETER satellite monitoring and
  enforcement. Background for lead 3.
- **Rodrigues (2025, BPSR), "Brazil's Journey Against Illegal Mining"** — documents the
  electronic invoice and the suspension of the good-faith presumption. A source for the
  2023 dates in lead 1.
- **Castro et al. (2025, PLoS NTD); Barbieri & dos Santos (2025, Lancet Reg. Health)** —
  mining, human mobility and malaria in the Yanomami territory. Descriptive support for
  outside crews (lead 2).
- **Araujo, Possebom & Setti (2026, arXiv 2509.06076)** — DETER enforcement instrumented
  by cloud cover reduces Amazon homicides by ~15% (~1,477/year). The cloud-cover
  instrument could serve lead 3.
- **De Salazar et al. (2024, Lancet Reg. Health Americas)** — malaria spillover into
  Indigenous communities in Guyana after Brazil's 2023 crackdown. Cross-border
  displacement evidence for lead 3 and outside crews for lead 2.
- **de Aguiar Barros et al. (2025, Malaria Journal)** — malaria before and after the 2023
  Yanomami intervention (descriptive).
- **Ribas & Galetti (2025, Global Ecol. Conserv.)** — Indigenous lands inhibit mining
  deforestation (counterfactual). Background for lead 4.
- **Moretz-Sohn & Costa (2026, OSF)** — protected-area erasure accelerates deforestation.
  Relevant to lead 4's protected-area-stability threat.
- **Mancha (2025, JDE)** — regulating junkyards dismantles a market for stolen goods,
  São Paulo. A fencing-market analog for lead 1.
- **Pereira, Pucci & Soares (2024), "Landing on Water" (IZA DP 17425)** — Brazil's
  shoot-down rule for drug flights moved cocaine traffic onto rivers from the Andean
  producer countries. Violence rose along those rivers and tracked coca production
  there. Uses the Amazon river network.
  - Lead 3: a displacement analog by the authors of lead 1's closest competitor.
  - Confounder: the rule took effect in July 2004 (Decreto 5.144), Parfitt's treatment year, and the
    Madeira comes out of Bolivia. Any post-2004 river-based treatment downstream of the
    Andes must separate trafficking. Also supports excluding external-cause infant
    deaths (ICD-10 V–Y).
  - CitEc lists 2 citers, not yet pulled.
- **Parfitt & Goulart (2026, JDE), heatwaves and births** — uses SINASC APGAR for about
  25 million births (null effect). Shows the APGAR field is usable at scale.
- **Damm, Börner & Gerber (2024)** — health benefits of reduced Amazon deforestation.
- **de Bakker et al. (2021); Vasconcellos et al. (2025)** — valuations of mercury health
  costs (MPF calculator; Yanomami burden of disease). Useful for welfare magnitudes.

## Method cautions and background (RePEc, 2026-10-26)

- **Fergusson, Saavedra & Vargas (2020 WP), "The perils of misusing remote sensing
  data"** — Hansen GFC measures tree cover, not native forest; with ground-validated
  data their conflict estimate flips sign. MapBiomas-based exposure is fine, but any
  lead-3/4 check on Hansen loss years (as in Oliveira 2025) should say so.
- **Moreno-Louzada, Figueira & Picchetti (2025, arXiv 2512.10076), "Microfoundations
  and the causal interpretation of price-exposure designs"** — designs that use a
  single commodity price as the instrument have contamination terms, and conventional
  standard errors over-reject. In their gold-mining × homicides application for the
  Amazon, the corrected SEs are about twice as large. Applies to any gold-price ×
  deposit exposure: Moreno-Louzada's birth-outcomes paper, Amaral et al., and our own
  price interactions. Cite it and report their SEs. In the corpus, previously
  unscreened.
- **D. H. C. Pereira (2023, UnB PhD, geography), detection of dredge-based gold mining
  ("garimpos embarcados") from satellite images** (http://repositorio.unb.br/handle/10482/48214).
  Land-cover maps such as MapBiomas miss river dredges. This is a possible measure for
  Madeira dredge mining, the Parfitt confounder, and for river garimpo generally.
  Unread.
- **Borges, Menezes-Filho & Moreno-Louzada (2024, ANPEC), Starlink and criminal
  capability in the Amazon** — satellite internet in remote areas. Background for
  lead 3, since Starlink is used to evade enforcement. Abstract not retrieved.
- **Soares, Pereira & Pucci (2021, Amazônia 2030 report), "Ilegalidade e Violência na
  Amazônia"** — the descriptive groundwork for P&P and "Landing on Water". Background.
- **Jimenez, Molina & Saavedra (2026, JEEM), "Power plays in the jungle"** — political
  alignment of governors and mayors raises deforestation about 40% via Colombia's
  regional environmental agencies. Background on agency capture for lead 3.
- **Prem, Saavedra & Vargas (2020, WD)**, end-of-conflict deforestation; **Dahis, de las
  Heras & Saavedra (2026, JAERE)**, young mayors and deforestation in Brazil. Background
  only.

## Verdict per claim

1. **Partly taken.** Parfitt (Madeira, foreign shock, mortality) and Moreno-Louzada
   (Legal Amazon, geology × price, anomalies) both claim downstream gold → infant health.
   What remains is distance-based exposure across all basins and few-cluster inference.
   The measured mercury link does not exist for the Amazon in the project's station data
   (HPC check, 2026-10-26; see the leads note's caveats). Parfitt's foreign-shock design is a
   natural robustness check, or a benchmark to replicate in the project's network; read
   in full, it is fragile (timing, dams, the 2004 drug-route shift, p-values) and
   apparently unpublished. Romero & Saavedra show the local-income/downstream-pollution
   split for Colombia only (unpublished).
2. **Open on the downstream outcome.** P&P own the 2013 law for violence, and Oliveira
   (2025) for mining deforestation inside Indigenous lands (read).
   Amaral et al. use the price channel. Nobody found yet takes the law downstream, or
   uses the 2023 reversal. Hemsley et al.'s post-2016 shock on the same exposure must be
   separated in the event study. The full policy → illegal mining → downstream newborns
   chain exists only for Colombia (Saavedra & Romero, thesis ch. 1, weak IV), and was
   dropped from the published EER.
3. **Open.** Malaria papers support the mobility story but do not model two groups or
   test the asymmetry of downstream spread.
4. **Open for a Brazilian displacement ratio and downstream health.** Displacement is
   documented descriptively for Peru (Dethier et al. 2023) and across the border into
   Guyana (De Salazar et al. 2024), and for crime in Mexico. No causal displacement ratio
   for garimpo within Brazil, and no downstream-health version, was found. The Colombian
   disclosure RCT (Saavedra 2022) gives an experimental benchmark: about a third of the
   gross reduction reappears in untargeted areas (11% gross, 7% net). That holds within
   municipalities only, in illegal share rather than area, for about 110 municipalities. Lead 3 asks the
   same question for armed raids, in Brazil, with downstream health.
5. **Not screened in depth.** Indigenous-land queries returned mostly descriptive and
   legal work. Ladino et al. (2021) give a design precedent for anticipation, and Romero
   & Saavedra (2019) one for title borders.

## Next actions

1. Done 2026-10-26: Parfitt read; Romero & Saavedra found and read; RePEc author lists
   for Pereira, Parfitt and Saavedra screened. Broad Google Scholar keyword queries
   skipped.
2. Done 2026-10-26: author re-screen of `corpus.jsonl`. It found five more unscreened
   records; the most important is Moreno-Louzada et al. (2025) on inference with
   price-exposure designs.
3. Optional: IDEAS author pages for Pucci, Moreno-Louzada, Romero and Rocha; CitEc
   citers of "Landing on Water".
4. Done 2026-10-26: the shoot-down rule (Lei 9.614/1998) took effect with Decreto 5.144
   of 16 July 2004, so it confounds Parfitt's 2004 treatment year.
5. Done 2026-10-26: Saavedra (2022) read. "Surrounding" means <1 km (the 8 neighbouring
   1 km cells); "untargeted" means the rest of the treated municipality. The displacement
   is in the illegal share, and is not significant in km².
6. Done 2026-10-26: findings carried over to `mining_economics_leads.md`.
