# Task: download FUNAI Indigenous-land data and send it to the HPC

*Temporary handoff for a Claude session on a laptop connected to a Brazilian VPN. Remove
this file once the data is on the HPC and the `protected_areas` source reads it.*

You are on a laptop connected to a Brazilian VPN. FUNAI's map server
(`geoserver.funai.gov.br`) does not answer from the HPC, which is why this download runs
here. The HPC already has the Indigenous-land polygons with their *current* stage, from
INPE TerraBrasilis (627 lands nationally). What it lacks, and what this task is mainly
for, is **when each land reached each stage**, above all the declaration (portaria
declaratória) and the homologation (decreto). The research needs to know which lands
were protected before 2013. Download FUNAI's own layers and tables, find the stage
dates, write a provenance note and a schema report, and rsync everything to the HPC. Do
not build any pipeline code.

## Rules

- Use FUNAI's normal downloads: the links on
  https://www.gov.br/funai/pt-br/atuacao/terras-indigenas/geoprocessamento-e-mapas and
  the GeoServer's standard WFS requests. If something is blocked even on the VPN, stop
  and tell me; do not try to get around it (no user-agent spoofing, no scraping around
  access controls).
- For other websites (Diário Oficial, Planalto, ISA), read their terms first, download
  only files they offer for download, and do not scrape page by page. If the dates exist
  only as web pages, report where and how many pages it would take; don't crawl them.
- Do not edit, re-encode or "clean" the downloaded files. Keep them exactly as served;
  unzip only into a separate folder for inspection.
- Work in a new, empty folder: `~/funai_<YYYY-MM-DD>/` (today's date). Run any Python
  that reads the files with `python -I`, with your scripts outside that folder.

## What to download

1. **The GeoServer layer list:**
   `https://geoserver.funai.gov.br/geoserver/Funai/ows?service=WFS&version=1.0.0&request=GetCapabilities`.
   Save the XML.
2. **The Indigenous-land layers,** as SHAPE-ZIP and as CSV (both are linked on the
   page above), with no feature cap. If a request returns exactly the `maxFeatures`
   value, raise it and download again.
   - `Funai:tis_poligonais` (land polygons; the main one);
   - `Funai:tis_poligonais_portarias` (polygons of lands with a declaration);
   - `Funai:tis_pontos` and `Funai:tis_pontos_portarias` (lands known only as points);
   - `Funai:tis_cr` and `Funai:tis_ctl` (regional coordination and local units),
     optional;
   - any other layer in the list that carries stage dates, decrees or ordinances.
3. **The attribute tables** linked on the same page
   (`http://mapas2.funai.gov.br/portal_mapas/ods/tis_*.ods`). They may hold more columns
   than the layers.
4. **Stage dates,** if the files above don't have them. Look, in this order, for:
   - date fields in the layers and tables (e.g. `data_*`, `portaria`, `decreto`,
     `homologacao`, `doc_*`);
   - a list or table published by FUNAI (gov.br) of lands by stage with their acts
     (portaria and decree numbers and dates), as a downloadable file;
   - an official downloadable list of homologation decrees (e.g. on planalto.gov.br)
     or a dataset on dados.gov.br;
   - Instituto Socioambiental's "Terras Indígenas no Brasil"
     (terrasindigenas.org.br), which records each land's stage history. Use it only if
     it offers a download and its terms allow research use; otherwise just report
     what it has.

Per land, the research needs at least the declaration and homologation dates, and
ideally also the study (GT ordinance), delimitation (RCID publication) and
regularisation (registry) dates.

## Checks and reports

Write these files into the download folder, next to the downloads.

### `PROVENANCE.md`

- For each file: source page, the exact download URL, download date and time, size in
  bytes, and any "last updated" date the source shows.
- Anything that failed, was capped or was missing.

Also write `SHA256SUMS` (`shasum -a 256 *` or `sha256sum *`) for every downloaded file.

### `SCHEMA.md`

- **Per layer and table:** format, encoding, CRS, feature or row count, and every
  column with its type, fill rate, distinct values and 5 examples. Show the full value
  list of the stage (`fase_ti`) and modality fields.
- **Keys:** the land code (`terrai_cod` or similar). Is it unique, and do the layers,
  the ODS tables and any date source share it? How many codes match between them?
- **Counts by stage,** nationally and for the Legal Amazon states (AC, AM, AP, MA, MT,
  PA, RO, RR, TO). Compare with the HPC's TerraBrasilis copy: 627 lands nationally, 386
  in the Legal Amazon layer.
- **Stage dates:** where each date comes from, and its coverage by current stage
  (e.g. the share of homologated lands with a homologation date). Count the Legal Amazon
  lands declared, and homologated, in each year since 2000, and list those homologated
  in 2013 or later.
- **Geometry:** invalid or empty geometries, and lands with several polygons.
- **Anything odd:** duplicated codes, dates out of range, stage values that disagree
  between sources.

## Transfer to the HPC

Copy the folder (downloads, `PROVENANCE.md`, `SCHEMA.md`, `SHA256SUMS`; not the
unzipped inspection copies):

```bash
rsync -av ~/funai_<YYYY-MM-DD>/ schulz0022@transfer12.scicore.unibas.ch:/scicore/home/meiera/schulz0022/projects/river-pollution-brazil/data/protected_areas/raw/funai_<YYYY-MM-DD>/
```

Create the target folder first if rsync complains
(`ssh schulz0022@transfer12.scicore.unibas.ch mkdir -p <path>`). After the copy, verify
the checksums on the HPC side: `ssh ... "cd <path> && sha256sum -c SHA256SUMS"`.

## Report back

End with a short message I can paste into the HPC session:

- the HPC path, the files and their sizes;
- the layers and tables found, their land counts, and the key that links them;
- whether stage dates exist, from which source, and their coverage;
- the Legal Amazon lands declared and homologated by year since 2000, and the number
  homologated in 2013 or later;
- anything that failed or looked wrong.
