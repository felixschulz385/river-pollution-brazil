# Task: download IBAMA enforcement data and send it to the HPC

*Temporary handoff for a Claude session on a laptop connected to a Brazilian VPN. Remove
this file once the data is on the HPC and the `enforcement` source is built.*

You are on a laptop connected to a Brazilian VPN. IBAMA's open-data portal
(https://dadosabertos.ibama.gov.br) blocks requests from outside Brazil, which is why
this download runs here and not on the HPC. Download the enforcement datasets below,
check them, write a provenance note and a schema report, and rsync everything to the
HPC. Do not build any pipeline code: the HPC session builds the `enforcement` source
from your report.

## Rules

- Use the portal's normal downloads (dataset pages, or the CKAN API at
  `/api/3/action/package_show?id=<dataset>`). If something is blocked even on the VPN,
  stop and tell me; do not try to get around it (no user-agent spoofing, no scraping
  around access controls).
- Do not edit, re-encode or "clean" the downloaded files. Keep the archives exactly as
  served; unzip them only into a separate folder for inspection.
- Work in a new, empty folder: `~/ibama_<YYYY-MM-DD>/` (today's date). Run any Python
  that reads the files with `python -I`, with your scripts outside that folder.

## What to download

From the "Fiscalização" datasets, the complete national files, all years. Names may
differ slightly; take what the portal lists under these topics:

1. **Autos de infração** (fines), e.g. dataset `fiscalizacao-auto-de-infracao`.
2. **Termos de embargo** (embargoes), e.g. `fiscalizacao-termo-de-embargo`.
3. **Termos de apreensão** (seizures), e.g. `fiscalizacao-termo-de-apreensao`.
4. **Termos de destruição / inutilização** (equipment destroyed on site), if the
   portal has it. This is the most important one for the research; if you can't find
   it, say so explicitly.
5. Any data dictionary or metadata document attached to these datasets.
6. Optional: the embargoed-area polygons (shapefile or GeoPackage, often named
   `adm_embargo_ibama_a` or similar), if offered as a direct download.

Prefer CSV over other formats when both exist. If a dataset is split by year or state,
download all parts.

## Checks and reports

Write two files into the download folder (next to the archives, not inside them).

### `PROVENANCE.md`

- For each file: dataset name and page URL, the exact download URL, download date and
  time, size in bytes, and the "last updated" date shown on the portal.
- Anything that failed or was missing.

Also write `SHA256SUMS` (`shasum -a 256 *` or `sha256sum *`) for every downloaded file.

### `SCHEMA.md`

For each table (inspect the unzipped copies):

- **Format:** encoding (UTF-8 or CP-1252/Latin-1), separator, quoting, header row, and
  row count.
- **Columns:** every column with its type and fill rate. For the important ones, 5
  example values and the number of distinct values. Important columns are:
  - dates (infraction, embargo, seizure dates);
  - municipality fields (name, IBGE code, UF);
  - coordinates (latitude/longitude, and their format);
  - infraction type or category, legal basis (`enquadramento`), and status or
    situation (e.g. cancelled, annulled);
  - fine amount;
  - free-text descriptions;
  - the seized or destroyed item description and quantity;
  - process or record IDs that link the tables.
- **Coverage:** row counts by year, and by year for the Legal Amazon states (AC, AM,
  AP, MA, MT, PA, RO, RR, TO).
- **Mining-related records:** rows whose description, category or item fields match
  (case- and accent-insensitive) any of `garimp`, `minera`, `ouro`, `lavra`, `draga`,
  `balsa`, `PC` (as a whole word, i.e. the hydraulic excavator), `escavadeira`,
  `retroescavadeira`, `mercúrio`/`mercurio`, `cassiterita`.
  - Counts by year, Legal Amazon vs. elsewhere.
  - 10 example rows.
  - Which fields carry the match. This decides how the HPC classifies mining records,
    so be concrete.
- **Linking:** do the seizure, destruction and embargo tables share an ID with the
  fines table? Check how many IDs match.
- **Anything odd:** duplicated records, years that look truncated, a format change
  between years, coordinates outside Brazil.

## Transfer to the HPC

Copy the folder (archives, `PROVENANCE.md`, `SCHEMA.md`, `SHA256SUMS`; not the
unzipped inspection copies):

```bash
rsync -av ~/ibama_<YYYY-MM-DD>/ schulz0022@transfer12.scicore.unibas.ch:/scicore/home/meiera/schulz0022/projects/river-pollution-brazil/data/enforcement/raw/ibama_<YYYY-MM-DD>/
```

Create `data/enforcement/raw/` first if rsync complains
(`ssh schulz0022@transfer12.scicore.unibas.ch mkdir -p <path>`). After the copy, verify
the checksums on the HPC side: `ssh ... "cd <path> && sha256sum -c SHA256SUMS"`.

## Report back

End with a short message I can paste into the HPC session:

- the HPC path;
- the files and their sizes;
- which of the four tables you found, and whether destruction records exist;
- the encoding and separator;
- the year coverage;
- the number of mining-related records per table and the fields that identify them;
- whether the tables link by ID;
- anything that failed.
