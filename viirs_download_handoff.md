# Task: send EOG VIIRS annual night-light composites to the HPC

*Temporary handoff for a Claude session on a laptop. Remove this file once the data is on
the HPC and the `nightlights` source reads it.*

The research uses the Earth Observation Group's (EOG, Colorado School of Mines) annual
VIIRS night-light composites (VNL) as a measure of garimpo activity in the Amazon. The
HPC can already read the 2012–2021 composites (VNL v2.1) from another project. What it
lacks is **2022 onward**, which EOG publishes as VNL v2.2. EOG requires a login, so the
user downloads the files by hand in a browser. Your job: check the downloads, write a
provenance note and a schema report, and rsync everything to the HPC. Do not build any
pipeline code.

## Rules

- **Do not log in to EOG, and do not handle the user's credentials.** The user downloads
  in a browser. If something is missing, tell the user which file to download; do not try
  to fetch it yourself with scripts, cookies or tokens, and do not try to get around the
  login.
- Do not edit, decompress in place, re-encode or rename the downloaded files. Keep them
  exactly as served (`.tif.gz`). For inspection, read them through GDAL's `/vsigzip/`, or
  decompress copies into a separate folder.
- Work in a new, empty folder: `~/eog_vnl_<YYYY-MM-DD>/` (today's date). Run any Python
  that reads the files with `python -I`, with your scripts outside that folder.

## What the user downloads

From `https://eogdata.mines.edu/nighttime_light/annual/v22/` (EOG's "VIIRS Nighttime
Lights", annual, version 2.2), into `~/eog_vnl_<YYYY-MM-DD>/`:

1. **For each year from 2022 to the latest available** (2024 or 2025): the composite of
   the full calendar year from the Suomi NPP satellite (`npp` in the file name, matching
   the v2.1 files on the HPC; take NOAA-20 `j01` files only if a year has no NPP
   composite, and say so), in these variants:
   - `average_masked` (the main measure: background, fire and aurora masked);
   - `median_masked`;
   - `cf_cvg` (cloud-free coverage, number of observations).

   The v2.1 files on the HPC are named like
   `VNL_v21_npp_2021_global_vcmslcfg_c202205302300.average_masked.dat.tif.gz`; the v2.2
   names differ in the version and creation stamp.
2. **Any README, release notes or "what changed in v2.2" document** on the same pages, as
   PDF or HTML saved from the browser.
3. **A record of the directory listing:** save the listing pages of
   `.../annual/v22/` and of each year folder (browser "Save page as", HTML only), so the
   provenance shows what was offered.

Each `average_masked` file is about 300 MB compressed (11.6 GB uncompressed); `cf_cvg`
can be 2 GB compressed.

### Ask, don't decide

Check the v22 listing for **years before 2022**. If v2.2 reprocesses 2012–2021, ask the
user whether to download those too (`average_masked` and `cf_cvg`, about 2.3 GB per year)
so the whole series is one version; the HPC's v2.1 files would then be a fallback. Report
what you found either way.

## Checks and reports

Write these files into the download folder, next to the downloads.

### `PROVENANCE.md`

- For each file: the page it came from, its exact URL (from the listing), the download
  date, its size in bytes, and the "last modified" date the listing shows.
- Which years and variants v22 offers, and which were downloaded; anything missing or
  failed.
- What the release notes say about v2.2 against v2.1 (method changes, whether values are
  comparable across versions), with the quote and where it is.

Also write `SHA256SUMS` (`shasum -a 256 *.gz` or `sha256sum *.gz`) for every downloaded
data file, and verify each `.gz` decompresses without error (`gzip -t`).

### `SCHEMA.md`

Per file, read through `/vsigzip/` with rasterio or `gdalinfo`:

- width, height, CRS, geotransform (origin and pixel size), data type, nodata value,
  block layout, compression;
- whether the grid is identical to the v2.1 grid on the HPC: 86401 × 33601 pixels,
  EPSG:4326, origin (−180, 75), pixel size 1/240° (15 arc-seconds), float32 for
  `average_masked`, uint16 for `cf_cvg`, no nodata value, one strip per row;
- value summary over Brazil only (lon −74.1 to −28.8, lat −33.8 to 5.3): min, max, mean,
  share of negative values, share above 1 nW/cm²/sr; and the same for the Legal Amazon
  box (lon −74 to −44, lat −18.1 to 5.3).

## Transfer to the HPC

Copy the folder (downloads, saved pages, `PROVENANCE.md`, `SCHEMA.md`, `SHA256SUMS`; not
any decompressed inspection copies):

```bash
rsync -av ~/eog_vnl_<YYYY-MM-DD>/ schulz0022@transfer12.scicore.unibas.ch:/scicore/home/meiera/schulz0022/projects/river-pollution-brazil/data/nightlights/raw/eog_vnl_<YYYY-MM-DD>/
```

Create the target folder first if rsync complains
(`ssh schulz0022@transfer12.scicore.unibas.ch mkdir -p <path>`). After the copy, verify
the checksums on the HPC side: `ssh ... "cd <path> && sha256sum -c SHA256SUMS"`.

## Report back

End with a short message I can paste into the HPC session:

- the HPC path, the files and their sizes;
- the years and variants, the satellite (NPP or NOAA-20) and the version of each;
- whether the grid matches v2.1 exactly, and the Brazil value summary;
- whether v2.2 covers 2012–2021 and what the release notes say about comparing v2.2 with
  v2.1;
- anything that failed or looked wrong.
