#!/usr/bin/env python3
"""Merge the Earth Engine mining-substance tiles into one GeoTIFF per year.

Input: the 41-band tiles written by `scripts/gee_export_mapbiomas_mining.js`
(`<prefix>-<row offset>-<col offset>.tif`, all on the asset's grid).

Output, for the land-cover pipeline (`src/data/sources/land_cover/mining.py`):
  - `mapbiomas_mining_substances_c11_<year>.tif`: one band, uint16, no nodata
    value, so pixels without mining (and tiles Earth Engine skipped as empty)
    read as class 0;
  - the pipeline legend (`ID`/`Class`/`Subclass` xlsx) built from MapBiomas's
    legend CSV;
  - national mined area by class and year (csv), in place of the Earth Engine
    table export.

Run on the HPC from the project root:
    python scripts/prepare_mapbiomas_mining.py [--n-jobs N]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from joblib import Parallel, delayed
from rasterio.transform import Affine
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.shared.slurm import resolve_n_jobs  # noqa: E402
from src.data.sources.land_cover.mining import (  # noqa: E402
    MINING_BAND_TEMPLATE,
    MINING_DATADIR,
    MINING_FILENAME_TEMPLATE,
    MINING_LEGEND_PATH,
    MINING_NATIONAL_AREA_PATH,
    MINING_SOURCE_LEGEND_PATH,
    MINING_TILES_DIR,
    build_mining_legend,
    pixel_area_km2,
)

TILE_OFFSETS = re.compile(r"-(\d{10})-(\d{10})\.tif$")


def read_tiles(tiles_dir):
    """Return the tiles with their pixel offsets, the export grid and band names."""
    tiles = []
    for path in sorted(Path(tiles_dir).glob("*.tif")):
        match = TILE_OFFSETS.search(path.name)
        if match is None:
            raise ValueError(f"Tile name without pixel offsets: {path.name}")
        row, col = map(int, match.groups())
        with rasterio.open(path) as src:
            tiles.append(dict(path=path, row=row, col=col, height=src.height, width=src.width,
                              transform=src.transform, crs=src.crs, bands=src.descriptions))
    if not tiles:
        raise FileNotFoundError(f"No tiles in {tiles_dir}")

    first = tiles[0]
    resolution = first["transform"].a
    origin_x = first["transform"].c - first["col"] * resolution
    origin_y = first["transform"].f + first["row"] * resolution
    for tile in tiles:
        expected = (origin_x + tile["col"] * resolution, origin_y - tile["row"] * resolution)
        if not np.allclose((tile["transform"].c, tile["transform"].f), expected, atol=resolution / 100):
            raise ValueError(f"{tile['path'].name} is off the common grid")
        if tile["bands"] != first["bands"] or tile["crs"] != first["crs"]:
            raise ValueError(f"{tile['path'].name} has different bands or CRS")

    grid = dict(
        crs=first["crs"],
        transform=Affine(resolution, 0.0, origin_x, 0.0, -resolution, origin_y),
        height=max(tile["row"] + tile["height"] for tile in tiles),
        width=max(tile["col"] + tile["width"] for tile in tiles),
    )
    return tiles, grid, first["bands"]


def merge_year(year, tiles, grid, bands, out_dir):
    """Write one year's mosaic and return its mined pixels and area by class."""
    band = bands.index(MINING_BAND_TEMPLATE.format(year=year)) + 1
    profile = dict(
        driver="GTiff", dtype="uint16", count=1, nodata=None,
        crs=grid["crs"], transform=grid["transform"], height=grid["height"], width=grid["width"],
        tiled=True, blockxsize=512, blockysize=512, compress="deflate", predictor=2,
        sparse_ok=True, bigtiff="yes",
    )
    resolution = grid["transform"].a
    top = grid["transform"].f
    pixels = {}
    area = {}
    out_path = Path(out_dir) / MINING_FILENAME_TEMPLATE.format(year=year)
    with rasterio.open(out_path, "w", **profile) as dst:
        for tile in tiles:
            with rasterio.open(tile["path"]) as src:
                data = src.read(band)
            dst.write(data, 1, window=Window(tile["col"], tile["row"], tile["width"], tile["height"]))
            rows, cols = np.nonzero(data)
            if rows.size == 0:
                continue
            values = data[rows, cols]
            row_area = pixel_area_km2(top - tile["row"] * resolution, resolution, tile["height"])
            classes, inverse = np.unique(values, return_inverse=True)
            tile_pixels = np.bincount(inverse)
            tile_area = np.bincount(inverse, weights=row_area[rows])
            for code, n, km2 in zip(classes.tolist(), tile_pixels, tile_area, strict=True):
                pixels[code] = pixels.get(code, 0) + int(n)
                area[code] = area.get(code, 0.0) + float(km2)
    return pd.DataFrame(
        {"year": year, "class": list(pixels), "pixels": list(pixels.values()), "area_km2": list(area.values())}
    )


def write_legend(source_legend_path, legend_path):
    source = pd.read_csv(source_legend_path)
    legend = build_mining_legend(source["class_id"])
    Path(legend_path).parent.mkdir(parents=True, exist_ok=True)
    legend.to_excel(legend_path, index=False)
    return legend


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tiles-dir", default=MINING_TILES_DIR)
    parser.add_argument("--out-dir", default=MINING_DATADIR)
    parser.add_argument("--source-legend", default=MINING_SOURCE_LEGEND_PATH)
    parser.add_argument("--legend", default=MINING_LEGEND_PATH)
    parser.add_argument("--area-csv", default=MINING_NATIONAL_AREA_PATH)
    parser.add_argument("--years", default=None, help="Comma-separated subset, e.g. 2019,2020.")
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()

    legend = write_legend(args.source_legend, args.legend)
    print(f"Legend: {len(legend)} codes -> {args.legend}")

    tiles, grid, bands = read_tiles(args.tiles_dir)
    years = [int(band.rsplit("_", 1)[1]) for band in bands]
    if args.years:
        years = [int(year) for year in args.years.split(",")]
    print(f"{len(tiles)} tiles, grid {grid['width']} x {grid['height']}, years {years[0]}-{years[-1]}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = Parallel(n_jobs=args.n_jobs or resolve_n_jobs(), verbose=10)(
        delayed(merge_year)(year, tiles, grid, bands, out_dir) for year in years
    )

    area = pd.concat(results, ignore_index=True).sort_values(["year", "class"])
    unknown = sorted(set(area["class"]) - set(legend["ID"]))
    if unknown:
        raise ValueError(f"Raster codes missing from the legend: {unknown}")
    Path(args.area_csv).parent.mkdir(parents=True, exist_ok=True)
    area.to_csv(args.area_csv, index=False)
    print(f"Wrote {len(years)} rasters to {out_dir} and national area to {args.area_csv}")


if __name__ == "__main__":
    main()
