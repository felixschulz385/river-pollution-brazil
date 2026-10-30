import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window, transform as window_transform

from src.data.sources.biomes.constants import BRAZIL_PROJECTED_CRS
from src.data.sources.biomes.preprocess import _load_adm2_boundaries
from src.data.sources.geology.constants import gpkg_path as geology_gpkg_path
from src.data.sources.geology.preprocess import load_gold_occurrences
from src.data.sources.land_cover.constants import derive_mun_id_from_adm2_id
from src.data.sources.protected_areas.preprocess import load_protected_areas

from .constants import (
    ADM2_COLUMNS,
    AVERAGE,
    COVERAGE,
    COVERAGE_COLUMN,
    DEFAULT_ADM2_OUTPUT_PATH,
    DOCUMENT_NAME_TAG,
    GOLD_BUFFER_M,
    LIT_THRESHOLD,
    MUN_ID_COLUMN,
    SATELLITE_COLUMN,
    VERSION_COLUMN,
    YEAR_COLUMN,
    ZONES,
    parse_name,
    raw_dir,
    raw_files,
    zone_columns,
)


logger = logging.getLogger(__name__)

# Zone suffix -> (flat pixel positions in the window, or None for every pixel;
# municipality number of each, 0 outside every municipality). Set before the
# worker processes fork, so they share it without pickling.
_ZONES = {}


def list_files(raw):
    """The `average_masked` and `cf_cvg` files under `raw`, one row per file, by the year and variant of their name."""
    files = pd.DataFrame(raw_files(raw), columns=["path", "year", "variant", "version", "satellite"])
    duplicated = files[files.duplicated(["year", "variant"], keep=False)]
    if not duplicated.empty:
        raise ValueError(f"Several files for one year and variant: {duplicated['path'].map(str).tolist()}")
    return files


def grid_window(transform, width, height, bounds):
    """The window of the grid covering `bounds` (west, south, east, north), snapped outward to whole pixels."""
    west, south, east, north = bounds
    col0 = max(int(np.floor((west - transform.c) / transform.a)), 0)
    col1 = min(int(np.ceil((east - transform.c) / transform.a)), width)
    row0 = max(int(np.floor((north - transform.f) / transform.e)), 0)
    row1 = min(int(np.ceil((south - transform.f) / transform.e)), height)
    return Window(col0, row0, col1 - col0, row1 - row0)


def _mask(geometries, shape, transform):
    if len(geometries) == 0:
        return np.zeros(shape, dtype=bool)
    return rasterize(((geometry, 1) for geometry in geometries), out_shape=shape, transform=transform, fill=0,
                     dtype="uint8").astype(bool)


def zones(codes, boundaries, protected, gold, shape, transform):
    """Per zone, the pixels and their municipality numbers (1 + position in `codes`); pixels by their centre."""
    index = rasterize(((geometry, code + 1) for geometry, code in zip(boundaries.geometry, codes)),
                      out_shape=shape, transform=transform, fill=0, dtype="int32").ravel()
    in_protected = _mask(protected.geometry, shape, transform).ravel() & (index > 0)
    in_gold = _mask(gold.geometry, shape, transform).ravel() & (index > 0)
    result = {}
    for suffix, mask in (("", None), ("_protected", in_protected), ("_gold5km", in_gold),
                         ("_gold5km_protected", in_protected & in_gold)):
        pixels = None if mask is None else np.flatnonzero(mask)
        result[suffix] = (pixels, index if pixels is None else index[pixels])
    return result


def _document_matches(path, document, year, variant):
    """Whether the original name in a file's TIFF tags names the same year and variant (files without it pass)."""
    if not document:
        logger.info("%s has no document name; identified by its file name.", path.name)
        return True
    parsed = parse_name(Path(document).name)
    if parsed is not None and parsed[:2] == (year, variant):
        return True
    logger.warning("%s holds %s; skipped.", path.name, Path(document).name)
    return False


def measure_file(path, year, variant, window, transform, n):
    """Zone sums of one file: lit pixels and radiance per zone (`average_masked`), or coverage (`cf_cvg`).

    Returns None when the file holds another year or variant than its name says."""
    with rasterio.open(f"/vsigzip/{path}") as source:
        if not _document_matches(path, source.tags().get(DOCUMENT_NAME_TAG), year, variant):
            return None
        if not source.transform.almost_equals(transform):
            raise ValueError(f"{path.name} is on another grid: {source.transform}")
        values = source.read(1, window=window).ravel()
    if variant == COVERAGE:
        pixels, ids = _ZONES[""]
        return {COVERAGE_COLUMN: np.bincount(ids, weights=values, minlength=n + 1)[1:]}
    values = np.clip(values, 0, None)
    result = {}
    for suffix, (pixels, ids) in _ZONES.items():
        zone = values if pixels is None else values[pixels]
        _, lit_column, radiance_column = zone_columns(suffix)
        result[lit_column] = np.bincount(ids, weights=zone > LIT_THRESHOLD, minlength=n + 1)[1:]
        result[radiance_column] = np.bincount(ids, weights=zone, minlength=n + 1)[1:]
    return result


def load_zone_inputs(root_dir=".", gadm_path=None, layer=None, adm2_id_column=None, protected_areas_path=None,
                     geology_gpkg=None):
    """Municipality polygons, the barred areas in force by 2012 and 5 km buffers of gold occurrences (EPSG:4326)."""
    boundaries = _load_adm2_boundaries(root_dir, gadm_path=gadm_path, layer=layer, adm2_id_column=adm2_id_column)
    boundaries[MUN_ID_COLUMN] = boundaries["adm2_id"].map(derive_mun_id_from_adm2_id)
    boundaries = boundaries.to_crs(4326)
    areas = load_protected_areas(root_dir, protected_areas_path)
    protected = areas[areas["mining_barred"] & areas["in_force_2012"]].to_crs(4326)
    gold = load_gold_occurrences(geology_gpkg or geology_gpkg_path(root_dir))
    buffers = gold.to_crs(BRAZIL_PROJECTED_CRS).buffer(GOLD_BUFFER_M).to_crs(4326)
    return boundaries[[MUN_ID_COLUMN, "geometry"]], protected, buffers.to_frame("geometry")


def build_nightlights(
    root_dir=".",
    raw=None,
    gadm_path=None,
    layer=None,
    adm2_id_column=None,
    protected_areas_path=None,
    geology_gpkg=None,
    output_path=None,
    workers=None,
):
    """Write night-light radiance, lit pixels and coverage per municipality, zone and year."""
    files = list_files(raw or raw_dir(root_dir))
    if files.empty:
        raise FileNotFoundError(f"No EOG VNL files under {raw or raw_dir(root_dir)}; they are placed by hand.")
    boundaries, protected, gold = load_zone_inputs(root_dir, gadm_path, layer, adm2_id_column, protected_areas_path,
                                                   geology_gpkg)
    codes, mun_ids = pd.factorize(boundaries[MUN_ID_COLUMN])

    with rasterio.open(f"/vsigzip/{files['path'].iloc[0]}") as source:
        grid, width, height = source.transform, source.width, source.height
    window = grid_window(grid, width, height, boundaries.total_bounds)
    transform = window_transform(window, grid)
    _ZONES.clear()
    _ZONES.update(zones(codes, boundaries, protected, gold, (int(window.height), int(window.width)), transform))
    n = len(mun_ids)
    logger.info("Window %s; %d municipalities; pixels in zones: %s", window, n,
                {suffix: len(ids) if pixels is not None else int((ids > 0).sum())
                 for suffix, (pixels, ids) in _ZONES.items()})

    workers = workers or int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork")) as pool:
        futures = {
            (row.year, row.variant): pool.submit(measure_file, row.path, row.year, row.variant, window, grid, n)
            for row in files.itertuples()
        }
        measured = {key: future.result() for key, future in futures.items()}

    pixels = {
        zone_columns(suffix)[0]: np.bincount(ids, minlength=n + 1)[1:]
        for suffix, (_, ids) in _ZONES.items()
    }
    frames = []
    for row in files[files["variant"] == AVERAGE].sort_values("year").itertuples():
        average = measured[(row.year, AVERAGE)]
        if average is None:
            continue
        coverage = measured.get((row.year, COVERAGE))
        frame = pd.DataFrame({MUN_ID_COLUMN: mun_ids, YEAR_COLUMN: row.year, VERSION_COLUMN: row.version,
                              SATELLITE_COLUMN: row.satellite, **pixels, **average})
        frame[COVERAGE_COLUMN] = np.nan
        if coverage is not None:
            frame[COVERAGE_COLUMN] = np.divide(coverage[COVERAGE_COLUMN], pixels["viirs_pixels"],
                                               out=np.full(n, np.nan), where=pixels["viirs_pixels"] > 0)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)[list(ADM2_COLUMNS)]
    count_columns = [column for suffix in ZONES for column in zone_columns(suffix)[:2]]
    result[count_columns] = result[count_columns].astype("int64")
    missing = sorted(set(files["year"]) - set(result.loc[result[COVERAGE_COLUMN].notna(), YEAR_COLUMN]))
    if missing:
        logger.warning("No coverage for years %s.", missing)

    resolved = Path(root_dir) / (output_path or DEFAULT_ADM2_OUTPUT_PATH)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(resolved, index=False)
    logger.info("Wrote night lights %s (%d-%d) to %s", result.shape, result[YEAR_COLUMN].min(),
                result[YEAR_COLUMN].max(), resolved)
    return result
