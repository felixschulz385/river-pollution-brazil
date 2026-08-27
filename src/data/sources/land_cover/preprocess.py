import logging
from multiprocessing import cpu_count
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
from joblib import Parallel, delayed

from .constants import (
    LAND_COVER_CLASS_PREFIX,
    LAND_COVER_TOTAL_COLUMN,
    TRENCH_ID_COLUMN,
    YEAR_COLUMN,
)
from src.data.sources import river_network as rn_module
from .schema import subclass_summary_id
from src.data.shared.spatial_tabular import (
    crop_unique_counts,
    deduplicate_drainage_polygons,
    is_extent_mismatch_error,
)


logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging for standalone land-cover execution."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _trench_ids(drainage_polygons):
    """Return trench identifiers for the deduplicated drainage polygons."""
    if TRENCH_ID_COLUMN not in drainage_polygons.columns:
        raise ValueError(
            f"Drainage polygons must include `{TRENCH_ID_COLUMN}` as an explicit column."
        )
    return drainage_polygons[TRENCH_ID_COLUMN].to_numpy()


def _class_column_name(class_id):
    """Return the canonical output column name for a land-cover class."""
    return f"{LAND_COVER_CLASS_PREFIX}{int(class_id)}"


def _mapped_classes_and_weights(values, counts, mapper):
    """Return finite mapped class ids and aligned weights for aggregation."""
    mapped = np.asarray(mapper(values), dtype=float)
    valid = np.isfinite(mapped)
    if not np.any(valid):
        return np.asarray([], dtype=int), np.asarray([], dtype=counts.dtype)
    return mapped[valid].astype(int), counts[valid]




def create_mappers(legend_path):
    """Create vectorized mappers from legend file."""
    legend = pd.read_excel(Path(legend_path))
    legend_class_dict = legend.set_index("ID").Class.to_dict()
    legend_subclass_dict = (
        legend.assign(
            _subclass_summary=legend.apply(
                lambda row: subclass_summary_id(row["Class"], row["Subclass"]),
                axis=1,
            )
        )
        .set_index("ID")["_subclass_summary"]
        .to_dict()
    )

    legend_class_dict_mapper = np.vectorize(
        lambda x: legend_class_dict.get(x, np.nan),
        otypes=[float],
    )
    legend_subclass_dict_mapper = np.vectorize(
        lambda x: (
            np.nan
            if (subclass_value := legend_subclass_dict.get(x, np.nan)) is None
            else subclass_value
        ),
        otypes=[float],
    )

    return legend_class_dict_mapper, legend_subclass_dict_mapper


# Plausible range for a MapBiomas collection year; used only to catch the
# extraction regex latching onto the wrong 4-digit token in a filename (e.g.
# a collection/version/tile id) rather than the actual year.
_MIN_PLAUSIBLE_YEAR = 1985
_MAX_PLAUSIBLE_YEAR = 2030


def get_files(datadir):
    """Return raster files keyed by year for the configured land-cover directory."""
    datadir = Path(datadir)
    files = sorted(path for path in datadir.iterdir() if path.suffix.lower() == ".tif")
    if not files:
        raise FileNotFoundError(f"No GeoTIFF files found in land-cover directory: {datadir}")

    file_series = pd.Series(files, dtype=object)
    extracted_years = file_series.astype(str).str.extract(r"_(\d{4})").iloc[:, 0]
    missing_years = file_series[extracted_years.isna()]
    if not missing_years.empty:
        raise ValueError(
            "Land-cover raster filenames must contain a four-digit year. "
            f"Invalid files: {[path.name for path in missing_years.tolist()[:10]]}"
        )

    years = extracted_years.astype(int)
    implausible = file_series[(years < _MIN_PLAUSIBLE_YEAR) | (years > _MAX_PLAUSIBLE_YEAR)]
    if not implausible.empty:
        raise ValueError(
            f"Extracted year outside the plausible range "
            f"[{_MIN_PLAUSIBLE_YEAR}, {_MAX_PLAUSIBLE_YEAR}]; the filename likely contains "
            f"another 4-digit token before the actual year. "
            f"Invalid files: {[path.name for path in implausible.tolist()[:10]]}"
        )

    return file_series.set_axis(years).sort_index()


def _empty_year_frame(trench_ids, year, output_columns):
    """Build the default all-zero result frame for one year."""
    data = np.zeros((len(trench_ids), len(output_columns)), dtype=np.int64)
    year_df = pd.DataFrame(data, columns=output_columns)
    year_df.insert(0, TRENCH_ID_COLUMN, trench_ids)
    year_df.insert(1, YEAR_COLUMN, year)
    return year_df


def _year_output_data(trench_ids, output_columns):
    """Allocate a writable dense output array for one processed year."""
    return np.zeros((len(trench_ids), len(output_columns)), dtype=np.int64)


def _accumulate_mapped_counts(
    row_data,
    values,
    counts,
    legend_mappers,
    column_positions,
):
    """Populate one output row from raw value counts.

    ``TOTAL`` is derived from the class-level mapper's own valid mask (the
    first entry in ``legend_mappers``) rather than from the raw pixel count,
    so it stays consistent with the class columns even if the raster's
    nodata pixels (e.g. MapBiomas class 0) aren't flagged as nodata in the
    GeoTIFF metadata and therefore survive as finite values.
    """
    if len(values) == 0:
        return

    total = 0
    for position, mapper in enumerate(legend_mappers):
        classes, weights = _mapped_classes_and_weights(values, counts, mapper)
        if len(classes) == 0:
            continue

        if position == 0:
            total = int(np.sum(weights))

        uniq, inv = np.unique(classes, return_inverse=True)
        agg = np.bincount(inv, weights=weights).astype(np.int64, copy=False)

        for class_id, count in zip(uniq, agg, strict=False):
            column_name = _class_column_name(class_id)
            if column_name in column_positions:
                row_data[column_positions[column_name]] = count

    row_data[column_positions[LAND_COVER_TOTAL_COLUMN]] = total


def _build_row_data(lc, geometry, legend_mappers, column_positions, n_columns):
    """Build one dense output row for a polygon."""
    row_data = np.zeros(n_columns, dtype=np.int64)
    values, counts = crop_unique_counts(lc, geometry)
    _accumulate_mapped_counts(
        row_data,
        values,
        counts,
        legend_mappers,
        column_positions,
    )
    return row_data


def _read_drainage_polygons(path):
    """Read drainage polygons from parquet or feather based on suffix."""
    path = Path(path)
    if path.suffix == ".parquet":
        return gpd.read_parquet(path)
    if path.suffix == ".feather":
        return gpd.read_feather(path)
    raise ValueError(
        f"Unsupported drainage polygon format: {path}. Expected .parquet or .feather."
    )


def process_year(
    year,
    file,
    polygon_items,
    output_columns,
    legend_mappers,
    log_level=None,
):
    """Process all polygons for a single year."""
    if log_level is not None:
        configure_logging(log_level)
    logger.info("Processing year %s", year)

    trench_ids = np.asarray([int(trench_id) for trench_id, _ in polygon_items], dtype=np.int64)
    output_data = _year_output_data(trench_ids, output_columns)
    column_positions = {column: idx for idx, column in enumerate(output_columns)}

    # Deliberately NOT caught here: a raster that fails to open (missing
    # file, corrupt GeoTIFF, transient disk/network hiccup on a SLURM node)
    # must abort this year rather than fall through to an all-zero
    # `output_data`, which downstream is indistinguishable from a year where
    # every drainage polygon genuinely has zero overlap. Let it propagate so
    # `Parallel` surfaces it as a failed preprocessing run instead of baking
    # a silently wrong "zero land cover" year into the output.
    raster_path = Path(file)
    with rxr.open_rasterio(raster_path, chunks=None, masked=True) as raster:
        lc = raster.squeeze(drop=True)

        n_polygons = len(polygon_items)
        n_success = 0
        n_no_overlap = 0
        n_errors = 0

        for idx, (trench_id, geometry) in enumerate(polygon_items):
            try:
                if geometry is None or geometry.is_empty:
                    n_errors += 1
                    continue

                try:
                    output_data[idx] = _build_row_data(
                        lc,
                        geometry,
                        legend_mappers,
                        column_positions,
                        len(output_columns),
                    )
                    n_success += 1
                except Exception as e:
                    if is_extent_mismatch_error(e):
                        n_no_overlap += 1
                        if n_no_overlap <= 10:
                            logger.debug(
                                "Polygon %s does not overlap raster extent",
                                trench_id,
                            )
                    else:
                        n_errors += 1
                        logger.warning("Error cropping polygon %s: %s", trench_id, e)
                    values, counts = np.array([]), np.array([])

                if (idx + 1) % 100000 == 0:
                    logger.info(
                        "Year %s: processed %s/%s polygons (success: %s, no_overlap: %s, errors: %s)",
                        year,
                        idx + 1,
                        n_polygons,
                        n_success,
                        n_no_overlap,
                        n_errors,
                    )

            except Exception as e:
                n_errors += 1
                logger.error(
                    "Unexpected error processing polygon %s in year %s: %s",
                    trench_id,
                    year,
                    e,
                )

        logger.info(
            "Completed year %s: %s successful, %s no overlap, %s errors",
            year,
            n_success,
            n_no_overlap,
            n_errors,
        )
    year_df = pd.DataFrame(output_data, columns=output_columns)
    year_df.insert(0, TRENCH_ID_COLUMN, trench_ids)
    year_df.insert(1, YEAR_COLUMN, year)
    return year_df


def _load_drainage_polygons(drainage_path, river_network_path):
    """Load and normalize preprocessing polygons from the selected source."""
    if river_network_path:
        logger.info("Loading river network from %s", river_network_path)
        network = rn_module.RiverNetwork()
        network.load(river_network_path)

        if network.drainage_areas is None:
            raise ValueError("River network does not have drainage_areas loaded")

        drainage_polygons = network.drainage_areas.to_crs(4326)
        logger.info("Loaded %d drainage areas from network", len(drainage_polygons))

        if "within_brazil" not in drainage_polygons.columns:
            raise ValueError(
                "Drainage areas missing 'within_brazil' column. "
                "Run river-network generate with --gadm-path to annotate this column."
            )

        n_before = len(drainage_polygons)
        drainage_polygons = drainage_polygons[drainage_polygons["within_brazil"]]
        logger.info(
            "Filtered to %d/%d drainage areas within Brazil",
            len(drainage_polygons),
            n_before,
        )
        return drainage_polygons

    logger.info("Loading drainage polygons from %s", drainage_path)
    drainage_polygons = deduplicate_drainage_polygons(
        _read_drainage_polygons(drainage_path)
    )
    logger.info("Loaded %d drainage polygons", len(drainage_polygons))
    return drainage_polygons


def preprocess_land_cover(self, n_jobs=None, river_network_path=None, output_path="data/land_cover/processed/extract/land_cover.parquet", log_level=None):
    """Preprocess land cover data by extracting values for drainage polygons."""
    if n_jobs is None:
        n_jobs = cpu_count()
    if log_level is not None:
        configure_logging(log_level)

    logger.info("Starting preprocessing with n_jobs=%s", n_jobs)

    drainage_polygons = _load_drainage_polygons(self.drainage_path, river_network_path)

    files = get_files(self.datadir)
    logger.info(
        "Found %d land cover files for years %s-%s",
        len(files),
        files.index.min(),
        files.index.max(),
    )
    polygon_items = list(
        zip(
            _trench_ids(drainage_polygons).tolist(),
            drainage_polygons.geometry.to_numpy(),
            strict=False,
        )
    )
    legend_mappers = create_mappers(self.legend_path)

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_year)(
            year,
            file,
            polygon_items,
            self.output_columns,
            legend_mappers,
            log_level,
        )
        for year, file in files.items()
    )

    logger.info("Merging all results...")
    final_df = pd.concat(results, ignore_index=True)
    final_df = final_df.sort_values([TRENCH_ID_COLUMN, YEAR_COLUMN]).reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".feather":
        final_df.to_feather(output_path)
    else:
        final_df.to_parquet(output_path, index=False)
    logger.info("Results saved to %s", output_path)

    return final_df
