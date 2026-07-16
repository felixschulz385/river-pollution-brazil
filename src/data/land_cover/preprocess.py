import logging
import os
from multiprocessing import cpu_count
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
from joblib import Parallel, delayed
from odc.geo.geom import Geometry

from .constants import (
    LAND_COVER_CLASS_PREFIX,
    LAND_COVER_TOTAL_COLUMN,
    TRENCH_ID_COLUMN,
    YEAR_COLUMN,
)
from .river_network_import import rn_module
from .schema import subclass_summary_id


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


def _build_year_multiindex(drainage_polygons, year):
    """Build the standard output index for one processed year."""
    return pd.MultiIndex.from_product(
        [_trench_ids(drainage_polygons), [year]],
        names=[TRENCH_ID_COLUMN, YEAR_COLUMN],
    )


def _class_column_name(class_id):
    """Return the canonical output column name for a land-cover class."""
    return f"{LAND_COVER_CLASS_PREFIX}{int(class_id)}"


def _class_column_names(class_ids):
    """Map numeric class identifiers to canonical output column names."""
    return [_class_column_name(class_id) for class_id in class_ids]


def _mapped_classes_and_weights(values, counts, mapper):
    """Return finite mapped class ids and aligned weights for aggregation."""
    mapped = np.asarray(mapper(values), dtype=float)
    valid = np.isfinite(mapped)
    if not np.any(valid):
        return np.asarray([], dtype=int), np.asarray([], dtype=counts.dtype)
    return mapped[valid].astype(int), counts[valid]


def deduplicate_drainage_polygons(drainage_polygons):
    """Keep the first drainage polygon for each trench_id."""
    if TRENCH_ID_COLUMN not in drainage_polygons.columns:
        raise ValueError(
            f"Drainage polygons must include `{TRENCH_ID_COLUMN}` as an explicit column."
        )
    drainage_polygons = drainage_polygons.drop_duplicates(
        subset=[TRENCH_ID_COLUMN],
        keep="first",
    )
    return drainage_polygons.set_index(TRENCH_ID_COLUMN, drop=False)


def add_crs(geom, crs=4326):
    return Geometry(geom, crs)


def create_mappers(legend_path):
    """Create vectorized mappers from legend file."""
    legend = pd.read_excel(legend_path)
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

    legend_class_dict_mapper = np.vectorize(lambda x: legend_class_dict.get(x, np.nan))
    legend_subclass_dict_mapper = np.vectorize(
        lambda x: (
            np.nan
            if (subclass_value := legend_subclass_dict.get(x, np.nan)) is None
            else subclass_value
        )
    )

    return legend_class_dict_mapper, legend_subclass_dict_mapper


def get_files(datadir):
    """Get sorted file list with year index."""
    files = pd.Series(os.listdir(datadir)).sort_values()
    years = files.str.extract(r"(\d{4,})").iloc[:, 0].astype("int")
    return files.set_axis(years)


def process_year(year, file, datadir, drainage_polygons, legend_path, output_columns, log_level=None):
    """Process all polygons for a single year."""
    if log_level is not None:
        configure_logging(log_level)
    logger.info("Processing year %s", year)

    try:
        legend_class_mapper, legend_subclass_mapper = create_mappers(legend_path)
        lc = rxr.open_rasterio(datadir + file, chunks=None)
    except Exception as e:
        logger.error("Failed to load raster for year %s: %s", year, e)
        mi = _build_year_multiindex(drainage_polygons, year)
        return pd.DataFrame(0, index=mi, columns=output_columns).reset_index()

    mi = _build_year_multiindex(drainage_polygons, year)
    year_df = pd.DataFrame(0, index=mi, columns=output_columns)

    n_polygons = len(drainage_polygons)
    n_success = 0
    n_no_overlap = 0
    n_errors = 0

    for idx, j in enumerate(drainage_polygons.index):
        try:
            geometry = drainage_polygons.loc[j, "geometry"]
            if geometry is None or geometry.is_empty:
                n_errors += 1
                continue

            try:
                cropped = lc.odc.crop(add_crs(geometry))
                values, counts = np.unique(cropped, return_counts=True)
                n_success += 1
            except (ValueError, RuntimeError, Exception) as e:
                error_msg = str(e)
                if "overlap" in error_msg.lower() or "extent" in error_msg.lower():
                    n_no_overlap += 1
                    if n_no_overlap <= 10:
                        logger.debug("Polygon %s does not overlap raster extent", j)
                else:
                    n_errors += 1
                    logger.warning("Error cropping polygon %s: %s", j, e)
                values, counts = np.array([]), np.array([])

            if len(values) > 0:
                year_df.loc[(j, year), LAND_COVER_TOTAL_COLUMN] = int(np.sum(counts))
                for mapper in (legend_class_mapper, legend_subclass_mapper):
                    classes, weights = _mapped_classes_and_weights(values, counts, mapper)
                    if len(classes) == 0:
                        continue

                    uniq, inv = np.unique(classes, return_inverse=True)
                    agg = np.bincount(inv, weights=weights)
                    year_df.loc[(j, year), _class_column_names(uniq)] = agg

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
            if "cannot convert float NaN to integer" in str(e):
                continue
            n_errors += 1
            logger.error("Unexpected error processing polygon %s in year %s: %s", j, year, e)

    logger.info(
        "Completed year %s: %s successful, %s no overlap, %s errors",
        year,
        n_success,
        n_no_overlap,
        n_errors,
    )
    return year_df.reset_index()


def preprocess_land_cover(self, n_jobs=None, river_network_path=None, output_path="data/land_cover_results.feather", log_level=None):
    """Preprocess land cover data by extracting values for drainage polygons."""
    if n_jobs is None:
        n_jobs = cpu_count()
    if log_level is not None:
        configure_logging(log_level)

    logger.info("Starting preprocessing with n_jobs=%s", n_jobs)

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
    else:
        logger.info("Loading drainage polygons from %s", self.drainage_path)
        drainage_polygons = deduplicate_drainage_polygons(
            gpd.read_feather(self.drainage_path)
        )
        logger.info("Loaded %d drainage polygons", len(drainage_polygons))

    files = get_files(self.datadir)
    logger.info(
        "Found %d land cover files for years %s-%s",
        len(files),
        files.index.min(),
        files.index.max(),
    )

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_year)(
            year,
            file,
            self.datadir,
            drainage_polygons,
            self.legend_path,
            self.output_columns,
            log_level,
        )
        for year, file in files.items()
    )

    logger.info("Merging all results...")
    final_df = pd.concat(results, ignore_index=True)
    final_df = final_df.sort_values([TRENCH_ID_COLUMN, YEAR_COLUMN]).reset_index(drop=True)

    output_path = Path(output_path)
    final_df.to_feather(output_path)
    logger.info("Results saved to %s", output_path)

    return final_df
