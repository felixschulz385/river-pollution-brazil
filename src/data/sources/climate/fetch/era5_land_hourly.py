from ..constants import BOUNDARY_HOURS, ERA5_LAND_SUBTYPE_DATASETS
from .common import (
    ERA5_AREA,
    ERA5_HOURS,
    days_in_month,
    retrieve_yearly_dataset_in_monthly_batches,
)
from .verify import verify_era5_grib_batch


DATASET = ERA5_LAND_SUBTYPE_DATASETS["era5_land_hourly"]
# total_precipitation, 2m_temperature, 2m_dewpoint_temperature, and the two
# volumetric_soil_water_layer variables are sourced from CDS's ARCO Zarr store
# instead (see fetch/era5_land_arco.py + preprocess/era5_land_arco.py) - they
# aren't offered there, so surface_runoff/sub_surface_runoff/potential_evaporation
# still go through this GRIB job-submission path.
VARIABLES = [
    "surface_runoff",
    "sub_surface_runoff",
    "potential_evaporation",
]
# GRIB short names for the variables above, as read back by _open_era5_dataset.
VERIFICATION_BANDS = ["sro", "ssro", "pev"]


def build_era5_land_hourly_request(year, month):
    return {
        "variable": VARIABLES,
        "year": [year],
        "month": [month],
        "day": days_in_month(year, month),
        "time": ERA5_HOURS,
        "area": ERA5_AREA,
        "data_format": "grib",
    }


def _next_year_month(year, month) -> tuple[str, str]:
    year, month = int(year), int(month)
    return (str(year + 1), "01") if month == 12 else (str(year), f"{month + 1:02d}")


def build_era5_land_hourly_boundary_request(year, month):
    """Request `(year, month)`'s "boundary" hours: the first `BOUNDARY_HOURS`
    UTC hours of the *following* month.

    `preprocess.era5_land.resample_era5l_hourly_to_daily` buckets by
    Brazil-local calendar day, which shifts every timestamp back
    `BRAZIL_UTC_OFFSET_HOURS` hours before flooring to a date -- so
    `(year, month)`'s own file is short its last local day by exactly these
    hours (they live in the following month's file). Fetched as a separate,
    tiny request (`BOUNDARY_HOURS` timesteps vs. ~720 for a full month)
    rather than widening the main request, since CDS's year/month/day/time
    request format can't mix days from two different months in one request.
    """
    next_year, next_month = _next_year_month(year, month)
    return {
        "variable": VARIABLES,
        "year": [next_year],
        "month": [next_month],
        "day": ["01"],
        "time": ERA5_HOURS[:BOUNDARY_HOURS],
        "area": ERA5_AREA,
        "data_format": "grib",
    }


def fetch_era5_land_hourly(root_dir="."):
    retrieve_yearly_dataset_in_monthly_batches(
        root_dir=root_dir,
        dataset=DATASET,
        request_factory=build_era5_land_hourly_boundary_request,
        output_subdir="era5_land_hourly",
        file_prefix="era5_land_hourly_boundary",
        max_running_remote_requests=1,
        verify_batch=lambda path: verify_era5_grib_batch(path, bands=VERIFICATION_BANDS),
    )
    return retrieve_yearly_dataset_in_monthly_batches(
        root_dir=root_dir,
        dataset=DATASET,
        request_factory=build_era5_land_hourly_request,
        output_subdir="era5_land_hourly",
        file_prefix="era5_land_hourly",
        max_running_remote_requests=1,
        verify_batch=lambda path: verify_era5_grib_batch(path, bands=VERIFICATION_BANDS),
    )
