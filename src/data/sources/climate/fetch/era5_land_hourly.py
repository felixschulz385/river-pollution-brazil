from ..constants import ERA5_LAND_SUBTYPE_DATASETS
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


def fetch_era5_land_hourly(root_dir="."):
    return retrieve_yearly_dataset_in_monthly_batches(
        root_dir=root_dir,
        dataset=DATASET,
        request_factory=build_era5_land_hourly_request,
        output_subdir="era5_land_hourly",
        file_prefix="era5_land_hourly",
        max_running_remote_requests=1,
        verify_batch=lambda path: verify_era5_grib_batch(path, bands=VERIFICATION_BANDS),
    )
