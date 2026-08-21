from ..constants import BRAZIL_UTC_OFFSET_HOURS, ERA5_LAND_SUBTYPE_DATASETS
from .common import ERA5_AREA, days_in_month, retrieve_yearly_dataset_in_monthly_batches
from .verify import verify_era5_grib_batch


DATASET = ERA5_LAND_SUBTYPE_DATASETS["era5_land_daily"]
VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
]
# GRIB short names for the variables above, as read back by _open_era5_dataset.
VERIFICATION_BANDS = ["2t", "2d", "swvl1", "swvl2"]


def build_era5_land_daily_request(year, month):
    return {
        "variable": VARIABLES,
        "year": year,
        "month": month,
        "day": days_in_month(year, month),
        "daily_statistic": "daily_mean",
        "frequency": "1_hourly",
        "time_zone": f"utc{BRAZIL_UTC_OFFSET_HOURS:+03d}:00",
        "area": ERA5_AREA,
    }


def fetch_era5_land_daily(root_dir="."):
    return retrieve_yearly_dataset_in_monthly_batches(
        root_dir=root_dir,
        dataset=DATASET,
        request_factory=build_era5_land_daily_request,
        output_subdir="era5_land_daily",
        file_prefix="era5_land_daily",
        max_running_remote_requests=1,
        verify_batch=lambda path: verify_era5_grib_batch(path, bands=VERIFICATION_BANDS),
    )
