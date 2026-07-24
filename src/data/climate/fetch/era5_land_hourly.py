from .common import (
    ERA5_AREA,
    ERA5_DAYS,
    ERA5_HOURS,
    retrieve_yearly_dataset_in_monthly_batches,
)


DATASET = "reanalysis-era5-land"
VARIABLES = [
    "total_precipitation",
    "surface_runoff",
    "sub_surface_runoff",
    "potential_evaporation",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
]


def build_era5_land_hourly_request(year, month):
    return {
        "variable": VARIABLES,
        "year": [year],
        "month": [month],
        "day": ERA5_DAYS,
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
    )
