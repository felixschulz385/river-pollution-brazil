from __future__ import annotations

from .common import load_cds_credentials

ARCO_BASE_URL = (
    "https://arco.datastores.ecmwf.int/cadl-arco-{chunks}-{group_id}/"
    "arco/reanalysis_era5_land/{group}/{chunks}Chunked.zarr"
)

# Only the ARCO variable groups this project needs. surface_runoff,
# sub_surface_runoff, and potential_evaporation are not offered by ARCO at
# all (confirmed against CDS's own variable tables) and stay on the GRIB
# job-submission path in fetch/era5_land_hourly.py.
ARCO_GROUPS = {
    "sfc-2m-temperature": {"id": "007", "vars": ["t2m", "d2m"]},
    "sfc-soil-water": {"id": "005", "vars": ["swvl1", "swvl2"]},
    "sfc-pressure-precipitation": {"id": "009", "vars": ["tp"]},
}


def arco_store_url(group: str, chunks: str = "geo") -> str:
    if group not in ARCO_GROUPS:
        raise ValueError(f"Unsupported ARCO group: {group!r}")
    return ARCO_BASE_URL.format(chunks=chunks, group_id=ARCO_GROUPS[group]["id"], group=group)


def open_arco_group_dataset(group: str, *, root_dir=".", chunks: str = "geo"):
    import xarray as xr

    key = load_cds_credentials(root_dir=root_dir)["key"]
    return xr.open_zarr(
        arco_store_url(group, chunks=chunks),
        consolidated=True,
        storage_options={"headers": {"Authorization": f"Bearer {key}"}},
    )


def fetch_era5_land_arco(root_dir=".", **kwargs):
    """ARCO is a live, always-queryable Zarr store, so there's no separate
    download step - opening it, slicing to our area, aggregating hourly to
    daily, and writing into the local store all happen here under `fetch`
    rather than `preprocess` (unlike the GRIB path, which genuinely has two
    distinct stages)."""
    from ..preprocess.era5_land_arco import preprocess_era5_land_arco

    return preprocess_era5_land_arco(root_dir=root_dir, **kwargs)
