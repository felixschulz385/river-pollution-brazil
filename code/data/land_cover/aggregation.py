import warnings

from .assembly import assemble_land_cover
from .constants import (
    ADM2_ASSEMBLY_VARIANT,
    DEFAULT_ADM2_UPSTREAM_OUTPUT_PATH,
    DEFAULT_ASSEMBLY_LAND_COVER_PATH,
    DEFAULT_RIVER_NETWORK_PATH,
)


def aggregate_along_rivers(
    self,
    land_cover_path=DEFAULT_ASSEMBLY_LAND_COVER_PATH,
    river_network_path=DEFAULT_RIVER_NETWORK_PATH,
    drainage_polygons_path=None,
    kernel=None,
    h=None,
    years=None,
    n_jobs=None,
    output_path=DEFAULT_ADM2_UPSTREAM_OUTPUT_PATH,
):
    """Backward-compatible wrapper around the ADM2 assembly variant."""
    ignored_arguments = {
        "drainage_polygons_path": drainage_polygons_path,
        "kernel": kernel,
        "h": h,
        "years": years,
    }
    provided_ignored_arguments = {
        name: value for name, value in ignored_arguments.items() if value is not None
    }
    if provided_ignored_arguments:
        warnings.warn(
            "aggregate_along_rivers() now delegates to the bucket-based `adm2` "
            "assembly variant. The following legacy arguments are ignored: "
            f"{sorted(provided_ignored_arguments)}.",
            DeprecationWarning,
            stacklevel=2,
        )

    return assemble_land_cover(
        self,
        variant=ADM2_ASSEMBLY_VARIANT,
        land_cover_path=land_cover_path,
        river_network_path=river_network_path,
        output_path=output_path,
        n_jobs=n_jobs,
    )
