from pathlib import Path
import zipfile

from .common import create_cds_client, should_skip_download, write_download_manifest


CLOUD_FRACTION_YEARS = [str(year) for year in range(1979, 2021)]
CLOUD_FRACTION_MONTHS = [f"{month:02d}" for month in range(1, 13)]


def _cloud_cover_archive_path(root_dir="."):
    return Path(root_dir) / "data" / "climate" / "raw" / "cloud_cover_clara_a3.zip"


def fetch_cloud_cover(root_dir="."):
    """
    Fetch Copernicus monthly cloud-fraction data and extract the downloaded archive.

    Note:
        This requires a Copernicus Climate Data Store account plus a configured
        local CDS API key.
    """
    archive_path = _cloud_cover_archive_path(root_dir)
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    request = {
        "format": "zip",
        "product_family": "clara_a3",
        "origin": "eumetsat",
        "variable": "cloud_fraction",
        "climate_data_record_type": "thematic_climate_data_record",
        "time_aggregation": "monthly_mean",
        "year": CLOUD_FRACTION_YEARS,
        "month": CLOUD_FRACTION_MONTHS,
    }

    if not should_skip_download(archive_path):
        client = create_cds_client(root_dir=root_dir)
        write_download_manifest(
            archive_path,
            dataset="satellite-cloud-properties",
            request=request,
            status="downloading",
        )
        try:
            client.retrieve(
                "satellite-cloud-properties",
                request,
                str(archive_path),
            )
        except Exception as exc:
            write_download_manifest(
                archive_path,
                dataset="satellite-cloud-properties",
                request=request,
                status="failed",
                error=str(exc),
            )
            raise
        write_download_manifest(
            archive_path,
            dataset="satellite-cloud-properties",
            request=request,
            status="downloaded",
        )

    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(archive_path.parent)

    return archive_path
