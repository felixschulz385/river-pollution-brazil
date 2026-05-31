#!/usr/bin/env python3

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import urlretrieve

BASE_URL = (
    "https://storage.googleapis.com/mapbiomas-public/"
    "initiatives/brasil/collection_10/lulc/coverage"
)
YEARS = range(1985, 2025)

LOCAL_DIR = Path(
    "/Users/felixschulz/Downloads/mapbiomas_land_cover"
)
REMOTE_USER = "schulz0022"
REMOTE_HOST = "transfer12.scicore.unibas.ch"
SSH_KEY = Path("/Users/felixschulz/.ssh/id_ed25519_scicore")
REMOTE_DIR = (
    "/scicore/home/meiera/schulz0022/projects/river-pollution-brazil/"
    "data/land_cover/raw/lc_mapbiomas10_30"
)


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def download_file(year: int, local_path: Path) -> None:
    url = f"{BASE_URL}/brazil_coverage_{year}.tif"
    print(f"Downloading {url}")
    urlretrieve(url, local_path)


def upload_file(local_path: Path) -> None:
    remote_target = f"{REMOTE_USER}@{REMOTE_HOST}:{REMOTE_DIR}"
    run(
        [
            "scp",
            "-i",
            str(SSH_KEY),
            str(local_path),
            remote_target,
        ]
    )


def main() -> int:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        for year in YEARS:
            filename = f"brazil_coverage_{year}.tif"
            local_path = LOCAL_DIR / filename

            download_file(year, local_path)
            upload_file(local_path)
            local_path.unlink()
            print(f"Uploaded and removed local file: {filename}")

    except Exception as exc:
        print(f"Failed: {exc}", file=sys.stderr)
        return 1

    try:
        shutil.rmtree(LOCAL_DIR)
    except OSError:
        pass

    print("All files transferred successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
