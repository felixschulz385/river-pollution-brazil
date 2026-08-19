from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# Bounds are in the raw GRIB units (see ERA5L_VAR_CONFIG's units_in in
# preprocess/era5_land.py): metres for accumulations, Kelvin for temperature,
# m3/m3 for soil water content.
ERA5L_VALUE_RANGES = {
    "tp": (0.0, 0.5),
    "sro": (-0.01, 0.5),
    "ssro": (-0.01, 0.5),
    # Potential evaporation is an accumulated hourly-forecast field; observed
    # minimums in this domain (hot/dry regions of Brazil) range down to ~-0.34,
    # well past an initial -0.1 guess that wasn't calibrated against real data.
    # Keep a wide margin below the worst observed value.
    "pev": (-0.5, 0.05),
    "2t": (180.0, 340.0),
    "2d": (180.0, 340.0),
    "swvl1": (0.0, 1.0),
    "swvl2": (0.0, 1.0),
}

# ERA5_AREA covers a bounding box around Brazil that includes a substantial
# stretch of Atlantic Ocean (ERA5-Land is a land-only dataset, so ocean cells
# are legitimately null every batch, not corrupted). Observed ocean coverage
# for this box is ~26% consistently across sro/ssro/pev; keep a wide margin
# above that so real ocean masking never trips verification, while still
# catching wholesale data loss (e.g. a mostly- or fully-empty field).
MAX_NULL_FRACTION = 0.5


@dataclass
class VerificationResult:
    ok: bool
    errors: list = field(default_factory=list)


def verify_era5_grib_batch(path: Path, *, bands) -> VerificationResult:
    from ..preprocess.era5_land import _open_era5_dataset

    errors = []
    try:
        dataset = _open_era5_dataset(Path(path))
    except Exception as exc:
        return VerificationResult(ok=False, errors=[f"Could not open {path.name} for verification: {exc}"])

    try:
        for band in bands:
            if band not in dataset.data_vars:
                errors.append(f"{band}: missing from {path.name}.")
                continue

            data_array = dataset[band]
            null_fraction = float(data_array.isnull().mean())
            if null_fraction > MAX_NULL_FRACTION:
                errors.append(
                    f"{band}: {null_fraction:.4%} of values are null "
                    f"(max allowed {MAX_NULL_FRACTION:.4%})."
                )
                continue

            lo, hi = ERA5L_VALUE_RANGES[band]
            observed_min = float(data_array.min())
            observed_max = float(data_array.max())
            if observed_min < lo or observed_max > hi:
                errors.append(
                    f"{band}: observed range [{observed_min}, {observed_max}] "
                    f"is outside the reasonable range [{lo}, {hi}]."
                )
    finally:
        dataset.close()

    return VerificationResult(ok=not errors, errors=errors)
