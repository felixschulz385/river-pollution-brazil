"""Reusable sanity checks and result dataclasses for source verification.

`CheckResult`/`VerificationResult` generalize `VerificationResult` from
`src/data/sources/climate/fetch/verify.py` (which checks raw GRIB batches at fetch
time) into a source-agnostic shape used to check preprocessed *outputs*.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CheckResult:
    name: str
    ok: bool
    message: str = ""


@dataclass
class VerificationResult:
    status: str  # "verified" | "failed" | "outstanding" | "not_present_locally"
    checks: list[CheckResult] = field(default_factory=list)
    fetch_completeness: dict | None = None

    @property
    def ok(self) -> bool:
        return self.status == "verified"


def check_required_columns(frame, required_columns, *, name: str = "required_columns") -> CheckResult:
    """Wrap `validate_required_columns` (assembly/land_cover schema) as a CheckResult."""
    from src.data.assembly.schema import validate_required_columns

    try:
        validate_required_columns(frame, required_columns, name)
    except ValueError as exc:
        return CheckResult(name=name, ok=False, message=str(exc))
    return CheckResult(
        name=name, ok=True, message=f"All {len(list(required_columns))} required columns present."
    )


def check_null_fraction(frame, column, *, max_null_fraction: float = 0.5, name: str | None = None) -> CheckResult:
    """Fail if `column`'s null share exceeds `max_null_fraction`."""
    name = name or f"null_fraction:{column}"
    if column not in frame.columns or frame.empty:
        return CheckResult(name=name, ok=False, message=f"Column '{column}' not present or frame is empty.")
    null_fraction = float(frame[column].isna().mean())
    ok = null_fraction <= max_null_fraction
    return CheckResult(
        name=name,
        ok=ok,
        message=f"{null_fraction:.2%} null (max allowed {max_null_fraction:.2%}).",
    )


def check_value_range(frame, column, *, lo: float, hi: float, name: str | None = None) -> CheckResult:
    """Fail if any observed value in `column` falls outside `[lo, hi]`."""
    name = name or f"value_range:{column}"
    if column not in frame.columns:
        return CheckResult(name=name, ok=False, message=f"Column '{column}' not present.")
    series = frame[column].dropna()
    if series.empty:
        return CheckResult(name=name, ok=False, message=f"Column '{column}' has no non-null values.")
    observed_min = float(series.min())
    observed_max = float(series.max())
    ok = observed_min >= lo and observed_max <= hi
    return CheckResult(
        name=name,
        ok=ok,
        message=f"Observed range [{observed_min}, {observed_max}], expected [{lo}, {hi}].",
    )


__all__ = [
    "CheckResult",
    "VerificationResult",
    "check_null_fraction",
    "check_required_columns",
    "check_value_range",
]
