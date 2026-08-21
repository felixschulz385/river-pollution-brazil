from __future__ import annotations

import pytest

from src.data.sources.land_cover.schema import normalize_optional_int


def test_normalize_optional_int_handles_null_and_numeric_scalars() -> None:
    assert normalize_optional_int(None) is None
    assert normalize_optional_int(float("nan")) is None
    assert normalize_optional_int("") is None
    assert normalize_optional_int(5) == 5
    assert normalize_optional_int("5") == 5


def test_normalize_optional_int_rejects_non_scalar_input() -> None:
    with pytest.raises(TypeError):
        normalize_optional_int([1, 2])
