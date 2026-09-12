from __future__ import annotations

import pandas as pd

from src.data.sources.land_cover.constants import non_municipality_adm2_mask


def test_non_municipality_adm2_mask_matches_string_cc2():
    cc2 = pd.Series(["4314902", "4300001", "4300002"])
    assert non_municipality_adm2_mask(cc2).tolist() == [False, True, True]


def test_non_municipality_adm2_mask_matches_float_typed_cc2():
    """Some GeoPackage/driver combinations round-trip CC_2 as a float OGR
    field (4300001.0) rather than text; a bare `.astype(str)` would produce
    "4300001.0" and silently never match the plain-string excluded codes."""
    cc2 = pd.Series([4314902.0, 4300001.0, 4300002.0])
    assert non_municipality_adm2_mask(cc2).tolist() == [False, True, True]
