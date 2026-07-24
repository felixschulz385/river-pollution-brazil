from __future__ import annotations

from pathlib import Path

from src.data.land_cover.core import LandCover
from src.data.land_cover.preprocess import get_files


def test_get_files_supports_mapbiomas_collection_10_naming(tmp_path: Path):
    for name in [
        "brazil_coverage_2024.tif",
        "brazil_coverage_1985.tif",
        "README.txt",
    ]:
        (tmp_path / name).touch()

    files = get_files(tmp_path)

    assert list(files.index) == [1985, 2024]
    assert [path.name for path in files.tolist()] == [
        "brazil_coverage_1985.tif",
        "brazil_coverage_2024.tif",
    ]


def test_land_cover_initializes_output_columns_from_selected_legend(monkeypatch):
    captured = {}

    def fake_get_output_columns(legend_path):
        captured["legend_path"] = legend_path
        return ["land_cover_total", "land_cover_class_1"]

    monkeypatch.setattr("src.data.land_cover.core.get_output_columns", fake_get_output_columns)

    land_cover = LandCover(legend_path=Path("/tmp/custom_legend.xlsx"))

    assert land_cover.legend_path == Path("/tmp/custom_legend.xlsx")
    assert captured["legend_path"] == Path("/tmp/custom_legend.xlsx")
    assert land_cover.output_columns == ["land_cover_total", "land_cover_class_1"]
