from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

from src.data.sources.river_network.__main__ import configure_parser, run


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    configure_parser(parser, include_action=False)
    return parser.parse_args(argv)


def test_run_builds_bbox_when_a_bound_is_exactly_zero() -> None:
    # A bbox spanning the equator (or the prime meridian) legitimately has a
    # bound of exactly 0.0. Using truthiness on the parsed args would treat
    # that 0.0 as "missing" and silently drop the whole bbox, loading the
    # unfiltered dataset instead.
    args = _parse_args(
        [
            "--gpkg-path",
            "network.gpkg",
            "--min-lon",
            "-60.0",
            "--min-lat",
            "0",
            "--max-lon",
            "-40.0",
            "--max-lat",
            "5.0",
        ]
    )

    with patch(
        "src.data.sources.river_network.__main__.RiverNetwork"
    ) as mock_network_cls:
        mock_network = MagicMock()
        mock_network_cls.return_value = mock_network

        run(args)

        _, kwargs = mock_network.generate.call_args
        assert kwargs["bbox"] is not None
        bounds = kwargs["bbox"].total_bounds
        assert list(bounds) == [-60.0, 0.0, -40.0, 5.0]


def test_run_leaves_bbox_none_when_a_bound_is_missing() -> None:
    args = _parse_args(
        [
            "--gpkg-path",
            "network.gpkg",
            "--min-lon",
            "-60.0",
            "--min-lat",
            "0",
            "--max-lon",
            "-40.0",
        ]
    )

    with patch(
        "src.data.sources.river_network.__main__.RiverNetwork"
    ) as mock_network_cls:
        mock_network = MagicMock()
        mock_network_cls.return_value = mock_network

        run(args)

        _, kwargs = mock_network.generate.call_args
        assert kwargs["bbox"] is None
