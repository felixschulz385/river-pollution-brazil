"""Sensor-data analysis API."""

from .groups import list_groups
from .prepare import build_analysis_data
from .runner import merge_suite, run_suite
from .specs import build_model_specs

__all__ = [
    "build_analysis_data",
    "build_model_specs",
    "faceted_distance_coefplot",
    "list_groups",
    "merge_suite",
    "run_plotly_app",
    "run_suite",
]


def faceted_distance_coefplot(*args, **kwargs):
    """Import plotting lazily so analysis code does not require matplotlib on import."""
    from .plots import faceted_distance_coefplot as _faceted_distance_coefplot

    return _faceted_distance_coefplot(*args, **kwargs)


def run_plotly_app(*args, **kwargs):
    """Import the Plotly app lazily so CLI usage stays optional."""
    from .plotly_app import run_plotly_app as _run_plotly_app

    return _run_plotly_app(*args, **kwargs)
