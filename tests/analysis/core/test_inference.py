"""Tests for the FE-demeaned wild-cluster bootstrap in src.analysis.core.inference."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pf = pytest.importorskip("pyfixest")
pytest.importorskip("wildboottest")

from src.analysis.core.inference import poisson_score_bootstrap, wild_cluster_bootstrap


def _panel(effect=0.0, n_basins=30, stations_per_basin=4, samples=12, seed=0):
    """Stations nested in basins, with station FE and a basin-correlated error."""
    rng = np.random.default_rng(seed)
    rows = []
    for basin in range(n_basins):
        basin_shock = rng.normal()
        for station in range(stations_per_basin):
            station_effect = rng.normal()
            for t in range(samples):
                x = rng.normal() + 0.5 * basin_shock
                y = effect * x + station_effect + basin_shock + rng.normal()
                rows.append((f"b{basin:02d}", basin, f"b{basin:02d}s{station}", t, x, rng.normal(), y))
    return pd.DataFrame(rows, columns=["basin", "basin_id", "station", "t", "x", "z", "y"])


def test_matches_pyfixest_one_hot_bootstrap():
    # With few FE, pyfixest's own (one-hot) wildboottest is feasible; the
    # demeaned implementation must give the same t statistic and p-value.
    # pyfixest passes cluster ids to wildboottest's numba kernels unconverted,
    # so its reference run needs the integer copy of the (string) basin code.
    data = _panel(effect=0.1)
    model = pf.feols("y ~ x + z | station + t", data=data, vcov={"CRV1": "basin"})

    ours = wild_cluster_bootstrap(model, "x", cluster="basin", reps=1999, seed=3)
    reference = model.wildboottest(param="x", reps=1999, cluster="basin_id", seed=3)

    assert ours["n_clusters"] == 30
    assert ours["weights"] == "rademacher"
    # The bootstrap p-value is invariant to the small-sample adjustment...
    assert ours["p_wcb"] == pytest.approx(float(reference["Pr(>|t|)"]), abs=0.005)
    # ...while the t statistic differs exactly by it: the one-hot design counts
    # every FE dummy in k, the demeaned design only the regressors.
    n_obs = model._N
    k_one_hot = model._model_matrix_one_hot()[1].shape[1]
    k_demeaned = model._X.shape[1]
    dof_ratio = np.sqrt((n_obs - k_one_hot) / (n_obs - k_demeaned))
    assert ours["t_wcb"] * dof_ratio == pytest.approx(float(reference["t value"]), rel=1e-3)


def test_rejects_a_true_effect_and_not_a_null_one():
    strong = pf.feols("y ~ x + z | station + t", data=_panel(effect=1.0), vcov={"CRV1": "basin"})
    assert wild_cluster_bootstrap(strong, "x", cluster="basin", reps=999)["p_wcb"] < 0.01

    null = pf.feols("y ~ x + z | station + t", data=_panel(effect=1.0), vcov={"CRV1": "basin"})
    assert wild_cluster_bootstrap(null, "z", cluster="basin", reps=999)["p_wcb"] > 0.01


def test_uses_webb_weights_with_few_clusters():
    data = _panel(n_basins=8)
    model = pf.feols("y ~ x + z | station", data=data, vcov={"CRV1": "basin"})
    assert wild_cluster_bootstrap(model, "x", cluster="basin", reps=999)["weights"] == "webb"


def _count_panel(effect=0.0, n_basins=40, units_per_basin=4, periods=10, seed=0):
    """Poisson counts with unit FE, an exposure offset and a basin-correlated shock."""
    rng = np.random.default_rng(seed)
    rows = []
    for basin in range(n_basins):
        basin_shock = rng.normal(scale=0.3)
        for unit in range(units_per_basin):
            unit_effect = rng.normal(scale=0.5)
            for t in range(periods):
                x = rng.normal() + basin_shock
                z = rng.normal()
                log_exposure = np.log(rng.integers(500, 5000))
                mean = np.exp(-4 + effect * x + 0.2 * z + unit_effect + 0.5 * basin_shock + log_exposure)
                rows.append((f"b{basin:02d}", f"b{basin:02d}u{unit}", t, x, z, log_exposure, rng.poisson(mean)))
    return pd.DataFrame(rows, columns=["basin", "unit", "t", "x", "z", "log_exposure", "y"])


def _fit_poisson(data):
    return pf.fepois("y ~ x + z | unit + t", data=data, offset="log_exposure", vcov={"CRV1": "basin"})


def test_poisson_score_bootstrap_rejects_a_real_effect():
    result = poisson_score_bootstrap(_fit_poisson(_count_panel(effect=0.3)), "x", cluster="basin", reps=1999, seed=1)

    assert result["n_clusters"] == 40
    assert result["weights"] == "rademacher"
    assert result["score_stat"] > 0
    assert result["p_wcb"] < 0.01


def test_poisson_score_bootstrap_does_not_reject_under_the_null():
    # Over 200 simulated null panels the test rejected 6% at the 5% level
    # (analytic CRV1: 8%); this panel is one that does not reject.
    model = _fit_poisson(_count_panel(effect=0.0, seed=1))

    result = poisson_score_bootstrap(model, "x", cluster="basin", reps=1999, seed=1)

    assert result["p_wcb"] > 0.05
    # Same seed, same draws.
    assert poisson_score_bootstrap(model, "x", cluster="basin", reps=1999, seed=1)["p_wcb"] == result["p_wcb"]


def test_poisson_score_bootstrap_uses_webb_weights_with_few_clusters():
    model = _fit_poisson(_count_panel(effect=0.0, n_basins=8))

    assert poisson_score_bootstrap(model, "x", cluster="basin", reps=999)["weights"] == "webb"


def test_poisson_score_bootstrap_rejects_unknown_parameter():
    with pytest.raises(ValueError, match="not a regressor"):
        poisson_score_bootstrap(_fit_poisson(_count_panel()), "w", cluster="basin", reps=99)
