"""Wild-cluster bootstrap inference for high-dimensional fixed-effect models.

`pyfixest`'s own `Feols.wildboottest` builds a dense one-hot matrix of every
fixed effect, which is infeasible for the sensor panel (~5,700 station FE on
~150k samples). By Frisch-Waugh-Lovell, the coefficients and the null-restricted
residuals of the FE model equal those of OLS on the FE-demeaned outcome and
regressors, which pyfixest already stores on the fit (`_Y`, `_X`). The
bootstrap here runs `wildboottest`'s WCR procedure on those demeaned arrays.
This is exact when the FE are nested within clusters (stations within Otto
basins); non-nested FE (biome x year/month) are projected out beforehand, the
standard treatment (as in fwildclusterboot's `fe` argument). The
small-sample adjustment counts only the non-FE regressors in k, so `t_wcb` is
larger than a one-hot design's t by sqrt((N - k_onehot) / (N - k)) (about 2%
on the sensor panel); the bootstrap p-value is unaffected, since that factor
scales the observed and bootstrapped t statistics alike.

`poisson_score_bootstrap` is the counterpart for PPML (`Fepois`) fits, whose
IRLS estimator has no demeaned OLS representation to resample: it perturbs the
cluster sums of the null-restricted score instead (Kline and Santos 2012).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Below this many clusters, Rademacher weights take too few distinct values;
# Webb's six-point distribution is the usual replacement (Webb 2014).
WEBB_WEIGHTS_MAX_CLUSTERS = 12


def wild_cluster_bootstrap(model, param, cluster, reps=9999, seed=0, weights_type=None):
    """Restricted (WCR) wild-cluster bootstrap test of H0: `param` = 0 for a fitted `pyfixest` Feols.

    Returns the analytic t statistic's bootstrap two-sided p-value together
    with the number of clusters and the weights used.
    """
    from wildboottest.wildboottest import WildboottestCL

    # wildboottest's numba kernels need integer cluster ids, not string codes.
    clusters = pd.factorize(model._data[cluster])[0].astype(np.int64)
    if (clusters < 0).any():
        raise ValueError(f"Cluster variable {cluster!r} is missing for some estimation rows.")
    n_clusters = int(clusters.max()) + 1
    if weights_type is None:
        weights_type = "webb" if n_clusters < WEBB_WEIGHTS_MAX_CLUSTERS else "rademacher"

    restriction = np.zeros(model._X.shape[1])
    restriction[list(model._coefnames).index(param)] = 1.0

    boot = WildboottestCL(
        X=model._X,
        Y=np.ravel(model._Y),
        cluster=clusters,
        R=restriction,
        B=reps,
        seed=seed,
        parallel=False,
    )
    boot.get_scores(bootstrap_type="11", impose_null=True, adj=True, cluster_adj=True)
    boot.get_weights(weights_type=weights_type)
    boot.get_numer()
    boot.get_denom()
    boot.get_tboot()
    boot.get_vcov()
    boot.get_tstat()
    boot.get_pvalue(pval_type="two-tailed")
    return pd.Series(
        {
            "Coefficient": param,
            "t_wcb": float(np.ravel(boot.t_stat)[0]),
            "p_wcb": float(np.ravel(boot.pvalue)[0]),
            "n_clusters": n_clusters,
            "weights": weights_type,
            "reps": int(boot.B),
        }
    )


def _bootstrap_weights(rng, reps, n_clusters, weights_type):
    if weights_type == "rademacher":
        return rng.choice([-1.0, 1.0], size=(reps, n_clusters))
    if weights_type == "webb":
        values = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
        return rng.choice(values, size=(reps, n_clusters))
    raise ValueError(f"Unknown weights_type {weights_type!r}.")


def poisson_score_bootstrap(model, param, cluster, reps=9999, seed=0, weights_type=None):
    """Restricted wild-cluster score bootstrap test of H0: `param` = 0 for a fitted `pyfixest` Fepois.

    The model is re-estimated without `param` (same FE, offset and sample). The
    efficient score of `param` at that null fit is `x_tilde * (y - mu)`, where
    `x_tilde` is `param` residualized on the FE (weighted demeaning) and the
    other regressors, with the null fit's Poisson weights `mu`. The statistic is
    the sum of the cluster scores over their root sum of squares; each draw
    multiplies the cluster scores by Rademacher (Webb below
    `WEBB_WEIGHTS_MAX_CLUSTERS` clusters) weights, without re-fitting. The
    terms of the model formula must be plain columns.
    """
    import pyfixest as pf
    from pyfixest.estimation import demean

    depvar, rest = (part.strip() for part in model._fml.split("~"))
    rhs, _, fixef = rest.partition("|")
    terms = [term.strip() for term in rhs.split("+")]
    if param not in terms:
        raise ValueError(f"{param!r} is not a regressor of {model._fml!r}.")
    others = [term for term in terms if term != param]
    fixed_effects = [fe.strip() for fe in fixef.split("+") if fe.strip()]

    data = model._data.copy()
    data["_offset"] = np.ravel(model._offset)
    restricted_formula = f"{depvar} ~ {' + '.join(others) or '1'}"
    if fixed_effects:
        restricted_formula += f" | {' + '.join(fixed_effects)}"
    restricted = pf.fepois(restricted_formula, data=data, offset="_offset", vcov="iid")

    sample = restricted._data
    mu = np.asarray(restricted.predict(type="response"), dtype=np.float64)
    y = sample[depvar].to_numpy(dtype=np.float64)
    design = sample[[param, *others]].to_numpy(dtype=np.float64)
    if fixed_effects:
        codes = np.column_stack([pd.factorize(sample[fe])[0] for fe in fixed_effects]).astype(np.uint64)
        design, converged = demean(design, codes, mu)
        if not converged:
            raise RuntimeError(f"Weighted demeaning on {fixed_effects} did not converge.")
    x_tilde, nuisance = design[:, 0], design[:, 1:]
    if nuisance.shape[1]:
        x_tilde = x_tilde - nuisance @ np.linalg.solve(nuisance.T @ (nuisance * mu[:, None]), nuisance.T @ (mu * x_tilde))

    clusters = pd.factorize(sample[cluster])[0]
    if (clusters < 0).any():
        raise ValueError(f"Cluster variable {cluster!r} is missing for some estimation rows.")
    cluster_scores = np.bincount(clusters, weights=x_tilde * (y - mu))
    n_clusters = len(cluster_scores)
    if weights_type is None:
        weights_type = "webb" if n_clusters < WEBB_WEIGHTS_MAX_CLUSTERS else "rademacher"

    statistic = cluster_scores.sum() / np.sqrt(np.square(cluster_scores).sum())
    draws = _bootstrap_weights(np.random.default_rng(seed), reps, n_clusters, weights_type)
    boot = (draws @ cluster_scores) / np.sqrt(np.square(draws) @ np.square(cluster_scores))
    return pd.Series(
        {
            "Coefficient": param,
            "score_stat": float(statistic),
            "p_wcb": float(np.mean(np.abs(boot) >= abs(statistic))),
            "n_clusters": n_clusters,
            "weights": weights_type,
            "reps": int(reps),
        }
    )
