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

The municipality-year designs with a national shock add three inferences that do
not rely on many treated clusters: randomization over simulated random-walk
paths of the gold price (`random_walk_p`), permutation of the exposure across
municipalities (`permutation_p`), and the price-exposure variance of
Moreno-Louzada, Figueira and Picchetti (`price_exposure_se`). The first two
recompute the coefficient by Frisch-Waugh-Lovell (`fwl_residuals`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pyfixest.estimation import demean as _pyfixest_demean

from src.analysis.core.defaults import FE, SEED

RANDOM_WALK_DRAWS = 1999
PERMUTATION_DRAWS = 999
PERMUTATION_CHUNK = 100  # draws per batch of demeaning

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


# --- Randomization inference ----------------------------------------------------------

def demean(values, data, fe, weights):
    """The columns of `values` net of the fixed effects in `fe` ("a + b", columns of `data`), weighted."""
    codes = np.column_stack([pd.factorize(data[key])[0] for key in fe.split(" + ")]).astype(np.uint64)
    residuals, converged = _pyfixest_demean(np.ascontiguousarray(values, dtype=np.float64), codes, weights)
    if not converged:
        raise RuntimeError(f"Demeaning on {fe} did not converge.")
    return residuals


def fwl_residuals(values, data, fe, weights, controls=()):
    """The columns of `values` net of the fixed effects and of the `controls` columns of `data`, weighted."""
    residuals = demean(values, data, fe, weights)
    if controls:
        control_values = demean(data[list(controls)].to_numpy(dtype=np.float64), data, fe, weights)
        sqrt_w = np.sqrt(weights)[:, None]
        coefficients = np.linalg.lstsq(sqrt_w * control_values, sqrt_w * residuals, rcond=None)[0]
        residuals = residuals - control_values @ coefficients
    return residuals


def random_walk_p(data, exposure, price, fe=FE, weighted=True, draws=RANDOM_WALK_DRAWS, seed=SEED, controls=()):
    """Randomization p-value of exposure x price over simulated random-walk price paths.

    Each draw replaces the price path by a driftless Gaussian random walk with
    the observed SD of yearly changes; its coefficient (by FWL, net of the FE
    and of the `controls` columns of `data`) is compared with the observed
    one. `data` holds the outcome as `y` and, if `weighted`, the weights as
    `weight`. Returns the observed coefficient and the two-sided p-value.
    """
    weights = data["weight"].to_numpy(dtype=np.float64) if weighted else np.ones(len(data))
    years = np.sort(data["year"].unique())
    path = data.groupby("year")[price].first().loc[years].to_numpy()
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, np.diff(path).std(ddof=1), size=(draws, len(years) - 1))
    paths = np.concatenate([np.zeros((draws, 1)), np.cumsum(steps, axis=1)], axis=1)
    position = np.searchsorted(years, data["year"].to_numpy())
    exposure_values = data[exposure].to_numpy(dtype=np.float64)
    columns = np.column_stack([
        data["y"].to_numpy(dtype=np.float64),
        exposure_values * path[position],
        exposure_values[:, None] * paths[:, position].T,
    ])
    residuals = fwl_residuals(columns, data, fe, weights, controls)
    y, z = residuals[:, 0], residuals[:, 1:]
    betas = (weights[:, None] * z * y[:, None]).sum(axis=0) / (weights[:, None] * z ** 2).sum(axis=0)
    observed, simulated = betas[0], betas[1:]
    return float(observed), float((1 + (np.abs(simulated) >= abs(observed)).sum()) / (1 + draws))


def permutation_p(data, exposure, exposure_terms, controls, periods, fe=FE, draws=PERMUTATION_DRAWS, seed=SEED,
                  strata=None, chunk=PERMUTATION_CHUNK):
    """Two-sided permutation p-values of the exposure terms, permuting the exposure across municipalities
    (within the groups of `strata`, a column constant within municipalities, if given).

    Each exposure term is the exposure times a function of the year: `periods`
    maps the part of the term's name after `__x__` to that function. Each draw
    reassigns the municipalities' exposure values at random, rebuilds every
    exposure term, and recomputes the coefficients by FWL net of the FE and the
    `controls`, unweighted. Returns the observed coefficients and the p-values.
    """
    weights = np.ones(len(data))
    municipalities, position = np.unique(data["mun_id"].to_numpy(), return_inverse=True)
    exposure_by_mun = data.groupby("mun_id")[exposure].first().reindex(municipalities).to_numpy(dtype=np.float64)
    groups = [np.arange(len(municipalities))] if strata is None else [
        np.flatnonzero(labels == label)
        for labels in [data.groupby("mun_id")[strata].first().reindex(municipalities).to_numpy()]
        for label in np.unique(labels)
    ]

    def permute(rng):
        permuted = exposure_by_mun.copy()
        for group in groups:
            permuted[group] = rng.permutation(exposure_by_mun[group])
        return permuted
    period_values = np.column_stack([
        periods[name.split("__x__")[1]](data["year"]).to_numpy(dtype=np.float64) for name in exposure_terms
    ])

    y = fwl_residuals(data["y"].to_numpy(dtype=np.float64)[:, None], data, fe, weights, controls)[:, 0]

    def betas(exposure_values):
        # exposure_values: municipalities x draws -> coefficients: draws x terms
        k = len(exposure_terms)
        stacked = np.concatenate(
            [exposure_values[position] * period_values[:, [j]] for j in range(k)], axis=1
        )
        residuals = fwl_residuals(stacked, data, fe, weights, controls)
        draws_here = exposure_values.shape[1]
        out = np.empty((draws_here, k))
        for d in range(draws_here):
            X = residuals[:, [j * draws_here + d for j in range(k)]]
            out[d] = np.linalg.lstsq(X, y, rcond=None)[0]
        return out

    observed = betas(exposure_by_mun[:, None])[0]
    rng = np.random.default_rng(seed)
    simulated = []
    for start in range(0, draws, chunk):
        n = min(chunk, draws - start)
        permuted = np.column_stack([permute(rng) for _ in range(n)])
        simulated.append(betas(permuted))
    simulated = np.concatenate(simulated)
    p_values = (1 + (np.abs(simulated) >= np.abs(observed)).sum(axis=0)) / (1 + draws)
    return observed, p_values


def price_exposure_se(model, term, exposure, price, weighted=True):
    """P-E standard error of `term` = `exposure` x `price` (Moreno-Louzada, Figueira and Picchetti).

    V = sum_t p~_t^2 (sum_i A_i w_it u_it)^2 / (sum_it w_it z~_it^2)^2, with p~
    the price demeaned over the sample years, A the exposure, u the residuals,
    z~ the regressor net of the FE and the other regressors, and w the weights.
    The randomness is in the price, assumed independent across years (their
    Assumption 4), so the score is summed by year with the exposure in place of
    the residualized regressor; p-values use the normal approximation of their
    Theorem 1. Returns the SE and the number of years.
    """
    data = model._data
    names = list(model._coefnames)
    X = np.asarray(model._X)  # FE-demeaned and multiplied by sqrt(w)
    j = names.index(term)
    z = X[:, j]
    others = np.delete(X, j, axis=1)
    if others.shape[1]:
        z = z - others @ np.linalg.lstsq(others, z, rcond=None)[0]
    sqrt_w = np.sqrt(np.asarray(model._weights).ravel()) if weighted else np.ones(len(z))
    weighted_residual = sqrt_w * np.ravel(model._u_hat)  # w_it * u_it
    year = data["year"].to_numpy()
    by_year = pd.DataFrame({"year": year, "score": data[exposure].to_numpy() * weighted_residual})
    sums = by_year.groupby("year")["score"].sum()
    prices = data.groupby("year")[price].first().loc[sums.index]
    demeaned = prices - prices.mean()
    variance = float((demeaned.to_numpy() ** 2 * sums.to_numpy() ** 2).sum()) / float((z ** 2).sum()) ** 2
    return np.sqrt(variance), len(sums)
