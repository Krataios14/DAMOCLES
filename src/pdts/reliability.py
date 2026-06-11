"""Failure probability estimation.

A limit state g(x) takes a dict of named sample arrays and returns a
margin: g < 0 is failure. The estimator reports an exact Clopper-Pearson
confidence interval for plain sampling, and a normal-approximation
interval for importance sampling where the indicator is weighted.

Importance sampling matters here because certified failure probabilities
sit around 1e-5 per cycle or lower. Hitting that with plain Monte Carlo
needs ~1e7 samples for a usable interval; shifting the dominant variable
toward its tail gets there in 1e4.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from .sampling import map_to_physical, sample_unit


@dataclass
class ReliabilityResult:
    pof: float
    ci_low: float
    ci_high: float
    n_samples: int
    n_failures: float          # effective count under IS, exact otherwise
    std_error: float
    method: str
    importance_sampled: bool = False
    samples: dict = field(default_factory=dict, repr=False)
    margin: np.ndarray = field(default=None, repr=False)

    @property
    def cov_estimator(self):
        """Coefficient of variation of the estimate itself. Below ~0.1
        is usually considered converged for risk reporting."""
        if self.pof == 0.0:
            return np.inf
        return self.std_error / self.pof

    def reliability_index(self):
        """Equivalent beta, the usual currency for comparing designs."""
        if self.pof <= 0.0:
            return np.inf
        return float(-stats.norm.ppf(self.pof))


def _clopper_pearson(k, n, confidence=0.95):
    alpha = 1.0 - confidence
    low = 0.0 if k == 0 else stats.beta.ppf(alpha / 2.0, k, n - k + 1)
    high = 1.0 if k == n else stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k)
    return float(low), float(high)


def estimate_pof(limit_state, variables, n=100_000, method="lhs", seed=None,
                 importance=None, confidence=0.95, keep_samples=False):
    """Estimate P(g < 0).

    limit_state : callable, dict[str, ndarray] -> ndarray of margins
    variables   : ordered dict name -> RandomVariable (nominal densities)
    importance  : optional dict name -> RandomVariable. Listed variables
                  are sampled from these proposal densities instead, and
                  the estimate is reweighted by the likelihood ratio.
    """
    if importance:
        unknown = set(importance) - set(variables)
        if unknown:
            raise ValueError(f"importance variables not in study: {sorted(unknown)}")

    sampling_vars = dict(variables)
    if importance:
        sampling_vars.update(importance)

    u = sample_unit(n, len(variables), method=method, seed=seed)
    n_eff = u.shape[0]  # sobol may round up
    x = map_to_physical(u, sampling_vars)

    if importance:
        log_w = np.zeros(n_eff)
        for name, proposal in importance.items():
            nominal = variables[name]
            log_w += np.log(nominal.pdf(x[name])) - np.log(proposal.pdf(x[name]))
        weights = np.exp(log_w)
    else:
        weights = None

    g = np.asarray(limit_state(x), dtype=float)
    if g.shape != (n_eff,):
        raise ValueError(f"limit state returned shape {g.shape}, expected ({n_eff},)")
    fail = g < 0.0

    if weights is None:
        k = int(fail.sum())
        pof = k / n_eff
        ci_low, ci_high = _clopper_pearson(k, n_eff, confidence)
        std_error = float(np.sqrt(max(pof * (1.0 - pof), 0.0) / n_eff))
        n_failures = float(k)
    else:
        contrib = fail * weights
        pof = float(contrib.mean())
        std_error = float(contrib.std(ddof=1) / np.sqrt(n_eff))
        z = stats.norm.ppf(0.5 + confidence / 2.0)
        ci_low = max(0.0, pof - z * std_error)
        ci_high = min(1.0, pof + z * std_error)
        n_failures = float(fail.sum())

    return ReliabilityResult(
        pof=float(pof), ci_low=ci_low, ci_high=ci_high,
        n_samples=n_eff, n_failures=n_failures, std_error=std_error,
        method=method, importance_sampled=importance is not None,
        samples=x if keep_samples else {},
        margin=g if keep_samples else None,
    )
