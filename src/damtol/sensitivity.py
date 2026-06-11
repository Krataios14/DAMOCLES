"""Variance-based global sensitivity analysis.

Saltelli's scheme with the Jansen estimators: first-order indices say
how much of the output variance each input drives alone, total indices
include all its interactions. The gap between them flags interaction
effects. Cost is (d + 2) model evaluations per base sample, so keep the
model vectorised.
"""

from __future__ import annotations

import numpy as np

from .sampling import map_to_physical, sample_unit


def sobol_indices(fn, variables, n=2**12, seed=None):
    """Estimate first-order and total Sobol indices.

    fn        : callable dict[str, ndarray] -> ndarray (the model output)
    variables : ordered dict name -> RandomVariable
    n         : base sample count (rounded up to a power of two)

    Returns dict name -> {"first": S_i, "total": T_i}.
    """
    names = list(variables)
    d = len(names)
    u = sample_unit(n, 2 * d, method="sobol", seed=seed)
    a, b = u[:, :d], u[:, d:]

    y_a = np.asarray(fn(map_to_physical(a, variables)), dtype=float)
    y_b = np.asarray(fn(map_to_physical(b, variables)), dtype=float)
    var = np.var(np.concatenate([y_a, y_b]), ddof=1)
    if var == 0.0:
        raise ValueError("model output has zero variance")

    out = {}
    for i, name in enumerate(names):
        ab_i = a.copy()
        ab_i[:, i] = b[:, i]
        y_abi = np.asarray(fn(map_to_physical(ab_i, variables)), dtype=float)
        first = float(np.mean(y_b * (y_abi - y_a)) / var)
        total = float(0.5 * np.mean((y_a - y_abi) ** 2) / var)
        out[name] = {"first": first, "total": total}
    return out


def rank_drivers(indices):
    """Names sorted by total index, biggest first. The variables worth
    spending characterisation money on are at the front."""
    return sorted(indices, key=lambda k: indices[k]["total"], reverse=True)
