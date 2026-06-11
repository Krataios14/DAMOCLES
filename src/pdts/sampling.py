"""Uniform sample generation on the unit hypercube.

Three schemes: plain pseudo-random, Latin hypercube, and scrambled Sobol.
LHS and Sobol converge faster than plain Monte Carlo for the smooth
integrands typical of structural reliability, often by an order of
magnitude in sample count for the same error.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc

METHODS = ("random", "lhs", "sobol")


def sample_unit(n, dims, method="lhs", seed=None):
    """Return an (m, dims) array of points in (0, 1).

    For Sobol, n is rounded up to the next power of two to preserve the
    balance properties of the sequence, so m may exceed n. Callers should
    use the returned length rather than the requested one.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if dims < 1:
        raise ValueError("dims must be at least 1")

    if method == "random":
        rng = np.random.default_rng(seed)
        u = rng.random((n, dims))
    elif method == "lhs":
        sampler = qmc.LatinHypercube(d=dims, seed=seed)
        u = sampler.random(n)
    elif method == "sobol":
        sampler = qmc.Sobol(d=dims, scramble=True, seed=seed)
        m = int(np.ceil(np.log2(n)))
        u = sampler.random_base2(m)
    else:
        raise ValueError(f"unknown method {method!r}, expected one of {METHODS}")

    # Guard the open interval: ppf(0) and ppf(1) are infinite for
    # unbounded distributions.
    tiny = np.finfo(float).tiny
    return np.clip(u, tiny, 1.0 - np.finfo(float).epsneg)


def map_to_physical(u, variables):
    """Transform unit samples to physical space.

    variables is an ordered dict name -> RandomVariable. Column i of u
    feeds variable i. Returns dict name -> 1-D array.
    """
    names = list(variables)
    if u.shape[1] != len(names):
        raise ValueError(f"u has {u.shape[1]} columns for {len(names)} variables")
    return {name: variables[name].ppf(u[:, i]) for i, name in enumerate(names)}
