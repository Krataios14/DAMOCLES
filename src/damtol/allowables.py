"""Statistical basis values for material properties.

A-basis: 99 % of the population exceeds the value with 95 % confidence.
B-basis: 90 % exceedance with 95 % confidence. These are the design
allowables of MMPDS and CMH-17. The one-sided tolerance factor for
normal data comes from the noncentral t distribution, which is exact.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def tolerance_factor(n, proportion=0.90, confidence=0.95):
    """Exact one-sided tolerance limit factor k for normal samples:
    P(at least `proportion` of the population > xbar - k*s) = confidence."""
    if n < 2:
        raise ValueError("need at least two specimens")
    z_p = stats.norm.ppf(proportion)
    return float(stats.nct.ppf(confidence, df=n - 1, nc=z_p * np.sqrt(n))
                 / np.sqrt(n))


def basis_value(data, proportion=0.90, confidence=0.95, dist="normal"):
    """Lower tolerance bound on the data.

    dist="normal"    : xbar - k * s
    dist="lognormal" : the same on log data, transformed back. Use this
                       for anything physically positive with right skew
                       (toughness, life, flaw size).
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 1:
        raise ValueError("data must be one-dimensional")
    n = x.shape[0]
    k = tolerance_factor(n, proportion, confidence)
    if dist == "normal":
        return float(np.mean(x) - k * np.std(x, ddof=1))
    if dist == "lognormal":
        if np.any(x <= 0):
            raise ValueError("lognormal basis needs positive data")
        lx = np.log(x)
        return float(np.exp(np.mean(lx) - k * np.std(lx, ddof=1)))
    raise ValueError(f"unknown dist {dist!r}")


def b_basis(data, dist="normal"):
    return basis_value(data, proportion=0.90, confidence=0.95, dist=dist)


def a_basis(data, dist="normal"):
    return basis_value(data, proportion=0.99, confidence=0.95, dist=dist)
