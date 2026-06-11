"""Random variable wrappers used to describe scattered inputs.

Every variable maps uniform samples in (0, 1) to physical values through
its inverse CDF, so the same definition works under plain Monte Carlo,
Latin hypercube and Sobol sampling without modification.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


class RandomVariable:
    """Base class. Subclasses set self._dist to a frozen scipy distribution."""

    _dist = None

    def ppf(self, u):
        return self._dist.ppf(u)

    def pdf(self, x):
        return self._dist.pdf(x)

    def cdf(self, x):
        return self._dist.cdf(x)

    def mean(self):
        return float(self._dist.mean())

    def std(self):
        return float(self._dist.std())

    def __repr__(self):
        params = ", ".join(f"{k}={v:g}" for k, v in self._params.items())
        return f"{type(self).__name__}({params})"


class Normal(RandomVariable):
    def __init__(self, mean, std):
        if std <= 0:
            raise ValueError("std must be positive")
        self._params = {"mean": mean, "std": std}
        self._dist = stats.norm(loc=mean, scale=std)


class Lognormal(RandomVariable):
    """Parametrised by the arithmetic mean and coefficient of variation,
    which is how material scatter is usually quoted."""

    def __init__(self, mean, cov):
        if mean <= 0 or cov <= 0:
            raise ValueError("mean and cov must be positive")
        self._params = {"mean": mean, "cov": cov}
        sigma_ln = np.sqrt(np.log(1.0 + cov**2))
        mu_ln = np.log(mean) - 0.5 * sigma_ln**2
        self._dist = stats.lognorm(s=sigma_ln, scale=np.exp(mu_ln))


class Weibull(RandomVariable):
    """Two-parameter Weibull, the usual choice for strength data."""

    def __init__(self, shape, scale):
        if shape <= 0 or scale <= 0:
            raise ValueError("shape and scale must be positive")
        self._params = {"shape": shape, "scale": scale}
        self._dist = stats.weibull_min(c=shape, scale=scale)


class Gumbel(RandomVariable):
    """Gumbel max, for extreme loads (gust, once-per-flight peak stress)."""

    def __init__(self, loc, scale):
        if scale <= 0:
            raise ValueError("scale must be positive")
        self._params = {"loc": loc, "scale": scale}
        self._dist = stats.gumbel_r(loc=loc, scale=scale)


class Uniform(RandomVariable):
    def __init__(self, low, high):
        if high <= low:
            raise ValueError("high must exceed low")
        self._params = {"low": low, "high": high}
        self._dist = stats.uniform(loc=low, scale=high - low)


class Deterministic(RandomVariable):
    """Fixed value. Lets a study treat any input as random or not
    without changing the analysis code."""

    def __init__(self, value):
        self._params = {"value": value}
        self.value = value

    def ppf(self, u):
        return np.full_like(np.asarray(u, dtype=float), self.value)

    def pdf(self, x):
        raise NotImplementedError("a deterministic input has no density")

    def cdf(self, x):
        return np.where(np.asarray(x, dtype=float) >= self.value, 1.0, 0.0)

    def mean(self):
        return float(self.value)

    def std(self):
        return 0.0


_DIST_MAP = {
    "normal": (Normal, ("mean", "std")),
    "lognormal": (Lognormal, ("mean", "cov")),
    "weibull": (Weibull, ("shape", "scale")),
    "gumbel": (Gumbel, ("loc", "scale")),
    "uniform": (Uniform, ("low", "high")),
    "deterministic": (Deterministic, ("value",)),
}


def from_spec(spec):
    """Build a variable from a config dict, e.g.
    {"dist": "lognormal", "mean": 9e-5, "cov": 0.5}."""
    spec = dict(spec)
    kind = spec.pop("dist", None)
    if kind not in _DIST_MAP:
        raise ValueError(f"unknown distribution {kind!r}, "
                         f"expected one of {sorted(_DIST_MAP)}")
    cls, keys = _DIST_MAP[kind]
    missing = [k for k in keys if k not in spec]
    if missing:
        raise ValueError(f"{kind} needs parameters {list(keys)}, missing {missing}")
    extra = [k for k in spec if k not in keys]
    if extra:
        raise ValueError(f"{kind} got unexpected parameters {extra}")
    return cls(**{k: spec[k] for k in keys})
