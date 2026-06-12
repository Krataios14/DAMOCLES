"""Fatigue crack growth under constant amplitude loading.

Conventions: stress in MPa, stress intensity in MPa*sqrt(m), crack size
in metres, life in cycles. K = Y(a) * sigma * sqrt(pi * a).

Everything is vectorised over samples so the Monte Carlo loop is a single
array pass. Sample sets are processed in chunks to keep the (samples x
grid) intermediates inside memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------- geometry

class Geometry:
    """Stress intensity geometry factor Y(a). a_max bounds the bisection
    for critical size and the validity of the correction."""

    a_max = 1.0

    def y(self, a):
        raise NotImplementedError


class ThroughCrack(Geometry):
    """Through crack in a wide plate, Y = 1."""

    def y(self, a):
        return np.ones_like(np.asarray(a, dtype=float))


class CenterCrack(Geometry):
    """Centre crack of half-length a in a plate of finite width W,
    Feddersen secant correction."""

    def __init__(self, width):
        if width <= 0:
            raise ValueError("width must be positive")
        self.width = width
        self.a_max = 0.49 * width

    def y(self, a):
        return np.sqrt(1.0 / np.cos(np.pi * np.asarray(a) / self.width))


class SurfaceCrack(Geometry):
    """Semicircular surface crack, deepest point.
    Y = 1.12 * 2/pi = 0.713 (free surface times embedded penny crack)."""

    def y(self, a):
        return np.full_like(np.asarray(a, dtype=float), 1.12 * 2.0 / np.pi)


class CornerCrack(Geometry):
    """Quarter-circular corner crack, two free surfaces.
    Y = 1.12^2 * 2/pi = 0.80."""

    def y(self, a):
        return np.full_like(np.asarray(a, dtype=float), 1.12**2 * 2.0 / np.pi)


class CustomGeometry(Geometry):
    """Any user-supplied Y(a), e.g. a Newman-Raju fit or an FE-derived
    weight function tabulated and interpolated."""

    def __init__(self, fn, a_max=1.0):
        self.fn = fn
        self.a_max = a_max

    def y(self, a):
        return np.asarray(self.fn(np.asarray(a, dtype=float)), dtype=float)


GEOMETRIES = {
    "through": ThroughCrack,
    "center": CenterCrack,
    "surface": SurfaceCrack,
    "corner": CornerCrack,
}


# ------------------------------------------------------------- growth laws

class ParisLaw:
    """da/dN = C * dK^m above the threshold, zero below.

    C may be a scalar or a per-sample array when growth rate scatter is
    itself a random variable.
    """

    def __init__(self, c, m, dk_threshold=0.0):
        self.c = c
        self.m = m
        self.dk_threshold = dk_threshold

    def _c_for(self, dk, sample_slice):
        c = np.asarray(self.c, dtype=float)
        if c.ndim == 0:
            return c
        if sample_slice is not None:
            c = c[sample_slice]
        return c[:, None] if dk.ndim == 2 else c

    def effective_dk(self, dk, stress_ratio):
        return dk

    def rate(self, dk, stress_ratio=0.0, a=None, kc=None, sample_slice=None):
        """Growth rate per cycle. `a` (crack size) and `kc` (per-sample
        toughness) are accepted for interface compatibility with laws
        whose rate depends on them; Paris and Walker ignore both."""
        dk = np.asarray(dk, dtype=float)
        dk_eff = self.effective_dk(dk, stress_ratio)
        c = self._c_for(dk, sample_slice)
        v = c * np.power(np.maximum(dk_eff, 1e-300), self.m)
        return np.where(dk_eff > self.dk_threshold, v, 0.0)


class WalkerLaw(ParisLaw):
    """Paris with the Walker mean stress correction:
    dK_eff = dK / (1 - R)^(1 - gamma). gamma = 1 recovers Paris."""

    def __init__(self, c, m, gamma, dk_threshold=0.0):
        super().__init__(c, m, dk_threshold)
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        self.gamma = gamma

    def effective_dk(self, dk, stress_ratio):
        r = np.clip(stress_ratio, -1.0, 0.99)
        return dk / np.power(1.0 - r, 1.0 - self.gamma)


GROWTH_LAWS = {"paris": ParisLaw, "walker": WalkerLaw}


# ------------------------------------------------------- fracture criteria

def critical_size(geometry, s_max, k_ic, tol=1e-12, max_iter=80):
    """Crack size at which K_max reaches the toughness, by bisection in
    log(a). Vectorised over samples. Capped at geometry.a_max when the
    toughness is never reached inside the correction's validity."""
    s_max = np.atleast_1d(np.asarray(s_max, dtype=float))
    k_ic = np.broadcast_to(np.asarray(k_ic, dtype=float), s_max.shape).copy()

    def overload(a):
        return geometry.y(a) * s_max * np.sqrt(np.pi * a) - k_ic

    lo = np.full_like(s_max, 1e-12)
    hi = np.full_like(s_max, geometry.a_max)

    already = overload(lo) >= 0.0          # fractures at any size
    never = overload(hi) <= 0.0            # never fractures inside validity
    log_lo, log_hi = np.log(lo), np.log(hi)
    for _ in range(max_iter):
        mid = 0.5 * (log_lo + log_hi)
        high = overload(np.exp(mid)) > 0.0
        log_hi = np.where(high, mid, log_hi)
        log_lo = np.where(high, log_lo, mid)
        if np.max(log_hi - log_lo) < tol:
            break
    a_c = np.exp(0.5 * (log_lo + log_hi))
    a_c[already] = lo[already]
    a_c[never] = hi[never]
    return a_c


# --------------------------------------------------------- life integration

@dataclass
class LifeResult:
    cycles_to_failure: np.ndarray   # (n,), inf where no growth
    a_critical: np.ndarray          # (n,)
    eval_cycles: np.ndarray | None  # (t,) requested checkpoints
    a_at: np.ndarray | None         # (n, t) crack size at each checkpoint

    def pof_at(self, cycles):
        """Fraction of samples failed at or before the given cycle count."""
        return float(np.mean(self.cycles_to_failure <= cycles))


def grow(a0, delta_sigma, geometry, law, k_ic, stress_ratio=0.0,
         eval_cycles=None, n_grid=250, chunk_size=100_000):
    """Integrate constant amplitude crack growth for every sample.

    a0, delta_sigma, k_ic : per-sample arrays (or scalars, broadcast)
    eval_cycles           : optional cycle counts at which to record the
                            crack size, used for inspection simulation

    Life is N = integral over a of da / (da/dN), computed on a log-spaced
    grid from a0 to the critical size with the trapezoid rule. dK grows
    monotonically with a under constant amplitude, so a crack below
    threshold at a0 never grows: its life is inf.
    """
    a0 = np.atleast_1d(np.asarray(a0, dtype=float))
    n = a0.shape[0]
    delta_sigma = np.broadcast_to(np.asarray(delta_sigma, dtype=float), (n,))
    k_ic = np.broadcast_to(np.asarray(k_ic, dtype=float), (n,))
    if np.any(a0 <= 0):
        raise ValueError("initial crack sizes must be positive")

    s_max = delta_sigma / (1.0 - stress_ratio)
    a_c = critical_size(geometry, s_max, k_ic)

    def rate_of(a_grid, sl):
        dk = geometry.y(a_grid) * delta_sigma[sl][:, None] * np.sqrt(np.pi * a_grid)
        return law.rate(dk, stress_ratio, a=a_grid, kc=k_ic[sl][:, None],
                        sample_slice=sl)

    return _integrate(a0, a_c, rate_of, eval_cycles, n_grid, chunk_size)


def grow_spectrum(a0, spectrum, geometry, law, k_ic, stress_scale=1.0,
                  eval_blocks=None, n_grid=250, chunk_size=100_000):
    """Integrate crack growth under a repeating load spectrum.

    spectrum     : object with .classes, an iterable of cycle classes
                   carrying delta_sigma [MPa], stress_ratio and count
                   (cycles per block), e.g. spectrum.Spectrum
    stress_scale : per-sample multiplier on every stress in the spectrum,
                   the natural place for load scatter (scalar or (n,))
    eval_blocks  : block counts at which to record crack size

    Life comes back in BLOCKS (e.g. flights), not cycles. With no load
    interaction modelled, the order of cycles inside a block does not
    change the integral, so the block growth rate is the count-weighted
    sum of the class rates and the same a-grid integration applies.
    """
    a0 = np.atleast_1d(np.asarray(a0, dtype=float))
    n = a0.shape[0]
    k_ic = np.broadcast_to(np.asarray(k_ic, dtype=float), (n,))
    stress_scale = np.broadcast_to(np.asarray(stress_scale, dtype=float), (n,))
    if np.any(a0 <= 0):
        raise ValueError("initial crack sizes must be positive")
    classes = list(spectrum.classes)
    if not classes:
        raise ValueError("spectrum has no cycle classes")

    # fracture is governed by the largest peak stress in the block
    peak = getattr(spectrum, "peak_stress", None)
    if peak is None:
        peak = max(c.delta_sigma / (1.0 - c.stress_ratio) for c in classes)
    a_c = critical_size(geometry, peak * stress_scale, k_ic)

    def rate_of(a_grid, sl):
        y = geometry.y(a_grid)
        root = np.sqrt(np.pi * a_grid)
        scale = stress_scale[sl][:, None]
        kc = k_ic[sl][:, None]
        v = np.zeros_like(a_grid)
        for c in classes:
            dk = y * scale * c.delta_sigma * root
            v += c.count * law.rate(dk, c.stress_ratio, a=a_grid, kc=kc,
                                    sample_slice=sl)
        return v

    return _integrate(a0, a_c, rate_of, eval_blocks, n_grid, chunk_size)


def _integrate(a0, a_c, rate_of, eval_steps, n_grid, chunk_size):
    n = a0.shape[0]
    n_f = np.empty(n)
    eval_steps = None if eval_steps is None else np.asarray(eval_steps, dtype=float)
    a_at = None if eval_steps is None else np.empty((n, eval_steps.shape[0]))

    for start in range(0, n, chunk_size):
        sl = slice(start, min(start + chunk_size, n))
        _integrate_chunk(a0[sl], a_c[sl], rate_of, n_grid, sl,
                         n_f[sl], None if a_at is None else a_at[sl],
                         eval_steps)

    return LifeResult(cycles_to_failure=n_f, a_critical=a_c,
                      eval_cycles=eval_steps, a_at=a_at)


def _integrate_chunk(a0, a_c, rate_of, n_grid, sample_slice,
                     out_nf, out_a_at, eval_cycles):
    m = a0.shape[0]
    burst = a_c <= a0                       # critical on arrival

    # log-spaced grid a0 -> a_c per sample; degenerate rows handled after
    safe_ac = np.maximum(a_c, a0 * (1.0 + 1e-9))
    t = np.linspace(0.0, 1.0, n_grid)[None, :]
    a_grid = np.exp(np.log(a0)[:, None] * (1.0 - t) + np.log(safe_ac)[:, None] * t)

    v = rate_of(a_grid, sample_slice)

    dormant = v[:, 0] <= 0.0                # below threshold at a0
    inv_v = np.where(v > 0.0, 1.0 / np.maximum(v, 1e-300), 0.0)

    da = np.diff(a_grid, axis=1)
    seg = 0.5 * (inv_v[:, 1:] + inv_v[:, :-1]) * da
    n_cum = np.concatenate([np.zeros((m, 1)), np.cumsum(seg, axis=1)], axis=1)

    nf = n_cum[:, -1].copy()
    nf[dormant] = np.inf
    nf[burst] = 0.0
    out_nf[:] = nf

    if eval_cycles is None:
        return
    rows = np.arange(m)
    for j, cyc in enumerate(eval_cycles):
        idx = np.sum(n_cum < cyc, axis=1)
        i0 = np.clip(idx - 1, 0, n_grid - 1)
        i1 = np.clip(idx, 0, n_grid - 1)
        n0, n1 = n_cum[rows, i0], n_cum[rows, i1]
        g0, g1 = a_grid[rows, i0], a_grid[rows, i1]
        frac = np.where(n1 > n0, (cyc - n0) / np.where(n1 > n0, n1 - n0, 1.0), 0.0)
        a_t = g0 + np.clip(frac, 0.0, 1.0) * (g1 - g0)
        a_t = np.where(cyc >= nf, a_c, a_t)     # already failed
        a_t = np.where(dormant, a0, a_t)        # never grew
        a_t = np.where(burst, a_c, a_t)
        out_a_at[:, j] = a_t
