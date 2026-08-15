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

from .retardation import (
    MAX_RETARDED_BLOCKS,
    MAX_RETARDED_CYCLES,
    MAX_RETARDED_WORK,
    _select_boundary_index,
)
from .spectrum import SpectrumSequence


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

    already = overload(lo) >= 0.0  # fractures at any size
    never = overload(hi) <= 0.0  # never fractures inside validity
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
    cycles_to_failure: np.ndarray  # (n,), inf where no growth
    a_critical: np.ndarray  # (n,)
    eval_cycles: np.ndarray | None  # (t,) requested checkpoints
    a_at: np.ndarray | None  # (n, t) crack size at each checkpoint

    def pof_at(self, cycles):
        """Fraction of samples failed at or before the given cycle count."""
        return float(np.mean(self.cycles_to_failure <= cycles))


def grow(
    a0,
    delta_sigma,
    geometry,
    law,
    k_ic,
    stress_ratio=0.0,
    eval_cycles=None,
    n_grid=250,
    chunk_size=100_000,
):
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
        return law.rate(
            dk, stress_ratio, a=a_grid, kc=k_ic[sl][:, None], sample_slice=sl
        )

    return _integrate(a0, a_c, rate_of, eval_cycles, n_grid, chunk_size)


def grow_spectrum(
    a0,
    spectrum,
    geometry,
    law,
    k_ic,
    stress_scale=1.0,
    eval_blocks=None,
    n_grid=250,
    chunk_size=100_000,
):
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
            v += c.count * law.rate(
                dk, c.stress_ratio, a=a_grid, kc=kc, sample_slice=sl
            )
        return v

    return _integrate(a0, a_c, rate_of, eval_blocks, n_grid, chunk_size)


def grow_spectrum_retarded(
    a0,
    sequence,
    geometry,
    law,
    k_ic,
    s_yield,
    stress_scale=1.0,
    max_cycles=1e8,
    max_blocks=10_000,
    chunk_size=100_000,
    eval_cycles=None,
    apply_retardation=True,
    max_work=MAX_RETARDED_WORK,
    _boundary_index=None,
):
    """Integrate an ordered spectrum, optionally with Willenborg retardation.

    The existing Willenborg cycle update is unchanged when
    ``apply_retardation`` is true. Disabling it supplies an ordered baseline
    without sorting or merging cycles. Life and checkpoints are in elapsed
    cycles, not repeating blocks.

    Horizons and inspection checkpoints must coincide exactly with cumulative
    ordered-record endpoints. A checkpoint observes the post-record crack;
    live samples retain that actual crack size, while failed samples carry the
    cycle-specific critical size at which failure was detected. If the next
    record's opening load establishes failure at the same elapsed endpoint,
    that endpoint observation is replaced by its critical size and ``Nf``
    equals the endpoint. Records are never partly executed or interpolated.
    ``LifeResult.a_critical`` remains the critical size for the highest mission
    peak.
    """
    a0 = np.asarray(a0, dtype=float)
    if a0.ndim == 0:
        a0 = a0.reshape(1)
    if a0.ndim != 1 or a0.size == 0:
        raise ValueError("a0 must be a non-empty one-dimensional array")
    n = a0.shape[0]
    k_ic = np.broadcast_to(np.asarray(k_ic, dtype=float), (n,))
    stress_scale = np.broadcast_to(np.asarray(stress_scale, dtype=float), (n,))
    if np.any(~np.isfinite(a0)) or np.any(a0 <= 0.0):
        raise ValueError("initial crack sizes must be finite and positive")
    if np.any(~np.isfinite(k_ic)) or np.any(k_ic <= 0.0):
        raise ValueError("fracture toughness must be finite and positive")
    if np.any(~np.isfinite(stress_scale)) or np.any(stress_scale <= 0.0):
        raise ValueError("stress_scale must be finite and positive")
    if not isinstance(sequence, SpectrumSequence):
        raise TypeError("sequence must be a SpectrumSequence")
    boundary_snapshot = _select_boundary_index(sequence, _boundary_index)
    if type(apply_retardation) is not bool:
        raise TypeError("apply_retardation must be a boolean")
    if apply_retardation:
        if isinstance(s_yield, (bool, str, bytes)) or not np.isscalar(s_yield):
            raise TypeError("s_yield must be a real number")
        if not np.isfinite(s_yield) or s_yield <= 0.0:
            raise ValueError("s_yield must be finite and positive")
    elif s_yield is not None:
        raise ValueError("s_yield must be omitted when retardation is disabled")

    if isinstance(max_cycles, (bool, str, bytes)) or not np.isscalar(max_cycles):
        raise TypeError("max_cycles must be a real number")
    max_cycles = float(max_cycles)
    if not np.isfinite(max_cycles) or max_cycles <= 0.0:
        raise ValueError("max_cycles must be finite and positive")
    if max_cycles > MAX_RETARDED_CYCLES:
        raise ValueError(f"max_cycles exceeds the limit of {MAX_RETARDED_CYCLES:,.0f}")
    boundary_snapshot.steps_to(max_cycles, "max_cycles")
    if isinstance(max_blocks, bool) or not isinstance(max_blocks, (int, np.integer)):
        raise TypeError("max_blocks must be an integer")
    if max_blocks <= 0 or max_blocks > MAX_RETARDED_BLOCKS:
        raise ValueError(f"max_blocks must be in [1, {MAX_RETARDED_BLOCKS:,}]")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)):
        raise TypeError("chunk_size must be an integer")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if isinstance(max_work, bool) or not isinstance(max_work, (int, np.integer)):
        raise TypeError("max_work must be an integer")
    if max_work <= 0 or max_work > MAX_RETARDED_WORK:
        raise ValueError(f"max_work must be in [1, {MAX_RETARDED_WORK:,}]")

    if eval_cycles is not None and isinstance(eval_cycles, (str, bytes)):
        raise TypeError("eval_cycles must be an ordered numeric sequence")
    eval_cycles = None if eval_cycles is None else np.asarray(eval_cycles, dtype=float)
    if eval_cycles is not None:
        if eval_cycles.ndim != 1:
            raise ValueError("eval_cycles must be one-dimensional")
        if eval_cycles.size > 10_000:
            raise ValueError("eval_cycles exceeds 10,000 checkpoints")
        if np.any(~np.isfinite(eval_cycles)) or np.any(eval_cycles <= 0.0):
            raise ValueError("eval_cycles must be finite and positive")
        if np.any(np.diff(eval_cycles) <= 0.0):
            raise ValueError("eval_cycles must be strictly increasing")
        horizon = min(max_cycles, max_blocks * sequence.total_count)
        if eval_cycles.size and eval_cycles[-1] > horizon:
            raise ValueError("eval_cycles exceed the integration horizon")
        boundary_snapshot.steps_many(eval_cycles, "eval_cycles entry")

    prospective_work = n * min(
        max_blocks * boundary_snapshot.n_cycles,
        boundary_snapshot.steps_to(max_cycles, "max_cycles"),
    )
    if prospective_work > max_work:
        raise ValueError(
            "ordered mission exceeds max_work before integration: "
            f"{prospective_work:,} > {max_work:,} sample-cycle steps"
        )

    a_c = critical_size(geometry, sequence.peak_stress * stress_scale, k_ic)
    n_f = np.full(n, np.inf)
    a_at = None if eval_cycles is None else np.empty((n, eval_cycles.size))

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        m = end - start
        a = a0[start:end].copy()
        kc = k_ic[start:end]
        scale = stress_scale[start:end]

        from .retardation import (
            effective_kr,
            init_state,
            plastic_zone_radius,
            residual_stress_intensity,
        )

        state = init_state(m) if apply_retardation else None
        chunk_done = np.zeros(m, dtype=bool)
        chunk_failed = np.zeros(m, dtype=bool)
        chunk_failure_size = np.full(m, np.nan)
        chunk_cycles = np.zeros(m)
        chunk_a_at = None if a_at is None else a_at[start:end]
        if chunk_a_at is not None:
            chunk_a_at.fill(np.nan)

        block = 0
        elapsed = 0.0
        while not chunk_done.all() and block < max_blocks:
            block += 1
            for cyc in sequence.cycles:
                alive = ~chunk_done
                if not alive.any():
                    break

                elapsed_after = elapsed + cyc.count
                a_sub = a[alive]
                kc_sub = kc[alive]
                sc_sub = scale[alive]
                alive_idx = np.where(alive)[0]

                # Fracture check before processing this cycle.
                s_max_cyc = cyc.delta_sigma / (1.0 - cyc.stress_ratio)
                ac_cyc = critical_size(geometry, s_max_cyc * sc_sub, kc_sub)
                newly_fractured = a_sub >= ac_cyc
                if newly_fractured.any():
                    failed_indices = alive_idx[newly_fractured]
                    chunk_done[failed_indices] = True
                    chunk_failed[failed_indices] = True
                    chunk_failure_size[failed_indices] = ac_cyc[newly_fractured]
                    if chunk_a_at is not None:
                        # This failure is detected on application of the next
                        # record's opening load, at the already-elapsed record
                        # endpoint.  Replace an observation written there by
                        # the prior record so a_at and Nf share one convention.
                        boundary_checkpoints = np.nonzero(eval_cycles == elapsed)[0]
                        for checkpoint in boundary_checkpoints:
                            chunk_a_at[failed_indices, checkpoint] = ac_cyc[
                                newly_fractured
                            ]

                alive2 = ~chunk_done
                if alive2.any():
                    a2 = a[alive2]
                    kc2 = kc[alive2]
                    sc2 = scale[alive2]
                    alive2_idx = np.where(alive2)[0]

                    y_a2 = geometry.y(a2)
                    root2 = np.sqrt(np.pi * a2)
                    dk2 = y_a2 * sc2 * cyc.delta_sigma * root2
                    k_max2 = dk2 / (1.0 - cyc.stress_ratio)

                    if apply_retardation:
                        # Existing AFGROW zone comparison and state update.
                        r_p2 = plastic_zone_radius(k_max2, s_yield)
                        current_boundary = a2 + r_p2
                        stored_boundary = state.a_overload[alive2] + state.r_p[alive2]
                        new_zone = ~state.active[alive2] | (
                            current_boundary > stored_boundary
                        )
                        contained = state.active[alive2] & ~new_zone
                        k_r = residual_stress_intensity(
                            a2,
                            k_max2,
                            state.k_max_ol[alive2],
                            state.r_p[alive2],
                            state.a_overload[alive2],
                            contained,
                        )
                        dk_eff, r_eff = effective_kr(dk2, cyc.stress_ratio, k_r)
                        if new_zone.any():
                            new_indices = alive2_idx[new_zone]
                            state.k_max_ol[new_indices] = k_max2[new_zone]
                            state.r_p[new_indices] = r_p2[new_zone]
                            state.a_overload[new_indices] = a2[new_zone]
                            state.active[new_indices] = True
                    else:
                        dk_eff = dk2
                        r_eff = cyc.stress_ratio

                    sample_indices = start + alive2_idx
                    rate = law.rate(
                        dk_eff,
                        r_eff,
                        a=a2,
                        kc=kc2,
                        sample_slice=sample_indices,
                    )
                    a[alive2_idx] += cyc.count * rate
                    chunk_cycles[alive2_idx] += cyc.count

                    ac2 = critical_size(geometry, s_max_cyc * sc2, kc2)
                    newly_fractured = a[alive2_idx] >= ac2
                    if newly_fractured.any():
                        failed_indices = alive2_idx[newly_fractured]
                        chunk_done[failed_indices] = True
                        chunk_failed[failed_indices] = True
                        chunk_failure_size[failed_indices] = ac2[newly_fractured]

                    run_out = (~chunk_done) & (chunk_cycles >= max_cycles)
                    if run_out.any():
                        chunk_done[run_out] = True

                if chunk_a_at is not None:
                    checkpoint_indices = np.nonzero(eval_cycles == elapsed_after)[0]
                    for checkpoint in checkpoint_indices:
                        failed_by_checkpoint = chunk_failed & (
                            chunk_cycles <= eval_cycles[checkpoint]
                        )
                        chunk_a_at[:, checkpoint] = np.where(
                            failed_by_checkpoint,
                            chunk_failure_size,
                            a,
                        )
                elapsed = elapsed_after

        n_f[start:end] = np.where(chunk_failed, chunk_cycles, np.inf)
        if chunk_a_at is not None and np.isnan(chunk_a_at).any():
            missing_rows, missing_cols = np.nonzero(np.isnan(chunk_a_at))
            for row, column in zip(missing_rows, missing_cols):
                if chunk_failed[row] and chunk_cycles[row] <= eval_cycles[column]:
                    chunk_a_at[row, column] = chunk_failure_size[row]
            if np.isnan(chunk_a_at).any():
                raise RuntimeError("an inspection checkpoint was not evaluated")

    return LifeResult(
        cycles_to_failure=n_f,
        a_critical=a_c,
        eval_cycles=eval_cycles,
        a_at=a_at,
    )


def _integrate(a0, a_c, rate_of, eval_steps, n_grid, chunk_size):
    n = a0.shape[0]
    n_f = np.empty(n)
    eval_steps = None if eval_steps is None else np.asarray(eval_steps, dtype=float)
    a_at = None if eval_steps is None else np.empty((n, eval_steps.shape[0]))

    for start in range(0, n, chunk_size):
        sl = slice(start, min(start + chunk_size, n))
        _integrate_chunk(
            a0[sl],
            a_c[sl],
            rate_of,
            n_grid,
            sl,
            n_f[sl],
            None if a_at is None else a_at[sl],
            eval_steps,
        )

    return LifeResult(
        cycles_to_failure=n_f, a_critical=a_c, eval_cycles=eval_steps, a_at=a_at
    )


def _integrate_chunk(
    a0, a_c, rate_of, n_grid, sample_slice, out_nf, out_a_at, eval_cycles
):
    m = a0.shape[0]
    burst = a_c <= a0  # critical on arrival

    # log-spaced grid a0 -> a_c per sample; degenerate rows handled after
    safe_ac = np.maximum(a_c, a0 * (1.0 + 1e-9))
    t = np.linspace(0.0, 1.0, n_grid)[None, :]
    a_grid = np.exp(np.log(a0)[:, None] * (1.0 - t) + np.log(safe_ac)[:, None] * t)

    v = rate_of(a_grid, sample_slice)

    dormant = v[:, 0] <= 0.0  # below threshold at a0
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
        a_t = np.where(cyc >= nf, a_c, a_t)  # already failed
        a_t = np.where(dormant, a0, a_t)  # never grew
        a_t = np.where(burst, a_c, a_t)
        out_a_at[:, j] = a_t
