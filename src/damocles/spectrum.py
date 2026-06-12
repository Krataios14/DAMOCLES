"""Load spectra and rainflow cycle counting.

Rainflow counting follows ASTM E1049-85 section 5.4.4 (the "rainflow
counting" practice, not the simplified range-pair variant): ranges are
paired into full cycles where an intervening larger range allows it,
and what remains at the end of the history is counted as half cycles.

A Spectrum is a set of cycle classes (stress range, stress ratio, count
per block) describing one repeating block of service loading, e.g. one
flight. Cycle classes whose peak stress is compressive do no damage in
a LEFM model (the crack is closed) and are dropped with a note in
`dropped_compressive`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _peaks_valleys(history):
    """Strip a load history to its sequence of reversal points."""
    x = np.asarray(history, dtype=float)
    if x.size < 2:
        return x
    # collapse plateaus first so the slope test below sees true reversals
    x = np.concatenate([x[:1], x[1:][x[1:] != x[:-1]]])
    if x.size < 3:
        return x
    interior = (x[1:-1] - x[:-2]) * (x[2:] - x[1:-1]) < 0.0
    keep = np.concatenate([[True], interior, [True]])
    return x[keep]


def rainflow(history):
    """Count cycles in a load history per ASTM E1049-85, 5.4.4.

    Returns a list of (stress_range, mean, count) with count 1.0 for
    full cycles and 0.5 for the residual half cycles.
    """
    pts = list(_peaks_valleys(history))
    counted = []
    stack = []
    start = 0  # index into stack of the current starting point

    for p in pts:
        stack.append(p)
        while len(stack) - start >= 3:
            x = abs(stack[-1] - stack[-2])
            y = abs(stack[-2] - stack[-3])
            if x < y:
                break
            if start == len(stack) - 3:
                # Y contains the starting point: half cycle, drop the
                # first point and move the start forward
                counted.append((y, 0.5 * (stack[-2] + stack[-3]), 0.5))
                stack.pop(-3)
            else:
                counted.append((y, 0.5 * (stack[-2] + stack[-3]), 1.0))
                del stack[-3:-1]

    # everything left counts as half cycles
    for a, b in zip(stack[:-1], stack[1:]):
        counted.append((abs(b - a), 0.5 * (a + b), 0.5))
    return counted


@dataclass(frozen=True)
class CycleClass:
    delta_sigma: float    # stress range [MPa]
    stress_ratio: float   # sigma_min / sigma_max
    count: float          # cycles per block


class Spectrum:
    """A repeating block of constant amplitude cycle classes."""

    def __init__(self, classes, dropped_compressive=0.0, peak_stress=None):
        self.classes = [c for c in classes if c.count > 0 and c.delta_sigma > 0]
        self.dropped_compressive = dropped_compressive
        if not self.classes:
            raise ValueError("spectrum has no damaging cycle classes")
        # fracture is governed by the true peak of the block; binning and
        # other reductions must not be allowed to move it
        self._peak = (peak_stress if peak_stress is not None else
                      max(c.delta_sigma / (1.0 - c.stress_ratio)
                          for c in self.classes))

    @classmethod
    def from_history(cls, history, repeats=1.0):
        """Build a spectrum from one block of stress history [MPa] via
        rainflow counting. `repeats` scales the counts if the history
        represents several blocks."""
        counted = rainflow(history)
        classes = []
        dropped = 0.0
        for rng, mean, count in counted:
            s_max = mean + 0.5 * rng
            s_min = mean - 0.5 * rng
            if s_max <= 0.0:
                dropped += count
                continue
            # crack only sees the tensile part of the excursion
            eff_min = max(s_min, 0.0)
            r = eff_min / s_max
            classes.append(CycleClass(delta_sigma=s_max - eff_min,
                                      stress_ratio=r,
                                      count=count / repeats))
        merged = _merge(classes)
        return cls(merged, dropped_compressive=dropped / repeats)

    @classmethod
    def constant_amplitude(cls, delta_sigma, stress_ratio=0.0, count=1.0):
        return cls([CycleClass(delta_sigma, stress_ratio, count)])

    @property
    def cycles_per_block(self):
        return sum(c.count for c in self.classes)

    @property
    def peak_stress(self):
        return self._peak

    def binned(self, n_bins=16, damage_exponent=3.0):
        """Merge classes into range bins to bound the cost of growth
        integration for long measured histories. Counts are conserved.
        Each bin's range is the damage-equivalent value preserving
        sum(count * range^p) with p = damage_exponent, which should be
        close to the growth law exponent m; an arithmetic mean would
        understate the damage of the large cycles in the bin."""
        if len(self.classes) <= n_bins:
            return self
        p = float(damage_exponent)
        ranges = np.array([c.delta_sigma for c in self.classes])
        edges = np.quantile(ranges, np.linspace(0.0, 1.0, n_bins + 1))
        which = np.clip(np.digitize(ranges, edges[1:-1], right=True), 0, n_bins - 1)
        out = []
        for b in range(n_bins):
            members = [c for c, k in zip(self.classes, which) if k == b]
            if not members:
                continue
            w = np.array([c.count for c in members])
            ds = np.array([c.delta_sigma for c in members])
            rr = np.array([c.stress_ratio for c in members])
            ds_eq = float(np.average(ds**p, weights=w) ** (1.0 / p))
            out.append(CycleClass(ds_eq, float(np.average(rr, weights=w)),
                                  float(w.sum())))
        return Spectrum(_merge(out), self.dropped_compressive,
                        peak_stress=self.peak_stress)


def _merge(classes):
    """Combine classes with identical (range, ratio)."""
    acc = {}
    for c in classes:
        key = (round(c.delta_sigma, 9), round(c.stress_ratio, 9))
        acc[key] = acc.get(key, 0.0) + c.count
    return [CycleClass(k[0], k[1], v) for k, v in sorted(acc.items())]
