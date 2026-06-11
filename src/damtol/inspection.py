"""Inspection planning against probability of detection curves.

The standard MIL-HDBK-1823 form: detection probability is a lognormal
CDF in crack size. An inspection at cycle t sees the crack at its
then-current size; a detected crack is repaired or the part retired, so
it no longer contributes to failure. The risk integrand for a sample
that fails at N_f is the product of miss probabilities over every
inspection that happens before the failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


class PODCurve:
    """POD(a) = Phi((ln a - ln a50) / sigma)."""

    def __init__(self, a50, sigma):
        if a50 <= 0 or sigma <= 0:
            raise ValueError("a50 and sigma must be positive")
        self.a50 = a50
        self.sigma = sigma

    @classmethod
    def from_a50_a90(cls, a50, a90):
        """Build from the two sizes usually quoted for an NDT technique."""
        if not a90 > a50 > 0:
            raise ValueError("need a90 > a50 > 0")
        sigma = (np.log(a90) - np.log(a50)) / stats.norm.ppf(0.90)
        return cls(a50, sigma)

    def pod(self, a):
        a = np.asarray(a, dtype=float)
        with np.errstate(divide="ignore"):
            z = (np.log(np.maximum(a, 1e-300)) - np.log(self.a50)) / self.sigma
        return stats.norm.cdf(z)

    def __repr__(self):
        return f"PODCurve(a50={self.a50:g}, sigma={self.sigma:g})"


@dataclass
class InspectionPlan:
    times: list          # cycle counts of each inspection
    pod: PODCurve

    @classmethod
    def at_interval(cls, interval, service_cycles, pod):
        """Inspections every `interval` cycles, excluding 0 and anything
        at or beyond the end of service (it could not prevent anything)."""
        if interval <= 0:
            raise ValueError("interval must be positive")
        times = list(np.arange(interval, service_cycles, interval, dtype=float))
        return cls(times=times, pod=pod)


@dataclass
class InspectionOutcome:
    pof_unmitigated: float    # no inspections at all
    pof_inspected: float      # with the plan applied
    mean_detections: float    # expected detections per part over the life
    times: list = field(default_factory=list)

    @property
    def risk_reduction(self):
        if self.pof_unmitigated == 0.0:
            return 1.0
        return 1.0 - self.pof_inspected / self.pof_unmitigated


def apply_plan(life, service_cycles, plan):
    """Evaluate an inspection plan against a LifeResult.

    The LifeResult must have been grown with eval_cycles covering every
    inspection time in the plan (study.run does this wiring).
    """
    nf = life.cycles_to_failure
    fails = nf <= service_cycles
    pof_unmitigated = float(np.mean(fails))

    if not plan.times:
        return InspectionOutcome(pof_unmitigated, pof_unmitigated, 0.0, [])

    if life.eval_cycles is None:
        raise ValueError("life result carries no crack size checkpoints")
    cols = {}
    for t in plan.times:
        match = np.nonzero(np.isclose(life.eval_cycles, t))[0]
        if match.size == 0:
            raise ValueError(f"no crack size checkpoint at inspection time {t}")
        cols[t] = match[0]

    n = nf.shape[0]
    miss_all = np.ones(n)
    detect_expectation = np.zeros(n)
    survived = np.ones(n)  # prob. the crack is still in service (undetected)
    for t in sorted(plan.times):
        a_t = life.a_at[:, cols[t]]
        pod_t = plan.pod.pod(a_t)
        active = (t < nf) & (t < service_cycles)   # part intact when inspected
        pod_t = np.where(active, pod_t, 0.0)
        detect_expectation += survived * pod_t
        survived = survived * (1.0 - pod_t)
        miss_all *= np.where(active, 1.0 - plan.pod.pod(a_t), 1.0)

    pof_inspected = float(np.mean(fails * miss_all))
    return InspectionOutcome(
        pof_unmitigated=pof_unmitigated,
        pof_inspected=pof_inspected,
        mean_detections=float(np.mean(detect_expectation)),
        times=sorted(plan.times),
    )


def sweep_intervals(grow_with_checkpoints, intervals, service_cycles, pod):
    """Risk versus inspection interval trade study.

    grow_with_checkpoints : callable taking eval_cycles and returning a
    LifeResult, so the caller controls geometry, law and samples.
    Returns a list of (interval, InspectionOutcome).
    """
    out = []
    for interval in intervals:
        plan = InspectionPlan.at_interval(interval, service_cycles, pod)
        life = grow_with_checkpoints(plan.times if plan.times else [service_cycles])
        out.append((interval, apply_plan(life, service_cycles, plan)))
    return out
