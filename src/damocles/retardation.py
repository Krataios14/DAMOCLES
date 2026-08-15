"""Willenborg overload retardation for ordered crack-growth spectra.

This module implements the original effective-stress formulation from
Willenborg, Engle, and Wood (1971).  A load establishes a controlling
plastic-zone boundary.  Later cycles are retarded while their own plastic
zones remain inside that boundary.

For plane stress, the overload-zone size is

    r_p,OL = (1 / pi) * (K_max,OL / sigma_y) ** 2

and the residual stress-intensity factor for a contained cycle is

    K_R = K_max,OL * sqrt(1 - (a_i - a_OL) / r_p,OL) - K_max,i.

Negative values of ``K_R`` are clipped to zero.  The same ``K_R`` is
subtracted from the maximum and minimum stress intensity of the cycle.  This
leaves delta K unchanged until the effective stress ratio becomes negative.
The implementation then follows the zero-effective-R convention used in
early applications of the model: ``R_eff`` is set to zero and
``delta K_eff`` is set to ``K_max_eff``.

Validity limits
---------------
- The plastic-zone expression is the plane-stress approximation.
- The model is an empirical LEFM load-interaction model and does not replace
  elastic-plastic analysis when the plastic zone is not small relative to the
  crack or remaining ligament.
- The zero-effective-R convention is one historical implementation choice;
  this implementation does not model negative-R crack-growth data separately.
- Predictions should be supported by representative spectrum-test data for
  safety-critical use.

References
----------
Willenborg, J. D., Engle, R. M., and Wood, H. A., "A Crack Growth
Retardation Model Using an Effective Stress Concept," AFFDL-TM-71-1-FBR,
1971. https://doi.org/10.21236/ADA956517

AFGROW Damage Tolerance Design Handbook, Section 5.2.1.2, equations 5.2.3
through 5.2.6.
"""

from __future__ import annotations

import hashlib
import struct
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from types import MappingProxyType

import numpy as np

from .spectrum import MAX_ORDERED_CYCLES, OrderedCycle, SpectrumSequence


MAX_RETARDED_CYCLES = 100_000_000.0
MAX_RETARDED_BLOCKS = 1_000_000
MAX_RETARDED_WORK = 100_000_000


def _positive_finite(value, name):
    if isinstance(value, (bool, str, bytes)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _positive_integer(value, name, maximum):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    if result > maximum:
        raise ValueError(f"{name} exceeds the limit of {maximum:,}")
    return result


def _half_cycle_ticks(value, name):
    """Convert an exact full/half-cycle boundary to integer half-cycle ticks."""
    value = _positive_finite(value, name)
    doubled = value * 2.0
    if not doubled.is_integer():
        raise ValueError(f"{name} must be an exact full/half-cycle boundary")
    return int(doubled)


@dataclass(frozen=True, slots=True)
class _BoundarySnapshot:
    """Private recomputed endpoint data trusted only within one call."""

    sequence: SpectrumSequence = field(repr=False, compare=False)
    sequence_fingerprint: str
    block_ticks: int
    n_cycles: int
    endpoint_steps: Mapping[int, int] = field(repr=False, compare=False)

    def steps_to(self, cycles, name="cycle horizon"):
        target_ticks = _half_cycle_ticks(cycles, name)
        complete_blocks, remainder = divmod(target_ticks, self.block_ticks)
        steps = complete_blocks * self.n_cycles
        if remainder == 0:
            return steps
        within_block = self.endpoint_steps.get(remainder)
        if within_block is None:
            raise ValueError(
                f"{name} must coincide with an ordered cycle-record endpoint"
            )
        return steps + within_block

    def steps_many(self, cycles, name="cycle horizon"):
        return tuple(self.steps_to(value, name) for value in cycles)


def _build_boundary_snapshot(sequence):
    """Recompute canonical endpoints and a full structural fingerprint."""
    if type(sequence) is not SpectrumSequence:
        raise TypeError("sequence must be an exact SpectrumSequence")
    try:
        cycles = object.__getattribute__(sequence, "cycles")
        peak_stress = object.__getattribute__(sequence, "peak_stress")
        dropped = object.__getattribute__(sequence, "dropped_compressive")
    except AttributeError as exc:
        raise ValueError("sequence structure is incomplete") from exc
    if type(cycles) is not tuple:
        raise TypeError("sequence cycles must be an exact tuple")
    if not 1 <= len(cycles) <= MAX_ORDERED_CYCLES:
        raise ValueError("sequence cycle count is outside its bounded range")
    if type(peak_stress) is not float or type(dropped) is not float:
        raise TypeError("sequence summary values must be floats")
    if not np.isfinite(peak_stress) or peak_stress <= 0.0:
        raise ValueError("sequence peak_stress must be finite and positive")
    if not np.isfinite(dropped) or dropped < 0.0:
        raise ValueError("sequence dropped_compressive must be finite and non-negative")

    fingerprint = hashlib.sha256(b"damocles-ordered-boundaries-v1\0")
    fingerprint.update(struct.pack("!dd", peak_stress, dropped))
    endpoint_steps = {}
    elapsed = 0
    maximum_cycle_peak = 0.0
    for expected_index, cycle in enumerate(cycles):
        if type(cycle) is not OrderedCycle:
            raise TypeError("sequence cycles must contain exact OrderedCycle values")
        try:
            delta_sigma = object.__getattribute__(cycle, "delta_sigma")
            stress_ratio = object.__getattribute__(cycle, "stress_ratio")
            count = object.__getattribute__(cycle, "count")
            index = object.__getattribute__(cycle, "index")
        except AttributeError as exc:
            raise ValueError("ordered cycle structure is incomplete") from exc
        if (
            type(delta_sigma) is not float
            or type(stress_ratio) is not float
            or type(count) is not float
            or type(index) is not int
        ):
            raise TypeError("ordered cycle canonical fields have invalid types")
        if (
            not np.isfinite(delta_sigma)
            or delta_sigma <= 0.0
            or not np.isfinite(stress_ratio)
            or not 0.0 <= stress_ratio < 1.0
            or count not in (0.5, 1.0)
            or index != expected_index
        ):
            raise ValueError("ordered cycle canonical fields are invalid")
        ticks = int(count * 2.0)
        elapsed += ticks
        endpoint_steps[elapsed] = expected_index + 1
        maximum_cycle_peak = max(maximum_cycle_peak, delta_sigma / (1.0 - stress_ratio))
        fingerprint.update(
            struct.pack("!dddq", delta_sigma, stress_ratio, count, index)
        )
    if peak_stress < maximum_cycle_peak:
        raise ValueError("sequence peak_stress cannot be below a cycle peak")
    if elapsed <= 0 or len(endpoint_steps) != len(cycles):
        raise ValueError("sequence must have positive, distinct record endpoints")
    return _BoundarySnapshot(
        sequence=sequence,
        sequence_fingerprint="sha256:" + fingerprint.hexdigest(),
        block_ticks=elapsed,
        n_cycles=len(cycles),
        endpoint_steps=MappingProxyType(endpoint_steps),
    )


_BOUNDARY_INDEX_REGISTRY = weakref.WeakKeyDictionary()


class OrderedBoundaryIndex:
    """Factory-prepared endpoint index, structurally revalidated before reuse.

    Python-level frozen syntax is not treated as an integrity boundary. Every
    accepted instance must be registered by :meth:`from_sequence`; its exact
    sequence identity, full canonical fingerprint, scalar fields, and bounded
    endpoint table are checked against a freshly recomputed private snapshot.
    """

    __slots__ = (
        "_sequence",
        "_sequence_fingerprint",
        "_block_ticks",
        "_n_cycles",
        "_endpoint_steps",
        "__weakref__",
    )

    def __init__(self, *_args, **_kwargs):
        raise TypeError(
            "OrderedBoundaryIndex cannot be constructed directly; "
            "use OrderedBoundaryIndex.from_sequence"
        )

    def __setattr__(self, _name, _value):
        raise AttributeError("OrderedBoundaryIndex is immutable")

    def __delattr__(self, _name):
        raise AttributeError("OrderedBoundaryIndex is immutable")

    @classmethod
    def from_sequence(cls, sequence):
        if cls is not OrderedBoundaryIndex:
            raise TypeError("OrderedBoundaryIndex cannot be subclassed")
        snapshot = _build_boundary_snapshot(sequence)
        instance = object.__new__(cls)
        object.__setattr__(instance, "_sequence", sequence)
        object.__setattr__(
            instance, "_sequence_fingerprint", snapshot.sequence_fingerprint
        )
        object.__setattr__(instance, "_block_ticks", snapshot.block_ticks)
        object.__setattr__(instance, "_n_cycles", snapshot.n_cycles)
        object.__setattr__(instance, "_endpoint_steps", snapshot.endpoint_steps)
        _BOUNDARY_INDEX_REGISTRY[instance] = snapshot
        return instance

    @property
    def block_ticks(self):
        return _validated_bound_snapshot(self).block_ticks

    @property
    def n_cycles(self):
        return _validated_bound_snapshot(self).n_cycles

    @property
    def endpoint_steps(self):
        return _validated_bound_snapshot(self).endpoint_steps

    def require_sequence(self, sequence):
        """Validate this prepared index against the exact supplied sequence."""
        _validated_boundary_snapshot(sequence, self)

    def steps_to(self, cycles, name="cycle horizon"):
        """Validate once, then return records ending exactly at ``cycles``."""
        return _validated_bound_snapshot(self).steps_to(cycles, name)

    def steps_many(self, cycles, name="cycle horizon"):
        """Validate once, then perform O(number of checkpoints) lookups."""
        return _validated_bound_snapshot(self).steps_many(cycles, name)


def _raw_index_fields(index):
    if type(index) is not OrderedBoundaryIndex:
        raise TypeError("boundary_index must be an OrderedBoundaryIndex")
    try:
        sequence = object.__getattribute__(index, "_sequence")
        fingerprint = object.__getattribute__(index, "_sequence_fingerprint")
        block_ticks = object.__getattribute__(index, "_block_ticks")
        n_cycles = object.__getattribute__(index, "_n_cycles")
        endpoint_steps = object.__getattribute__(index, "_endpoint_steps")
    except AttributeError as exc:
        raise ValueError("boundary index structure is incomplete") from exc
    if type(sequence) is not SpectrumSequence:
        raise TypeError("boundary index sequence must be an exact SpectrumSequence")
    if type(fingerprint) is not str:
        raise TypeError("boundary index fingerprint must be text")
    if type(block_ticks) is not int:
        raise TypeError("boundary index block_ticks must be an integer")
    if block_ticks <= 0:
        raise ValueError("boundary index block_ticks must be positive")
    if type(n_cycles) is not int:
        raise TypeError("boundary index n_cycles must be an integer")
    if not 1 <= n_cycles <= MAX_ORDERED_CYCLES:
        raise ValueError("boundary index n_cycles is outside its bounded range")
    if type(endpoint_steps) is not MappingProxyType:
        raise TypeError("boundary index endpoint table must be read-only")
    return sequence, fingerprint, block_ticks, n_cycles, endpoint_steps


def _validated_boundary_snapshot(sequence, boundary_index=None):
    recomputed = _build_boundary_snapshot(sequence)
    if boundary_index is None:
        return recomputed
    raw = _raw_index_fields(boundary_index)
    raw_sequence, fingerprint, block_ticks, n_cycles, endpoint_steps = raw
    try:
        registered = _BOUNDARY_INDEX_REGISTRY[boundary_index]
    except KeyError as exc:
        raise ValueError(
            "boundary index was not prepared by OrderedBoundaryIndex.from_sequence"
        ) from exc
    if raw_sequence is not sequence or registered.sequence is not sequence:
        raise ValueError(
            "boundary index belongs to a different SpectrumSequence instance"
        )
    if registered.sequence_fingerprint != recomputed.sequence_fingerprint:
        raise ValueError("boundary index source sequence changed after preparation")
    if fingerprint != registered.sequence_fingerprint:
        raise ValueError("boundary index fingerprint is inconsistent")
    if block_ticks != registered.block_ticks or block_ticks != recomputed.block_ticks:
        raise ValueError("boundary index block_ticks is inconsistent")
    if n_cycles != registered.n_cycles or n_cycles != recomputed.n_cycles:
        raise ValueError("boundary index n_cycles is inconsistent")
    if endpoint_steps is not registered.endpoint_steps:
        raise ValueError("boundary index endpoint table is inconsistent")
    if len(registered.endpoint_steps) != len(recomputed.endpoint_steps):
        raise ValueError("boundary index endpoint table is inconsistent")
    missing = object()
    for ticks, expected_step in recomputed.endpoint_steps.items():
        registered_step = registered.endpoint_steps.get(ticks, missing)
        if type(registered_step) is not int or registered_step != expected_step:
            raise ValueError("boundary index endpoint table is inconsistent")
    return recomputed


def _validated_bound_snapshot(index):
    sequence, *_ = _raw_index_fields(index)
    return _validated_boundary_snapshot(sequence, index)


def _select_boundary_index(sequence, boundary_index):
    """Return a private recomputed snapshot, never caller-owned index data."""
    return _validated_boundary_snapshot(sequence, boundary_index)


def cycle_steps_to_boundary(
    sequence, cycles, name="cycle horizon", boundary_index=None
):
    """Return records ending exactly at ``cycles`` or reject the boundary.

    Ordered records have counts of exactly 0.5 or 1.0, so integer half-cycle
    ticks avoid floating tolerances. A half-cycle value can still fall inside
    a full-cycle record; only cumulative record endpoints are accepted.
    """
    snapshot = _select_boundary_index(sequence, boundary_index)
    return snapshot.steps_to(cycles, name)


def prospective_cycle_steps(sequence, max_cycles, max_blocks, boundary_index=None):
    """Exactly bound ordered records evaluated for one non-failing sample."""
    max_blocks = _positive_integer(max_blocks, "max_blocks", MAX_RETARDED_BLOCKS)
    snapshot = _select_boundary_index(sequence, boundary_index)
    steps_to_cycle_limit = snapshot.steps_to(max_cycles, "max_cycles")
    return min(max_blocks * snapshot.n_cycles, steps_to_cycle_limit)


@dataclass(frozen=True)
class WillenborgConfig:
    """Bounded high-level configuration for an ordered mission study.

    ``enabled=False`` retains the ordered cycle-by-cycle integration but
    applies no load-interaction correction. This makes the no-retardation
    baseline explicit without converting the mission to merged classes.
    Omitted cycle and block limits are derived from the study service life.
    """

    enabled: bool = True
    yield_strength: float | None = None
    max_cycles: float | None = None
    max_blocks: int | None = None
    max_work: int = MAX_RETARDED_WORK

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be a boolean")
        if self.enabled:
            if self.yield_strength is None:
                raise ValueError(
                    "yield_strength is required when Willenborg is enabled"
                )
            yield_strength = _positive_finite(self.yield_strength, "yield_strength")
        else:
            if self.yield_strength is not None:
                raise ValueError(
                    "yield_strength must be omitted when retardation is disabled"
                )
            yield_strength = None

        max_cycles = self.max_cycles
        if max_cycles is not None:
            max_cycles = _positive_finite(max_cycles, "max_cycles")
            if max_cycles > MAX_RETARDED_CYCLES:
                raise ValueError(
                    f"max_cycles exceeds the limit of {MAX_RETARDED_CYCLES:,.0f}"
                )
        max_blocks = self.max_blocks
        if max_blocks is not None:
            max_blocks = _positive_integer(
                max_blocks, "max_blocks", MAX_RETARDED_BLOCKS
            )
        max_work = _positive_integer(self.max_work, "max_work", MAX_RETARDED_WORK)
        object.__setattr__(self, "yield_strength", yield_strength)
        object.__setattr__(self, "max_cycles", max_cycles)
        object.__setattr__(self, "max_blocks", max_blocks)
        object.__setattr__(self, "max_work", max_work)

    @classmethod
    def disabled(cls, **limits):
        """Return an explicit ordered no-retardation baseline."""
        return cls(enabled=False, **limits)

    @classmethod
    def from_spec(cls, spec):
        """Parse the strict ``retardation`` mapping used by study files."""
        if not isinstance(spec, dict):
            raise TypeError("retardation must be a mapping")
        values = dict(spec)
        allowed = {"model", "yield_strength", "max_cycles", "max_blocks", "max_work"}
        extra = set(values) - allowed
        if extra:
            raise ValueError(f"retardation has unexpected keys {sorted(extra)}")
        model = values.pop("model", None)
        if model == "willenborg":
            if "yield_strength" not in values:
                raise ValueError("willenborg retardation needs yield_strength")
            return cls(enabled=True, **values)
        if model == "none":
            if "yield_strength" in values:
                raise ValueError(
                    "yield_strength is not allowed when retardation model is none"
                )
            return cls.disabled(**values)
        raise ValueError("retardation model must be 'willenborg' or 'none'")

    def to_spec(self):
        """Return a JSON/YAML-safe representation for evidence manifests."""
        result = {
            "model": "willenborg" if self.enabled else "none",
            "max_work": self.max_work,
        }
        if self.yield_strength is not None:
            result["yield_strength"] = self.yield_strength
        if self.max_cycles is not None:
            result["max_cycles"] = self.max_cycles
        if self.max_blocks is not None:
            result["max_blocks"] = self.max_blocks
        return result

    def resolve(self, sequence, service_cycles, n_samples, boundary_index=None):
        """Resolve limits and reject oversized work before sampling.

        Returns ``(max_cycles, max_blocks, prospective_cycle_steps)``.
        The work estimate is an upper bound on sample-cycle evaluations up
        to ``max_cycles`` and is independent of early fracture.
        """
        boundary_snapshot = _select_boundary_index(sequence, boundary_index)
        service_cycles = _positive_finite(service_cycles, "service_cycles")
        boundary_snapshot.steps_to(service_cycles, "service_cycles")
        n_samples = _positive_integer(n_samples, "n_samples", MAX_RETARDED_WORK)
        max_cycles = service_cycles if self.max_cycles is None else self.max_cycles
        if max_cycles < service_cycles:
            raise ValueError("max_cycles must cover the complete service life")
        if max_cycles > MAX_RETARDED_CYCLES:
            raise ValueError(
                f"resolved max_cycles exceeds the limit of {MAX_RETARDED_CYCLES:,.0f}"
            )
        boundary_snapshot.steps_to(max_cycles, "max_cycles")

        max_cycle_ticks = _half_cycle_ticks(max_cycles, "max_cycles")
        block_ticks = boundary_snapshot.block_ticks
        required_blocks = (max_cycle_ticks + block_ticks - 1) // block_ticks
        max_blocks = required_blocks if self.max_blocks is None else self.max_blocks
        if max_blocks < required_blocks:
            raise ValueError("max_blocks must cover max_cycles for this sequence")
        if max_blocks > MAX_RETARDED_BLOCKS:
            raise ValueError(
                f"resolved max_blocks exceeds the limit of {MAX_RETARDED_BLOCKS:,}"
            )

        cycle_steps = min(
            max_blocks * boundary_snapshot.n_cycles,
            boundary_snapshot.steps_to(max_cycles, "max_cycles"),
        )
        prospective_work = n_samples * cycle_steps
        if prospective_work > self.max_work:
            raise ValueError(
                "ordered mission exceeds max_work before sampling: "
                f"{prospective_work:,} > {self.max_work:,} sample-cycle steps"
            )
        return max_cycles, max_blocks, prospective_work


@dataclass
class WillenborgState:
    """Per-sample state for the currently controlling plastic zone.

    ``a_overload + r_p`` is the stored plastic-zone boundary.  A later cycle
    replaces this state only if its unretarded boundary extends farther.
    """

    k_max_ol: np.ndarray
    r_p: np.ndarray
    a_overload: np.ndarray
    active: np.ndarray


def init_state(n):
    """Create state for ``n`` samples with no controlling load recorded."""
    if n < 0:
        raise ValueError("n must be non-negative")
    return WillenborgState(
        k_max_ol=np.zeros(n),
        r_p=np.zeros(n),
        a_overload=np.zeros(n),
        active=np.zeros(n, dtype=bool),
    )


def plastic_zone_radius(k_max, s_yield):
    """Return the Willenborg plane-stress plastic-zone size.

    Parameters use MPa sqrt(m) for ``k_max`` and MPa for ``s_yield``, giving
    the zone size in metres.  The ``1 / pi`` coefficient is the full forward
    zone extent used by the Willenborg formulation; no additional factor of
    two is applied to the stored boundary.
    """
    if not np.isfinite(s_yield) or s_yield <= 0.0:
        raise ValueError("s_yield must be finite and positive")
    k_max = np.asarray(k_max, dtype=float)
    if np.any(~np.isfinite(k_max)) or np.any(k_max < 0.0):
        raise ValueError("k_max must be finite and non-negative")
    return (1.0 / np.pi) * (k_max / s_yield) ** 2


def residual_stress_intensity(a, k_max, k_max_ol, r_p, a_overload, active):
    """Return the original Willenborg residual intensity ``K_R``.

    This is equation 5.2.4 of the AFGROW handbook.  ``active`` identifies
    samples whose current plastic zone is contained by a previously stored
    zone.  Values outside a valid stored zone return zero.
    """
    a, k_max, k_max_ol, r_p, a_overload, active = np.broadcast_arrays(
        np.asarray(a, dtype=float),
        np.asarray(k_max, dtype=float),
        np.asarray(k_max_ol, dtype=float),
        np.asarray(r_p, dtype=float),
        np.asarray(a_overload, dtype=float),
        np.asarray(active, dtype=bool),
    )

    k_r = np.zeros(a.shape, dtype=float)
    valid = active & (r_p > 0.0)
    remaining = np.zeros(a.shape, dtype=float)
    np.divide(a - a_overload, r_p, out=remaining, where=valid)
    remaining = np.clip(1.0 - remaining, 0.0, 1.0)
    candidate = k_max_ol * np.sqrt(remaining) - k_max
    k_r[valid] = np.maximum(candidate[valid], 0.0)
    return k_r


def effective_kr(dk, stress_ratio, k_r):
    """Apply ``K_R`` and return ``(delta_K_eff, R_eff)``.

    The same residual intensity is subtracted from both ends of the cycle.
    If that would make ``R_eff`` negative, the historical zero-R convention
    is applied by setting ``K_min_eff`` to zero.
    """
    dk, stress_ratio, k_r = np.broadcast_arrays(
        np.asarray(dk, dtype=float),
        np.asarray(stress_ratio, dtype=float),
        np.asarray(k_r, dtype=float),
    )
    if np.any(stress_ratio >= 1.0):
        raise ValueError("stress_ratio must be less than 1")

    k_max = dk / (1.0 - stress_ratio)
    k_min = k_max * stress_ratio
    k_max_eff = np.maximum(k_max - np.maximum(k_r, 0.0), 0.0)
    k_min_eff = np.maximum(k_min - np.maximum(k_r, 0.0), 0.0)
    dk_eff = np.maximum(k_max_eff - k_min_eff, 0.0)

    r_eff = np.zeros_like(dk_eff)
    np.divide(k_min_eff, k_max_eff, out=r_eff, where=k_max_eff > 0.0)
    return dk_eff, r_eff
