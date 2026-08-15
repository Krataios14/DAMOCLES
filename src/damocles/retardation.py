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

from dataclasses import dataclass

import numpy as np


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


def residual_stress_intensity(a, k_max, k_max_ol, r_p, a_overload,
                              active):
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
