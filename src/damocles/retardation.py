"""Overload-retardation models for spectrum crack growth.

Implements the Willenborg (1971) retardation model as an opt-in
addition to the no-interaction spectrum integration.

The Willenborg model tracks the maximum stress intensity factor K_max
encountered so far.  When a new K_max is set (an overload), the Irwin
plastic zone radius is computed and the crack length at overload is
recorded.  Retardation continues while the current crack tip lies inside
the overload plastic zone, i.e. while

    a < a_OL + 2 * r_p          (plane stress)

where a_OL is the crack length at the time of overload and

    r_p = (1 / pi) * (K_max_OL / sigma_y)^2

is the Irwin estimate.

Within the zone, a residual compressive stress intensity K_R is
computed:

    K_R = f_R * K_max_OL,   f_R = r_p / (2 * (a - a_OL + r_p))

and subtracted from both K_max and K_min of every subsequent cycle:

    K_max_eff = K_max - K_R
    K_min_eff = max(K_min - K_R, 0)

Validity limits
---------------
- Plane stress assumed throughout (conservative for most metals).
- The Irwin zone estimate underestimates r_p for large overloads.
- The model does not account for crack closure.

References
----------
Willenborg, J.D., Engle, R.M. and Wood, H.A., "A Crack Growth
Retardation Spectrum Model," AFFDL-TM-71-1-FBR, 1971.

Broek, D., "Elementary Engineering Fracture Mechanics," 4th ed.,
Springer, 1986.  Chapter 12.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WillenborgState:
    """Per-sample retardation state carried across cycles.

    Attributes
    ----------
    k_max_ol : np.ndarray
        K_max at the time of the most recent overload [MPa sqrt(m)].
        Used for the residual stress computation K_R = f_R * k_max_ol.
    r_p : np.ndarray
        Plastic zone radius at the overload [m].
    a_overload : np.ndarray
        Crack length at the time of overload [m].
    active : np.ndarray
        True where retardation is currently active.
    """

    k_max_ol: np.ndarray
    r_p: np.ndarray
    a_overload: np.ndarray
    active: np.ndarray


def plastic_zone_radius(k_max, s_yield):
    """Irwin plastic zone radius (plane stress).

    r_p = (1 / pi) * (K / sigma_y)^2
    """
    k_max = np.asarray(k_max, dtype=float)
    return (1.0 / np.pi) * (k_max / s_yield) ** 2


def init_state(n):
    """Create initial retardation state with no overload recorded."""
    return WillenborgState(
        k_max_ol=np.zeros(n),
        r_p=np.zeros(n),
        a_overload=np.zeros(n),
        active=np.zeros(n, dtype=bool),
    )


def retardation_factor(a, r_p, a_overload, active):
    """Willenborg retardation factor f_R.

    f_R = r_p / (2 * (a - a_OL + r_p))

    Retardation is active while a < a_OL + 2 * r_p.
    """
    a = np.asarray(a, dtype=float)
    r_p = np.asarray(r_p, dtype=float)
    a_ol = np.asarray(a_overload, dtype=float)
    f = np.zeros_like(a)
    zone_boundary = a_ol + 2.0 * r_p
    inside_zone = active & (a < zone_boundary)
    denom = 2.0 * (a - a_ol + r_p)
    safe = inside_zone & (denom > 0.0)
    f[safe] = r_p[safe] / denom[safe]
    return f


def effective_kr(dk, k_max_ol, stress_ratio, f_r):
    """Compute effective K after Willenborg retardation.

    K_R = f_R * K_max_OL
    K_max_eff = K_max - K_R
    K_min_eff = max(K_min - K_R, 0)
    """
    dk = np.asarray(dk, dtype=float)
    k_max_ol = np.asarray(k_max_ol, dtype=float)
    f_r = np.asarray(f_r, dtype=float)

    k_max_cyc = dk / np.maximum(1.0 - stress_ratio, 1e-300)
    k_min_cyc = k_max_cyc * stress_ratio
    k_r = f_r * k_max_ol

    k_max_eff = np.maximum(k_max_cyc - k_r, 0.0)
    k_min_eff = np.maximum(k_min_cyc - k_r, 0.0)

    dk_eff = np.maximum(k_max_eff - k_min_eff, 0.0)
    with np.errstate(invalid="ignore"):
        r_eff = np.where(k_max_eff > 0.0, k_min_eff / k_max_eff, 0.0)

    return dk_eff, r_eff
