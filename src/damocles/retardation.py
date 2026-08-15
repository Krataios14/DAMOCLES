"""Overload-retardation models for spectrum crack growth.

Implements the Willenborg (1971) retardation model as an opt-in
addition to the no-interaction spectrum integration.  The present
no-interaction calculation remains the default; retardation is enabled
by calling ``grow_spectrum_retarded`` instead of ``grow_spectrum``.

References
----------
Willenborg, J.D., Engle, R.M. and Wood, H.A., "A Crack Growth
Retardation Spectrum Model," AFFDL-TM-71-1-FBR, 1971.

Broek, D., "Elementary Engineering Fracture Mechanics," 4th ed.,
Springer, 1986.  Chapter 12 reviews retardation models including
Willenborg.

Forman, R.G. and Shivakumar, V., "Growth Rate Equation for
Part-through Crack in Residual Stress Fields," NASA TM-88963, 1986.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WillenborgState:
    """Per-sample retardation state carried across cycles.

    Attributes
    ----------
    k_max : np.ndarray
        Maximum stress intensity factor encountered so far [MPa sqrt(m)].
    r_p : np.ndarray
        Plastic zone radius at the overload, computed from ``k_max``
        using the Irwin estimate.  Updated when a new maximum is set.
    active : np.ndarray
        Boolean mask; True where retardation is currently active
        (crack has not yet grown beyond the overload plastic zone).
    """

    k_max: np.ndarray
    r_p: np.ndarray
    active: np.ndarray


def plastic_zone_radius(k_max, s_yield):
    """Irwin plastic zone radius: r_p = (1/pi) * (K_max / sigma_y)^2.

    Parameters
    ----------
    k_max : array-like
        Maximum stress intensity factor [MPa sqrt(m)].
    s_yield : float
        Material yield strength [MPa].

    Returns
    -------
    np.ndarray
        Plastic zone radius [m].
    """
    k_max = np.asarray(k_max, dtype=float)
    return (1.0 / np.pi) * (k_max / s_yield) ** 2


def init_state(k_max_0, s_yield):
    """Create initial retardation state from the first cycle's K_max.

    Parameters
    ----------
    k_max_0 : array-like
        Stress intensity factor from the first cycle [MPa sqrt(m)].
    s_yield : float
        Material yield strength [MPa].

    Returns
    -------
    WillenborgState
    """
    k = np.asarray(k_max_0, dtype=float)
    rp = plastic_zone_radius(k, s_yield)
    # The first cycle establishes the baseline; retardation activates
    # only when a *subsequent* cycle exceeds this baseline K_max.
    return WillenborgState(k_max=k.copy(), r_p=rp.copy(),
                           active=np.zeros_like(k, dtype=bool))


def retardation_factor(a, r_p, active):
    """Willenborg retardation factor f_R.

    ``f_R = r_p / (2 * (a + r_p))``

    When the crack tip has advanced beyond the overload plastic zone
    (``a > 2 * r_p``), retardation is inactive and ``f_R = 0``.

    Parameters
    ----------
    a : np.ndarray
        Current crack size [m].
    r_p : np.ndarray
        Plastic zone radius at the overload [m].
    active : np.ndarray
        Boolean mask of samples where retardation is active.

    Returns
    -------
    np.ndarray
        Retardation factor in [0, 0.5].
    """
    a = np.asarray(a, dtype=float)
    r_p = np.asarray(r_p, dtype=float)
    f = np.zeros_like(a)
    inside_zone = active & (a <= 2.0 * r_p)
    denom = 2.0 * (a + r_p)
    safe = inside_zone & (denom > 0.0)
    f[safe] = r_p[safe] / denom[safe]
    return f


def effective_dk(dk, f_r):
    """Apply retardation to the stress intensity range.

    ``dK_eff = dK * (1 - f_R)``

    Parameters
    ----------
    dk : np.ndarray
        Unretarded stress intensity range [MPa sqrt(m)].
    f_r : np.ndarray
        Retardation factor from :func:`retardation_factor`.

    Returns
    -------
    np.ndarray
        Effective (retarded) stress intensity range.
    """
    return np.asarray(dk, dtype=float) * (1.0 - np.asarray(f_r, dtype=float))
