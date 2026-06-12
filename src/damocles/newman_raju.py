"""Newman-Raju stress intensity factor equations for part-through cracks
in finite plates under tension.

Source: NASA TM-85793 (Newman & Raju, 1984), "Stress-Intensity Factor
Equations for Cracks in Three-Dimensional Finite Bodies Subjected to
Tension and Bending Loads", equations 14-19 and 26-29 (surface crack)
and 37-51 (corner crack). The coefficients below were cross-checked
against NASA TP-1578 (1979) and NASA TM-83200 (1981); the three sources
agree exactly. Engineering Fracture Mechanics 15 (1981) 185-192 is the
journal version of the surface-crack set.

Conventions: a = crack depth, c = surface half-length, t = plate
thickness, b = plate half-width (surface crack) or width (corner crack),
phi = parametric angle, 0 at the free surface, pi/2 at maximum depth.
K = sigma * sqrt(pi * a / Q) * F, so the Y(a) factor used by the growth
code is F / sqrt(Q). Stated accuracy: within +-5% of the underlying
finite element results for a/t <= 0.8, 0.2 <= a/c <= 2, c/b < 0.5.
"""

from __future__ import annotations

import numpy as np

from .fracture import Geometry


def shape_factor_q(aspect):
    """Q = square of the complete elliptic integral of the second kind,
    Rawe approximation (TM-85793 eqs. 2a-2b), max K error ~0.13%."""
    aspect = np.asarray(aspect, dtype=float)
    return np.where(
        aspect <= 1.0,
        1.0 + 1.464 * np.maximum(aspect, 0.0) ** 1.65,
        1.0 + 1.464 * (1.0 / aspect) ** 1.65,
    )


def surface_crack_f(a, c, t, b, phi):
    """Boundary correction F for a semi-elliptical surface crack under
    remote tension (TM-85793 eqs. 15-19 for a/c <= 1, 26-29 for a/c > 1).
    All arguments broadcast."""
    a = np.asarray(a, dtype=float)
    c = np.asarray(c, dtype=float)
    ac = a / c
    at = a / t
    sin_p, cos_p = np.sin(phi), np.cos(phi)

    # a/c <= 1 branch (eqs. 16-19, 10)
    m1_lo = 1.13 - 0.09 * ac
    m2_lo = -0.54 + 0.89 / (0.2 + ac)
    m3_lo = 0.5 - 1.0 / (0.65 + ac) + 14.0 * (1.0 - np.minimum(ac, 1.0)) ** 24
    g_lo = 1.0 + (0.1 + 0.35 * at**2) * (1.0 - sin_p) ** 2
    f_phi_lo = (ac**2 * cos_p**2 + sin_p**2) ** 0.25

    # a/c > 1 branch (eqs. 26-29, 13)
    ca = np.where(ac > 0, 1.0 / ac, np.inf)
    m1_hi = np.sqrt(np.minimum(ca, 1e30)) * (1.0 + 0.04 * ca)
    m2_hi = 0.2 * ca**4
    m3_hi = -0.11 * ca**4
    g_hi = 1.0 + (0.1 + 0.35 * ca * at**2) * (1.0 - sin_p) ** 2
    f_phi_hi = (ca**2 * sin_p**2 + cos_p**2) ** 0.25

    lo = ac <= 1.0
    m = np.where(lo, m1_lo + m2_lo * at**2 + m3_lo * at**4,
                 m1_hi + m2_hi * at**2 + m3_hi * at**4)
    g = np.where(lo, g_lo, g_hi)
    f_phi = np.where(lo, f_phi_lo, f_phi_hi)

    # finite width (eq. 11), sec in radians, valid c/b < 0.5
    fw = np.sqrt(1.0 / np.cos(np.pi * c / (2.0 * b) * np.sqrt(at)))
    return m * g * f_phi * fw


def corner_crack_f(a, c, t, b, phi):
    """Boundary correction F for a quarter-elliptical corner crack under
    remote tension (TM-85793 eqs. 38-44 for a/c <= 1, 47-51)."""
    a = np.asarray(a, dtype=float)
    c = np.asarray(c, dtype=float)
    ac = a / c
    at = a / t
    ct = c / t
    sin_p, cos_p = np.sin(phi), np.cos(phi)

    m1_lo = 1.08 - 0.03 * ac
    m2_lo = -0.44 + 1.06 / (0.3 + ac)
    m3_lo = -0.5 + 0.25 * ac + 14.8 * (1.0 - np.minimum(ac, 1.0)) ** 15
    g1_lo = 1.0 + (0.08 + 0.4 * at**2) * (1.0 - sin_p) ** 3
    g2_lo = 1.0 + (0.08 + 0.15 * at**2) * (1.0 - cos_p) ** 3
    f_phi_lo = (ac**2 * cos_p**2 + sin_p**2) ** 0.25

    ca = np.where(ac > 0, 1.0 / ac, np.inf)
    m1_hi = np.sqrt(np.minimum(ca, 1e30)) * (1.08 - 0.03 * ca)
    m2_hi = 0.375 * ca**2
    m3_hi = -0.25 * ca**2
    # note: (c/t), not (a/t), in the a/c > 1 g functions (eqs. 50-51)
    g1_hi = 1.0 + (0.08 + 0.4 * ct**2) * (1.0 - sin_p) ** 3
    g2_hi = 1.0 + (0.08 + 0.15 * ct**2) * (1.0 - cos_p) ** 3
    f_phi_hi = (ca**2 * sin_p**2 + cos_p**2) ** 0.25

    lo = ac <= 1.0
    m = np.where(lo, m1_lo + m2_lo * at**2 + m3_lo * at**4,
                 m1_hi + m2_hi * at**2 + m3_hi * at**4)
    g1 = np.where(lo, g1_lo, g1_hi)
    g2 = np.where(lo, g2_lo, g2_hi)
    f_phi = np.where(lo, f_phi_lo, f_phi_hi)

    # finite width (eq. 44), the 1984 extension, valid c/b < 0.5
    lam = (c / b) * np.sqrt(at)
    fw = 1.0 - 0.2 * lam + 9.4 * lam**2 - 19.4 * lam**3 + 27.1 * lam**4
    return m * g1 * g2 * f_phi * fw


class _NewmanRajuBase(Geometry):
    """Y(a) at fixed aspect ratio a/c, evaluated at a chosen point on the
    crack front. K = Y(a) * sigma * sqrt(pi * a)."""

    _f = None  # set by subclass

    def __init__(self, thickness, width, aspect=1.0, tip="depth"):
        if thickness <= 0 or width <= 0:
            raise ValueError("thickness and width must be positive")
        if not 0.2 <= aspect <= 2.0:
            raise ValueError("Newman-Raju equations are fitted for "
                             "0.2 <= a/c <= 2.0")
        if tip not in ("depth", "surface", "max"):
            raise ValueError("tip must be 'depth', 'surface' or 'max'")
        self.thickness = thickness
        self.width = width
        self.aspect = aspect
        self.tip = tip
        # a/t < 1 and c/b < 0.5 bound the fit; growth stops at whichever
        # comes first
        self.a_max = min(0.98 * thickness, 0.49 * width * aspect)

    def _y_at(self, a, phi):
        c = a / self.aspect
        f = type(self)._f(a, c, self.thickness, self.width, phi)
        return f / np.sqrt(shape_factor_q(self.aspect))

    def y(self, a):
        a = np.asarray(a, dtype=float)
        if self.tip == "depth":
            return self._y_at(a, np.pi / 2.0)
        if self.tip == "surface":
            return self._y_at(a, 0.0)
        return np.maximum(self._y_at(a, np.pi / 2.0), self._y_at(a, 0.0))


class NewmanRajuSurfaceCrack(_NewmanRajuBase):
    """Semi-elliptical surface crack in a finite plate, remote tension."""
    _f = staticmethod(surface_crack_f)


class NewmanRajuCornerCrack(_NewmanRajuBase):
    """Quarter-elliptical corner crack in a finite plate, remote tension."""
    _f = staticmethod(corner_crack_f)


# make the solutions reachable from YAML study files
from .fracture import GEOMETRIES  # noqa: E402

GEOMETRIES["nr-surface"] = NewmanRajuSurfaceCrack
GEOMETRIES["nr-corner"] = NewmanRajuCornerCrack
