"""The NASGRO (Forman-Mettu) fatigue crack growth equation.

da/dN = C * [((1-f)/(1-R)) * dK]^n * (1 - dKth/dK)^p / (1 - Kmax/Kc)^q

with Newman's plasticity-induced crack opening function f and a
threshold that depends on stress ratio and crack size. Sources:

- Mettu, Shivakumar, Beek, Yeh, Williams, Forman, McMahon, Newman,
  "NASGRO 3.0 - A software for analyzing aging aircraft", NASA NTRS
  19990028759 (equation 1 and threshold equation 2, a0 = 0.0381 mm).
- Newman, "A crack opening stress equation for fatigue crack growth",
  Int. J. Fracture 24 (1984) R131-R135 (the f function), as reprinted
  in e.g. Maierhofer et al., Eng. Fract. Mech. 59 (2014) and the
  IntechOpen chapter "Good Practice for Fatigue Crack Growth Curves
  Description".

The equation is unit-agnostic as long as C, dK and Kc are consistent;
the bundled material database stores published US-customary constants
and converts to SI (m/cycle, MPa sqrt(m)) on load.

A note on the threshold parameter: the equation below reproduces the
NASA primary source, in which the constant (called DK0 there, DK1 in
the later NASGRO 4 materials files) equals the long-crack threshold at
R = 0. NASGRO 4 reparametrised the threshold in a way that is not
fully public; treating DK1 as the R = 0 threshold is the documented,
slightly conservative reading used here.
"""

from __future__ import annotations

import numpy as np

# NASGRO's fixed intrinsic crack length, 0.0015 in (El Haddad style
# small-crack correction), in metres
A0_INTRINSIC_M = 0.0381e-3


def newman_opening_function(stress_ratio, alpha, smax_sigma0):
    """Newman's crack opening ratio f = Kop/Kmax.

    alpha       : constraint factor, 1 = plane stress to 3 = plane strain
    smax_sigma0 : ratio of peak stress to flow stress, 0.3 typical
    """
    r = float(stress_ratio)
    a0 = (0.825 - 0.34 * alpha + 0.05 * alpha**2) * \
        np.cos(np.pi / 2.0 * smax_sigma0) ** (1.0 / alpha)
    a1 = (0.415 - 0.071 * alpha) * smax_sigma0
    a3 = 2.0 * a0 + a1 - 1.0
    a2 = 1.0 - a0 - a1 - a3
    if r >= 0.0:
        poly = a0 + a1 * r + a2 * r**2 + a3 * r**3
        return max(r, poly), a0
    if r >= -2.0:
        return a0 + a1 * r, a0
    raise ValueError("Newman's f is defined for R >= -2")


class NasgroLaw:
    """Forman-Mettu equation with Newman closure.

    c, n, p, q   : curve fit constants (c may be a per-sample array)
    dk1          : long-crack threshold at R = 0 (0 disables threshold)
    cth_plus/minus : threshold spread exponents for R >= 0 / R < 0
    kc           : fracture toughness for the instability term; usually
                   left None and supplied per-sample by the integrator
    a0_intrinsic : small-crack El Haddad length (NASGRO fixes 0.0381 mm)
    """

    def __init__(self, c, n, p, q, dk1=0.0, cth_plus=0.0, cth_minus=0.1,
                 alpha=2.0, smax_sigma0=0.3, kc=None,
                 a0_intrinsic=A0_INTRINSIC_M):
        self.c = c
        self.n = n
        self.p = p
        self.q = q
        self.dk1 = dk1
        self.cth_plus = cth_plus
        self.cth_minus = cth_minus
        self.alpha = alpha
        self.smax_sigma0 = smax_sigma0
        self.kc = kc
        self.a0_intrinsic = a0_intrinsic

    def _c_for(self, dk, sample_slice):
        c = np.asarray(self.c, dtype=float)
        if c.ndim == 0:
            return c
        if sample_slice is not None:
            c = c[sample_slice]
        return c[:, None] if dk.ndim == 2 else c

    def threshold(self, stress_ratio, a=None):
        """dKth per the NASA threshold equation; equals dk1 for a long
        crack at R = 0."""
        if self.dk1 <= 0.0:
            return 0.0
        r = float(np.clip(stress_ratio, -2.0, 0.95))
        f, a0_coef = newman_opening_function(r, self.alpha, self.smax_sigma0)
        cth = self.cth_plus if r >= 0.0 else self.cth_minus
        spread = ((1.0 - f) / ((1.0 - a0_coef) * (1.0 - r))) ** (1.0 + cth * r)
        if a is None:
            small_crack = 1.0
        else:
            a = np.asarray(a, dtype=float)
            small_crack = np.sqrt(a / (a + self.a0_intrinsic))
        return self.dk1 * small_crack / spread

    def rate(self, dk, stress_ratio=0.0, a=None, kc=None, sample_slice=None):
        dk = np.asarray(dk, dtype=float)
        r = float(np.clip(stress_ratio, -2.0, 0.95))
        f, _ = newman_opening_function(r, self.alpha, self.smax_sigma0)

        c = self._c_for(dk, sample_slice)
        dk_eff = ((1.0 - f) / (1.0 - r)) * dk
        v = c * np.power(np.maximum(dk_eff, 1e-300), self.n)

        dk_th = self.threshold(r, a)
        above = dk > np.maximum(dk_th, 1e-300)
        v = v * np.power(np.where(above, 1.0 - dk_th / np.maximum(dk, 1e-300),
                                  1.0), self.p)

        kc_eff = kc if kc is not None else self.kc
        if kc_eff is not None:
            k_max = dk / (1.0 - r)
            margin = 1.0 - k_max / np.asarray(kc_eff, dtype=float)
            stable = margin > 1e-9
            # rate diverges at instability; the life integrand 1/v -> 0
            v = np.where(stable,
                         v / np.power(np.where(stable, margin, 1.0), self.q),
                         np.inf)
        return np.where(above, v, 0.0)


# make the law reachable from YAML study files
from .fracture import GROWTH_LAWS  # noqa: E402

GROWTH_LAWS["nasgro"] = NasgroLaw
