"""NASGRO equation checks: the Newman closure function against hand
computation, the threshold limits from the NASA source, and growth rate
cross-checks against the published FCGD spline tables (which are an
independent fit of the same data, so agreement within a factor is the
right expectation, not equality)."""

import numpy as np
import pytest

from damocles.fracture import ThroughCrack, grow
from damocles.nasgro import NasgroLaw, newman_opening_function

# 2024-T3 clad/bare sheet L-T constants, DOT/FAA/AR-05/15 fig. B-4a,
# in US units (in/cycle, ksi sqrt(in)); the equation is unit-agnostic
LAW_2024 = dict(c=0.800e-8, n=3.200, p=0.250, q=1.000, dk1=1.22,
                cth_plus=1.21, cth_minus=0.10, alpha=2.0, smax_sigma0=0.30)
KC_2024 = 74.3


def test_newman_closure_hand_values():
    # alpha = 2, Smax/sigma0 = 0.3:
    # A0 = (0.825 - 0.68 + 0.2) * cos(0.15 pi)^(1/2) = 0.3257
    f0, a0 = newman_opening_function(0.0, 2.0, 0.3)
    assert a0 == pytest.approx(0.32566, abs=2e-4)
    assert f0 == pytest.approx(a0)  # at R = 0 the polynomial equals A0
    # at high R the opening level rides just above R itself
    f7, _ = newman_opening_function(0.7, 2.0, 0.3)
    assert f7 == pytest.approx(0.71250, abs=5e-4)
    assert f7 > 0.7


def test_closure_function_validity():
    with pytest.raises(ValueError):
        newman_opening_function(-3.0, 2.0, 0.3)


def test_threshold_limits():
    law = NasgroLaw(**LAW_2024)
    # long crack at R = 0: threshold equals dk1 by construction
    assert law.threshold(0.0) == pytest.approx(1.22, rel=1e-9)
    # the small-crack term knocks it down by sqrt(a/(a+a0))
    a = law.a0_intrinsic
    assert law.threshold(0.0, a=a) == pytest.approx(1.22 / np.sqrt(2.0), rel=1e-9)
    # threshold shrinks as R rises
    assert law.threshold(0.7) < law.threshold(0.3) < law.threshold(0.0)


def test_rate_zero_below_threshold():
    law = NasgroLaw(**LAW_2024)
    assert law.rate(np.array([1.0]), 0.0)[0] == 0.0
    assert law.rate(np.array([2.0]), 0.0)[0] > 0.0


def test_stress_ratio_accelerates_growth():
    law = NasgroLaw(**LAW_2024)
    dk = np.array([10.0])
    assert law.rate(dk, 0.5)[0] > law.rate(dk, 0.0)[0]
    assert law.rate(dk, 0.7)[0] > law.rate(dk, 0.5)[0]


def test_cross_check_against_fcgd_spline():
    # DOT/FAA/AR-05/15 table 2(a), R = 0 spline points for 2024-T3:
    # da/dN 3.2e-6 at dK 8.5678; 1.0e-5 at 14.4607; 1.0e-4 at 28.4466.
    # Spline and NASGRO fit are independent fits of scattered data;
    # they agree within a factor, which is what we assert.
    law = NasgroLaw(**LAW_2024, kc=KC_2024)
    for dk, expected in [(8.5678, 3.2e-6), (14.4607, 1.0e-5),
                         (28.4466, 1.0e-4)]:
        got = law.rate(np.array([dk]), 0.0)[0]
        assert expected / 1.8 < got < expected * 1.8


def test_instability_term_diverges_at_kc():
    law = NasgroLaw(**LAW_2024, kc=KC_2024)
    near = law.rate(np.array([KC_2024 * 0.999]), 0.0)[0]
    assert np.isinf(law.rate(np.array([KC_2024 * 1.001]), 0.0)[0])
    assert near > law.rate(np.array([KC_2024 * 0.9]), 0.0)[0] * 10


def test_per_sample_c_and_growth_integration():
    # SI version of the 2024-T3 constants via the materials loader is
    # tested separately; here just exercise the law inside grow()
    c = np.array([2e-10, 4e-10])
    law = NasgroLaw(c=c, n=3.2, p=0.25, q=1.0, dk1=1.34,
                    cth_plus=1.21, alpha=2.0, smax_sigma0=0.3)
    res = grow(np.array([1e-3, 1e-3]), 80.0, ThroughCrack(), law, 81.6)
    assert np.all(np.isfinite(res.cycles_to_failure))
    assert res.cycles_to_failure[0] > res.cycles_to_failure[1]


def test_dormant_below_threshold_in_growth():
    law = NasgroLaw(c=2e-10, n=3.2, p=0.25, q=1.0, dk1=1.34,
                    cth_plus=1.21, alpha=2.0, smax_sigma0=0.3)
    res = grow(np.array([1e-5]), 10.0, ThroughCrack(), law, 81.6)
    # dK at a0 = 10 * sqrt(pi e-5) = 0.056, far below threshold
    assert np.isinf(res.cycles_to_failure[0])
