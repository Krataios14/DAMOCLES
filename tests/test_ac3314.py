"""The AC 33.14-1 calibration test case, the qualification evidence the
Advisory Circular itself prescribes for probabilistic rotor codes:

  "Test case results in the ranges from 1.27E-09 to 1.93E-09 (for the
  'no inspection' case) and from 8.36E-10 to 1.53E-09 (for the 'with
  in-service inspection' case) are considered acceptable."
                                  - AC 33.14-1, Section 3, Calibration
"""

import numpy as np
import pytest

from damocles.ac3314 import (
    ACCEPTANCE_NO_INSPECTION, ACCEPTANCE_WITH_INSPECTION, ExceedanceCurve,
    TabulatedPOD, build_zones, hoop_stress, run_test_case,
)


def test_hoop_stress_matches_ac_contours():
    # the AC's stress figure tops out at 572.4 MPa at the bore
    assert hoop_stress(0.300) == pytest.approx(572.4, abs=4.0)
    # decreasing toward the rim, where the lowest contour band is 400
    r = np.linspace(0.300, 0.425, 50)
    s = hoop_stress(r)
    assert np.all(np.diff(s) < 0)
    assert 370.0 < s[-1] < 400.0


def test_zone_volumes_match_ac_table_a1_1():
    zones = {z.name: z for z in build_zones(18)}
    # AC Table A1-1: bore surface zone 94.9 cm^3, rim surface 134.37 cm^3
    assert zones["bore skin"].volume * 1e6 == pytest.approx(94.9, rel=0.03)
    assert zones["rim skin"].volume * 1e6 == pytest.approx(134.37, rel=0.03)
    # the zones partition the disk: total = pi (R1^2 - R2^2) L exactly
    total = sum(z.volume for z in build_zones(18))
    assert total == pytest.approx(np.pi * (0.425**2 - 0.3**2) * 0.1, rel=1e-9)


def test_exceedance_curve_2001_versus_chg1():
    old = ExceedanceCurve.test_case_2001()
    new = ExceedanceCurve.hard_alpha(3, 3)
    # both monotone decreasing
    a = np.logspace(2.1, 4.2, 40)
    assert np.all(np.diff(old.exceedance(a)) < 0)
    assert np.all(np.diff(new.exceedance(a)) < 0)
    # the documented divergence in the risk-dominant region: the 2017
    # tabulation sits several times higher around 3000 sq mils
    assert new.exceedance(3000.0) / old.exceedance(3000.0) > 4.0


def test_chg1_table_verbatim_endpoints():
    new = ExceedanceCurve.hard_alpha(3, 3)
    assert new.exceedance(3.52) == pytest.approx(8.76, rel=1e-3)
    assert new.exceedance(111062.0) == pytest.approx(6.73e-3, rel=1e-3)


def test_loglog_extrapolation_continues_slope():
    c = ExceedanceCurve.test_case_2001()
    inside = c.exceedance(np.array([15000.0, 18000.0]))
    beyond = c.exceedance(np.array([30000.0, 60000.0]))
    assert beyond[0] < inside[-1]
    assert beyond[1] < beyond[0] > 0.0


def test_pod_curve_anchors():
    pod = TabulatedPOD.from_ac("ut-3fbh-cal")
    assert pod.pod(19648.0) == pytest.approx(0.50, abs=0.01)
    assert pod.pod(39205.0) == pytest.approx(0.90, abs=0.01)
    assert pod.pod(100.0) == 0.0
    assert pod.pod(80000.0) == 1.0
    with pytest.raises(KeyError):
        TabulatedPOD.from_ac("xray")


def test_calibration_no_inspection_inside_acceptance_band():
    r = run_test_case(inspection=False, n_rings=18)
    lo, hi = ACCEPTANCE_NO_INSPECTION
    assert lo <= r.events_per_cycle <= hi, r.events_per_cycle


def test_calibration_with_inspection_inside_acceptance_band():
    r = run_test_case(inspection=True, n_rings=18)
    lo, hi = ACCEPTANCE_WITH_INSPECTION
    assert lo <= r.events_per_cycle <= hi, r.events_per_cycle
    # and the inspection must reduce the risk
    base = run_test_case(inspection=False, n_rings=18)
    assert r.events_per_cycle < base.events_per_cycle


def test_zoning_refinement_converges_downward():
    # zone-max stress makes coarse zoning conservative; refinement must
    # reduce the answer monotonically toward the continuum limit
    vals = [run_test_case(inspection=False, n_rings=n).events_per_cycle
            for n in (6, 12, 24, 48)]
    assert all(a > b for a, b in zip(vals, vals[1:]))
    assert vals[-1] / vals[-2] > 0.93  # approaching the continuum limit


def test_monte_carlo_agrees_with_quadrature():
    q = run_test_case(inspection=True, n_rings=12, method="quadrature")
    m = run_test_case(inspection=True, n_rings=12, method="montecarlo",
                      n_samples=200_000, seed=3)
    assert m.events_per_cycle == pytest.approx(q.events_per_cycle, rel=0.02)


def test_bore_region_dominates_risk():
    r = run_test_case(inspection=False, n_rings=12)
    inner = sum(v for k, v in r.zone_risk.items()
                if k in ("bore skin", "interior 0", "face skin 0",
                         "interior 1", "face skin 1"))
    assert inner / r.pof_service > 0.4
