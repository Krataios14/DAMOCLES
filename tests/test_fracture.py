"""Crack growth validation against the closed-form Paris integration.

For constant Y and no threshold:
N = (a0^(1-m/2) - ac^(1-m/2)) / ((m/2 - 1) * C * (Y*dsigma*sqrt(pi))^m)
"""

import numpy as np
import pytest

from pdts.fracture import (
    CenterCrack, CornerCrack, CustomGeometry, LifeResult, ParisLaw,
    SurfaceCrack, ThroughCrack, WalkerLaw, critical_size, grow,
)

C, M = 1e-11, 3.0
DSIGMA = 100.0
KIC = 60.0


def paris_closed_form(a0, ac, y, dsigma, c, m):
    num = a0 ** (1 - m / 2) - ac ** (1 - m / 2)
    den = (m / 2 - 1) * c * (y * dsigma * np.sqrt(np.pi)) ** m
    return num / den


def test_critical_size_through_crack_analytic():
    ac = critical_size(ThroughCrack(), s_max=DSIGMA, k_ic=KIC)
    expected = (KIC / (DSIGMA * np.sqrt(np.pi))) ** 2
    assert ac[0] == pytest.approx(expected, rel=1e-6)


def test_life_matches_closed_form():
    a0 = np.array([1e-3])
    res = grow(a0, DSIGMA, ThroughCrack(), ParisLaw(C, M), KIC, n_grid=400)
    expected = paris_closed_form(1e-3, res.a_critical[0], 1.0, DSIGMA, C, M)
    assert res.cycles_to_failure[0] == pytest.approx(expected, rel=5e-3)


def test_life_closed_form_other_exponent():
    a0 = np.array([5e-4])
    law = ParisLaw(2e-12, 3.6)
    res = grow(a0, 150.0, ThroughCrack(), law, 80.0, n_grid=400)
    ac = res.a_critical[0]
    expected = paris_closed_form(5e-4, ac, 1.0, 150.0, 2e-12, 3.6)
    assert res.cycles_to_failure[0] == pytest.approx(expected, rel=5e-3)


def test_per_sample_paris_constant():
    a0 = np.full(3, 1e-3)
    c = np.array([1e-11, 2e-11, 4e-11])
    res = grow(a0, DSIGMA, ThroughCrack(), ParisLaw(c, M), KIC)
    lives = res.cycles_to_failure
    # life scales as 1/C
    assert lives[0] / lives[1] == pytest.approx(2.0, rel=1e-6)
    assert lives[0] / lives[2] == pytest.approx(4.0, rel=1e-6)


def test_threshold_means_infinite_life():
    res = grow(np.array([1e-5]), 50.0, ThroughCrack(),
               ParisLaw(C, M, dk_threshold=5.0), KIC)
    # dK at a0 = 50 * sqrt(pi * 1e-5) = 0.28, well below 5
    assert np.isinf(res.cycles_to_failure[0])


def test_critical_on_arrival():
    res = grow(np.array([0.5]), DSIGMA, ThroughCrack(), ParisLaw(C, M), KIC)
    assert res.cycles_to_failure[0] == 0.0


def test_bigger_flaw_means_shorter_life():
    a0 = np.array([1e-4, 5e-4, 2e-3])
    res = grow(a0, DSIGMA, ThroughCrack(), ParisLaw(C, M), KIC)
    assert np.all(np.diff(res.cycles_to_failure) < 0)


def test_walker_reduces_to_paris_at_gamma_one():
    a0 = np.array([1e-3])
    p = grow(a0, DSIGMA, ThroughCrack(), ParisLaw(C, M), KIC, stress_ratio=0.5)
    w = grow(a0, DSIGMA, ThroughCrack(), WalkerLaw(C, M, gamma=1.0), KIC,
             stress_ratio=0.5)
    assert w.cycles_to_failure[0] == pytest.approx(p.cycles_to_failure[0], rel=1e-9)


def test_walker_higher_r_shortens_life():
    a0 = np.array([1e-3])
    law = WalkerLaw(C, M, gamma=0.6)
    low = grow(a0, DSIGMA, ThroughCrack(), law, KIC, stress_ratio=0.05)
    high = grow(a0, DSIGMA, ThroughCrack(), law, KIC, stress_ratio=0.5)
    assert high.cycles_to_failure[0] < low.cycles_to_failure[0]


def test_geometry_factors_ordering():
    a = np.array([1e-3])
    assert SurfaceCrack().y(a)[0] == pytest.approx(0.713, abs=1e-3)
    assert CornerCrack().y(a)[0] == pytest.approx(0.798, abs=1e-3)
    # center crack approaches infinity towards the plate edge
    cc = CenterCrack(width=0.1)
    assert cc.y(np.array([0.001]))[0] == pytest.approx(1.0, abs=0.01)
    assert cc.y(np.array([0.045]))[0] > 2.0


def test_custom_geometry():
    geom = CustomGeometry(lambda a: 1.12 * np.ones_like(a), a_max=0.2)
    ac = critical_size(geom, s_max=DSIGMA, k_ic=KIC)
    expected = (KIC / (1.12 * DSIGMA * np.sqrt(np.pi))) ** 2
    assert ac[0] == pytest.approx(expected, rel=1e-6)


def test_crack_size_checkpoints():
    a0 = np.array([1e-3])
    res = grow(a0, DSIGMA, ThroughCrack(), ParisLaw(C, M), KIC,
               eval_cycles=[0.0, 1e5, 5e5, 1e9], n_grid=400)
    a_t = res.a_at[0]
    assert a_t[0] == pytest.approx(1e-3, rel=1e-6)        # nothing yet
    assert a_t[1] > a_t[0] and a_t[2] > a_t[1]            # monotone growth
    assert a_t[3] == pytest.approx(res.a_critical[0])     # failed by then


def test_chunking_is_invisible():
    rng = np.random.default_rng(0)
    a0 = rng.uniform(1e-4, 2e-3, 1000)
    big = grow(a0, DSIGMA, ThroughCrack(), ParisLaw(C, M), KIC,
               eval_cycles=[2e5], chunk_size=1_000_000)
    small = grow(a0, DSIGMA, ThroughCrack(), ParisLaw(C, M), KIC,
                 eval_cycles=[2e5], chunk_size=137)
    assert np.allclose(big.cycles_to_failure, small.cycles_to_failure)
    assert np.allclose(big.a_at, small.a_at)


def test_pof_at():
    res = LifeResult(cycles_to_failure=np.array([100.0, 200.0, np.inf]),
                     a_critical=np.zeros(3), eval_cycles=None, a_at=None)
    assert res.pof_at(150.0) == pytest.approx(1.0 / 3.0)
