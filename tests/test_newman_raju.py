"""Newman-Raju equations checked against the published limit cases in
NASA TM-85793 and the classical embedded-crack results."""

import numpy as np
import pytest

from damocles.fracture import ParisLaw, grow
from damocles.newman_raju import (
    NewmanRajuCornerCrack, NewmanRajuSurfaceCrack, corner_crack_f,
    shape_factor_q, surface_crack_f,
)


def test_shape_factor_semicircle():
    # a/c = 1: Q = 1 + 1.464 = 2.464, both branches must agree there
    assert shape_factor_q(1.0) == pytest.approx(2.464)
    assert shape_factor_q(1.0 + 1e-12) == pytest.approx(2.464, rel=1e-6)


def test_semicircular_shallow_limit():
    # TM-85793 sanity case: a/c = 1, a/t -> 0, deepest point:
    # F = M1 = 1.04, K = 1.04/sqrt(2.464) * S * sqrt(pi a) = 0.6625 S sqrt(pi a)
    f = surface_crack_f(a=1e-6, c=1e-6, t=1.0, b=10.0, phi=np.pi / 2)
    assert f == pytest.approx(1.04, rel=1e-3)
    geom = NewmanRajuSurfaceCrack(thickness=1.0, width=10.0, aspect=1.0)
    assert geom.y(np.array([1e-6]))[0] == pytest.approx(0.6625, abs=2e-3)


def test_corner_crack_through_thickness_limit():
    # TM-85793 eq. 45: a/c = 1, a/t -> 1, phi = 0 reduces to
    # K = 1.11 * S * sqrt(pi c) * fw, i.e. F/sqrt(Q) -> 1.11 for wide plates
    f = corner_crack_f(a=0.999, c=0.999, t=1.0, b=1000.0, phi=0.0)
    assert f / np.sqrt(shape_factor_q(1.0)) == pytest.approx(1.11, rel=0.01)


def test_deepest_point_governs_shallow_semicircular():
    geom_d = NewmanRajuSurfaceCrack(1.0, 10.0, aspect=1.0, tip="depth")
    geom_s = NewmanRajuSurfaceCrack(1.0, 10.0, aspect=1.0, tip="surface")
    a = np.array([0.01])
    # for a/c = 1 the surface point carries the 1.1-ish free surface term
    # and actually exceeds the depth point; 'max' must take the larger
    geom_m = NewmanRajuSurfaceCrack(1.0, 10.0, aspect=1.0, tip="max")
    y_d, y_s, y_m = geom_d.y(a)[0], geom_s.y(a)[0], geom_m.y(a)[0]
    assert y_m == pytest.approx(max(y_d, y_s))


def test_finite_width_amplifies():
    wide = NewmanRajuSurfaceCrack(thickness=0.01, width=1.0, aspect=1.0)
    narrow = NewmanRajuSurfaceCrack(thickness=0.01, width=0.025, aspect=1.0)
    a = np.array([5e-3])
    assert narrow.y(a)[0] > wide.y(a)[0]


def test_y_grows_with_depth_ratio():
    geom = NewmanRajuSurfaceCrack(thickness=0.01, width=1.0, aspect=1.0)
    a = np.array([1e-4, 2e-3, 5e-3, 8e-3])
    y = geom.y(a)
    assert np.all(np.diff(y) > 0)


def test_aspect_validity_and_tip_validation():
    with pytest.raises(ValueError, match="0.2"):
        NewmanRajuSurfaceCrack(0.01, 1.0, aspect=0.1)
    with pytest.raises(ValueError, match="tip"):
        NewmanRajuSurfaceCrack(0.01, 1.0, tip="middle")


def test_a_max_respects_thickness_and_width():
    geom = NewmanRajuSurfaceCrack(thickness=0.005, width=1.0, aspect=1.0)
    assert geom.a_max == pytest.approx(0.0049)
    geom2 = NewmanRajuSurfaceCrack(thickness=1.0, width=0.02, aspect=1.0)
    assert geom2.a_max == pytest.approx(0.0098)


def test_grows_under_engine_like_loading():
    # end to end: NR surface crack works inside the growth integrator
    geom = NewmanRajuSurfaceCrack(thickness=0.02, width=0.2, aspect=1.0)
    res = grow(np.array([2e-4]), 400.0, geom, ParisLaw(9.25e-13, 3.87), 64.5)
    assert 0 < res.cycles_to_failure[0] < np.inf
    # the crack cannot grow past the plate validity bound
    assert res.a_critical[0] <= geom.a_max


def test_corner_vs_surface_severity():
    # a corner crack of equal size is more severe than a surface crack
    s = NewmanRajuSurfaceCrack(0.02, 0.2, aspect=1.0)
    c = NewmanRajuCornerCrack(0.02, 0.2, aspect=1.0)
    a = np.array([1e-3])
    assert c.y(a)[0] > s.y(a)[0]
