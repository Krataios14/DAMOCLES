import numpy as np
import pytest

from pdts.allowables import a_basis, b_basis, basis_value, tolerance_factor


def test_published_k_factors():
    # one-sided normal tolerance factors, MMPDS/CMH-17 tables
    assert tolerance_factor(10, 0.90, 0.95) == pytest.approx(2.355, abs=0.005)
    assert tolerance_factor(10, 0.99, 0.95) == pytest.approx(3.981, abs=0.01)
    assert tolerance_factor(100, 0.90, 0.95) == pytest.approx(1.527, abs=0.005)


def test_k_decreases_with_sample_size():
    ks = [tolerance_factor(n) for n in (5, 10, 30, 100, 1000)]
    assert all(a > b for a, b in zip(ks, ks[1:]))


def test_a_basis_below_b_basis_below_mean():
    rng = np.random.default_rng(2)
    data = rng.normal(450.0, 25.0, 30)
    assert a_basis(data) < b_basis(data) < np.mean(data)


def test_b_basis_coverage():
    # the whole point of the 95 % confidence: across many datasets the
    # B-basis should fall below the true 10th percentile ~95 % of the time
    rng = np.random.default_rng(7)
    true_p10 = 100.0 - 1.2816 * 10.0
    hits = sum(
        b_basis(rng.normal(100.0, 10.0, 15)) <= true_p10
        for _ in range(2000)
    )
    assert 0.93 < hits / 2000 < 0.97


def test_lognormal_basis_stays_positive():
    rng = np.random.default_rng(3)
    data = rng.lognormal(np.log(50.0), 0.6, 12)  # heavy right skew
    assert basis_value(data, dist="lognormal") > 0.0


def test_input_validation():
    with pytest.raises(ValueError):
        tolerance_factor(1)
    with pytest.raises(ValueError):
        basis_value(np.array([[1.0, 2.0]]))
    with pytest.raises(ValueError):
        basis_value(np.array([1.0, -2.0, 3.0]), dist="lognormal")
    with pytest.raises(ValueError):
        basis_value(np.array([1.0, 2.0, 3.0]), dist="gamma")
