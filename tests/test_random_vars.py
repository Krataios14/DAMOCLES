import numpy as np
import pytest

from pdts.random_vars import (
    Deterministic, Gumbel, Lognormal, Normal, Uniform, Weibull, from_spec,
)


def test_lognormal_mean_cov_parametrisation():
    rv = Lognormal(mean=100.0, cov=0.3)
    u = np.linspace(1e-6, 1 - 1e-6, 200001)
    x = rv.ppf(u)
    # quantile average converges to the distribution mean
    assert rv.mean() == pytest.approx(100.0, rel=1e-9)
    assert np.mean(x) == pytest.approx(100.0, rel=0.01)
    assert rv.std() / rv.mean() == pytest.approx(0.3, rel=1e-9)


def test_normal_round_trip():
    rv = Normal(mean=50.0, std=5.0)
    assert rv.cdf(rv.ppf(0.975)) == pytest.approx(0.975)
    assert rv.ppf(0.5) == pytest.approx(50.0)


def test_deterministic_is_flat():
    rv = Deterministic(7.5)
    out = rv.ppf(np.array([0.01, 0.5, 0.99]))
    assert np.all(out == 7.5)
    assert rv.std() == 0.0


def test_weibull_and_gumbel_basic_shape():
    w = Weibull(shape=10.0, scale=500.0)
    assert 400.0 < w.mean() < 500.0
    g = Gumbel(loc=100.0, scale=10.0)
    assert g.mean() > 100.0  # right-skewed


def test_uniform_bounds():
    rv = Uniform(2.0, 4.0)
    x = rv.ppf(np.array([0.0, 1.0]))
    assert x[0] == pytest.approx(2.0)
    assert x[1] == pytest.approx(4.0)


def test_from_spec_builds_and_validates():
    rv = from_spec({"dist": "lognormal", "mean": 9e-5, "cov": 0.5})
    assert isinstance(rv, Lognormal)
    with pytest.raises(ValueError, match="unknown distribution"):
        from_spec({"dist": "cauchy", "loc": 0})
    with pytest.raises(ValueError, match="missing"):
        from_spec({"dist": "normal", "mean": 1.0})
    with pytest.raises(ValueError, match="unexpected"):
        from_spec({"dist": "uniform", "low": 0, "high": 1, "mean": 2})


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        Normal(0.0, -1.0)
    with pytest.raises(ValueError):
        Lognormal(-5.0, 0.2)
    with pytest.raises(ValueError):
        Uniform(3.0, 3.0)
