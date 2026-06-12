import numpy as np
import pytest

from damocles.random_vars import Normal, Uniform
from damocles.sampling import map_to_physical, sample_unit


def test_shapes_and_open_interval():
    for method in ("random", "lhs"):
        u = sample_unit(1000, 3, method=method, seed=1)
        assert u.shape == (1000, 3)
        assert u.min() > 0.0 and u.max() < 1.0


def test_sobol_rounds_up_to_power_of_two():
    u = sample_unit(1000, 2, method="sobol", seed=1)
    assert u.shape == (1024, 2)


def test_lhs_stratification():
    # every one of n equal-width bins gets exactly one point per dimension
    n = 64
    u = sample_unit(n, 2, method="lhs", seed=3)
    for j in range(2):
        bins = np.floor(u[:, j] * n).astype(int)
        assert len(np.unique(bins)) == n


def test_reproducible_with_seed():
    a = sample_unit(100, 2, method="lhs", seed=42)
    b = sample_unit(100, 2, method="lhs", seed=42)
    assert np.array_equal(a, b)


def test_map_to_physical_columns_match_names():
    variables = {"x": Normal(0.0, 1.0), "y": Uniform(10.0, 20.0)}
    u = sample_unit(500, 2, method="lhs", seed=7)
    x = map_to_physical(u, variables)
    assert set(x) == {"x", "y"}
    assert abs(np.mean(x["x"])) < 0.2
    assert 10.0 < x["y"].min() and x["y"].max() < 20.0


def test_map_to_physical_dimension_mismatch():
    with pytest.raises(ValueError):
        map_to_physical(np.zeros((10, 3)), {"x": Normal(0, 1)})


def test_unknown_method():
    with pytest.raises(ValueError):
        sample_unit(10, 1, method="halton")
