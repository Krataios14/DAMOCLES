"""Validation on the Ishigami function, whose Sobol indices are known
in closed form: S1 = 0.3139, S2 = 0.4424, S3 = 0,
T1 = 0.5576, T2 = 0.4424, T3 = 0.2437 (a = 7, b = 0.1).
"""

import numpy as np
import pytest

from pdts.random_vars import Uniform
from pdts.sensitivity import rank_drivers, sobol_indices

A, B = 7.0, 0.1


def ishigami(x):
    return (np.sin(x["x1"]) + A * np.sin(x["x2"]) ** 2
            + B * x["x3"] ** 4 * np.sin(x["x1"]))


VARS = {name: Uniform(-np.pi, np.pi) for name in ("x1", "x2", "x3")}


def test_ishigami_indices():
    idx = sobol_indices(ishigami, VARS, n=2**13, seed=8)
    assert idx["x1"]["first"] == pytest.approx(0.3139, abs=0.03)
    assert idx["x2"]["first"] == pytest.approx(0.4424, abs=0.03)
    assert idx["x3"]["first"] == pytest.approx(0.0, abs=0.03)
    assert idx["x1"]["total"] == pytest.approx(0.5576, abs=0.03)
    assert idx["x2"]["total"] == pytest.approx(0.4424, abs=0.03)
    assert idx["x3"]["total"] == pytest.approx(0.2437, abs=0.03)


def test_x3_matters_only_through_interaction():
    idx = sobol_indices(ishigami, VARS, n=2**13, seed=8)
    assert idx["x3"]["total"] > 0.15
    assert abs(idx["x3"]["first"]) < 0.05


def test_rank_drivers():
    idx = sobol_indices(ishigami, VARS, n=2**12, seed=1)
    assert rank_drivers(idx)[0] == "x1"


def test_zero_variance_raises():
    with pytest.raises(ValueError, match="zero variance"):
        sobol_indices(lambda x: np.zeros_like(x["x1"]), {"x1": Uniform(0, 1)},
                      n=128)
