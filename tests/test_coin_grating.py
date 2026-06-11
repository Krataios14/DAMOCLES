"""Regression test for the original 2024 coin-through-grating problem,
now running on the reliability engine and checked against quadrature."""

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

_EXAMPLE = pathlib.Path(__file__).parent.parent / "examples" / "coin_grating.py"
_spec = importlib.util.spec_from_file_location("coin_grating", _EXAMPLE)
coin = importlib.util.module_from_spec(_spec)
sys.modules["coin_grating"] = coin
_spec.loader.exec_module(coin)

from damtol import estimate_pof  # noqa: E402


def test_engine_matches_quadrature():
    exact = coin.analytic_probability()
    res = estimate_pof(coin.coin_limit_state, coin.VARIABLES, n=2**16,
                       method="sobol", seed=2024)
    assert res.pof == pytest.approx(exact, abs=0.005)
    assert res.ci_low - 0.005 <= exact <= res.ci_high + 0.005


def test_quadrature_is_stable():
    assert coin.analytic_probability(5001) == pytest.approx(
        coin.analytic_probability(50001), abs=1e-6)


def test_limit_state_sign_convention():
    # dead centre of a slot, flat-topped hexagon: falls through
    x0 = coin.SLOT_X0[0] + coin.SLOT_W / 2.0
    s = {"x": np.array([x0, 0.0]),
         "y": np.array([5.0, 5.0]),
         "theta": np.array([np.pi / 6.0, 0.0])}
    g = coin.coin_limit_state(s)
    assert g[0] < 0.0   # falls through
    assert g[1] > 0.0   # sitting on the solid edge of the grating
