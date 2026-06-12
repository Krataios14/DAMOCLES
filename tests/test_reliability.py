"""Validation against the classic R-S problem, which has an exact answer.

R ~ N(mu_R, s_R), S ~ N(mu_S, s_S), g = R - S.
beta = (mu_R - mu_S) / sqrt(s_R^2 + s_S^2), pof = Phi(-beta).
"""

import numpy as np
import pytest
from scipy import stats

from damocles.random_vars import Normal
from damocles.reliability import estimate_pof


MU_R, S_R = 10.0, 1.0
MU_S, S_S = 5.0, 1.0
BETA = (MU_R - MU_S) / np.sqrt(S_R**2 + S_S**2)
POF_TRUE = stats.norm.cdf(-BETA)  # 2.035e-4

VARIABLES = {"R": Normal(MU_R, S_R), "S": Normal(MU_S, S_S)}


def g(x):
    return x["R"] - x["S"]


def test_plain_lhs_brackets_truth():
    res = estimate_pof(g, VARIABLES, n=400_000, method="lhs", seed=11)
    assert res.ci_low <= POF_TRUE <= res.ci_high
    assert res.pof == pytest.approx(POF_TRUE, rel=0.35)


def test_sobol_runs_and_is_close():
    res = estimate_pof(g, VARIABLES, n=2**18, method="sobol", seed=5)
    assert res.pof == pytest.approx(POF_TRUE, rel=0.35)


def test_importance_sampling_efficiency():
    # design point of g = R - S is at R = S = 7.5; centre both proposals there
    proposal = {"R": Normal(7.5, 1.0), "S": Normal(7.5, 1.0)}
    res = estimate_pof(g, VARIABLES, n=20_000, method="random", seed=3,
                       importance=proposal)
    assert res.importance_sampled
    assert res.pof == pytest.approx(POF_TRUE, rel=0.10)
    # plain MC at n=20k and p=2e-4 has estimator CoV ~ 0.5; demand 5x better
    assert res.cov_estimator < 0.10


def test_reliability_index_round_trip():
    res = estimate_pof(g, VARIABLES, n=400_000, method="lhs", seed=11)
    assert res.reliability_index() == pytest.approx(BETA, abs=0.15)


def test_zero_failures_gives_zero_with_upper_bound():
    safe = {"R": Normal(100.0, 1.0), "S": Normal(5.0, 1.0)}
    res = estimate_pof(g, safe, n=10_000, method="lhs", seed=1)
    assert res.pof == 0.0
    assert res.ci_low == 0.0
    assert 0.0 < res.ci_high < 1e-3  # rule-of-three style bound


def test_importance_name_validation():
    with pytest.raises(ValueError, match="not in study"):
        estimate_pof(g, VARIABLES, n=100, importance={"T": Normal(0, 1)})


def test_keep_samples():
    res = estimate_pof(g, VARIABLES, n=1000, method="lhs", seed=2,
                       keep_samples=True)
    assert set(res.samples) == {"R", "S"}
    assert res.margin.shape == (1000,)
