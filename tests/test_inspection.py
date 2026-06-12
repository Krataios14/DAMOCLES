import numpy as np
import pytest

from damocles.fracture import LifeResult
from damocles.inspection import InspectionPlan, PODCurve, apply_plan


def make_life(nf, eval_cycles, a_at):
    nf = np.asarray(nf, dtype=float)
    return LifeResult(cycles_to_failure=nf,
                      a_critical=np.full_like(nf, 0.01),
                      eval_cycles=np.asarray(eval_cycles, dtype=float),
                      a_at=np.asarray(a_at, dtype=float))


def test_pod_curve_shape():
    pod = PODCurve(a50=1e-3, sigma=0.5)
    assert pod.pod(1e-3) == pytest.approx(0.5)
    assert pod.pod(1e-6) < 0.01
    assert pod.pod(1e-1) > 0.99
    a = np.logspace(-5, -1, 50)
    assert np.all(np.diff(pod.pod(a)) >= 0)


def test_pod_from_a50_a90():
    pod = PODCurve.from_a50_a90(a50=0.8e-3, a90=2.0e-3)
    assert pod.pod(0.8e-3) == pytest.approx(0.5, abs=1e-9)
    assert pod.pod(2.0e-3) == pytest.approx(0.9, abs=1e-9)


def test_interval_plan_excludes_service_end():
    pod = PODCurve(1e-3, 0.5)
    plan = InspectionPlan.at_interval(5000, 20000, pod)
    assert plan.times == [5000.0, 10000.0, 15000.0]


def test_no_inspections_equals_base_pof():
    life = make_life([100.0, 5000.0, np.inf], [10000.0], [[1e-3], [1e-3], [1e-4]])
    out = apply_plan(life, 10000.0, InspectionPlan([], PODCurve(1e-3, 0.5)))
    assert out.pof_inspected == out.pof_unmitigated == pytest.approx(2.0 / 3.0)


def test_hand_computed_case():
    # three parts, one inspection at t=200, service 1000
    # part 0 fails at 100, before the inspection: cannot be saved
    # part 1 fails at 300: saved with probability POD(a at 200)
    # part 2 never fails
    pod = PODCurve(a50=1e-3, sigma=0.4)
    a_insp = np.array([[5e-3], [2e-3], [1e-4]])
    life = make_life([100.0, 300.0, np.inf], [200.0], a_insp)
    out = apply_plan(life, 1000.0, InspectionPlan([200.0], pod))

    p_detect_part1 = pod.pod(2e-3)
    expected = (1.0 + (1.0 - p_detect_part1)) / 3.0
    assert out.pof_unmitigated == pytest.approx(2.0 / 3.0)
    assert out.pof_inspected == pytest.approx(expected)
    assert 0.0 < out.risk_reduction < 1.0


def test_perfect_detection_leaves_only_early_failures():
    # essentially certain detection of any crack bigger than 1 micron
    pod = PODCurve(a50=1e-6, sigma=0.1)
    a_insp = np.array([[1e-3], [1e-3], [1e-3], [1e-3]])
    life = make_life([100.0, 600.0, 900.0, np.inf], [500.0], a_insp)
    out = apply_plan(life, 1000.0, InspectionPlan([500.0], pod))
    # only the t=100 failure happens before anyone can look
    assert out.pof_inspected == pytest.approx(0.25, abs=1e-6)


def test_useless_pod_changes_nothing():
    pod = PODCurve(a50=10.0, sigma=0.1)  # can only see 10 m cracks
    a_insp = np.array([[1e-3], [2e-3]])
    life = make_life([300.0, 700.0], [200.0], a_insp)
    out = apply_plan(life, 1000.0, InspectionPlan([200.0], pod))
    assert out.pof_inspected == pytest.approx(out.pof_unmitigated)


def test_missing_checkpoint_raises():
    life = make_life([100.0], [200.0], [[1e-3]])
    with pytest.raises(ValueError, match="checkpoint"):
        apply_plan(life, 1000.0, InspectionPlan([300.0], PODCurve(1e-3, 0.5)))


def test_more_inspections_never_increase_risk():
    rng = np.random.default_rng(4)
    n = 5000
    nf = rng.lognormal(np.log(8000), 0.6, n)
    times = [2000.0, 4000.0, 6000.0]
    # crack grows with time, so later checkpoints see bigger cracks
    a_at = np.column_stack([np.minimum(1e-4 * (nf / t) ** -1.5, 5e-3)
                            for t in times])
    life = make_life(nf, times, a_at)
    pod = PODCurve(1e-3, 0.5)
    one = apply_plan(life, 10000.0, InspectionPlan([4000.0], pod))
    three = apply_plan(life, 10000.0, InspectionPlan(times, pod))
    assert three.pof_inspected <= one.pof_inspected <= one.pof_unmitigated
