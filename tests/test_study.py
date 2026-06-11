import numpy as np
import pytest

from damtol.random_vars import Deterministic, Lognormal, Normal
from damtol.fracture import ThroughCrack, ParisLaw
from damtol.inspection import InspectionPlan, PODCurve
from damtol.study import DamageToleranceStudy, build_study


def small_spec(**overrides):
    spec = {
        "name": "unit test study",
        "variables": {
            "initial_flaw": {"dist": "lognormal", "mean": 1.0e-3, "cov": 0.5},
            "stress_range": {"dist": "normal", "mean": 100.0, "std": 5.0},
            "toughness": {"dist": "normal", "mean": 60.0, "std": 3.0},
        },
        "geometry": {"type": "through"},
        "growth": {"law": "paris", "c": 1.0e-11, "m": 3.0},
        "service_cycles": 500000,
        "analysis": {"samples": 5000, "method": "lhs", "seed": 1},
    }
    spec.update(overrides)
    return spec


def test_build_and_run_from_spec():
    res = build_study(small_spec()).run()
    assert res.n_samples == 5000
    assert 0.0 < res.pof < 1.0
    assert res.ci_low <= res.pof <= res.ci_high
    assert np.all(np.diff(res.pof_curve) >= 0)  # cdf is monotone
    assert "P(failure)" in res.summary()


def test_certain_failure_and_certain_survival():
    hopeless = build_study(small_spec(service_cycles=1.0e12)).run()
    assert hopeless.pof > 0.99
    relaxed = small_spec(service_cycles=10.0)
    relaxed["variables"]["initial_flaw"] = {"dist": "deterministic",
                                            "value": 1.0e-6}
    assert build_study(relaxed).run().pof == 0.0


def test_inspection_in_study_reduces_risk():
    spec = small_spec(service_cycles=600000)
    spec["inspection"] = {"interval": 150000.0,
                          "pod_a50": 2.0e-3, "pod_a90": 5.0e-3}
    res = build_study(spec).run()
    assert res.inspection is not None
    assert res.inspection.pof_inspected < res.inspection.pof_unmitigated
    assert "inspected" in res.summary()


def test_random_paris_c_via_spec():
    spec = small_spec()
    spec["variables"]["paris_c"] = {"dist": "lognormal",
                                    "mean": 1.0e-11, "cov": 0.4}
    del spec["growth"]["c"]
    res = build_study(spec).run()
    assert 0.0 < res.pof < 1.0


def test_sensitivity_in_study():
    res = build_study(small_spec()).run(sensitivity=True)
    assert set(res.sensitivity) == {"initial_flaw", "stress_range", "toughness"}
    # flaw size scatter dominates life scatter in this setup
    totals = {k: v["total"] for k, v in res.sensitivity.items()}
    assert max(totals, key=totals.get) == "initial_flaw"
    assert "variance drivers" in res.summary()


def test_target_verdict_in_summary():
    spec = small_spec()
    spec["analysis"]["target_pof"] = 1.0e-9
    text = build_study(spec).run().summary()
    assert "EXCEEDS target" in text


def test_missing_required_variable():
    with pytest.raises(ValueError, match="missing"):
        DamageToleranceStudy(
            "x", {"initial_flaw": Deterministic(1e-3)},
            ThroughCrack(), ParisLaw(1e-11, 3.0), 1000.0)


def test_unknown_geometry_and_law():
    with pytest.raises(ValueError, match="geometry"):
        build_study(small_spec(geometry={"type": "ellipse"}))
    with pytest.raises(ValueError, match="growth law"):
        build_study(small_spec(growth={"law": "forman", "c": 1e-11, "m": 3.0}))


def test_direct_construction():
    study = DamageToleranceStudy(
        "direct",
        {"initial_flaw": Lognormal(1e-3, 0.5),
         "stress_range": Normal(100.0, 5.0),
         "toughness": Deterministic(60.0)},
        ThroughCrack(), ParisLaw(1e-11, 3.0),
        service_cycles=500000,
        inspection_plan=InspectionPlan.at_interval(
            150000, 500000, PODCurve(2e-3, 0.5)),
        n_samples=2000, seed=3)
    res = study.run()
    assert res.inspection is not None
