"""High-level ordered mission and Willenborg study contracts."""

from copy import deepcopy

import numpy as np
import pytest
import yaml

import damocles
import damocles.study as study_module
from damocles.cli import main as cli_main
from damocles.retardation import WillenborgConfig
from damocles.spectrum import SpectrumSequence
from damocles.study import build_study


def mission_spec(ranges, **overrides):
    spec = {
        "name": "ordered mission",
        "variables": {
            "initial_flaw": {"dist": "deterministic", "value": 1.0e-3},
            "stress_scale": {"dist": "deterministic", "value": 1.0},
            "toughness": {"dist": "deterministic", "value": 100.0},
        },
        "geometry": {"type": "through"},
        "growth": {"law": "paris", "c": 1.0e-10, "m": 3.0},
        "spectrum": {
            "type": "ordered",
            "cycles": [{"delta_sigma": value, "stress_ratio": 0.0} for value in ranges],
        },
        "retardation": {
            "model": "willenborg",
            "yield_strength": 1000.0,
            "max_cycles": 100.0,
            "max_blocks": 1,
            "max_work": 1_000,
        },
        "service_cycles": 100.0,
        "analysis": {"samples": 1, "method": "lhs", "seed": 7},
    }
    spec.update(overrides)
    return spec


def test_ordered_mission_contract_is_versioned_as_0_3():
    assert damocles.__version__ == "0.3.0"
    assert damocles.WillenborgConfig is WillenborgConfig
    assert damocles.SpectrumSequence is SpectrumSequence


def test_high_level_study_preserves_the_known_82_vs_78_order_effect():
    overload_first = [1500.0] + [1000.0] * 500
    overload_later = [1000.0] * 10 + [1500.0] + [1000.0] * 490

    first = build_study(mission_spec(overload_first)).run()
    later = build_study(mission_spec(overload_later)).run()

    assert first.lives[0] == 82.0
    assert later.lives[0] == 78.0
    assert first.loading == later.loading == "ordered-mission"
    assert first.retardation == later.retardation == "willenborg"
    assert first.life_unit == later.life_unit == "cycles"


def test_explicit_disabled_retardation_equals_the_omitted_baseline():
    ranges = [1000.0] * 100
    omitted = mission_spec(ranges)
    omitted.pop("retardation")
    explicit = deepcopy(omitted)
    explicit["retardation"] = {
        "model": "none",
        "max_cycles": 100.0,
        "max_blocks": 1,
        "max_work": 1_000,
    }

    result_omitted = build_study(omitted).run()
    result_explicit = build_study(explicit).run()

    np.testing.assert_array_equal(result_omitted.lives, result_explicit.lives)
    assert result_omitted.pof == result_explicit.pof
    assert result_omitted.retardation == result_explicit.retardation == "none"


def test_seeded_mission_study_is_exactly_repeatable():
    spec = mission_spec([1500.0] + [1000.0] * 100)
    spec["variables"]["initial_flaw"] = {
        "dist": "lognormal",
        "mean": 1.0e-3,
        "cov": 0.05,
    }
    spec["analysis"]["samples"] = 16
    spec["retardation"]["max_work"] = 2_000

    first = build_study(spec).run(curve_points=10)
    second = build_study(spec).run(curve_points=10)

    np.testing.assert_array_equal(first.lives, second.lives)
    np.testing.assert_array_equal(first.pof_curve, second.pof_curve)
    assert first.pof == second.pof
    assert first.prospective_work == second.prospective_work == 1_600


def test_yaml_parser_and_public_configs_round_trip_without_reordering():
    spec = mission_spec([1500.0, 1000.0, 1200.0])
    spec["service_cycles"] = 3.0
    spec["retardation"]["max_cycles"] = 3.0
    decoded = yaml.safe_load(yaml.safe_dump(spec, sort_keys=True))
    study = build_study(decoded)

    assert [cycle.delta_sigma for cycle in study.spectrum.cycles] == [
        1500.0,
        1000.0,
        1200.0,
    ]
    assert SpectrumSequence.from_cycles(study.spectrum.to_cycles()) == study.spectrum
    assert WillenborgConfig.from_spec(study.retardation.to_spec()) == study.retardation


def test_cli_runs_the_ordered_yaml_contract(tmp_path, capsys):
    spec = mission_spec([1500.0] + [1000.0] * 100)
    path = tmp_path / "mission.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=True), encoding="utf-8")

    assert cli_main([str(path)]) == 0
    output = capsys.readouterr().out
    assert "loading            : ordered-mission" in output
    assert "load interaction   : willenborg" in output


def test_cli_reports_unsupported_ordered_sensitivity_without_a_traceback(
    tmp_path, capsys
):
    path = tmp_path / "mission.yaml"
    path.write_text(
        yaml.safe_dump(mission_spec([1000.0] * 100), sort_keys=True),
        encoding="utf-8",
    )

    assert cli_main([str(path), "--sensitivity"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error: Sobol sensitivity is not available" in captured.err


def test_ordered_inspections_flow_through_cycle_checkpoints():
    spec = mission_spec([100.0] * 10)
    spec["service_cycles"] = 10.0
    spec["retardation"] = {
        "model": "none",
        "max_work": 100,
    }
    spec["inspection"] = {
        "times": [2.0, 5.0, 9.0],
        "pod_a50": 2.0e-3,
        "pod_a90": 4.0e-3,
    }
    result = build_study(spec).run(curve_points=5)

    assert result.inspection is not None
    assert result.inspection.times == [2.0, 5.0, 9.0]
    assert result.inspection.pof_inspected == result.inspection.pof_unmitigated


def test_live_checkpoint_crack_drives_pod_and_residual_risk():
    class FixedGrowthLaw:
        def rate(self, dk, _stress_ratio, **_kwargs):
            return np.full_like(np.asarray(dk, dtype=float), 1.0e-3)

    sequence = SpectrumSequence.from_cycles(
        [
            {"delta_sigma": 1000.0, "stress_ratio": 0.0},
            {"delta_sigma": 1000.0, "stress_ratio": 0.0},
            {"delta_sigma": 1500.0, "stress_ratio": 0.0},
        ]
    )
    pod = damocles.PODCurve.from_a50_a90(1.5e-3, 2.5e-3)
    study = damocles.DamageToleranceStudy(
        "live checkpoint POD",
        variables={
            "initial_flaw": damocles.Deterministic(1.0e-3),
            "stress_scale": damocles.Deterministic(1.0),
            "toughness": damocles.Deterministic(100.0),
        },
        geometry=damocles.ThroughCrack(),
        growth_law=FixedGrowthLaw(),
        service_cycles=3.0,
        inspection_plan=damocles.InspectionPlan([1.0], pod),
        n_samples=1,
        method="lhs",
        seed=7,
        spectrum=sequence,
        retardation=WillenborgConfig.disabled(max_cycles=3.0, max_blocks=1, max_work=3),
    )
    result = study.run(curve_points=3)

    expected = 1.0 - float(pod.pod(2.0e-3))
    incorrectly_capped = 1.0 - float(pod.pod((100.0 / 1500.0) ** 2 / np.pi))
    assert result.lives[0] == 2.0
    assert result.inspection.pof_inspected == pytest.approx(expected)
    assert result.inspection.pof_inspected != pytest.approx(incorrectly_capped)


def test_failure_at_checkpoint_is_not_available_for_detection():
    class FixedGrowthLaw:
        def rate(self, dk, _stress_ratio, **_kwargs):
            return np.full_like(np.asarray(dk, dtype=float), 1.0e-3)

    sequence = SpectrumSequence.from_cycles(
        [
            {"delta_sigma": 1000.0, "stress_ratio": 0.0},
            {"delta_sigma": 1000.0, "stress_ratio": 0.0},
            {"delta_sigma": 1500.0, "stress_ratio": 0.0},
        ]
    )
    study = damocles.DamageToleranceStudy(
        "opening-load failure POD",
        variables={
            "initial_flaw": damocles.Deterministic(1.0e-3),
            "stress_scale": damocles.Deterministic(1.0),
            "toughness": damocles.Deterministic(100.0),
        },
        geometry=damocles.ThroughCrack(),
        growth_law=FixedGrowthLaw(),
        service_cycles=3.0,
        inspection_plan=damocles.InspectionPlan(
            [2.0],
            damocles.PODCurve.from_a50_a90(1.0e-6, 2.0e-6),
        ),
        n_samples=1,
        method="lhs",
        seed=7,
        spectrum=sequence,
        retardation=WillenborgConfig.disabled(
            max_cycles=3.0,
            max_blocks=1,
            max_work=3,
        ),
    )
    result = study.run(curve_points=3)

    assert result.lives[0] == 2.0
    assert result.inspection.pof_unmitigated == 1.0
    assert result.inspection.pof_inspected == 1.0
    assert result.inspection.mean_detections == 0.0


def test_study_reuses_its_exact_prepared_index_inside_growth(monkeypatch):
    captured = []
    original = study_module.grow_spectrum_retarded

    def capture_index(*args, **kwargs):
        captured.append(kwargs.get("_boundary_index"))
        return original(*args, **kwargs)

    monkeypatch.setattr(study_module, "grow_spectrum_retarded", capture_index)
    study = build_study(mission_spec([1000.0] * 100))
    prepared = study._mission_boundaries
    study.run(curve_points=3)

    assert captured == [prepared]
    assert captured[0] is prepared


def test_study_revalidates_prepared_index_before_sampling(monkeypatch):
    study = build_study(mission_spec([1000.0] * 100))
    object.__setattr__(study._mission_boundaries, "_block_ticks", 0)

    def sampling_must_not_start(*_args, **_kwargs):
        raise AssertionError("sampling began")

    monkeypatch.setattr("damocles.study.sample_unit", sampling_must_not_start)
    with pytest.raises(ValueError, match="block_ticks must be positive"):
        study.run(curve_points=3)


@pytest.mark.parametrize(
    "spec",
    [
        {"model": "willenborg", "yield_strength": True},
        {"model": "willenborg", "yield_strength": 1000.0, "max_cycles": np.inf},
        {"model": "willenborg", "yield_strength": 1000.0, "max_blocks": 1.5},
        {"model": "none", "yield_strength": 1000.0},
        {"model": "none", "max_work": 100_000_001},
        {"model": "unknown"},
        {"model": "none", "extra": 1},
    ],
)
def test_willenborg_config_rejects_hostile_types_and_bounds(spec):
    with pytest.raises((TypeError, ValueError)):
        WillenborgConfig.from_spec(spec)


@pytest.mark.parametrize("samples", [True, 1.5, "2", 0])
def test_mission_sample_count_is_strict(samples):
    spec = mission_spec([1000.0] * 100)
    spec["analysis"]["samples"] = samples
    with pytest.raises((TypeError, ValueError), match="n_samples"):
        build_study(spec)


@pytest.mark.parametrize(
    "service_cycles",
    [
        True,
        "100",
        np.inf,
        0.0,
        0.5,
        np.nextafter(100.0, 0.0),
        np.nextafter(100.0, np.inf),
    ],
)
def test_mission_service_horizon_is_strict(service_cycles):
    spec = mission_spec([1000.0] * 100)
    spec["service_cycles"] = service_cycles
    with pytest.raises((TypeError, ValueError), match="service_cycles"):
        build_study(spec)


def test_mission_work_cap_fails_during_build_before_sampling(monkeypatch):
    spec = mission_spec([1000.0] * 100)
    spec["analysis"]["samples"] = 2
    spec["retardation"]["max_work"] = 100

    def sampling_must_not_start(*_args, **_kwargs):
        raise AssertionError("sampling began")

    monkeypatch.setattr("damocles.study.sample_unit", sampling_must_not_start)
    with pytest.raises(ValueError, match="max_work"):
        build_study(spec)


def test_limits_must_cover_service_life_before_sampling():
    too_few_cycles = mission_spec([1000.0] * 100)
    too_few_cycles["retardation"]["max_cycles"] = 99.0
    with pytest.raises(ValueError, match="complete service life"):
        build_study(too_few_cycles)

    too_few_blocks = mission_spec([1000.0] * 10)
    too_few_blocks["retardation"]["max_blocks"] = 1
    with pytest.raises(ValueError, match="max_blocks"):
        build_study(too_few_blocks)


def test_derived_limits_respect_the_absolute_cycle_and_block_caps():
    too_many_cycles = mission_spec([1000.0])
    too_many_cycles["service_cycles"] = 100_000_001.0
    too_many_cycles["retardation"] = {
        "model": "none",
        "max_work": 100_000_000,
    }
    with pytest.raises(ValueError, match="resolved max_cycles"):
        build_study(too_many_cycles)

    too_many_blocks = mission_spec([1000.0])
    too_many_blocks["service_cycles"] = 1_000_001.0
    too_many_blocks["retardation"] = {
        "model": "none",
        "max_work": 100_000_000,
    }
    with pytest.raises(ValueError, match="resolved max_blocks"):
        build_study(too_many_blocks)


def test_sensitivity_rejects_before_ordered_growth(monkeypatch):
    study = build_study(mission_spec([1000.0] * 100))

    def sampling_must_not_start(*_args, **_kwargs):
        raise AssertionError("sampling began")

    monkeypatch.setattr("damocles.study.sample_unit", sampling_must_not_start)
    with pytest.raises(ValueError, match="Sobol sensitivity"):
        study.run(sensitivity=True)


@pytest.mark.parametrize(
    "inspection",
    [
        {"times": [5.0, 2.0], "pod_a50": 1e-3, "pod_a90": 2e-3},
        {"times": [2.0, 2.0], "pod_a50": 1e-3, "pod_a90": 2e-3},
        {"times": [0.0], "pod_a50": 1e-3, "pod_a90": 2e-3},
        {"times": [100.0], "pod_a50": 1e-3, "pod_a90": 2e-3},
        {"times": [0.5], "pod_a50": 1e-3, "pod_a90": 2e-3},
        {
            "times": [np.nextafter(1.0, np.inf)],
            "pod_a50": 1e-3,
            "pod_a90": 2e-3,
        },
    ],
)
def test_ordered_inspection_times_have_strict_semantics(inspection):
    spec = mission_spec([1000.0] * 100, inspection=inspection)
    with pytest.raises(ValueError, match="inspection"):
        build_study(spec)
