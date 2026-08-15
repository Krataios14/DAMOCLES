"""Tests for ordered spectrum growth with Willenborg retardation."""

from types import MappingProxyType

import numpy as np
import pytest

import damocles.retardation as retardation_module
from damocles.fracture import (
    ParisLaw,
    ThroughCrack,
    grow,
    grow_spectrum_retarded,
)
from damocles.nasgro import NasgroLaw
from damocles.retardation import (
    MAX_RETARDED_BLOCKS,
    OrderedBoundaryIndex,
    cycle_steps_to_boundary,
    effective_kr,
    init_state,
    plastic_zone_radius,
    prospective_cycle_steps,
    residual_stress_intensity,
)
from damocles.spectrum import MAX_ORDERED_CYCLES, OrderedCycle, SpectrumSequence


DS_CA = 1000.0
LAW = ParisLaw(1e-11, 3.0)
GEO = ThroughCrack()
K_IC = 100.0
S_YIELD = 500.0


def _sequence(ranges, stress_ratio=0.0):
    """Build a sequence of full cycles without invoking cycle counting."""
    cycles = [
        OrderedCycle(float(ds), stress_ratio, 1.0, idx) for idx, ds in enumerate(ranges)
    ]
    peak = max(ds / (1.0 - stress_ratio) for ds in ranges)
    return SpectrumSequence(cycles, peak, 0.0)


def _scalar_reference_life(a0, sequence, c, exponent, k_ic, s_yield, max_blocks=10_000):
    """Independent scalar implementation of AFGROW equations 5.2.3-5.2.6."""
    a = float(a0)
    elapsed = 0.0
    state = None

    for _ in range(max_blocks):
        for cycle in sequence.cycles:
            s_max = cycle.delta_sigma / (1.0 - cycle.stress_ratio)
            a_critical = (k_ic / s_max) ** 2 / np.pi
            if a >= a_critical:
                return elapsed

            k_max = s_max * np.sqrt(np.pi * a)
            k_min = k_max * cycle.stress_ratio
            r_p = (1.0 / np.pi) * (k_max / s_yield) ** 2
            boundary = a + r_p

            if state is None or boundary > state[3]:
                k_r = 0.0
                state = (a, k_max, r_p, boundary)
            else:
                a_ol, k_max_ol, r_p_ol, _ = state
                remaining = max(1.0 - (a - a_ol) / r_p_ol, 0.0)
                k_required = k_max_ol * np.sqrt(remaining)
                k_r = max(k_required - k_max, 0.0)

            k_max_eff = max(k_max - k_r, 0.0)
            k_min_eff = max(k_min - k_r, 0.0)
            dk_eff = max(k_max_eff - k_min_eff, 0.0)
            a += cycle.count * c * dk_eff**exponent
            elapsed += cycle.count

            if a >= a_critical:
                return elapsed

    return np.inf


class TestPlasticZoneRadius:
    def test_plane_stress_formula(self):
        k_max = np.array([10.0])
        expected = (1.0 / np.pi) * (k_max / 500.0) ** 2
        np.testing.assert_allclose(
            plastic_zone_radius(k_max, 500.0), expected, rtol=1e-12
        )

    def test_vectorized(self):
        result = plastic_zone_radius(np.array([10.0, 20.0, 30.0]), 500.0)
        assert result.shape == (3,)
        assert np.all(np.diff(result) > 0.0)

    @pytest.mark.parametrize("s_yield", [0.0, -1.0, np.inf, np.nan])
    def test_rejects_invalid_yield_strength(self, s_yield):
        with pytest.raises(ValueError, match="s_yield"):
            plastic_zone_radius(10.0, s_yield)

    def test_rejects_invalid_k_max(self):
        with pytest.raises(ValueError, match="k_max"):
            plastic_zone_radius(np.array([10.0, -1.0]), 500.0)


class TestState:
    def test_initial_state_has_no_controlling_zone(self):
        state = init_state(3)
        assert state.k_max_ol.shape == (3,)
        assert state.r_p.shape == (3,)
        assert state.a_overload.shape == (3,)
        assert not state.active.any()
        assert np.all(state.k_max_ol == 0.0)
        assert np.all(state.r_p == 0.0)
        assert np.all(state.a_overload == 0.0)

    def test_rejects_negative_sample_count(self):
        with pytest.raises(ValueError, match="non-negative"):
            init_state(-1)


class TestResidualStressIntensity:
    def test_matches_afgrow_equation_5_2_4(self):
        # K_R = 30*sqrt(1 - (0.006-0.005)/0.004) - 24
        expected = 30.0 * np.sqrt(0.75) - 24.0
        result = residual_stress_intensity(0.006, 24.0, 30.0, 0.004, 0.005, True)
        assert float(result) == pytest.approx(expected, rel=1e-12)

    def test_depends_on_current_cycle_k_max(self):
        result = residual_stress_intensity(
            np.array([0.006, 0.006]),
            np.array([20.0, 24.0]),
            30.0,
            0.004,
            0.005,
            True,
        )
        assert result[0] > result[1] > 0.0

    def test_zero_when_current_zone_reaches_boundary(self):
        k_required = 30.0 * np.sqrt(0.75)
        result = residual_stress_intensity(0.006, k_required, 30.0, 0.004, 0.005, True)
        assert float(result) == pytest.approx(0.0, abs=1e-15)

    def test_zero_after_overload_zone_is_exhausted(self):
        result = residual_stress_intensity(0.009, 10.0, 30.0, 0.004, 0.005, True)
        assert float(result) == pytest.approx(0.0, abs=1e-15)

    def test_inactive_state_returns_zero(self):
        result = residual_stress_intensity(0.006, 20.0, 30.0, 0.004, 0.005, False)
        assert float(result) == pytest.approx(0.0, abs=1e-15)


class TestEffectiveKr:
    def test_no_residual_leaves_cycle_unchanged(self):
        dk_eff, r_eff = effective_kr(20.0, 0.2, 0.0)
        assert float(dk_eff) == pytest.approx(20.0)
        assert float(r_eff) == pytest.approx(0.2)

    def test_equal_shift_preserves_delta_k_for_positive_effective_minimum(self):
        dk_eff, r_eff = effective_kr(20.0, 0.2, 2.0)
        assert float(dk_eff) == pytest.approx(20.0)
        assert float(r_eff) == pytest.approx(3.0 / 23.0)

    def test_zero_r_cutoff_reduces_range(self):
        dk_eff, r_eff = effective_kr(20.0, 0.2, 10.0)
        assert float(dk_eff) == pytest.approx(15.0)
        assert float(r_eff) == pytest.approx(0.0)

    def test_residual_can_arrest_cycle(self):
        dk_eff, r_eff = effective_kr(20.0, 0.0, 25.0)
        assert float(dk_eff) == pytest.approx(0.0)
        assert float(r_eff) == pytest.approx(0.0)

    def test_vectorized(self):
        dk_eff, r_eff = effective_kr(np.array([20.0, 20.0]), 0.0, np.array([0.0, 5.0]))
        np.testing.assert_allclose(dk_eff, [20.0, 15.0])
        np.testing.assert_allclose(r_eff, [0.0, 0.0])

    def test_rejects_invalid_stress_ratio(self):
        with pytest.raises(ValueError, match="stress_ratio"):
            effective_kr(20.0, 1.0, 0.0)


class TestConstantAmplitudeAgreement:
    @pytest.mark.parametrize("a0", [1e-4, 5e-4, 1e-3])
    def test_matches_existing_growth_path(self, a0):
        law = ParisLaw(3e-11, 3.0)
        expected = grow([a0], DS_CA, GEO, law, K_IC).cycles_to_failure[0]
        result = grow_spectrum_retarded(
            [a0], _sequence([DS_CA]), GEO, law, K_IC, S_YIELD
        )
        assert result.cycles_to_failure[0] == pytest.approx(expected, rel=0.01, abs=1.0)

    def test_multiple_samples(self):
        law = ParisLaw(3e-11, 3.0)
        a0 = np.array([1e-4, 5e-4, 1e-3])
        expected = grow(a0, DS_CA, GEO, law, K_IC).cycles_to_failure
        result = grow_spectrum_retarded(a0, _sequence([DS_CA]), GEO, law, K_IC, S_YIELD)
        np.testing.assert_allclose(
            result.cycles_to_failure, expected, rtol=0.01, atol=1.0
        )


class TestLoadInteraction:
    def test_single_overload_delays_growth(self):
        law = ParisLaw(1e-10, 3.0)
        baseline = grow_spectrum_retarded(
            [1e-3],
            _sequence([1000.0] * 501),
            GEO,
            law,
            K_IC,
            1000.0,
            max_blocks=1,
        )
        overloaded = grow_spectrum_retarded(
            [1e-3],
            _sequence([1500.0] + [1000.0] * 500),
            GEO,
            law,
            K_IC,
            1000.0,
            max_blocks=1,
        )
        assert baseline.cycles_to_failure[0] == 51.0
        assert overloaded.cycles_to_failure[0] == 82.0

    def test_overload_order_changes_life(self):
        law = ParisLaw(1e-10, 3.0)
        first = [1500.0] + [1000.0] * 500
        later = [1000.0] * 10 + [1500.0] + [1000.0] * 490
        life_first = grow_spectrum_retarded(
            [1e-3],
            _sequence(first),
            GEO,
            law,
            K_IC,
            1000.0,
            max_blocks=1,
        ).cycles_to_failure[0]
        life_later = grow_spectrum_retarded(
            [1e-3],
            _sequence(later),
            GEO,
            law,
            K_IC,
            1000.0,
            max_blocks=1,
        ).cycles_to_failure[0]
        assert life_first == 82.0
        assert life_later == 78.0


class TestSpectrumSequence:
    def test_ordered_history_preserves_order(self):
        sequence = SpectrumSequence.from_history_ordered([1500.0, 0.0, 1000.0, 0.0])
        assert [cycle.delta_sigma for cycle in sequence.cycles[:2]] == [1500.0, 1000.0]

    def test_ordered_history_records_true_peak(self):
        sequence = SpectrumSequence.from_history_ordered(
            [100.0, 0.0, 200.0, 0.0, 150.0, 0.0]
        )
        assert sequence.peak_stress == pytest.approx(200.0)
        assert sequence.total_count > 0.0

    def test_ordered_history_drops_compressive_cycles(self):
        with pytest.raises(ValueError, match="no damaging"):
            SpectrumSequence.from_history_ordered(
                [-100.0, -200.0, -100.0, -150.0, -100.0]
            )


class TestVectorizedSamples:
    def test_different_initial_cracks_have_ordered_lives(self):
        law = ParisLaw(1e-10, 3.0)
        result = grow_spectrum_retarded(
            np.array([1e-4, 5e-4, 1e-3]),
            _sequence([DS_CA]),
            GEO,
            law,
            K_IC,
            S_YIELD,
        )
        assert np.all(np.diff(result.cycles_to_failure) < 0.0)

    def test_per_sample_stress_scale(self):
        law = ParisLaw(1e-10, 3.0)
        result = grow_spectrum_retarded(
            np.full(3, 1e-3),
            _sequence([DS_CA]),
            GEO,
            law,
            K_IC,
            S_YIELD,
            stress_scale=np.array([0.8, 1.0, 1.2]),
        )
        assert np.all(np.diff(result.cycles_to_failure) < 0.0)

    def test_per_sample_paris_coefficients_with_chunking(self):
        law = ParisLaw(np.array([1e-10, 2e-10]), 3.0)
        result = grow_spectrum_retarded(
            np.array([1e-4, 1e-4]),
            _sequence([DS_CA]),
            GEO,
            law,
            K_IC,
            S_YIELD,
            chunk_size=1,
        )
        assert np.all(np.isfinite(result.cycles_to_failure))
        assert result.cycles_to_failure[0] > result.cycles_to_failure[1]

    def test_surviving_subset_uses_matching_material_samples(self):
        law = ParisLaw(np.array([1e-10, 2e-10, 3e-10]), 3.0)
        result = grow_spectrum_retarded(
            np.array([1.0, 1e-4, 2e-4]),
            _sequence([DS_CA]),
            GEO,
            law,
            K_IC,
            S_YIELD,
            chunk_size=3,
        )
        np.testing.assert_allclose(result.cycles_to_failure, [0.0, 151.0, 66.0])

    def test_chunking_is_invisible_after_samples_fail(self):
        a0 = np.array([1.0, 1e-4, 2e-4])
        law = ParisLaw(np.array([1e-10, 2e-10, 3e-10]), 3.0)
        one = grow_spectrum_retarded(
            a0, _sequence([DS_CA]), GEO, law, K_IC, S_YIELD, chunk_size=1
        )
        all_at_once = grow_spectrum_retarded(
            a0, _sequence([DS_CA]), GEO, law, K_IC, S_YIELD, chunk_size=3
        )
        np.testing.assert_array_equal(
            one.cycles_to_failure, all_at_once.cycles_to_failure
        )

    def test_nasgro_accepts_per_sample_effective_stress_ratios(self):
        law = NasgroLaw(c=np.array([1e-9, 2e-9]), n=3.0, p=0.0, q=0.0, dk1=0.0)
        result = grow_spectrum_retarded(
            np.array([1e-3, 1e-3]),
            _sequence([750.0, 500.0]),
            GEO,
            law,
            K_IC,
            S_YIELD,
            max_cycles=10_000,
        )
        np.testing.assert_allclose(result.cycles_to_failure, [99.0, 51.0])


class TestLifecycleAndValidation:
    def test_crack_already_above_critical_fails_at_zero(self):
        result = grow_spectrum_retarded(
            [1.0], _sequence([DS_CA]), GEO, LAW, K_IC, S_YIELD
        )
        assert result.cycles_to_failure[0] == 0.0

    def test_crossing_critical_size_fails_on_same_cycle(self):
        a_critical = (K_IC / DS_CA) ** 2 / np.pi
        law = ParisLaw(1e-6, 3.0)
        result = grow_spectrum_retarded(
            [0.99 * a_critical], _sequence([DS_CA]), GEO, law, K_IC, S_YIELD
        )
        assert result.cycles_to_failure[0] == 1.0

    def test_max_cycles_is_runout(self):
        result = grow_spectrum_retarded(
            [1e-4],
            _sequence([DS_CA]),
            GEO,
            ParisLaw(1e-30, 3.0),
            K_IC,
            S_YIELD,
            max_cycles=100,
        )
        assert result.cycles_to_failure[0] == np.inf

    def test_below_threshold_cycles_still_count_toward_runout(self):
        law = ParisLaw(1e-11, 3.0, dk_threshold=50.0)
        result = grow_spectrum_retarded(
            [1e-4], _sequence([5.0]), GEO, law, K_IC, S_YIELD, max_cycles=500
        )
        assert result.cycles_to_failure[0] == np.inf

    @pytest.mark.parametrize(
        ("keyword", "value", "message"),
        [
            ("s_yield", 0.0, "s_yield"),
            ("max_cycles", 0.0, "max_cycles"),
            ("max_blocks", 0, "max_blocks"),
            ("chunk_size", 0, "chunk_size"),
        ],
    )
    def test_rejects_invalid_integration_limits(self, keyword, value, message):
        kwargs = {"s_yield": S_YIELD, keyword: value}
        with pytest.raises(ValueError, match=message):
            grow_spectrum_retarded([1e-3], _sequence([DS_CA]), GEO, LAW, K_IC, **kwargs)

    def test_ordered_baseline_records_post_record_checkpoints(self):
        a0 = 1.0e-4
        law = ParisLaw(1.0e-11, 3.0)
        sequence = SpectrumSequence.from_cycles(
            [
                {"delta_sigma": 100.0, "stress_ratio": 0.0, "count": 0.5},
                {"delta_sigma": 100.0, "stress_ratio": 0.0, "count": 0.5},
            ]
        )
        result = grow_spectrum_retarded(
            [a0],
            sequence,
            GEO,
            law,
            K_IC,
            None,
            max_cycles=1.0,
            max_blocks=1,
            eval_cycles=[0.5, 1.0],
            apply_retardation=False,
        )
        first_rate = law.rate(np.array([100.0 * np.sqrt(np.pi * a0)]))[0]
        after_first = a0 + 0.5 * first_rate
        second_rate = law.rate(np.array([100.0 * np.sqrt(np.pi * after_first)]))[0]
        assert result.a_at[0, 0] == pytest.approx(after_first)
        assert result.a_at[0, 1] == pytest.approx(after_first + 0.5 * second_rate)
        np.testing.assert_array_equal(result.eval_cycles, [0.5, 1.0])

    def test_ordered_baseline_is_repeatable(self):
        sequence = _sequence([1500.0] + [1000.0] * 100)
        kwargs = dict(
            a0=np.array([1e-3, 1.1e-3]),
            sequence=sequence,
            geometry=GEO,
            law=ParisLaw(1e-10, 3.0),
            k_ic=K_IC,
            s_yield=None,
            max_cycles=100,
            max_blocks=1,
            eval_cycles=[10.0, 50.0],
            apply_retardation=False,
        )
        first = grow_spectrum_retarded(**kwargs)
        second = grow_spectrum_retarded(**kwargs)
        np.testing.assert_array_equal(first.cycles_to_failure, second.cycles_to_failure)
        np.testing.assert_array_equal(first.a_at, second.a_at)

    @pytest.mark.parametrize(
        ("keyword", "value", "message"),
        [
            ("max_cycles", np.inf, "max_cycles"),
            ("max_cycles", True, "max_cycles"),
            ("max_blocks", 1.5, "max_blocks"),
            ("max_blocks", True, "max_blocks"),
            ("max_work", 0, "max_work"),
            ("apply_retardation", 1, "apply_retardation"),
        ],
    )
    def test_rejects_hostile_limit_types(self, keyword, value, message):
        kwargs = {"s_yield": S_YIELD, keyword: value}
        with pytest.raises((TypeError, ValueError), match=message):
            grow_spectrum_retarded([1e-3], _sequence([DS_CA]), GEO, LAW, K_IC, **kwargs)

    def test_work_cap_fails_before_geometry_evaluation(self):
        class ExplodingGeometry:
            def y(self, _a):
                raise AssertionError("heavy geometry work began")

        sequence = _sequence([1000.0] * 100)
        with pytest.raises(ValueError, match="max_work"):
            grow_spectrum_retarded(
                [1e-3, 1e-3],
                sequence,
                ExplodingGeometry(),
                LAW,
                K_IC,
                S_YIELD,
                max_cycles=100,
                max_blocks=1,
                max_work=100,
            )

    @pytest.mark.parametrize(
        "checkpoints",
        [[2.0, 1.0], [1.0, 1.0], [0.0], [np.inf], [[1.0]]],
    )
    def test_rejects_invalid_inspection_checkpoints(self, checkpoints):
        with pytest.raises(ValueError, match="eval_cycles"):
            grow_spectrum_retarded(
                [1e-3],
                _sequence([DS_CA]),
                GEO,
                LAW,
                K_IC,
                S_YIELD,
                max_cycles=2,
                max_blocks=2,
                eval_cycles=checkpoints,
            )

    @pytest.mark.parametrize(
        ("keyword", "value"),
        [
            ("max_cycles", 0.5),
            ("max_cycles", np.nextafter(1.0, 0.0)),
            ("max_cycles", np.nextafter(1.0, np.inf)),
        ],
    )
    def test_rejects_horizons_inside_or_adjacent_to_a_record_before_geometry(
        self, keyword, value
    ):
        class ExplodingGeometry:
            def y(self, _a):
                raise AssertionError("geometry evaluation began")

        with pytest.raises(ValueError, match="max_cycles"):
            grow_spectrum_retarded(
                [1e-3],
                _sequence([DS_CA]),
                ExplodingGeometry(),
                LAW,
                K_IC,
                S_YIELD,
                max_blocks=2,
                **{keyword: value},
            )

    @pytest.mark.parametrize(
        "checkpoint",
        [0.5, np.nextafter(1.0, 0.0), np.nextafter(1.0, np.inf)],
    )
    def test_rejects_checkpoints_inside_or_adjacent_to_a_record(self, checkpoint):
        with pytest.raises(ValueError, match="eval_cycles entry"):
            grow_spectrum_retarded(
                [1e-3],
                _sequence([DS_CA]),
                GEO,
                LAW,
                K_IC,
                S_YIELD,
                max_cycles=2.0,
                max_blocks=2,
                eval_cycles=[checkpoint],
            )

    def test_failure_checkpoint_is_never_above_critical_size(self):
        a_critical = (K_IC / DS_CA) ** 2 / np.pi
        result = grow_spectrum_retarded(
            [0.99 * a_critical],
            _sequence([DS_CA]),
            GEO,
            ParisLaw(1e-6, 3.0),
            K_IC,
            S_YIELD,
            max_cycles=1.0,
            max_blocks=1,
            eval_cycles=[1.0],
        )
        assert result.cycles_to_failure[0] == 1.0
        assert result.a_at[0, 0] == pytest.approx(result.a_critical[0])
        assert result.a_at[0, 0] <= result.a_critical[0]

    def test_live_low_load_checkpoint_preserves_actual_crack_size(self):
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
        result = grow_spectrum_retarded(
            [1.0e-3],
            sequence,
            GEO,
            FixedGrowthLaw(),
            K_IC,
            None,
            max_cycles=3.0,
            max_blocks=1,
            eval_cycles=[1.0],
            apply_retardation=False,
        )
        assert result.cycles_to_failure[0] == 2.0
        assert result.a_at[0, 0] == pytest.approx(2.0e-3)
        assert result.a_at[0, 0] > result.a_critical[0]

    def test_opening_load_failure_replaces_checkpoint_at_same_endpoint(self):
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
        result = grow_spectrum_retarded(
            [1.0e-3],
            sequence,
            GEO,
            FixedGrowthLaw(),
            K_IC,
            None,
            max_cycles=3.0,
            max_blocks=1,
            eval_cycles=[1.0, 2.0, 3.0],
            apply_retardation=False,
        )

        high_cycle_critical = (K_IC / 1500.0) ** 2 / np.pi
        assert result.cycles_to_failure[0] == 2.0
        assert result.a_at[0, 0] == pytest.approx(2.0e-3)
        assert result.a_at[0, 1] == pytest.approx(high_cycle_critical)
        assert result.a_at[0, 2] == pytest.approx(high_cycle_critical)

    def test_opening_load_failure_only_replaces_newly_failed_sample(self):
        class FixedGrowthLaw:
            def rate(self, dk, _stress_ratio, **_kwargs):
                return np.full_like(np.asarray(dk, dtype=float), 5.0e-4)

        sequence = SpectrumSequence.from_cycles(
            [
                {"delta_sigma": 1000.0, "stress_ratio": 0.0},
                {"delta_sigma": 1000.0, "stress_ratio": 0.0},
                {"delta_sigma": 1500.0, "stress_ratio": 0.0},
            ]
        )
        result = grow_spectrum_retarded(
            [1.0e-3, 1.0e-4],
            sequence,
            GEO,
            FixedGrowthLaw(),
            K_IC,
            None,
            max_cycles=3.0,
            max_blocks=1,
            eval_cycles=[2.0],
            apply_retardation=False,
        )

        high_cycle_critical = (K_IC / 1500.0) ** 2 / np.pi
        assert result.cycles_to_failure[0] == 2.0
        assert result.a_at[0, 0] == pytest.approx(high_cycle_critical)
        assert result.a_at[1, 0] == pytest.approx(1.1e-3)

    def test_failure_and_later_checkpoints_use_cycle_specific_critical_size(self):
        class FixedGrowthLaw:
            def rate(self, dk, _stress_ratio, **_kwargs):
                return np.full_like(np.asarray(dk, dtype=float), 3.0e-3)

        sequence = SpectrumSequence.from_cycles(
            [
                {"delta_sigma": 1000.0, "stress_ratio": 0.0},
                {"delta_sigma": 1500.0, "stress_ratio": 0.0},
            ]
        )
        result = grow_spectrum_retarded(
            [1.0e-3],
            sequence,
            GEO,
            FixedGrowthLaw(),
            K_IC,
            None,
            max_cycles=2.0,
            max_blocks=1,
            eval_cycles=[1.0, 2.0],
            apply_retardation=False,
        )
        low_cycle_critical = (K_IC / 1000.0) ** 2 / np.pi
        assert result.cycles_to_failure[0] == 1.0
        np.testing.assert_allclose(result.a_at[0], low_cycle_critical)
        assert low_cycle_critical > result.a_critical[0]


class TestProspectiveWorkAccounting:
    class CountingNoGrowthLaw:
        def __init__(self):
            self.sample_cycle_evaluations = 0

        def rate(self, dk, _stress_ratio, **_kwargs):
            values = np.asarray(dk, dtype=float)
            self.sample_cycle_evaluations += values.size
            return np.zeros_like(values)

    def test_randomized_counter_exactly_matches_executed_rate_work(self):
        rng = np.random.default_rng(20260816)
        for _ in range(64):
            record_count = int(rng.integers(1, 10))
            counts = rng.choice([0.5, 1.0], size=record_count)
            sequence = SpectrumSequence.from_cycles(
                [
                    {
                        "delta_sigma": float(rng.uniform(10.0, 100.0)),
                        "stress_ratio": 0.0,
                        "count": float(count),
                    }
                    for count in counts
                ]
            )
            target_steps = int(rng.integers(1, 4 * record_count + 1))
            horizon = sum(
                sequence.cycles[index % record_count].count
                for index in range(target_steps)
            )
            max_blocks = (target_steps + record_count - 1) // record_count
            expected_steps = prospective_cycle_steps(sequence, horizon, max_blocks)
            assert expected_steps == target_steps

            law = self.CountingNoGrowthLaw()
            sample_count = int(rng.integers(1, 8))
            grow_spectrum_retarded(
                np.full(sample_count, 1e-4),
                sequence,
                GEO,
                law,
                1e9,
                None,
                max_cycles=horizon,
                max_blocks=max_blocks,
                chunk_size=2,
                apply_retardation=False,
            )
            assert law.sample_cycle_evaluations == sample_count * expected_steps

    @pytest.mark.parametrize(
        "adjacent",
        [np.nextafter(1.0, 0.0), np.nextafter(1.0, np.inf)],
    )
    def test_adjacent_float_cannot_be_counted_as_an_exact_boundary(self, adjacent):
        sequence = _sequence([DS_CA])
        with pytest.raises(ValueError, match="exact full/half-cycle"):
            prospective_cycle_steps(sequence, adjacent, 2)

    def test_half_cycle_value_inside_full_record_is_not_a_boundary(self):
        with pytest.raises(ValueError, match="record endpoint"):
            cycle_steps_to_boundary(_sequence([DS_CA]), 0.5)

    @pytest.mark.parametrize(
        ("max_blocks", "error", "message"),
        [
            (True, TypeError, "must be an integer"),
            (0, ValueError, "must be positive"),
            (-1, ValueError, "must be positive"),
            (1.0, TypeError, "must be an integer"),
            (10**100, ValueError, "exceeds the limit"),
            ([], TypeError, "must be an integer"),
        ],
    )
    def test_public_counter_rejects_invalid_max_blocks(
        self, max_blocks, error, message
    ):
        with pytest.raises(error, match=message):
            prospective_cycle_steps(_sequence([DS_CA]), 1.0, max_blocks)

    def test_public_counter_accepts_absolute_block_cap_only(self):
        sequence = _sequence([DS_CA])
        assert (
            prospective_cycle_steps(
                sequence,
                1.0,
                MAX_RETARDED_BLOCKS,
            )
            == 1
        )
        with pytest.raises(ValueError, match="exceeds the limit"):
            prospective_cycle_steps(
                sequence,
                1.0,
                MAX_RETARDED_BLOCKS + 1,
            )

    def test_max_caps_use_one_index_build_then_one_lookup_per_checkpoint(
        self, monkeypatch
    ):
        sequence = SpectrumSequence.from_cycles(
            {"delta_sigma": 100.0, "stress_ratio": 0.0}
            for _ in range(MAX_ORDERED_CYCLES)
        )
        index = OrderedBoundaryIndex.from_sequence(sequence)
        assert index.n_cycles == MAX_ORDERED_CYCLES
        assert len(index.endpoint_steps) == MAX_ORDERED_CYCLES

        assert type(index.endpoint_steps) is MappingProxyType
        validation_calls = 0
        lookup_calls = 0
        original_validation = retardation_module._build_boundary_snapshot
        original_lookup = retardation_module._BoundarySnapshot.steps_to

        def counted_validation(sequence):
            nonlocal validation_calls
            validation_calls += 1
            return original_validation(sequence)

        def counted_lookup(self, cycles, name="cycle horizon"):
            nonlocal lookup_calls
            lookup_calls += 1
            return original_lookup(self, cycles, name)

        monkeypatch.setattr(
            retardation_module, "_build_boundary_snapshot", counted_validation
        )
        monkeypatch.setattr(
            retardation_module._BoundarySnapshot, "steps_to", counted_lookup
        )
        checkpoint_count = 10_000
        checkpoints = np.arange(1, checkpoint_count + 1, dtype=float)
        steps = index.steps_many(checkpoints, "checkpoint")

        assert steps[0] == 1
        assert steps[-1] == checkpoint_count
        assert validation_calls == 1
        assert lookup_calls == checkpoint_count


class TestOrderedBoundaryIndexIntegrity:
    def test_direct_constructor_rejects_forged_and_mutable_tables(self):
        with pytest.raises(TypeError, match="cannot be constructed directly"):
            OrderedBoundaryIndex()
        with pytest.raises(TypeError, match="cannot be constructed directly"):
            OrderedBoundaryIndex(0, True, {0: True})
        with pytest.raises(TypeError, match="cannot be constructed directly"):
            OrderedBoundaryIndex(
                block_ticks=2,
                n_cycles=1,
                endpoint_steps={2: 99},
            )

    def test_prepared_index_is_frozen_and_endpoint_table_is_read_only(self):
        index = OrderedBoundaryIndex.from_sequence(_sequence([100.0, 200.0]))
        assert type(index.endpoint_steps) is MappingProxyType
        with pytest.raises(TypeError):
            index.endpoint_steps[2] = 99
        with pytest.raises((AttributeError, TypeError)):
            index._block_ticks = 0

    @pytest.mark.parametrize(
        ("field", "value", "error", "message"),
        [
            ("_block_ticks", 0, ValueError, "block_ticks must be positive"),
            ("_block_ticks", True, TypeError, "block_ticks must be an integer"),
            ("_n_cycles", 0, ValueError, "n_cycles is outside"),
            ("_n_cycles", True, TypeError, "n_cycles must be an integer"),
        ],
    )
    def test_low_level_scalar_replacement_fails_with_typed_error(
        self, field, value, error, message
    ):
        sequence = _sequence([100.0, 200.0])
        index = OrderedBoundaryIndex.from_sequence(sequence)
        object.__setattr__(index, field, value)

        with pytest.raises(error, match=message):
            cycle_steps_to_boundary(sequence, 1.0, boundary_index=index)
        with pytest.raises(error, match=message):
            index.steps_to(1.0)

    @pytest.mark.parametrize(
        ("replacement", "error", "message"),
        [
            ({2: 1, 4: 2}, TypeError, "endpoint table must be read-only"),
            (MappingProxyType({2: 1, 4: 2}), ValueError, "table is inconsistent"),
            (MappingProxyType({0: 1, 4: 2}), ValueError, "table is inconsistent"),
            (MappingProxyType({2: True, 4: 2}), ValueError, "table is inconsistent"),
            (MappingProxyType({2: 99, 4: 2}), ValueError, "table is inconsistent"),
        ],
    )
    def test_low_level_endpoint_table_replacement_is_rejected(
        self, replacement, error, message
    ):
        sequence = _sequence([100.0, 200.0])
        index = OrderedBoundaryIndex.from_sequence(sequence)
        object.__setattr__(index, "_endpoint_steps", replacement)

        with pytest.raises(error, match=message):
            prospective_cycle_steps(sequence, 2.0, 1, boundary_index=index)

    def test_low_level_fingerprint_and_identity_replacement_are_rejected(self):
        sequence = _sequence([100.0, 200.0])
        other = _sequence([100.0, 200.0])

        fingerprint_index = OrderedBoundaryIndex.from_sequence(sequence)
        object.__setattr__(
            fingerprint_index,
            "_sequence_fingerprint",
            "sha256:" + "0" * 64,
        )
        with pytest.raises(ValueError, match="fingerprint is inconsistent"):
            cycle_steps_to_boundary(
                sequence,
                1.0,
                boundary_index=fingerprint_index,
            )

        identity_index = OrderedBoundaryIndex.from_sequence(sequence)
        object.__setattr__(identity_index, "_sequence", other)
        with pytest.raises(ValueError, match="different SpectrumSequence"):
            cycle_steps_to_boundary(
                sequence,
                1.0,
                boundary_index=identity_index,
            )

    def test_prepared_index_detects_source_sequence_change(self):
        sequence = _sequence([100.0, 200.0])
        index = OrderedBoundaryIndex.from_sequence(sequence)
        object.__setattr__(sequence.cycles[0], "delta_sigma", 90.0)

        with pytest.raises(ValueError, match="source sequence changed"):
            cycle_steps_to_boundary(sequence, 1.0, boundary_index=index)

    def test_unregistered_exact_type_instances_are_rejected(self):
        sequence = _sequence([100.0, 200.0])
        prepared = OrderedBoundaryIndex.from_sequence(sequence)

        incomplete = object.__new__(OrderedBoundaryIndex)
        with pytest.raises(ValueError, match="structure is incomplete"):
            cycle_steps_to_boundary(sequence, 1.0, boundary_index=incomplete)

        unregistered = object.__new__(OrderedBoundaryIndex)
        for field in (
            "_sequence",
            "_sequence_fingerprint",
            "_block_ticks",
            "_n_cycles",
            "_endpoint_steps",
        ):
            object.__setattr__(
                unregistered,
                field,
                object.__getattribute__(prepared, field),
            )
        with pytest.raises(ValueError, match="was not prepared"):
            cycle_steps_to_boundary(sequence, 1.0, boundary_index=unregistered)

    def test_corrupt_index_fails_before_geometry_without_arithmetic_leak(self):
        class ExplodingGeometry:
            def y(self, _a):
                raise AssertionError("geometry evaluation began")

        sequence = _sequence([100.0])
        index = OrderedBoundaryIndex.from_sequence(sequence)
        object.__setattr__(index, "_block_ticks", 0)
        object.__setattr__(index, "_endpoint_steps", MappingProxyType({0: True}))

        with pytest.raises(ValueError, match="block_ticks must be positive"):
            grow_spectrum_retarded(
                [1.0e-3],
                sequence,
                ExplodingGeometry(),
                LAW,
                K_IC,
                S_YIELD,
                max_cycles=1.0,
                max_blocks=1,
                _boundary_index=index,
            )

    @pytest.mark.parametrize(
        ("cycles", "error", "message"),
        [
            (True, TypeError, "real number"),
            (0, ValueError, "positive"),
        ],
    )
    def test_boolean_and_zero_endpoint_ticks_are_rejected(self, cycles, error, message):
        index = OrderedBoundaryIndex.from_sequence(_sequence([100.0]))
        with pytest.raises(error, match=message):
            index.steps_to(cycles)

    def test_equal_but_distinct_sequence_cannot_reuse_index(self):
        original = _sequence([100.0, 200.0])
        equal_copy = _sequence([100.0, 200.0])
        assert equal_copy == original
        assert equal_copy is not original
        index = OrderedBoundaryIndex.from_sequence(original)

        with pytest.raises(ValueError, match="different SpectrumSequence"):
            cycle_steps_to_boundary(
                equal_copy,
                1.0,
                boundary_index=index,
            )
        with pytest.raises(ValueError, match="different SpectrumSequence"):
            prospective_cycle_steps(
                equal_copy,
                2.0,
                1,
                boundary_index=index,
            )

    def test_external_non_index_injection_is_rejected(self):
        sequence = _sequence([100.0])
        with pytest.raises(TypeError, match="boundary_index"):
            cycle_steps_to_boundary(
                sequence,
                1.0,
                boundary_index={2: 1},
            )

    def test_growth_rejects_index_from_another_sequence_before_geometry(self):
        class ExplodingGeometry:
            def y(self, _a):
                raise AssertionError("geometry evaluation began")

        indexed = _sequence([100.0])
        other = _sequence([100.0])
        index = OrderedBoundaryIndex.from_sequence(indexed)
        with pytest.raises(ValueError, match="different SpectrumSequence"):
            grow_spectrum_retarded(
                [1.0e-3],
                other,
                ExplodingGeometry(),
                LAW,
                K_IC,
                S_YIELD,
                max_cycles=1.0,
                max_blocks=1,
                _boundary_index=index,
            )


class TestIndependentReference:
    def test_scalar_cycle_by_cycle_reference(self):
        sequence = _sequence([1500.0] + [1000.0] * 500)
        expected = _scalar_reference_life(
            1e-3,
            sequence,
            1e-10,
            3.0,
            K_IC,
            1000.0,
            max_blocks=1,
        )
        result = grow_spectrum_retarded(
            [1e-3],
            sequence,
            GEO,
            ParisLaw(1e-10, 3.0),
            K_IC,
            1000.0,
            max_blocks=1,
        )

        # The fixed value makes accidental changes to the reference helper
        # visible as well as comparing it with the vectorized implementation.
        assert expected == 82.0
        assert result.cycles_to_failure[0] == expected
