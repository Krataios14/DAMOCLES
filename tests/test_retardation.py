"""Tests for ordered spectrum growth with Willenborg retardation."""

import numpy as np
import pytest

from damocles.fracture import (
    ParisLaw,
    ThroughCrack,
    grow,
    grow_spectrum_retarded,
)
from damocles.nasgro import NasgroLaw
from damocles.retardation import (
    effective_kr,
    init_state,
    plastic_zone_radius,
    residual_stress_intensity,
)
from damocles.spectrum import OrderedCycle, SpectrumSequence


DS_CA = 1000.0
LAW = ParisLaw(1e-11, 3.0)
GEO = ThroughCrack()
K_IC = 100.0
S_YIELD = 500.0


def _sequence(ranges, stress_ratio=0.0):
    """Build a sequence of full cycles without invoking cycle counting."""
    cycles = [
        OrderedCycle(float(ds), stress_ratio, 1.0, idx)
        for idx, ds in enumerate(ranges)
    ]
    peak = max(ds / (1.0 - stress_ratio) for ds in ranges)
    return SpectrumSequence(cycles, peak, 0.0)


def _scalar_reference_life(a0, sequence, c, exponent, k_ic, s_yield,
                           max_blocks=10_000):
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
            plastic_zone_radius(k_max, 500.0), expected, rtol=1e-12)

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
        result = residual_stress_intensity(
            0.006, 24.0, 30.0, 0.004, 0.005, True)
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
        result = residual_stress_intensity(
            0.006, k_required, 30.0, 0.004, 0.005, True)
        assert float(result) == pytest.approx(0.0, abs=1e-15)

    def test_zero_after_overload_zone_is_exhausted(self):
        result = residual_stress_intensity(
            0.009, 10.0, 30.0, 0.004, 0.005, True)
        assert float(result) == pytest.approx(0.0, abs=1e-15)

    def test_inactive_state_returns_zero(self):
        result = residual_stress_intensity(
            0.006, 20.0, 30.0, 0.004, 0.005, False)
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
        dk_eff, r_eff = effective_kr(
            np.array([20.0, 20.0]), 0.0, np.array([0.0, 5.0]))
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
            [a0], _sequence([DS_CA]), GEO, law, K_IC, S_YIELD)
        assert result.cycles_to_failure[0] == pytest.approx(
            expected, rel=0.01, abs=1.0)

    def test_multiple_samples(self):
        law = ParisLaw(3e-11, 3.0)
        a0 = np.array([1e-4, 5e-4, 1e-3])
        expected = grow(a0, DS_CA, GEO, law, K_IC).cycles_to_failure
        result = grow_spectrum_retarded(
            a0, _sequence([DS_CA]), GEO, law, K_IC, S_YIELD)
        np.testing.assert_allclose(
            result.cycles_to_failure, expected, rtol=0.01, atol=1.0)


class TestLoadInteraction:
    def test_single_overload_delays_growth(self):
        law = ParisLaw(1e-10, 3.0)
        baseline = grow_spectrum_retarded(
            [1e-3], _sequence([1000.0] * 501), GEO, law, K_IC, 1000.0,
            max_blocks=1,
        )
        overloaded = grow_spectrum_retarded(
            [1e-3], _sequence([1500.0] + [1000.0] * 500), GEO, law,
            K_IC, 1000.0, max_blocks=1,
        )
        assert baseline.cycles_to_failure[0] == 51.0
        assert overloaded.cycles_to_failure[0] == 82.0

    def test_overload_order_changes_life(self):
        law = ParisLaw(1e-10, 3.0)
        first = [1500.0] + [1000.0] * 500
        later = [1000.0] * 10 + [1500.0] + [1000.0] * 490
        life_first = grow_spectrum_retarded(
            [1e-3], _sequence(first), GEO, law, K_IC, 1000.0,
            max_blocks=1,
        ).cycles_to_failure[0]
        life_later = grow_spectrum_retarded(
            [1e-3], _sequence(later), GEO, law, K_IC, 1000.0,
            max_blocks=1,
        ).cycles_to_failure[0]
        assert life_first == 82.0
        assert life_later == 78.0


class TestSpectrumSequence:
    def test_ordered_history_preserves_order(self):
        sequence = SpectrumSequence.from_history_ordered(
            [1500.0, 0.0, 1000.0, 0.0])
        assert [cycle.delta_sigma for cycle in sequence.cycles[:2]] == [
            1500.0, 1000.0]

    def test_ordered_history_records_true_peak(self):
        sequence = SpectrumSequence.from_history_ordered(
            [100.0, 0.0, 200.0, 0.0, 150.0, 0.0])
        assert sequence.peak_stress == pytest.approx(200.0)
        assert sequence.total_count > 0.0

    def test_ordered_history_drops_compressive_cycles(self):
        with pytest.raises(ValueError, match="no damaging"):
            SpectrumSequence.from_history_ordered(
                [-100.0, -200.0, -100.0, -150.0, -100.0])


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
            np.array([1e-4, 1e-4]), _sequence([DS_CA]), GEO, law,
            K_IC, S_YIELD, chunk_size=1,
        )
        assert np.all(np.isfinite(result.cycles_to_failure))
        assert result.cycles_to_failure[0] > result.cycles_to_failure[1]

    def test_surviving_subset_uses_matching_material_samples(self):
        law = ParisLaw(np.array([1e-10, 2e-10, 3e-10]), 3.0)
        result = grow_spectrum_retarded(
            np.array([1.0, 1e-4, 2e-4]), _sequence([DS_CA]), GEO, law,
            K_IC, S_YIELD, chunk_size=3,
        )
        np.testing.assert_allclose(result.cycles_to_failure, [0.0, 151.0, 66.0])

    def test_chunking_is_invisible_after_samples_fail(self):
        a0 = np.array([1.0, 1e-4, 2e-4])
        law = ParisLaw(np.array([1e-10, 2e-10, 3e-10]), 3.0)
        one = grow_spectrum_retarded(
            a0, _sequence([DS_CA]), GEO, law, K_IC, S_YIELD, chunk_size=1)
        all_at_once = grow_spectrum_retarded(
            a0, _sequence([DS_CA]), GEO, law, K_IC, S_YIELD, chunk_size=3)
        np.testing.assert_array_equal(
            one.cycles_to_failure, all_at_once.cycles_to_failure)

    def test_nasgro_accepts_per_sample_effective_stress_ratios(self):
        law = NasgroLaw(
            c=np.array([1e-9, 2e-9]), n=3.0, p=0.0, q=0.0, dk1=0.0)
        result = grow_spectrum_retarded(
            np.array([1e-3, 1e-3]), _sequence([750.0, 500.0]), GEO,
            law, K_IC, S_YIELD, max_cycles=10_000)
        np.testing.assert_allclose(result.cycles_to_failure, [99.0, 51.0])


class TestLifecycleAndValidation:
    def test_crack_already_above_critical_fails_at_zero(self):
        result = grow_spectrum_retarded(
            [1.0], _sequence([DS_CA]), GEO, LAW, K_IC, S_YIELD)
        assert result.cycles_to_failure[0] == 0.0

    def test_crossing_critical_size_fails_on_same_cycle(self):
        a_critical = (K_IC / DS_CA) ** 2 / np.pi
        law = ParisLaw(1e-6, 3.0)
        result = grow_spectrum_retarded(
            [0.99 * a_critical], _sequence([DS_CA]), GEO, law,
            K_IC, S_YIELD)
        assert result.cycles_to_failure[0] == 1.0

    def test_max_cycles_is_runout(self):
        result = grow_spectrum_retarded(
            [1e-4], _sequence([DS_CA]), GEO, ParisLaw(1e-30, 3.0),
            K_IC, S_YIELD, max_cycles=100)
        assert result.cycles_to_failure[0] == np.inf

    def test_below_threshold_cycles_still_count_toward_runout(self):
        law = ParisLaw(1e-11, 3.0, dk_threshold=50.0)
        result = grow_spectrum_retarded(
            [1e-4], _sequence([5.0]), GEO, law, K_IC, S_YIELD,
            max_cycles=500)
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
            grow_spectrum_retarded(
                [1e-3], _sequence([DS_CA]), GEO, LAW, K_IC,
                **kwargs)


class TestIndependentReference:
    def test_scalar_cycle_by_cycle_reference(self):
        sequence = _sequence([1500.0] + [1000.0] * 500)
        expected = _scalar_reference_life(
            1e-3, sequence, 1e-10, 3.0, K_IC, 1000.0,
            max_blocks=1,
        )
        result = grow_spectrum_retarded(
            [1e-3], sequence, GEO, ParisLaw(1e-10, 3.0), K_IC,
            1000.0, max_blocks=1,
        )

        # The fixed value makes accidental changes to the reference helper
        # visible as well as comparing it with the vectorized implementation.
        assert expected == 82.0
        assert result.cycles_to_failure[0] == expected
