"""Tests for the Willenborg overload-retardation model."""

import numpy as np
import pytest

from damocles.fracture import (
    ParisLaw, ThroughCrack, grow, grow_spectrum, grow_spectrum_retarded,
)
from damocles.spectrum import Spectrum, SpectrumSequence
from damocles.retardation import (
    WillenborgState, init_state, plastic_zone_radius,
    retardation_factor, effective_kr,
)

DS_CA = 1000.0
LAW = ParisLaw(1e-11, 3.0)
GEO = ThroughCrack()
K_IC = 100.0
S_YIELD = 500.0


# ---------------------------------------------------------------- unit tests


class TestPlasticZoneRadius:
    def test_formula(self):
        k = np.array([10.0])
        sy = 500.0
        expected = (1.0 / np.pi) * (10.0 / 500.0) ** 2
        rp = plastic_zone_radius(k, sy)
        assert rp[0] == pytest.approx(expected, rel=1e-12)

    def test_vectorised(self):
        k = np.array([10.0, 20.0, 30.0])
        rp = plastic_zone_radius(k, 500.0)
        assert rp.shape == (3,)
        assert rp[1] > rp[0] > 0
        assert rp[2] > rp[1]


class TestInitState:
    def test_no_overload(self):
        s = init_state(3)
        assert s.k_max_ol.shape == (3,)
        assert s.r_p.shape == (3,)
        assert s.a_overload.shape == (3,)
        assert not s.active.any()
        assert np.all(s.k_max_ol == 0.0)
        assert np.all(s.r_p == 0.0)
        assert np.all(s.a_overload == 0.0)


class TestRetardationFactor:
    def test_zero_when_crack_beyond_zone(self):
        a = np.array([0.01])
        r_p = np.array([0.001])
        a_ol = np.array([0.0])
        active = np.array([True])
        f = retardation_factor(a, r_p, a_ol, active)
        assert f[0] == pytest.approx(0.0, abs=1e-15)

    def test_max_at_overload(self):
        a = np.array([0.005])
        r_p = np.array([0.005])
        a_ol = np.array([0.005])
        active = np.array([True])
        f = retardation_factor(a, r_p, a_ol, active)
        assert f[0] == pytest.approx(0.5, abs=1e-15)

    def test_inactive_gives_zero(self):
        a = np.array([0.001])
        r_p = np.array([0.005])
        a_ol = np.array([0.0])
        active = np.array([False])
        f = retardation_factor(a, r_p, a_ol, active)
        assert f[0] == 0.0

    def test_intermediate(self):
        a = np.array([0.008])
        r_p = np.array([0.003])
        a_ol = np.array([0.005])
        active = np.array([True])
        f = retardation_factor(a, r_p, a_ol, active)
        # zone boundary = 0.005 + 2*0.003 = 0.011; inside since 0.008 < 0.011
        # denom = 2*(0.008 - 0.005 + 0.003) = 0.012
        # f_r = 0.003 / 0.012 = 0.25
        assert f[0] == pytest.approx(0.25, abs=1e-15)

    def test_zone_boundary_uses_a_overload(self):
        a = np.array([0.005])
        r_p = np.array([0.002])
        a_ol = np.array([0.003])
        active = np.array([True])
        # zone = 0.003 + 2*0.002 = 0.007; crack at 0.005 < 0.007, inside
        f = retardation_factor(a, r_p, a_ol, active)
        assert f[0] > 0.0


class TestEffectiveKr:
    def test_no_retardation(self):
        dk = np.array([20.0])
        k_max = np.array([20.0])
        f_r = np.array([0.0])
        dk_eff, r_eff = effective_kr(dk, k_max, 0.0, f_r)
        assert dk_eff[0] == pytest.approx(20.0, rel=1e-12)
        assert r_eff[0] == pytest.approx(0.0, abs=1e-15)

    def test_full_retardation_zero_r(self):
        dk = np.array([20.0])
        k_max = np.array([20.0])
        f_r = np.array([1.0])
        dk_eff, r_eff = effective_kr(dk, k_max, 0.0, f_r)
        # K_max_eff = 20*(1-1) = 0, K_min_eff = max(0-20,0) = 0
        assert dk_eff[0] == pytest.approx(0.0, abs=1e-15)
        assert r_eff[0] == pytest.approx(0.0, abs=1e-15)

    def test_half_retardation_with_ratio(self):
        dk = np.array([20.0])
        k_max = np.array([25.0])  # R = 0.2, K_min = 5
        f_r = np.array([0.5])
        dk_eff, r_eff = effective_kr(dk, k_max, 0.2, f_r)
        # K_R = 0.5*25 = 12.5
        # K_max_eff = 25*(1-0.5) = 12.5
        # K_min_eff = max(5 - 12.5, 0) = 0
        # dk_eff = 12.5 - 0 = 12.5
        # r_eff = 0/12.5 = 0
        assert dk_eff[0] == pytest.approx(12.5, rel=1e-12)
        assert r_eff[0] == pytest.approx(0.0, abs=1e-15)

    def test_partial_retardation_preserves_ratio(self):
        dk = np.array([20.0])
        k_max = np.array([25.0])  # R = 0.2, K_min = 5
        f_r = np.array([0.1])
        dk_eff, r_eff = effective_kr(dk, k_max, 0.2, f_r)
        # K_R = 0.1*25 = 2.5
        # K_max_eff = 25*0.9 = 22.5
        # K_min_eff = max(5 - 2.5, 0) = 2.5
        # dk_eff = 22.5 - 2.5 = 20.0 (same range!)
        # r_eff = 2.5/22.5 = 0.111...
        assert dk_eff[0] == pytest.approx(20.0, rel=1e-12)
        assert r_eff[0] == pytest.approx(2.5 / 22.5, rel=1e-12)


# ---------------------------------------------------------------- integration tests


class TestConstantAmplitudeAgreement:
    def test_single_sample(self):
        a0 = np.array([1e-3])
        ca = grow(a0, DS_CA, GEO, LAW, K_IC)
        seq = SpectrumSequence.from_history_ordered([DS_CA, 0.0])
        ret = grow_spectrum_retarded(a0, seq, GEO, LAW, K_IC, S_YIELD)

        assert np.isfinite(ca.cycles_to_failure[0])
        assert np.isfinite(ret.cycles_to_failure[0])
        ratio = ret.cycles_to_failure[0] / ca.cycles_to_failure[0]
        assert 0.8 < ratio < 1.5, f"ratio={ratio:.3f}"

    def test_multiple_samples(self):
        a0 = np.array([1e-4, 5e-4, 1e-3])
        ca = grow(a0, DS_CA, GEO, LAW, K_IC)
        seq = SpectrumSequence.from_history_ordered([DS_CA, 0.0])
        ret = grow_spectrum_retarded(a0, seq, GEO, LAW, K_IC, S_YIELD)

        for i in range(len(a0)):
            if np.isfinite(ca.cycles_to_failure[i]) and ca.cycles_to_failure[i] > 0:
                if np.isfinite(ret.cycles_to_failure[i]):
                    ratio = ret.cycles_to_failure[i] / ca.cycles_to_failure[i]
                    assert 0.8 < ratio < 1.5, f"i={i}, ratio={ratio:.3f}"


class TestOverloadDelaysGrowth:
    def test_overload_increases_life(self):
        a0 = np.array([1e-3])
        ca_seq = SpectrumSequence.from_history_ordered([DS_CA, 0.0])
        ca = grow_spectrum_retarded(a0, ca_seq, GEO, LAW, K_IC, S_YIELD)
        ol_seq = SpectrumSequence.from_history_ordered(
            [1500.0, 0.0, DS_CA, 0.0])
        ol = grow_spectrum_retarded(a0, ol_seq, GEO, LAW, K_IC, S_YIELD)

        assert np.isfinite(ca.cycles_to_failure[0])
        assert np.isfinite(ol.cycles_to_failure[0])
        assert ol.cycles_to_failure[0] > ca.cycles_to_failure[0]


class TestOrderMatters:
    def test_ol_then_ca_vs_ca_then_ol(self):
        a0 = np.array([1e-3])
        s1 = SpectrumSequence.from_history_ordered(
            [1500.0, 0.0, DS_CA, 0.0])
        r1 = grow_spectrum_retarded(a0, s1, GEO, LAW, K_IC, S_YIELD)
        s2 = SpectrumSequence.from_history_ordered(
            [DS_CA, 0.0, 1500.0, 0.0])
        r2 = grow_spectrum_retarded(a0, s2, GEO, LAW, K_IC, S_YIELD)

        assert np.isfinite(r1.cycles_to_failure[0])
        assert np.isfinite(r2.cycles_to_failure[0])
        assert r1.cycles_to_failure[0] != r2.cycles_to_failure[0]


class TestSpectrumSequence:
    def test_from_history_order(self):
        h = [100.0, 0.0, 200.0, 0.0, 150.0, 0.0]
        seq = SpectrumSequence.from_history_ordered(h)
        assert seq.n_cycles > 0
        assert seq.peak_stress == pytest.approx(200.0)

    def test_from_history_drops_compressive(self):
        h = [-100.0, -200.0, -100.0, -150.0, -100.0]
        with pytest.raises(ValueError, match="no damaging"):
            SpectrumSequence.from_history_ordered(h)

    def test_total_count(self):
        h = [100.0, 0.0, 200.0, 0.0, 150.0, 0.0]
        seq = SpectrumSequence.from_history_ordered(h)
        assert seq.total_count > 0

    def test_ordered_preserves_sequence(self):
        h = [1500.0, 0.0, 1000.0, 0.0]
        seq = SpectrumSequence.from_history_ordered(h)
        assert seq.cycles[0].delta_sigma == 1500.0
        assert seq.cycles[1].delta_sigma == 1000.0


class TestVectorisedMonteCarlo:
    def test_different_a0_different_life(self):
        a0 = np.array([1e-4, 5e-4, 1e-3])
        seq = SpectrumSequence.from_history_ordered(
            [1500.0, 0.0, DS_CA, 0.0])
        result = grow_spectrum_retarded(a0, seq, GEO, LAW, K_IC, S_YIELD)
        assert result.cycles_to_failure[0] > result.cycles_to_failure[1]
        assert result.cycles_to_failure[1] > result.cycles_to_failure[2]

    def test_stress_scale(self):
        a0 = np.array([1e-3, 1e-3, 1e-3])
        scale = np.array([0.8, 1.0, 1.2])
        seq = SpectrumSequence.from_history_ordered(
            [1500.0, 0.0, DS_CA, 0.0])
        result = grow_spectrum_retarded(
            a0, seq, GEO, LAW, K_IC, S_YIELD, stress_scale=scale)
        assert result.cycles_to_failure[0] > result.cycles_to_failure[1]
        assert result.cycles_to_failure[1] > result.cycles_to_failure[2]


class TestEdgeCases:
    def test_crack_already_above_critical(self):
        a0 = np.array([1.0])
        seq = SpectrumSequence.from_history_ordered([DS_CA, 0.0])
        result = grow_spectrum_retarded(a0, seq, GEO, LAW, K_IC, S_YIELD)
        assert result.cycles_to_failure[0] == 0.0

    def test_max_cycles_runout(self):
        law = ParisLaw(1e-30, 3.0)
        a0 = np.array([1e-4])
        seq = SpectrumSequence.from_history_ordered([DS_CA, 0.0])
        result = grow_spectrum_retarded(
            a0, seq, GEO, law, K_IC, S_YIELD, max_cycles=100)
        assert result.cycles_to_failure[0] == np.inf

    def test_below_threshold_cycles_counted(self):
        law = ParisLaw(1e-11, 3.0, dk_threshold=50.0)
        a0 = np.array([1e-4])
        seq = SpectrumSequence.from_history_ordered([5.0, 0.0])
        result = grow_spectrum_retarded(
            a0, seq, GEO, law, K_IC, S_YIELD, max_cycles=500)
        assert result.cycles_to_failure[0] == np.inf

    def test_per_sample_paris_with_chunking(self):
        c = np.array([1e-11, 2e-11])
        law = ParisLaw(c, 3.0)
        a0 = np.array([1e-4, 1e-4])
        seq = SpectrumSequence.from_history_ordered([DS_CA, 0.0])
        result = grow_spectrum_retarded(
            a0, seq, GEO, law, K_IC, S_YIELD, chunk_size=1)
        assert np.isfinite(result.cycles_to_failure[0])
        assert np.isfinite(result.cycles_to_failure[1])
        assert result.cycles_to_failure[0] > result.cycles_to_failure[1]

    def test_no_repeated_overload_reactivation(self):
        a0 = np.array([1e-3])
        seq = SpectrumSequence.from_history_ordered(
            [1500.0, 0.0] + [DS_CA, 0.0] * 5)
        result = grow_spectrum_retarded(a0, seq, GEO, LAW, K_IC, S_YIELD)
        assert np.isfinite(result.cycles_to_failure[0])


class TestReferenceValidation:
    def test_independent_hand_calculation(self):
        """Verify retarded CA matches the analytical Paris integral.

        For Paris law with m=3 and R=0, the closed-form life is:
            N = 2 / ((m-2) * C * (Y*sigma*sqrt(pi))^(m) * (a0^(1-m/2) - ac^(1-m/2)))
        with m=3 simplifying to:
            N = 2 / (C * (Y*sigma*sqrt(pi))^3) * (1/sqrt(a0) - 1/sqrt(ac))
        """
        a0_val = 1e-3
        ac_val = (K_IC / 1000.0) ** 2 / np.pi
        c, m = 1e-11, 3.0
        y_sig_root_pi = 1000.0 * np.sqrt(np.pi)
        expected = (2.0 / (c * y_sig_root_pi**m)) * (
            1.0 / np.sqrt(a0_val) - 1.0 / np.sqrt(ac_val))

        a0 = np.array([a0_val])
        seq = SpectrumSequence.from_history_ordered([DS_CA, 0.0])
        result = grow_spectrum_retarded(a0, seq, GEO, LAW, K_IC, S_YIELD)
        ratio = result.cycles_to_failure[0] / expected
        assert 0.8 < ratio < 1.5, (
            f"Expected ~{expected:.0f} cycles, got {result.cycles_to_failure[0]:.0f}, "
            f"ratio={ratio:.3f}"
        )

    def test_single_overload_retardation_ratio(self):
        a0 = np.array([1e-3])
        ca_seq = SpectrumSequence.from_history_ordered([DS_CA, 0.0])
        ca = grow_spectrum_retarded(a0, ca_seq, GEO, LAW, K_IC, S_YIELD)
        ol_seq = SpectrumSequence.from_history_ordered(
            [1500.0, 0.0] + [DS_CA, 0.0] * 20)
        ol = grow_spectrum_retarded(a0, ol_seq, GEO, LAW, K_IC, S_YIELD)

        assert np.isfinite(ca.cycles_to_failure[0])
        assert np.isfinite(ol.cycles_to_failure[0])
        ratio = ol.cycles_to_failure[0] / ca.cycles_to_failure[0]
        assert ratio > 1.5, f"Expected retardation ratio > 1.5, got {ratio}"

    def test_constant_amplitude_matches_grow(self):
        a0 = np.array([1e-4, 5e-4, 1e-3])
        ca = grow(a0, DS_CA, GEO, LAW, K_IC)
        seq = SpectrumSequence.from_history_ordered([DS_CA, 0.0])
        ret = grow_spectrum_retarded(a0, seq, GEO, LAW, K_IC, S_YIELD)

        for i in range(len(a0)):
            if np.isfinite(ca.cycles_to_failure[i]) and ca.cycles_to_failure[i] > 0:
                if np.isfinite(ret.cycles_to_failure[i]):
                    ratio = ret.cycles_to_failure[i] / ca.cycles_to_failure[i]
                    assert 0.8 < ratio < 1.5, f"i={i}, ratio={ratio:.3f}"
