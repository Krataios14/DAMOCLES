"""Tests for the Willenborg overload-retardation model."""

import numpy as np
import pytest

from damocles.fracture import (
    ParisLaw, ThroughCrack, grow, grow_spectrum, grow_spectrum_retarded,
)
from damocles.spectrum import Spectrum, SpectrumSequence
from damocles.retardation import (
    WillenborgState, init_state, plastic_zone_radius,
    retardation_factor, effective_dk,
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


class TestRetardationFactor:
    def test_zero_when_crack_beyond_zone(self):
        a = np.array([0.01])
        r_p = np.array([0.001])
        active = np.array([True])
        f = retardation_factor(a, r_p, active)
        assert f[0] == pytest.approx(0.0, abs=1e-15)

    def test_max_at_overload(self):
        a = np.array([0.0])
        r_p = np.array([0.005])
        active = np.array([True])
        f = retardation_factor(a, r_p, active)
        assert f[0] == pytest.approx(0.5, abs=1e-15)

    def test_inactive_gives_zero(self):
        a = np.array([0.001])
        r_p = np.array([0.005])
        active = np.array([False])
        f = retardation_factor(a, r_p, active)
        assert f[0] == 0.0

    def test_intermediate(self):
        a = np.array([0.003])
        r_p = np.array([0.003])
        active = np.array([True])
        f = retardation_factor(a, r_p, active)
        assert f[0] == pytest.approx(0.25, abs=1e-15)


class TestEffectiveDk:
    def test_no_retardation(self):
        dk = np.array([20.0])
        f_r = np.array([0.0])
        assert effective_dk(dk, f_r)[0] == pytest.approx(20.0)

    def test_half_retardation(self):
        dk = np.array([20.0])
        f_r = np.array([0.5])
        assert effective_dk(dk, f_r)[0] == pytest.approx(10.0)


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
        # CA only
        ca_seq = SpectrumSequence.from_history_ordered([DS_CA, 0.0])
        ca = grow_spectrum_retarded(a0, ca_seq, GEO, LAW, K_IC, S_YIELD)
        # OL then CA
        ol_seq = SpectrumSequence.from_history_ordered(
            [1500.0, 0.0, DS_CA, 0.0])
        ol = grow_spectrum_retarded(a0, ol_seq, GEO, LAW, K_IC, S_YIELD)

        assert np.isfinite(ca.cycles_to_failure[0])
        assert np.isfinite(ol.cycles_to_failure[0])
        assert ol.cycles_to_failure[0] > ca.cycles_to_failure[0]


class TestOrderMatters:
    def test_ol_then_ca_vs_ca_then_ol(self):
        a0 = np.array([1e-3])
        # OL first
        s1 = SpectrumSequence.from_history_ordered(
            [1500.0, 0.0, DS_CA, 0.0])
        r1 = grow_spectrum_retarded(a0, s1, GEO, LAW, K_IC, S_YIELD)
        # CA first then OL
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


class TestReferenceValidation:
    def test_single_overload_retardation_ratio(self):
        a0 = np.array([1e-3])
        # CA only
        ca_seq = SpectrumSequence.from_history_ordered([DS_CA, 0.0])
        ca = grow_spectrum_retarded(a0, ca_seq, GEO, LAW, K_IC, S_YIELD)
        # OL then many CA cycles (retardation applies to many cycles)
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
