"""Rainflow counting validated against the worked example in
ASTM E1049-85 section 5.4.4, and spectrum growth validated against
constant amplitude limits."""

import numpy as np
import pytest

from damocles.fracture import ParisLaw, ThroughCrack, grow, grow_spectrum
from damocles.spectrum import CycleClass, Spectrum, _peaks_valleys, rainflow

# the standard's example history, peaks and valleys labelled A through I
ASTM_HISTORY = [-2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0]

# (range, count) pairs from the standard's table for that history
ASTM_COUNTS = {
    (3.0, 0.5): 1,   # A-B
    (4.0, 0.5): 1,   # B-C
    (8.0, 0.5): 2,   # C-D and G-H
    (9.0, 0.5): 1,   # D-G
    (4.0, 1.0): 1,   # E-F
    (6.0, 0.5): 1,   # H-I
}


def test_rainflow_astm_e1049_worked_example():
    counted = rainflow(ASTM_HISTORY)
    seen = {}
    for rng, _mean, cnt in counted:
        seen[(rng, cnt)] = seen.get((rng, cnt), 0) + 1
    assert seen == ASTM_COUNTS


def test_rainflow_total_damage_content():
    # every reversal of the history must be accounted for: the total
    # cycle count of an n-reversal sequence is (n-1)/2
    counted = rainflow(ASTM_HISTORY)
    assert sum(c for _, _, c in counted) == pytest.approx(4.0)


def test_peaks_valleys_strips_monotone_points():
    out = _peaks_valleys([0.0, 1.0, 2.0, 3.0, 1.0, 1.0, 4.0, 0.0])
    assert np.allclose(out, [0.0, 3.0, 1.0, 4.0, 0.0])


def test_single_class_spectrum_equals_constant_amplitude():
    a0 = np.array([1e-3, 2e-3])
    law = ParisLaw(1e-11, 3.0)
    ca = grow(a0, 100.0, ThroughCrack(), law, 60.0)
    sp = grow_spectrum(a0, Spectrum.constant_amplitude(100.0), ThroughCrack(),
                       law, 60.0)
    assert np.allclose(ca.cycles_to_failure, sp.cycles_to_failure)
    assert np.allclose(ca.a_critical, sp.a_critical)


def test_count_scaling_halves_block_life():
    a0 = np.array([1e-3])
    law = ParisLaw(1e-11, 3.0)
    one = grow_spectrum(a0, Spectrum.constant_amplitude(100.0, count=1.0),
                        ThroughCrack(), law, 60.0)
    two = grow_spectrum(a0, Spectrum.constant_amplitude(100.0, count=2.0),
                        ThroughCrack(), law, 60.0)
    assert two.cycles_to_failure[0] == pytest.approx(
        one.cycles_to_failure[0] / 2.0, rel=1e-9)


def test_mixed_spectrum_bounded_by_pure_components():
    a0 = np.array([1e-3])
    law = ParisLaw(1e-11, 3.0)
    lo = grow_spectrum(a0, Spectrum.constant_amplitude(80.0), ThroughCrack(),
                       law, 60.0).cycles_to_failure[0]
    hi = grow_spectrum(a0, Spectrum.constant_amplitude(120.0), ThroughCrack(),
                       law, 60.0).cycles_to_failure[0]
    mixed = Spectrum([CycleClass(80.0, 0.0, 0.5), CycleClass(120.0, 0.0, 0.5)])
    mid = grow_spectrum(a0, mixed, ThroughCrack(), law, 60.0).cycles_to_failure[0]
    assert hi < mid < lo


def test_stress_scale_per_sample():
    a0 = np.full(3, 1e-3)
    law = ParisLaw(1e-11, 3.0)
    res = grow_spectrum(a0, Spectrum.constant_amplitude(100.0), ThroughCrack(),
                        law, 60.0, stress_scale=np.array([0.8, 1.0, 1.25]))
    assert np.all(np.diff(res.cycles_to_failure) < 0)


def test_from_history_tension_only_conversion():
    sp = Spectrum.from_history(ASTM_HISTORY)
    # all classes must carry positive range and a ratio in [0, 1)
    for c in sp.classes:
        assert c.delta_sigma > 0.0
        assert 0.0 <= c.stress_ratio < 1.0
    # counts conserved (no compressive-peak classes in this history)
    assert sp.dropped_compressive == 0.0
    assert sp.cycles_per_block == pytest.approx(4.0)


def test_compressive_history_raises_when_nothing_tensile():
    with pytest.raises(ValueError, match="no damaging"):
        Spectrum.from_history([-10.0, -2.0, -10.0, -1.0, -10.0, -2.0, -10.0])


def test_binned_conserves_counts():
    rng = np.random.default_rng(1)
    history = rng.normal(100.0, 40.0, 4001)
    sp = Spectrum.from_history(history)
    bn = sp.binned(12)
    assert len(bn.classes) <= 12
    assert bn.cycles_per_block == pytest.approx(sp.cycles_per_block)
    # binning should barely move the growth answer
    a0 = np.array([2e-3])
    law = ParisLaw(1e-11, 3.0)
    full = grow_spectrum(a0, sp, ThroughCrack(), law, 60.0)
    binned = grow_spectrum(a0, bn, ThroughCrack(), law, 60.0)
    assert binned.cycles_to_failure[0] == pytest.approx(
        full.cycles_to_failure[0], rel=0.05)
