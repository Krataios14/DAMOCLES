import numpy as np
import pytest

from damocles.fracture import ParisLaw, ThroughCrack, WalkerLaw, grow
from damocles.materials import KSI, KSI_SQRT_IN, available, get, growth_law
from damocles.nasgro import NasgroLaw
from damocles.study import build_study


def test_every_entry_has_a_source():
    for name in available():
        rec = get(name)
        assert "source" in rec and len(rec["source"]) > 20, name


def test_si_entry_passes_through():
    rec = get("ti-6al-4v-ac33.14")
    assert rec["paris"]["c"] == pytest.approx(9.25e-13)
    assert rec["paris"]["m"] == pytest.approx(3.87)
    assert rec["toughness"]["kic"] == pytest.approx(64.5)
    assert rec["strength"]["ys"] == pytest.approx(834.0)


def test_us_entry_converts_to_si():
    rec = get("2024-t3-sheet-lt")
    # dk1: 1.22 ksi sqrt(in) -> MPa sqrt(m)
    assert rec["nasgro"]["dk1"] == pytest.approx(1.22 * KSI_SQRT_IN)
    # C converts so that the same physical rate comes out:
    # da/dN = C_us * dK_us^n [in] = C_si * dK_si^n [m]
    c_si = rec["nasgro"]["c"]
    n = rec["nasgro"]["n"]
    dk_si = 10.0
    rate_si = c_si * dk_si**n
    rate_us_in = 0.800e-8 * (dk_si / KSI_SQRT_IN) ** n
    assert rate_si == pytest.approx(rate_us_in * 0.0254, rel=1e-9)
    assert rec["strength"]["ys"] == pytest.approx(53.0 * KSI)


def test_growth_law_builders():
    assert isinstance(growth_law("2024-t3-sheet-lt"), NasgroLaw)
    assert isinstance(growth_law("2024-t3-sheet-lt", "walker"), WalkerLaw)
    assert isinstance(growth_law("ti-6al-4v-ac33.14"), ParisLaw)
    with pytest.raises(KeyError, match="no 'nasgro'"):
        growth_law("ti-6al-4v-ac33.14", "nasgro")
    with pytest.raises(KeyError, match="unknown material"):
        growth_law("unobtainium")


def test_database_law_grows_in_si():
    law = growth_law("2024-t3-sheet-lt")
    kc = get("2024-t3-sheet-lt")["toughness"]["kc"]
    res = grow(np.array([1e-3]), 80.0, ThroughCrack(), law, kc)
    life = res.cycles_to_failure[0]
    assert 1e3 < life < 1e7  # a sane order of magnitude for sheet at 80 MPa


def test_material_reference_in_study_spec():
    spec = {
        "name": "db material study",
        "variables": {
            "initial_flaw": {"dist": "lognormal", "mean": 1.0e-3, "cov": 0.4},
            "stress_range": {"dist": "normal", "mean": 80.0, "std": 4.0},
            "toughness": {"dist": "normal", "mean": 81.6, "std": 4.0},
        },
        "geometry": {"type": "through"},
        "growth": {"material": "2024-t3-sheet-lt"},
        "service_cycles": 100000,
        "analysis": {"samples": 2000, "seed": 5},
    }
    res = build_study(spec).run()
    assert 0.0 <= res.pof <= 1.0
