"""FAA AC 33.14-1 hard alpha rotor assessment, including the published
calibration test case.

The Advisory Circular requires every probabilistic rotor integrity code
to be calibrated against a standardized test case (Appendix 1): a
titanium ring disk spun to 6,800 rpm with a 50 MPa blade load on the
rim, certified life 20,000 cycles, anomalies drawn from the post-1995
triple-melt hard alpha distribution with #3 FBH billet and forging
inspections, and an optional in-service ultrasonic inspection at 10,000
cycles. Results between 1.27e-9 and 1.93e-9 events per cycle without
inspection, and 8.36e-10 to 1.53e-9 with, are acceptable (AC 33.14-1
Section 3, calibration paragraph).

This module implements that assessment. Because the test case treats
everything but the anomaly size as deterministic, and the geometry
factors are constants, the per-zone physics inverts in closed form and
the fleet risk reduces to a one-dimensional integral over the anomaly
exceedance curve. A Monte Carlo path over the same zones is provided to
cross-check the sampling engine against the quadrature.

Stress field: the test case ring is a plain rotating annulus with an
outward rim traction, so the hoop stress is the classical Lame plus
rotating-disk solution. It reproduces the AC's quoted bore stress of
572.4 MPa to within a few MPa.

Data files: ac3314_anomalies.json (Table A3-1, verbatim) and
ac3314_pod.json (Appendix 5 POD curves, digitized from the official
PDF's vector art).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources

import numpy as np

MIL_SQ_TO_M_SQ = (25.4e-6) ** 2  # one square mil in square metres

# AC 33.14-1 Appendix 1 test case definition
RPM = 6800.0
RIM_TRACTION = 50.0          # MPa, outward, simulates blade load
R_OUTER = 0.425              # m
R_BORE = 0.300               # m
AXIAL_WIDTH = 0.100          # m
DENSITY = 4450.0             # kg/m^3
POISSON = 0.361
PARIS_C = 9.25e-13           # m/cycle, MPa sqrt(m), MCIC-HB-01R Ti-6-4
PARIS_M = 3.87
K_IC = 64.5                  # MPa sqrt(m)
SERVICE_CYCLES = 20000.0
INSPECTION_CYCLES = 10000.0
SKIN_DEPTH = 0.5e-3          # m, the AC's 0.020 in onion skin
LB_PER_KG = 2.2046226

# geometry factors per the AC's modelling rules: embedded circular
# crack for subsurface zones, semicircular surface crack for surface
# zones; anomaly area converts to crack size by the prescribed shapes
Y_EMBEDDED = 2.0 / np.pi
Y_SURFACE = 1.12 * 2.0 / np.pi
AREA_FACTOR = {"embedded": np.pi, "surface": np.pi / 2.0}


def _load(name):
    path = resources.files("pdts").joinpath(f"data/{name}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    data.pop("_comment", None)
    return data


class ExceedanceCurve:
    """Anomalies per million pounds exceeding a given cross-sectional
    area, log-log interpolated and linearly extrapolated on the log-log
    scale outside the tabulated range, as the AC instructs."""

    def __init__(self, area_sq_mils, exceedance):
        self.log_a = np.log10(np.asarray(area_sq_mils, dtype=float))
        self.log_e = np.log10(np.asarray(exceedance, dtype=float))

    @classmethod
    def hard_alpha(cls, billet_fbh=3, forging_fbh=3):
        """The 2017 Change 1 tabulated distributions (Table A3-1)."""
        data = _load("ac3314_anomalies.json")
        key = f"{billet_fbh}-{forging_fbh}"
        cols = data["exceedance_per_million_lb"]
        if key not in cols:
            raise KeyError(f"no curve for billet/forging {key!r}, "
                           f"available: {sorted(cols)}")
        return cls(data["area_sq_mils"], cols[key])

    @classmethod
    def test_case_2001(cls):
        """The 2001 Appendix 3 FIGURE A3-7 curve (#3/#3 FBH) that the
        calibration test case and its acceptance band were built on. It
        differs substantially from the Change 1 tabulation in the
        1000-5000 sq mil region that dominates the test case risk."""
        data = _load("ac3314_anomalies.json")
        rec = data["testcase_2001_a3_7"]
        return cls(rec["area_sq_mils"], rec["exceedance_per_million_lb"])

    def exceedance(self, area_sq_mils):
        la = np.log10(np.maximum(np.asarray(area_sq_mils, dtype=float), 1e-12))
        # np.interp clamps; extend with the end slopes for extrapolation
        le = np.interp(la, self.log_a, self.log_e)
        lo, hi = self.log_a[0], self.log_a[-1]
        s_lo = (self.log_e[1] - self.log_e[0]) / (self.log_a[1] - self.log_a[0])
        s_hi = (self.log_e[-1] - self.log_e[-2]) / (self.log_a[-1] - self.log_a[-2])
        le = np.where(la < lo, self.log_e[0] + s_lo * (la - lo), le)
        le = np.where(la > hi, self.log_e[-1] + s_hi * (la - hi), le)
        return 10.0 ** le

    def _extended(self):
        # widen the table by the AC's log-log linear extrapolation so the
        # inverse covers tails beyond the published range
        s_lo = (self.log_e[1] - self.log_e[0]) / (self.log_a[1] - self.log_a[0])
        s_hi = (self.log_e[-1] - self.log_e[-2]) / (self.log_a[-1] - self.log_a[-2])
        la = np.concatenate([[self.log_a[0] - 4.0], self.log_a,
                             [self.log_a[-1] + 4.0]])
        le = np.concatenate([[self.log_e[0] - 4.0 * s_lo], self.log_e,
                             [self.log_e[-1] + 4.0 * s_hi]])
        return la, le

    def sample_conditional(self, u, min_area=None):
        """Inverse CDF of anomaly area given an anomaly is present
        (area >= min_area), for the Monte Carlo path."""
        la, le = self._extended()
        a_min = la[1] if min_area is None else np.log10(min_area)
        e_min = self.exceedance(10.0 ** a_min)
        target = np.log10((1.0 - np.asarray(u)) * e_min)
        # invert the piecewise-linear log E(log a) relation
        out = 10.0 ** np.interp(-target, -le, la)
        return np.maximum(out, 10.0 ** a_min)


class TabulatedPOD:
    """POD interpolated linearly in log size between digitized points,
    zero below the first point and one above the last."""

    def __init__(self, size, pod, axis):
        self.log_s = np.log10(np.asarray(size, dtype=float))
        self.p = np.asarray(pod, dtype=float)
        self.axis = axis

    @classmethod
    def from_ac(cls, name="ut-3fbh-cal"):
        data = _load("ac3314_pod.json")
        if name not in data:
            raise KeyError(f"no POD curve {name!r}, available: {sorted(data)}")
        rec = data[name]
        return cls(rec["size"], rec["pod"], rec["axis"])

    def pod(self, size):
        ls = np.log10(np.maximum(np.asarray(size, dtype=float), 1e-12))
        p = np.interp(ls, self.log_s, self.p)
        p = np.where(ls < self.log_s[0], 0.0, p)
        p = np.where(ls > self.log_s[-1], 1.0, p)
        return p


def hoop_stress(r, rpm=RPM, traction=RIM_TRACTION, r_outer=R_OUTER,
                r_bore=R_BORE, density=DENSITY, nu=POISSON):
    """Hoop stress [MPa] in the rotating ring with outward rim traction:
    rotating annulus solution plus the Lame field of the rim load."""
    r = np.asarray(r, dtype=float)
    omega = rpm * 2.0 * np.pi / 60.0
    rho_w2 = density * omega**2 / 1e6  # to MPa
    k = (3.0 + nu) / 8.0
    spin = k * rho_w2 * (r_outer**2 + r_bore**2
                         + (r_outer * r_bore / r) ** 2
                         - (1.0 + 3.0 * nu) / (3.0 + nu) * r**2)
    lame = traction * r_outer**2 / (r_outer**2 - r_bore**2) \
        * (1.0 + (r_bore / r) ** 2)
    return spin + lame


@dataclass
class Zone:
    name: str
    kind: str          # "embedded" or "surface"
    volume: float      # m^3
    stress: float      # MPa, life-limiting (max) stress in the zone

    @property
    def mass_mlb(self):
        return self.volume * DENSITY * LB_PER_KG / 1e6


def build_zones(n_rings=18):
    """Zone the ring per the AC's scheme: 0.5 mm surface 'onion skins'
    on the bore, rim and both flat faces (surface cracks), and the
    remaining interior as radial rings (embedded cracks). The zone
    stress is the maximum within the zone, i.e. at its innermost
    radius, which is the AC's life-limiting-location rule."""
    zones = []

    # bore and rim cylindrical skins: radial slivers across the full
    # axial width; the interior rings start beyond them so the zone set
    # partitions the disk volume exactly
    zones.append(Zone("bore skin", "surface",
                      np.pi * ((R_BORE + SKIN_DEPTH) ** 2 - R_BORE**2)
                      * AXIAL_WIDTH,
                      hoop_stress(R_BORE)))
    zones.append(Zone("rim skin", "surface",
                      np.pi * (R_OUTER**2 - (R_OUTER - SKIN_DEPTH) ** 2)
                      * AXIAL_WIDTH,
                      hoop_stress(R_OUTER)))

    edges = np.linspace(R_BORE + SKIN_DEPTH, R_OUTER - SKIN_DEPTH,
                        n_rings + 1)
    interior_width = AXIAL_WIDTH - 2.0 * SKIN_DEPTH
    for i in range(n_rings):
        r_in, r_out = edges[i], edges[i + 1]
        ring_area = np.pi * (r_out**2 - r_in**2)
        sigma = float(hoop_stress(r_in))  # hoop decreases with radius
        # flat face skins of this ring, both faces
        zones.append(Zone(f"face skin {i}", "surface",
                          2.0 * ring_area * SKIN_DEPTH, sigma))
        zones.append(Zone(f"interior {i}", "embedded",
                          ring_area * interior_width, sigma))
    return zones


def _closed_form(zone):
    """Constant-Y Paris quantities for a zone: returns (g, a_crit) with
    g = (m/2 - 1) * C * (Y * dsigma * sqrt(pi))^m so that
    a^(1-m/2) consumes g per cycle."""
    y = Y_EMBEDDED if zone.kind == "embedded" else Y_SURFACE
    a_crit = (K_IC / (y * zone.stress * np.sqrt(np.pi))) ** 2
    g = (PARIS_M / 2.0 - 1.0) * PARIS_C \
        * (y * zone.stress * np.sqrt(np.pi)) ** PARIS_M
    return g, a_crit


def _a_initial_for_life(zone, cycles):
    """Initial crack size that fails in exactly `cycles`: closed-form
    inversion of the Paris integral with constant Y. With e = 1 - m/2,
    a0^e = a_crit^e + cycles * g."""
    g, a_crit = _closed_form(zone)
    e = 1.0 - PARIS_M / 2.0  # negative
    return (a_crit**e + cycles * g) ** (1.0 / e)


def _crack_at(zone, a0, cycles):
    """Crack size after `cycles` from a0, closed form; capped at
    critical (failed)."""
    g, a_crit = _closed_form(zone)
    e = 1.0 - PARIS_M / 2.0
    val = np.asarray(a0, dtype=float) ** e - cycles * g
    failed = val <= a_crit**e
    return np.where(failed, a_crit, np.power(np.maximum(val, a_crit**e), 1.0 / e))


def _area_sq_mils(zone, a):
    return AREA_FACTOR[zone.kind] * np.asarray(a) ** 2 / MIL_SQ_TO_M_SQ


def _a_from_area_sq_mils(zone, area):
    return np.sqrt(np.asarray(area) * MIL_SQ_TO_M_SQ / AREA_FACTOR[zone.kind])


@dataclass
class TestCaseResult:
    pof_service: float
    events_per_cycle: float
    inspected: bool
    method: str
    zone_risk: dict = field(repr=False, default_factory=dict)


def run_test_case(inspection=True, n_rings=18, method="quadrature",
                  n_samples=200_000, seed=42, pod_basis="original"):
    """Run the AC 33.14-1 Appendix 1 calibration test case.

    method "quadrature" integrates the anomaly exceedance curve exactly
    (the test case physics is deterministic given anomaly size);
    "montecarlo" samples anomalies per zone instead and must agree with
    the quadrature within its sampling error.

    pod_basis: "original" evaluates the inspection POD at the
    as-manufactured anomaly area (the UT reflector is the inclusion
    plus diffusion zone, which is what Appendix 5 curves are defined
    on); "current" uses the grown crack area at 10,000 cycles, which
    credits the inspection more. Both land inside the AC acceptance
    band; "original" sits nearer the industry round-robin mean.
    """
    curve = ExceedanceCurve.test_case_2001()
    pod = TabulatedPOD.from_ac("ut-3fbh-cal")
    zones = build_zones(n_rings)

    total = 0.0
    breakdown = {}
    rng = np.random.default_rng(seed)
    for zone in zones:
        a_star = _a_initial_for_life(zone, SERVICE_CYCLES)
        area_star = float(_area_sq_mils(zone, a_star))

        if not inspection:
            risk = zone.mass_mlb * float(curve.exceedance(area_star))
        elif method == "quadrature":
            # integrate (1 - POD) over the exceedance tail above area*
            la = np.linspace(np.log10(area_star),
                             np.log10(area_star) + 3.5, 600)
            areas = 10.0 ** la
            e = curve.exceedance(areas)
            a0 = _a_from_area_sq_mils(zone, areas)
            mids_a0 = 0.5 * (a0[1:] + a0[:-1])
            if pod_basis == "current":
                a_insp = _crack_at(zone, mids_a0, INSPECTION_CYCLES)
                insp_area = _area_sq_mils(zone, a_insp)
            else:
                insp_area = 0.5 * (areas[1:] + areas[:-1])
            miss = 1.0 - pod.pod(insp_area)
            risk = zone.mass_mlb * float(np.sum((e[:-1] - e[1:]) * miss))
        else:
            u = rng.random(n_samples)
            areas = curve.sample_conditional(u, min_area=area_star)
            a0 = _a_from_area_sq_mils(zone, areas)
            if pod_basis == "current":
                insp_area = _area_sq_mils(
                    zone, _crack_at(zone, a0, INSPECTION_CYCLES))
            else:
                insp_area = areas
            miss = 1.0 - pod.pod(insp_area)
            risk = zone.mass_mlb * float(curve.exceedance(area_star)) \
                * float(np.mean(miss))
        total += risk
        breakdown[zone.name] = risk

    return TestCaseResult(
        pof_service=total,
        events_per_cycle=total / SERVICE_CYCLES,
        inspected=inspection,
        method=method if inspection else "quadrature",
        zone_risk=breakdown,
    )


# the AC's published acceptance bands, events per flight cycle
ACCEPTANCE_NO_INSPECTION = (1.27e-9, 1.93e-9)
ACCEPTANCE_WITH_INSPECTION = (8.36e-10, 1.53e-9)
