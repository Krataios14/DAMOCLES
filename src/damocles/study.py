"""End-to-end damage tolerance study.

Reserved variable names:
  initial_flaw  initial crack size [m]
  stress_range  constant amplitude stress range per cycle [MPa]
  toughness     fracture toughness K_Ic [MPa sqrt(m)]
  paris_c       optional, growth law coefficient as a random variable

Any of them may be Deterministic, so the same study definition covers
everything from a quick deterministic check to a full probabilistic run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from .fracture import GEOMETRIES, GROWTH_LAWS, grow, grow_spectrum
from .spectrum import CycleClass, Spectrum
from .inspection import InspectionOutcome, InspectionPlan, PODCurve, apply_plan
from .random_vars import from_spec
from .reliability import _clopper_pearson
from .sampling import map_to_physical, sample_unit
from .sensitivity import rank_drivers, sobol_indices

REQUIRED = ("initial_flaw", "stress_range", "toughness")
# under spectrum loading the per-cycle stress is set by the spectrum and
# load scatter enters as a multiplier instead
SPECTRUM_REQUIRED = ("initial_flaw", "stress_scale", "toughness")


@dataclass
class StudyResult:
    name: str
    n_samples: int
    method: str
    service_cycles: float
    pof: float                       # unmitigated, at end of service
    ci_low: float
    ci_high: float
    pof_curve_cycles: np.ndarray
    pof_curve: np.ndarray
    inspection: InspectionOutcome | None
    inspection_plan: InspectionPlan | None
    sensitivity: dict | None
    lives: np.ndarray = field(repr=False, default=None)
    target_pof: float | None = None

    def reliability_index(self):
        if self.pof <= 0.0:
            return np.inf
        return float(-stats.norm.ppf(self.pof))

    def summary(self):
        lines = []
        bar = "=" * 64
        lines.append(bar)
        lines.append(f"  {self.name}")
        lines.append(bar)
        lines.append(f"  samples            : {self.n_samples:,} ({self.method})")
        lines.append(f"  service life       : {self.service_cycles:,.0f} cycles")
        lines.append("")
        lines.append(f"  P(failure), no inspection : {self.pof:.3e}")
        lines.append(f"    95% CI                  : "
                     f"[{self.ci_low:.3e}, {self.ci_high:.3e}]")
        lines.append(f"    reliability index beta  : {self.reliability_index():.2f}")
        per_cycle = self.pof / self.service_cycles if self.service_cycles else 0.0
        lines.append(f"    mean hazard per cycle   : {per_cycle:.3e}")
        if self.inspection is not None:
            insp = self.inspection
            lines.append("")
            times = ", ".join(f"{t:,.0f}" for t in insp.times)
            lines.append(f"  inspections at            : {times}")
            lines.append(f"  P(failure), inspected     : {insp.pof_inspected:.3e}")
            lines.append(f"    risk reduction          : {insp.risk_reduction:.1%}")
            lines.append(f"    expected detections/part: {insp.mean_detections:.3e}")
        if self.target_pof is not None:
            achieved = (self.inspection.pof_inspected
                        if self.inspection is not None else self.pof)
            verdict = "MEETS" if achieved <= self.target_pof else "EXCEEDS"
            lines.append("")
            lines.append(f"  target P(failure)         : {self.target_pof:.1e}  "
                         f"-> {verdict} target")
        if self.sensitivity:
            lines.append("")
            lines.append("  variance drivers (total Sobol index on log-life):")
            for name in rank_drivers(self.sensitivity):
                s = self.sensitivity[name]
                lines.append(f"    {name:<14} total={s['total']:.3f}  "
                             f"first={s['first']:.3f}")
        lines.append(bar)
        return "\n".join(lines)


class DamageToleranceStudy:
    def __init__(self, name, variables, geometry, growth_law, service_cycles,
                 stress_ratio=0.0, inspection_plan=None, n_samples=200_000,
                 method="lhs", seed=None, target_pof=None, spectrum=None):
        required = SPECTRUM_REQUIRED if spectrum is not None else REQUIRED
        missing = [k for k in required if k not in variables]
        if missing:
            raise ValueError(f"study needs variables {list(required)}, "
                             f"missing {missing}")
        self.name = name
        self.variables = variables
        self.geometry = geometry
        self.growth_law = growth_law
        self.service_cycles = float(service_cycles)
        self.stress_ratio = stress_ratio
        self.inspection_plan = inspection_plan
        self.n_samples = n_samples
        self.method = method
        self.seed = seed
        self.target_pof = target_pof
        self.spectrum = spectrum

    def _grow(self, x, eval_cycles=None):
        law = self.growth_law
        if "paris_c" in x:
            law = type(law)(**{**law.__dict__, "c": x["paris_c"]})
        if self.spectrum is not None:
            return grow_spectrum(x["initial_flaw"], self.spectrum,
                                 self.geometry, law, x["toughness"],
                                 stress_scale=x["stress_scale"],
                                 eval_blocks=eval_cycles)
        return grow(x["initial_flaw"], x["stress_range"], self.geometry, law,
                    x["toughness"], stress_ratio=self.stress_ratio,
                    eval_cycles=eval_cycles)

    def run(self, sensitivity=False, curve_points=60):
        u = sample_unit(self.n_samples, len(self.variables),
                        method=self.method, seed=self.seed)
        x = map_to_physical(u, self.variables)
        n = u.shape[0]

        eval_cycles = self.inspection_plan.times if self.inspection_plan else None
        life = self._grow(x, eval_cycles=eval_cycles)

        k = int(np.sum(life.cycles_to_failure <= self.service_cycles))
        pof = k / n
        ci_low, ci_high = _clopper_pearson(k, n)

        curve_cycles = np.linspace(0.0, self.service_cycles, curve_points + 1)[1:]
        curve = np.array([life.pof_at(c) for c in curve_cycles])

        outcome = None
        if self.inspection_plan is not None:
            outcome = apply_plan(life, self.service_cycles, self.inspection_plan)

        sens = None
        if sensitivity:
            sens = sobol_indices(self._log_life, self.variables,
                                 n=2**11, seed=self.seed)

        return StudyResult(
            name=self.name, n_samples=n, method=self.method,
            service_cycles=self.service_cycles,
            pof=pof, ci_low=ci_low, ci_high=ci_high,
            pof_curve_cycles=curve_cycles, pof_curve=curve,
            inspection=outcome, inspection_plan=self.inspection_plan,
            sensitivity=sens, lives=life.cycles_to_failure,
            target_pof=self.target_pof,
        )

    def _log_life(self, x):
        life = self._grow(x)
        # cap so the Sobol estimator is not dominated by run-outs
        capped = np.minimum(life.cycles_to_failure, 100.0 * self.service_cycles)
        return np.log10(np.maximum(capped, 1.0))


def build_study(spec):
    """Build a study from a parsed YAML/JSON dict. See examples/ for the
    schema."""
    variables = {name: from_spec(s) for name, s in spec["variables"].items()}

    geo = dict(spec["geometry"])
    geo_type = geo.pop("type")
    if geo_type not in GEOMETRIES:
        raise ValueError(f"unknown geometry {geo_type!r}, "
                         f"expected one of {sorted(GEOMETRIES)}")
    geometry = GEOMETRIES[geo_type](
        **{k: (v if isinstance(v, str) else float(v)) for k, v in geo.items()})

    gr = dict(spec["growth"])
    if "material" in gr:
        from .materials import growth_law as material_growth_law
        law = material_growth_law(gr.pop("material"), kind=gr.pop("law", None))
        if gr:
            raise ValueError(f"unexpected growth keys with 'material': {sorted(gr)}")
    else:
        law_name = gr.pop("law")
        if law_name not in GROWTH_LAWS:
            raise ValueError(f"unknown growth law {law_name!r}, "
                             f"expected one of {sorted(GROWTH_LAWS)}")
        if "paris_c" not in variables:
            if "c" not in gr:
                raise ValueError("growth law needs 'c', or supply a 'paris_c' variable")
        else:
            gr.pop("c", None)
        law = GROWTH_LAWS[law_name](c=float(gr.pop("c", 1.0)),
                                    **{k: float(v) for k, v in gr.items()})

    analysis = spec.get("analysis", {})
    service = float(spec["service_cycles"])

    plan = None
    if "inspection" in spec:
        insp = spec["inspection"]
        pod = PODCurve.from_a50_a90(float(insp["pod_a50"]), float(insp["pod_a90"]))
        if "times" in insp:
            plan = InspectionPlan([float(t) for t in insp["times"]], pod)
        else:
            plan = InspectionPlan.at_interval(float(insp["interval"]), service, pod)

    target = analysis.get("target_pof")
    return DamageToleranceStudy(
        name=spec.get("name", "unnamed study"),
        variables=variables,
        geometry=geometry,
        growth_law=law,
        service_cycles=service,
        stress_ratio=float(spec.get("stress_ratio", 0.0)),
        inspection_plan=plan,
        n_samples=int(analysis.get("samples", 200_000)),
        method=analysis.get("method", "lhs"),
        seed=analysis.get("seed"),
        target_pof=None if target is None else float(target),
    )
