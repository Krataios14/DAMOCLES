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
from numbers import Real

import numpy as np
from scipy import stats

from .fracture import (
    GEOMETRIES,
    GROWTH_LAWS,
    grow,
    grow_spectrum,
    grow_spectrum_retarded,
)
from .spectrum import SpectrumSequence
from .inspection import InspectionOutcome, InspectionPlan, PODCurve, apply_plan
from .random_vars import from_spec
from .reliability import _clopper_pearson
from .retardation import OrderedBoundaryIndex, WillenborgConfig
from .sampling import METHODS, map_to_physical, sample_unit
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
    pof: float  # unmitigated, at end of service
    ci_low: float
    ci_high: float
    pof_curve_cycles: np.ndarray
    pof_curve: np.ndarray
    inspection: InspectionOutcome | None
    inspection_plan: InspectionPlan | None
    sensitivity: dict | None
    lives: np.ndarray = field(repr=False, default=None)
    target_pof: float | None = None
    loading: str = "constant-amplitude"
    life_unit: str = "cycles"
    retardation: str | None = None
    prospective_work: int | None = None

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
        lines.append(
            f"  service life       : {self.service_cycles:,.0f} {self.life_unit}"
        )
        lines.append(f"  loading            : {self.loading}")
        if self.retardation is not None:
            lines.append(f"  load interaction   : {self.retardation}")
        lines.append("")
        lines.append(f"  P(failure), no inspection : {self.pof:.3e}")
        lines.append(
            f"    95% CI                  : [{self.ci_low:.3e}, {self.ci_high:.3e}]"
        )
        lines.append(f"    reliability index beta  : {self.reliability_index():.2f}")
        per_unit = self.pof / self.service_cycles if self.service_cycles else 0.0
        unit = self.life_unit[:-1] if self.life_unit.endswith("s") else self.life_unit
        lines.append(f"    mean hazard per {unit:<8}: {per_unit:.3e}")
        if self.inspection is not None:
            insp = self.inspection
            lines.append("")
            times = ", ".join(f"{t:,.0f}" for t in insp.times)
            lines.append(f"  inspections at            : {times}")
            lines.append(f"  P(failure), inspected     : {insp.pof_inspected:.3e}")
            lines.append(f"    risk reduction          : {insp.risk_reduction:.1%}")
            lines.append(f"    expected detections/part: {insp.mean_detections:.3e}")
        if self.target_pof is not None:
            achieved = (
                self.inspection.pof_inspected
                if self.inspection is not None
                else self.pof
            )
            verdict = "MEETS" if achieved <= self.target_pof else "EXCEEDS"
            lines.append("")
            lines.append(
                f"  target P(failure)         : {self.target_pof:.1e}  "
                f"-> {verdict} target"
            )
        if self.sensitivity:
            lines.append("")
            lines.append("  variance drivers (total Sobol index on log-life):")
            for name in rank_drivers(self.sensitivity):
                s = self.sensitivity[name]
                lines.append(
                    f"    {name:<14} total={s['total']:.3f}  first={s['first']:.3f}"
                )
        lines.append(bar)
        return "\n".join(lines)


class DamageToleranceStudy:
    def __init__(
        self,
        name,
        variables,
        geometry,
        growth_law,
        service_cycles,
        stress_ratio=0.0,
        inspection_plan=None,
        n_samples=200_000,
        method="lhs",
        seed=None,
        target_pof=None,
        spectrum=None,
        retardation=None,
    ):
        required = SPECTRUM_REQUIRED if spectrum is not None else REQUIRED
        missing = [k for k in required if k not in variables]
        if missing:
            raise ValueError(
                f"study needs variables {list(required)}, missing {missing}"
            )
        self.name = name
        self.variables = variables
        self.geometry = geometry
        self.growth_law = growth_law
        self.service_cycles = float(service_cycles)
        if not np.isfinite(self.service_cycles) or self.service_cycles <= 0.0:
            raise ValueError("service_cycles must be finite and positive")
        self.stress_ratio = stress_ratio
        self.inspection_plan = inspection_plan
        if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
            raise TypeError("n_samples must be an integer")
        if n_samples < 1:
            raise ValueError("n_samples must be at least 1")
        self.n_samples = n_samples
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}, expected one of {METHODS}")
        self.method = method
        self.seed = seed
        self.target_pof = target_pof
        self.spectrum = spectrum
        if retardation is not None and not isinstance(retardation, WillenborgConfig):
            raise TypeError("retardation must be a WillenborgConfig or None")
        if isinstance(spectrum, SpectrumSequence):
            self._mission_boundaries = OrderedBoundaryIndex.from_sequence(spectrum)
            self.retardation = (
                WillenborgConfig.disabled() if retardation is None else retardation
            )
            self._validate_mission_inspections()
            self._resolve_mission_limits(self._sample_count())
        else:
            if retardation is not None:
                raise ValueError("retardation requires an ordered SpectrumSequence")
            self._mission_boundaries = None
            self.retardation = None

    def _sample_count(self):
        if self.method == "sobol":
            return 1 << int(np.ceil(np.log2(self.n_samples)))
        return self.n_samples

    def _validate_mission_inspections(self):
        if self.inspection_plan is None:
            return
        try:
            times = np.asarray(self.inspection_plan.times, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError("inspection times must be numeric") from exc
        if times.ndim != 1 or times.size > 10_000:
            raise ValueError(
                "ordered mission inspections must be a list of at most "
                "10,000 checkpoints"
            )
        if (
            np.any(~np.isfinite(times))
            or np.any(times <= 0.0)
            or np.any(times >= self.service_cycles)
        ):
            raise ValueError(
                "inspection times must be finite, positive, and before service life"
            )
        if np.any(np.diff(times) <= 0.0):
            raise ValueError(
                "ordered mission inspection times must be strictly increasing"
            )
        self._mission_boundaries.steps_many(times, "inspection checkpoint")

    def _resolve_mission_limits(self, n_samples):
        return self.retardation.resolve(
            self.spectrum,
            self.service_cycles,
            n_samples,
            self._mission_boundaries,
        )

    def _grow(self, x, eval_cycles=None):
        law = self.growth_law
        if "paris_c" in x:
            law = type(law)(**{**law.__dict__, "c": x["paris_c"]})
        if self.spectrum is not None:
            if isinstance(self.spectrum, SpectrumSequence):
                max_cycles, max_blocks, _ = self._resolve_mission_limits(
                    len(x["initial_flaw"])
                )
                return grow_spectrum_retarded(
                    x["initial_flaw"],
                    self.spectrum,
                    self.geometry,
                    law,
                    x["toughness"],
                    s_yield=self.retardation.yield_strength,
                    stress_scale=x["stress_scale"],
                    max_cycles=max_cycles,
                    max_blocks=max_blocks,
                    eval_cycles=eval_cycles,
                    apply_retardation=self.retardation.enabled,
                    max_work=self.retardation.max_work,
                    _boundary_index=self._mission_boundaries,
                )
            return grow_spectrum(
                x["initial_flaw"],
                self.spectrum,
                self.geometry,
                law,
                x["toughness"],
                stress_scale=x["stress_scale"],
                eval_blocks=eval_cycles,
            )
        return grow(
            x["initial_flaw"],
            x["stress_range"],
            self.geometry,
            law,
            x["toughness"],
            stress_ratio=self.stress_ratio,
            eval_cycles=eval_cycles,
        )

    def run(self, sensitivity=False, curve_points=60):
        if isinstance(sensitivity, np.bool_):
            sensitivity = bool(sensitivity)
        if type(sensitivity) is not bool:
            raise TypeError("sensitivity must be a boolean")
        if isinstance(self.spectrum, SpectrumSequence) and sensitivity:
            raise ValueError(
                "Sobol sensitivity is not available for ordered mission "
                "studies; run bounded parameter sweeps explicitly"
            )
        if isinstance(curve_points, bool) or not isinstance(
            curve_points, (int, np.integer)
        ):
            raise TypeError("curve_points must be an integer")
        if curve_points < 1 or curve_points > 10_000:
            raise ValueError("curve_points must be in [1, 10,000]")
        prospective_work = None
        if isinstance(self.spectrum, SpectrumSequence):
            _, _, prospective_work = self._resolve_mission_limits(self._sample_count())
        u = sample_unit(
            self.n_samples, len(self.variables), method=self.method, seed=self.seed
        )
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
            sens = sobol_indices(
                self._log_life, self.variables, n=2**11, seed=self.seed
            )

        if isinstance(self.spectrum, SpectrumSequence):
            loading = "ordered-mission"
            life_unit = "cycles"
            retardation = "willenborg" if self.retardation.enabled else "none"
        elif self.spectrum is not None:
            loading = "repeating-block"
            life_unit = "blocks"
            retardation = None
        else:
            loading = "constant-amplitude"
            life_unit = "cycles"
            retardation = None
        return StudyResult(
            name=self.name,
            n_samples=n,
            method=self.method,
            service_cycles=self.service_cycles,
            pof=pof,
            ci_low=ci_low,
            ci_high=ci_high,
            pof_curve_cycles=curve_cycles,
            pof_curve=curve,
            inspection=outcome,
            inspection_plan=self.inspection_plan,
            sensitivity=sens,
            lives=life.cycles_to_failure,
            target_pof=self.target_pof,
            loading=loading,
            life_unit=life_unit,
            retardation=retardation,
            prospective_work=prospective_work,
        )

    def _log_life(self, x):
        life = self._grow(x)
        # cap so the Sobol estimator is not dominated by run-outs
        capped = np.minimum(life.cycles_to_failure, 100.0 * self.service_cycles)
        return np.log10(np.maximum(capped, 1.0))


def build_study(spec):
    """Build a study from a parsed YAML/JSON dict. See examples/ for the
    schema."""
    if not isinstance(spec, dict):
        raise TypeError("study definition must be a mapping")
    variables = {name: from_spec(s) for name, s in spec["variables"].items()}

    spectrum = None
    if "spectrum" in spec:
        spectrum_spec = spec["spectrum"]
        if not isinstance(spectrum_spec, dict):
            raise TypeError("spectrum must be a mapping")
        extra = set(spectrum_spec) - {"type", "cycles"}
        if extra:
            raise ValueError(f"spectrum has unexpected keys {sorted(extra)}")
        if spectrum_spec.get("type") != "ordered":
            raise ValueError("spectrum type must be 'ordered'")
        if "cycles" not in spectrum_spec:
            raise ValueError("ordered spectrum needs cycles")
        spectrum = SpectrumSequence.from_cycles(spectrum_spec["cycles"])

    retardation = None
    if "retardation" in spec:
        retardation = WillenborgConfig.from_spec(spec["retardation"])

    geo = dict(spec["geometry"])
    geo_type = geo.pop("type")
    if geo_type not in GEOMETRIES:
        raise ValueError(
            f"unknown geometry {geo_type!r}, expected one of {sorted(GEOMETRIES)}"
        )
    geometry = GEOMETRIES[geo_type](
        **{k: (v if isinstance(v, str) else float(v)) for k, v in geo.items()}
    )

    gr = dict(spec["growth"])
    if "material" in gr:
        from .materials import growth_law as material_growth_law

        law = material_growth_law(gr.pop("material"), kind=gr.pop("law", None))
        if gr:
            raise ValueError(f"unexpected growth keys with 'material': {sorted(gr)}")
    else:
        law_name = gr.pop("law")
        if law_name not in GROWTH_LAWS:
            raise ValueError(
                f"unknown growth law {law_name!r}, "
                f"expected one of {sorted(GROWTH_LAWS)}"
            )
        if "paris_c" not in variables:
            if "c" not in gr:
                raise ValueError("growth law needs 'c', or supply a 'paris_c' variable")
        else:
            gr.pop("c", None)
        law = GROWTH_LAWS[law_name](
            c=float(gr.pop("c", 1.0)), **{k: float(v) for k, v in gr.items()}
        )

    analysis = spec.get("analysis", {})
    if not isinstance(analysis, dict):
        raise TypeError("analysis must be a mapping")
    raw_service = spec["service_cycles"]
    if isinstance(spectrum, SpectrumSequence) and (
        isinstance(raw_service, (bool, str, bytes)) or not isinstance(raw_service, Real)
    ):
        raise TypeError("ordered mission service_cycles must be a real number")
    service = float(raw_service)
    raw_samples = analysis.get("samples", 200_000)
    samples = (
        raw_samples if isinstance(spectrum, SpectrumSequence) else int(raw_samples)
    )

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
        n_samples=samples,
        method=analysis.get("method", "lhs"),
        seed=analysis.get("seed"),
        target_pof=None if target is None else float(target),
        spectrum=spectrum,
        retardation=retardation,
    )
