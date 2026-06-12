# Damage Tolerance Monte Carlo Life Estimation Suite (DAMOCLES)

[![pypi](https://img.shields.io/pypi/v/damocles?label=pypi)](https://pypi.org/project/damocles/)
[![tests](https://img.shields.io/github/actions/workflow/status/Krataios14/DAMOCLES/ci.yml?label=tests)](https://github.com/Krataios14/DAMOCLES/actions/workflows/ci.yml)
[![python](https://img.shields.io/pypi/pyversions/damocles)](https://pypi.org/project/damocles/)
[![license](https://img.shields.io/github/license/Krataios14/DAMOCLES)](LICENSE)

Monte Carlo fatigue crack growth for safety-critical metallic structure,
with the NASGRO equation, Newman-Raju stress intensity solutions,
inspection planning against probability-of-detection curves, and the FAA
AC 33.14-1 hard alpha rotor assessment. Every physics and statistics
module is verified against an exact or published reference, and the
package reproduces the FAA's own calibration test case inside the
acceptance bands the Advisory Circular publishes for qualifying
probabilistic rotor codes.

## What it does

You describe a component as distributions instead of point values:
initial flaw size, stress per cycle (constant amplitude or a rainflow
counted spectrum), fracture toughness, growth rate coefficient. For each
study DAMOCLES produces:

- probability of failure over the service life, with an exact
  Clopper-Pearson confidence interval and the equivalent reliability
  index beta
- the same number with a scheduled inspection program applied: residual
  risk, risk reduction, and expected detections per part, given an NDT
  capability stated as a POD curve (lognormal a50/a90 or the digitized
  AC 33.14-1 curves)
- the probability-of-failure history over cycles, for setting intervals
- total and first-order Sobol indices on log life, so the input scatter
  that actually drives the answer is identified before money is spent
  characterising one that does not
- for titanium rotors: the zone-based AC 33.14-1 hard alpha assessment
  with the published anomaly exceedance curves and ultrasonic POD

The sampling engine offers plain Monte Carlo, Latin hypercube and
scrambled Sobol sequences, plus importance sampling with likelihood
reweighting for failure probabilities of 1e-5 and below, where plain
sampling wastes nearly every draw.

## Why the calibration test case matters

AC 33.14-1 Section 3 requires each engine manufacturer to calibrate its
probabilistic tools against the Appendix 1 test case, a titanium ring
disk with fully specified geometry, loading, material and anomaly
distribution: "Test case results in the ranges from 1.27E-09 to 1.93E-09
(for the 'no inspection' case) and from 8.36E-10 to 1.53E-09 (for the
'with in-service inspection' case) are considered acceptable." That is
the qualification gate the FAA wrote for codes like DARWIN. DAMOCLES runs
the test case in closed form (the test case physics is deterministic
given anomaly size, so the risk reduces to a one-dimensional integral
over the exceedance curve) and again by Monte Carlo through its own
sampling engine, and lands inside both bands. The run is a unit test, so
the claim is re-verified on every commit.

One detail discovered on the way: the 2017 Change 1 tabulation of the
anomaly distribution (Table A3-1) differs from the original 2001 figure
A3-7 by factors of several in the 1,000 to 5,000 square mil region that
dominates the test case risk. The acceptance band predates Change 1, so
the calibration must run on the 2001 curve. DAMOCLES bundles both vintages,
digitized from the official PDFs, and documents which one each
assessment uses.

## The data

Everything numeric ships with provenance:

- NASGRO 4.0 equation fits for 2024-T3 sheet, 7075-T651 plate, 2014-T6
  sheet and 7475-T7351 plate, read from DOT/FAA/AR-05/15 (the FAA
  fatigue crack growth database report, Forman et al., 2005), with
  Walker fits where published. Stored in the printed US customary units
  and converted to SI on load.
- The generic Ti-6Al-4V average-air Paris fit specified by the AC
  33.14-1 test case (MCIC-HB-01R via the AC).
- The complete AC 33.14-1 Change 1 Table A3-1 hard alpha exceedance
  table (184 rows, four billet/forging inspection levels), verbatim,
  plus the 2001 figure A3-7 curve digitized from the official PDF's
  vector art.
- The five AC 33.14-1 Appendix 5 POD curves (UT at three calibration
  levels, FPI, eddy current), digitized the same way; the AC publishes
  them only as graphs.

None of this is design data. The sources are public; check them, and
substitute your own basis values for anything that matters.

## How the numbers hold up

Every claim is enforced by the test suite on each commit (CI on Linux
and Windows, Python 3.10 to 3.14). Measured results:

| Check | Reference | Result |
| --- | --- | --- |
| AC 33.14-1 calibration, no inspection | acceptance 1.27e-9 to 1.93e-9 events/cycle | 1.44e-9, PASS |
| AC 33.14-1 calibration, with inspection at 10,000 cycles | acceptance 8.36e-10 to 1.53e-9 | 1.28e-9, PASS |
| Quadrature vs the Monte Carlo engine on the inspected test case | internal cross-check | agree within 0.5% |
| Crack growth life integration | closed-form Paris solution, two exponents | within 0.5% |
| Failure probability estimator | exact normal R-S problem, pof = 2.035e-4 | CI brackets truth |
| Importance sampling | same reference, equal sample count | estimator CoV 5x below plain MC |
| Sobol sensitivity indices | Ishigami closed-form values | within 0.03 absolute |
| Tolerance factors kB, kA | published MMPDS/CMH-17 values (n=10: 2.355, 3.981) | match to 0.005 |
| B-basis confidence | simulation across 2,000 datasets | 95% coverage confirmed |
| NASGRO rates, 2024-T3 | the FCGD spline fits of the same data | within a factor of 1.8 |
| Newman-Raju limit cases | published F = 1.04 and 1.11 limits, NASA TM-85793 | match within 1% |
| Rainflow counting | ASTM E1049-85 worked example | exact cycle table |
| Hoop stress of the AC test disk | AC quoted bore stress 572.4 MPa | 574.2 MPa analytic |

## Quickstart

Needs Python 3.10+.

```
pip install damocles
```

The worked examples and the test suite live in the repository:

```
git clone https://github.com/Krataios14/DAMOCLES
cd DAMOCLES
pip install -e .[dev]
damocles examples/ti64_disk_bore.yaml --sensitivity --plot out/
python examples/ac3314_test_case.py
python -m pytest -q
```

The first command runs a corner crack study at a compressor disk bore on
the AC Ti-6-4 material and prints:

```
  P(failure), no inspection : 9.295e-03
    95% CI                  : [8.879e-03, 9.725e-03]
    reliability index beta  : 2.35

  inspections at            : 4,000, 8,000, 12,000, 16,000
  P(failure), inspected     : 8.221e-05
    risk reduction          : 99.1%
    expected detections/part: 2.712e-01

  target P(failure)         : 1.0e-03  -> MEETS target

  variance drivers (total Sobol index on log-life):
    initial_flaw   total=0.554
    paris_c        total=0.371
    stress_range   total=0.077
    toughness      total=0.000
```

Read the bottom block before trusting the top one: here toughness
scatter is irrelevant and flaw size dominates, so the productive next
step is flaw characterisation, not more toughness coupons. A second
worked example covers a 2024-T3 fuselage skin panel on the FCGD NASGRO
fit with eddy current inspections.

## Using it as a library

```python
from damocles import (Lognormal, Normal, DamageToleranceStudy,
                      InspectionPlan, PODCurve, NewmanRajuCornerCrack,
                      material_growth_law)

study = DamageToleranceStudy(
    "disk bore",
    variables={
        "initial_flaw": Lognormal(mean=1.2e-4, cov=0.6),   # metres
        "stress_range": Normal(mean=480.0, std=25.0),      # MPa
        "toughness":    Normal(mean=64.5, std=4.0),        # MPa sqrt(m)
    },
    geometry=NewmanRajuCornerCrack(thickness=0.025, width=0.08),
    growth_law=material_growth_law("ti-6al-4v-ac33.14"),
    service_cycles=20_000,
    inspection_plan=InspectionPlan.at_interval(
        4_000, 20_000, PODCurve.from_a50_a90(0.4e-3, 1.0e-3)),
    n_samples=200_000, method="lhs", seed=42)
print(study.run(sensitivity=True).summary())
```

The pieces work standalone: `estimate_pof` takes any limit state over
named random variables (it is a general structural reliability engine),
`grow` and `grow_spectrum` integrate crack growth for arbitrary sample
arrays, `rainflow` counts a stress history per ASTM E1049, `b_basis`
and `a_basis` turn coupon data into allowables, and `ac3314.run_test_case`
is the calibration run. `CustomGeometry` accepts any Y(a) callable for
FE-derived solutions.

## Interpreting the study output

| Field | Meaning |
| --- | --- |
| `pof`, `ci_low`, `ci_high` | failure probability at end of service, exact binomial CI |
| `reliability_index()` | equivalent beta, for comparing designs |
| `inspection.pof_inspected` | residual risk with the inspection plan applied |
| `inspection.risk_reduction` | 1 - inspected/unmitigated |
| `inspection.mean_detections` | expected detections per part over the life |
| `sensitivity[name]["total"]` | total Sobol index of that input on log life |
| `pof_curve_cycles`, `pof_curve` | the failure probability history |

Run-outs (cracks below threshold that never grow) carry infinite life
and are reported as such, not silently dropped.

## Design allowables

`damocles.allowables` computes one-sided lower tolerance bounds from
measured samples: B-basis (90% coverage at 95% confidence) and A-basis
(99/95), using exact normal-theory factors from the noncentral t
distribution, on raw or log-transformed data. Use it to turn your own
coupon results into study inputs. The MMPDS/CMH-17 definitions apply;
the factors match the published tables and the confidence level is
verified by simulation in the test suite.

If you need the toughness input itself predicted from composition with
finite-sample guarantees, that is a different problem and a different
tool: FTQS (Fracture Toughness Qualification Suite). DAMOCLES consumes
toughness as a distribution; FTQS is one defensible way to get one for
alloys you have not tested.

## Limitations, stated plainly

- LEFM with no load interaction: constant amplitude or repeating-block
  spectra, no retardation. Order effects within a block are exactly
  irrelevant under this assumption; overload retardation is not
  modelled at all.
- One dominant crack per part. No multi-site damage, no continuing
  damage after repair (detected parts leave the fleet).
- The AC 33.14 module covers the Appendix 1 ring disk class of problem:
  axisymmetric stress, the AC's prescribed crack shapes, no surface
  transition of subsurface cracks. It is a calibration-grade
  implementation, not a general rotor risk code.
- The AC POD curves and the 2001 anomaly curve are digitized from the
  official PDFs' vector art (the FAA never published them as numbers).
  Estimated digitization accuracy is half a percent in POD and one to
  two percent in size.
- NASGRO 4's threshold reparametrisation is not fully public; DK1 is
  read as the long-crack R = 0 threshold per the NASA primary source,
  which is the slightly conservative interpretation, and is documented
  in nasgro.py.
- Inputs are independent random variables. Correlation requires a
  custom limit state.
- This is not a certified tool and nothing here is design data. It is
  built to certification-grade verification standards (every number
  traced, every claim tested), which is a different statement.

## Repository layout

```
src/damocles/random_vars.py   input distributions
src/damocles/sampling.py      MC / LHS / Sobol sample generation
src/damocles/reliability.py   pof estimation, exact CIs, importance sampling
src/damocles/fracture.py      growth laws, geometry factors, life integration
src/damocles/newman_raju.py   NASA TM-85793 surface and corner crack solutions
src/damocles/nasgro.py        Forman-Mettu equation with Newman closure
src/damocles/spectrum.py      ASTM E1049 rainflow, spectrum blocks
src/damocles/inspection.py    POD curves, inspection plans, risk arithmetic
src/damocles/ac3314.py        AC 33.14-1 hard alpha assessment + calibration
src/damocles/materials.py     cited material database (data/materials.json)
src/damocles/sensitivity.py   Sobol indices, Saltelli/Jansen estimators
src/damocles/allowables.py    A- and B-basis tolerance bounds
src/damocles/study.py         YAML-driven studies
src/damocles/cli.py           command line entry point
docs/theory.md                the equations and assumptions
docs/verification.md          claim -> reference -> test matrix
examples/                     disk bore, skin panel, AC test case, coin
tests/                        112 tests, all against external references
```

The original repository was a high school Monte Carlo toy that dropped
hexagonal coins through a slotted grating. That problem is still here,
in `examples/coin_grating.py`, now solved on the reliability engine and
checked against quadrature; everything else was torn down and rebuilt.

## Tests

```
python -m pytest -q
```

112 tests, a few seconds. The verification matrix in
`docs/verification.md` maps each capability to its reference and its
test.

## References

- FAA AC 33.14-1, Damage Tolerance for High Energy Turbine Engine
  Rotors, 2001, and Change 1, 2017. Anomaly distributions, POD curves,
  the calibration test case and its acceptance bands.
- Forman, Shivakumar, Cardinal, Williams, McKeighan. Fatigue Crack
  Growth Database for Damage Tolerance Analysis. DOT/FAA/AR-05/15, 2005.
  NASGRO and Walker fits with the underlying data.
- Newman, Raju. Stress-Intensity Factor Equations for Cracks in
  Three-Dimensional Finite Bodies Subjected to Tension and Bending
  Loads. NASA TM-85793, 1984. Also TP-1578 (1979) and TM-83200 (1981),
  and Eng. Fract. Mech. 15 (1981) 185-192.
- Mettu, Shivakumar, Beek, Yeh, Williams, Forman, McMahon, Newman.
  NASGRO 3.0: A software for analyzing aging aircraft. NASA NTRS
  19990028759. The NASGRO equation and threshold form.
- Newman. A crack opening stress equation for fatigue crack growth.
  Int. J. Fracture 24 (1984). The closure function.
- Leverant, McClung, Millwater, Enright et al. Turbine Rotor Material
  Design. DOT/FAA/AR-00/64, 2000. DARWIN's analysis of the AC test case.
- ASTM E1049-85, Standard Practices for Cycle Counting in Fatigue
  Analysis. The rainflow rules and the worked example used as an
  acceptance test.
- MIL-HDBK-1823A, Nondestructive Evaluation System Reliability
  Assessment, 2009. The lognormal POD model.
- MMPDS / CMH-17 for the A- and B-basis definitions.

## License

MIT.
