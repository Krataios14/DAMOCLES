# pdts

Probabilistic damage tolerance in Python: Monte Carlo fatigue crack growth,
inspection planning against NDT detection curves, and the supporting
statistics (variance reduction, Sobol sensitivity, A/B-basis allowables).

This repo started as a high school project that dropped hexagonal coins
through a slotted grating to estimate a probability. I work in aerospace
now, and the grown-up version of "what fraction of random drops falls
through" turns out to be a question people stake engine certifications on:
given the scatter in initial flaw size, stress, growth rate and toughness,
what fraction of a fleet grows a crack to failure before the end of service,
and how much does an inspection program buy you. So I gutted the repo and
rebuilt it around that question. The original coin problem is still here, in
`examples/coin_grating.py`, running as a regression test on the new engine.

This is the same class of analysis that tools like DARWIN do for rotor
integrity under FAA AC 33.14, in a few thousand lines of plain numpy/scipy
you can read in an afternoon. It is not a certified tool and the example
material data is illustrative, not design data. Treat it as an engineering
sandbox: fast trade studies, sanity checks, teaching, and as a second
opinion next to whatever your company actually certifies with.

## What it does

You describe a component as distributions instead of point values: initial
flaw size (lognormal, usually), stress range per cycle, fracture toughness,
Paris coefficient. The tool grows a crack for every sample with a Paris or
Walker law through a stress intensity geometry factor, finds the cycle count
where K hits the toughness, and reports the probability of failure over the
service life with a proper confidence interval.

Then the part that matters for maintenance: you give it an NDT capability
as a POD curve (the a50/a90 sizes your inspectors actually demonstrate) and
an inspection interval, and it tells you the residual risk with inspections
in place, the risk reduction, and the expected number of detections per
part. That is the trade between inspection cost and fleet risk, quantified.

A few things engineers usually have to go elsewhere for are built in:

- Latin hypercube and scrambled Sobol sampling, because plain Monte Carlo
  wastes most of its samples when failure probabilities are small
- importance sampling for the genuinely rare-event cases (1e-5 and below)
- exact Clopper-Pearson confidence intervals, including the zero-failure case
- Sobol sensitivity indices on log-life, so you know which input scatter is
  actually driving the answer before you spend money characterising the
  wrong one
- one-sided tolerance bounds (A-basis and B-basis values) computed exactly
  from the noncentral t distribution, for turning coupon data into inputs

## Quick start

Needs Python 3.10+.

```
pip install -e .
pdts examples/ti64_disk_bore.yaml --sensitivity --plot out/
```

which prints something like:

```
================================================================
  Ti-6Al-4V compressor disk bore, corner crack
================================================================
  samples            : 200,000 (lhs)
  service life       : 20,000 cycles

  P(failure), no inspection : 1.133e-02
    95% CI                  : [1.087e-02, 1.180e-02]
    reliability index beta  : 2.28
    mean hazard per cycle   : 5.665e-07

  inspections at            : 4,000, 8,000, 12,000, 16,000
  P(failure), inspected     : 8.035e-04
    risk reduction          : 92.9%
    expected detections/part: 8.912e-02

  target P(failure)         : 1.0e-03  -> MEETS target

  variance drivers (total Sobol index on log-life):
    initial_flaw   total=0.554  first=0.547
    paris_c        total=0.367  first=0.372
    stress_range   total=0.081  first=0.063
    toughness      total=0.000  first=-0.000
================================================================
```

Read that bottom block before trusting the top one. In this study the
toughness scatter is irrelevant and the flaw size distribution dominates,
so the honest next step is better flaw characterisation, not more toughness
coupons.

The study file is plain YAML, see `examples/`. There is a second worked
example for a 7075-T6 fuselage skin panel with eddy current inspections.

## Using it as a library

```python
import numpy as np
from pdts import (Lognormal, Normal, CornerCrack, WalkerLaw,
                    DamageToleranceStudy, InspectionPlan, PODCurve)

study = DamageToleranceStudy(
    "disk bore",
    variables={
        "initial_flaw": Lognormal(mean=6e-5, cov=0.6),    # metres
        "stress_range": Normal(mean=400.0, std=20.0),     # MPa
        "toughness":    Normal(mean=75.0, std=5.0),       # MPa sqrt(m)
        "paris_c":      Lognormal(mean=2.5e-12, cov=0.45),
    },
    geometry=CornerCrack(),
    growth_law=WalkerLaw(c=1.0, m=3.87, gamma=0.65, dk_threshold=2.0),
    service_cycles=20_000,
    stress_ratio=0.05,
    inspection_plan=InspectionPlan.at_interval(
        4_000, 20_000, PODCurve.from_a50_a90(0.4e-3, 1.0e-3)),
    n_samples=200_000, method="lhs", seed=42,
)
result = study.run(sensitivity=True)
print(result.summary())
```

The lower-level pieces work on their own. `estimate_pof` takes any limit
state over named random variables, so it is a general structural
reliability engine; `grow` integrates crack growth for arbitrary sample
arrays; `b_basis` and `a_basis` take a 1-D array of coupon results.
`CustomGeometry` accepts any Y(a) callable, so a Newman-Raju fit or an
FE-derived weight function drops straight in.

## How it is checked

Every physics and statistics module is tested against something exact
rather than against itself:

- crack growth integration against the closed-form Paris life
- the reliability engine against the analytic normal R-S problem,
  including the importance sampling variance reduction
- Sobol estimators against the known Ishigami indices
- tolerance factors against published MMPDS/CMH-17 k values, plus a
  simulation check that B-basis coverage really is 95 percent
- the coin problem against a quadrature solution

```
python -m pytest
```

66 tests, a couple of seconds.

## Units

MPa for stress, MPa sqrt(m) for stress intensity, metres for crack size,
cycles for life. The Paris C must be consistent with those units. Mixing
ksi sqrt(in) data into this will hurt; convert first.

## Honest limitations

Constant amplitude loading per cycle (no spectrum, no retardation), simple
handbook geometry factors unless you supply your own, lognormal POD with
detected cracks assumed removed from service, and no crack initiation
model: life starts at an existing flaw, in the damage tolerance tradition.
All of these are deliberate scope choices, not oversights. Spectrum loading
with cycle counting is the thing I would add next.

## License

MIT.
