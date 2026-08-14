# Contributing to DAMOCLES

DAMOCLES welcomes contributions to its fracture mechanics, reliability
methods, inspection models, material data, command-line interface, examples,
and verification documentation. Because the package is intended for
safety-critical analysis research, a small well-verified correction is more
valuable than a broad feature with unclear numerical behavior.

## Before starting

Please open an issue before adding a physical model, probability method,
material dataset, file format, or public API. Small fixes and documentation
corrections can go directly to a pull request. The `good first issue` and
`help wanted` labels indicate work whose scope is already defined.

This is not a certified design tool. Contributions must not weaken the existing
limitations or present typical material values as design allowables.

## Development setup

DAMOCLES supports Python 3.10 and newer.

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

CI runs the suite on several Python versions and on Linux and Windows. If your
change affects plotting or the CLI, also run the relevant worked example, such
as:

```bash
python examples/ac3314_test_case.py
damocles examples/ti64_disk_bore.yaml --sensitivity --plot out/
```

## Numerical and physical changes

Every implemented model should be traceable to a primary paper, standard,
advisory circular, or public technical report. Include the equation, variable
definitions, units, domain of validity, and the exact reference location in code
or `docs/theory.md`.

Tests should use an independent reference whenever possible:

- a closed-form limit;
- a worked example from the source;
- a digitized or tabulated published value;
- a separately derived quadrature result; or
- a statistical coverage experiment with a fixed seed.

Avoid tests that merely duplicate the implementation's arithmetic. Numerical
tolerances should reflect source accuracy, discretization, and sampling error.
For Monte Carlo changes, report uncertainty or convergence rather than requiring
one seed to equal an unexplained constant.

Preserve vectorized operation over samples and deterministic results for a fixed
seed. New sampling behavior should be tested for both statistical correctness
and reproducibility.

## Material and reference data

Data records must include a source and retain the units in which the source
printed the constants; conversion to SI belongs in the loader. State material
condition, orientation, stress ratio, environment, and thickness wherever the
source provides them. Do not combine unlike datasets under one material name.

Digitized curves need the original document, figure or table number,
digitization method, and an estimate of digitization accuracy. Confirm that the
source can be redistributed before committing extracted data.

Update `docs/verification.md` whenever a capability or its reference test is
added or materially changed.

## Code and tests

Keep modules focused and match the surrounding type and documentation style.
Add regression coverage at the lowest useful layer, then an integration test if
the behavior flows through `DamageToleranceStudy` or the CLI. Existing behavior
must remain the default when a new physical effect is optional.

Do not commit generated plots, result directories, caches, or local environments.

## Pull requests

Please include:

- a concise description of the failure or capability;
- the physical or statistical reference;
- the verification method and test commands;
- any unit, compatibility, runtime, or memory implications; and
- updates to the theory, verification matrix, examples, or limitations where
  appropriate.

By contributing, you agree that your contribution may be distributed under the
MIT license used by this repository.
