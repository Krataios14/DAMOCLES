# Theory manual

Units everywhere: stress in MPa, stress intensity in MPa sqrt(m), crack
size in metres, life in cycles (or blocks, where a spectrum defines the
block). This document states the equations the code implements and the
assumptions behind them. The companion `verification.md` maps each one to
the test that checks it.

## 1. Problem statement

A component contains an initial flaw of size a0 drawn from a population.
Under cyclic stress the flaw grows; the part fails when the stress
intensity at the largest load reaches the fracture toughness. The
quantity of interest is the probability that failure happens before the
end of the service life, with and without scheduled inspections, given
distributions for flaw size, stress, growth rate constant and toughness.
This is the standard probabilistic damage tolerance formulation used for
engine rotor integrity under FAA AC 33.14-1 and for airframe structure
under the damage tolerance provisions of 14 CFR 25.571.

## 2. Stress intensity and critical size

K = Y(a) * sigma * sqrt(pi * a)

Y(a) is the geometry correction. Built in: through crack (Y = 1), centre
crack with the Feddersen finite width secant, and constant front-factor
approximations for semicircular surface (0.713) and quarter-circular
corner (0.798) cracks; arbitrary Y(a) callables are accepted for anything
better (see the Newman-Raju section below once implemented). The critical
size ac solves Y(ac) sigma_max sqrt(pi ac) = Kc by bisection in log a,
clamped to the validity limit of the geometry correction.

## 3. Crack growth laws

Paris: da/dN = C (dK)^m for dK above the threshold, zero below.
Walker: the same with dK replaced by dK / (1-R)^(1-gamma), which folds
the stress ratio dependence into an effective range; gamma = 1 recovers
Paris.

C may be a per-sample array, which is how growth rate scatter enters the
Monte Carlo. Heat-to-heat and lab-to-lab scatter in C is typically
lognormal with a coefficient of variation between 0.3 and 0.5.

## 4. Life integration

Under constant amplitude (or a fixed repeating block, section 6) the
growth rate is a function of crack size alone, so life is the quadrature

N(a) = integral from a0 to a of da' / v(a')

evaluated with the trapezoid rule on a log-spaced grid from a0 to ac.
This avoids cycle-by-cycle stepping entirely and vectorises across
samples; the grid is fine enough that the result matches the closed-form
Paris integration to a few parts in a thousand (see verification). Since
dK increases monotonically with a under constant amplitude, a crack
below threshold at a0 never starts growing and its life is infinite
(a run-out, reported as such).

The same cumulative N(a) curve gives the crack size at any earlier
cycle count by inverse interpolation, which is what the inspection model
consumes.

## 5. Reliability estimation

The limit state for service life Ns is g = Nf - Ns; failure is g < 0.
The estimator options are plain Monte Carlo, Latin hypercube, and
scrambled Sobol sequences, all driven through the inverse CDF of each
input. Binomial confidence intervals are exact Clopper-Pearson, so the
zero-failure case yields a proper one-sided bound rather than zero.

For rare events, importance sampling replaces the density of selected
inputs with a proposal q and reweights each failure indicator by the
likelihood ratio f/q. The estimate is the weighted mean; its standard
error comes from the sample variance of the weighted indicator. The
verification case demonstrates a 5x reduction in estimator coefficient
of variation at equal sample count, equivalent to roughly 25x fewer
samples for the same precision.

## 6. Spectrum loading

A measured stress history is reduced to cycles by rainflow counting per
ASTM E1049-85 section 5.4.4, retaining half cycles for the residual.
Cycles whose peak stress is compressive are dropped (the crack is closed
and LEFM transfers no range); for the rest, the compressive part of the
excursion is clipped at zero, consistent with treating negative R as
R = 0 for growth purposes.

The counted cycles form a repeating block. With no load interaction
model (no retardation), the order of cycles within a block does not
affect the total per-block growth, so the block rate is the
count-weighted sum of class rates:

v_block(a) = sum_i n_i * v(dK_i(a), R_i)

and the same a-grid quadrature applies, with life measured in blocks.
This is exact under the no-interaction assumption, not an approximation.
Long histories may be binned; bins preserve cycle count and the
damage-equivalent range (sum n dS^p conserved, p close to the growth
exponent), and never alter the true block peak stress that governs
fracture.

Retardation (e.g. Willenborg) is deliberately out of scope for now: it
would force cycle-by-cycle integration and its parameters are rarely
known at the fleet statistics level this tool targets.

## 7. Inspections

NDT capability is a probability of detection curve, lognormal in crack
size (MIL-HDBK-1823A form): POD(a) = Phi((ln a - ln a50)/sigma). An
inspection at cycle t sees the crack at its then-current size a(t); a
detected crack is removed from service. For a sample that would fail at
Nf, the probability that it survives the program to fail in service is
the product of miss probabilities over all inspections that occur before
Nf. The fleet risk with inspection is the mean of that product over
failing samples. Expected detections per part accumulate as
sum_j POD_j * prod_{k<j} (1 - POD_k).

## 8. Sensitivity

Variance-based Sobol indices on log life, estimated with the Saltelli
sampling scheme and Jansen estimators. First-order indices measure what
a variable drives alone; total indices include interactions. They are
the rational basis for deciding which input deserves better
characterisation, and they are cheap relative to the value: (d+2) model
runs of the base sample.

## 9. Basis values

A-basis (99 percent exceedance) and B-basis (90 percent) at 95 percent
confidence, from the exact one-sided normal tolerance factor
k = nct.ppf(conf, n-1, z_p sqrt(n)) / sqrt(n), applied to raw or
log-transformed data. These are the MMPDS / CMH-17 definitions; the
factors match the published tables and the confidence level is verified
by simulation.

## 10. Assumptions and scope

- LEFM throughout; no plasticity correction, no initiation life.
- Constant amplitude or repeating-block loading; no retardation.
- A single dominant crack per part; no continuing damage or multi-site
  interaction.
- Detected cracks leave the fleet (repair is not modelled as a renewal
  process).
- Inputs are independent random variables. Correlation can be smuggled
  in through a custom limit state if needed, but the study interface
  does not expose it yet.
