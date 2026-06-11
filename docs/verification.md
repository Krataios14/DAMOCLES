# Verification matrix

Every physics or statistics capability is verified against an exact or
published reference, not against the code's own earlier output. The table
maps each claim to the reference and to the test that enforces it on every
commit (CI runs the suite on Linux and Windows, Python 3.10 to 3.14).

| Capability | Reference | Test |
| --- | --- | --- |
| Constant amplitude crack growth life | Closed-form Paris integration, N = (a0^(1-m/2) - ac^(1-m/2)) / ((m/2-1) C (Y ds sqrt(pi))^m), two exponent values | `test_fracture.py::test_life_matches_closed_form`, `test_life_closed_form_other_exponent` |
| Critical crack size | Analytic ac = (Kc / (Y smax sqrt(pi)))^2 for constant Y | `test_fracture.py::test_critical_size_through_crack_analytic`, `test_custom_geometry` |
| Walker mean stress correction | Reduces exactly to Paris at gamma = 1; higher R shortens life | `test_fracture.py::test_walker_reduces_to_paris_at_gamma_one`, `test_walker_higher_r_shortens_life` |
| Failure probability estimator | Analytic normal R-S problem, pof = Phi(-beta) = 2.035e-4 | `test_reliability.py::test_plain_lhs_brackets_truth`, `test_sobol_runs_and_is_close` |
| Importance sampling | Same R-S reference; estimator CoV at least 5x below plain MC at equal n | `test_reliability.py::test_importance_sampling_efficiency` |
| Confidence intervals | Exact Clopper-Pearson, including the zero-failure bound | `test_reliability.py::test_zero_failures_gives_zero_with_upper_bound` |
| Latin hypercube stratification | One point per bin in every dimension, by construction | `test_sampling.py::test_lhs_stratification` |
| Sobol sensitivity indices | Ishigami function closed-form indices (S1 = 0.3139, S2 = 0.4424, S3 = 0, T3 = 0.2437) | `test_sensitivity.py::test_ishigami_indices` |
| One-sided tolerance factors | Published MMPDS/CMH-17 k values (n = 10: kB = 2.355, kA = 3.981; n = 100: kB = 1.527) | `test_allowables.py::test_published_k_factors` |
| B-basis confidence level | Simulation: bound falls below the true 10th percentile in 95 percent of repeated datasets | `test_allowables.py::test_b_basis_coverage` |
| Rainflow counting | ASTM E1049-85 section 5.4.4 worked example, exact cycle table | `test_spectrum.py::test_rainflow_astm_e1049_worked_example` |
| Spectrum growth | Single-class spectrum reproduces constant amplitude exactly; count scaling is exact | `test_spectrum.py::test_single_class_spectrum_equals_constant_amplitude`, `test_count_scaling_halves_block_life` |
| Inspection risk arithmetic | Hand-computed three-part case; limiting cases (perfect POD, useless POD, no inspections) | `test_inspection.py::test_hand_computed_case` and neighbours |
| Newman-Raju surface crack | Published limit case: a/c = 1, a/t -> 0, deepest point gives F = 1.04, Y = 0.6625 (NASA TM-85793) | `test_newman_raju.py::test_semicircular_shallow_limit` |
| Newman-Raju corner crack | Published limit case: a/t -> 1, phi = 0 reduces to 1.11 St sqrt(pi c) fw within 1% (TM-85793 eq. 45) | `test_newman_raju.py::test_corner_crack_through_thickness_limit` |
| Newman closure function | Hand evaluation of A0 at alpha = 2, Smax/sigma0 = 0.3 (0.3257) and f at R = 0.7 | `test_nasgro.py::test_newman_closure_hand_values` |
| NASGRO threshold | Long crack at R = 0 returns DK1 exactly; El Haddad small-crack knockdown is sqrt(a/(a+a0)) | `test_nasgro.py::test_threshold_limits` |
| NASGRO rates | Cross-checked against the independent FCGD spline fits of the same 2024-T3 data (DOT/FAA/AR-05/15 table 2), within a factor of 1.8 at three points | `test_nasgro.py::test_cross_check_against_fcgd_spline` |
| Unit conversion | US customary to SI for C, dK1, toughness and strength is self-consistent at the rate level | `test_materials.py::test_us_entry_converts_to_si` |
| AC 33.14-1 calibration, no inspection | FAA acceptance band 1.27e-9 to 1.93e-9 events/cycle (AC 33.14-1 Section 3) | `test_ac3314.py::test_calibration_no_inspection_inside_acceptance_band` |
| AC 33.14-1 calibration, with inspection | FAA acceptance band 8.36e-10 to 1.53e-9 events/cycle | `test_ac3314.py::test_calibration_with_inspection_inside_acceptance_band` |
| AC 33.14-1 stress field | Analytic rotating-annulus + Lame field reproduces the AC's bore stress (572.4 MPa) within 4 MPa; zone volumes match Table A1-1 | `test_ac3314.py::test_hoop_stress_matches_ac_contours`, `test_zone_volumes_match_ac_table_a1_1` |
| Quadrature vs Monte Carlo | The closed-form risk integral and the sampling engine agree within 2% on the inspected test case | `test_ac3314.py::test_monte_carlo_agrees_with_quadrature` |
| Numerical implementation invariants | Chunked integration bit-identical to single pass; binning conserves cycle counts | `test_fracture.py::test_chunking_is_invisible`, `test_spectrum.py::test_binned_conserves_counts` |
| Geometric reliability regression | Original coin-through-grating problem against 1-D quadrature | `test_coin_grating.py::test_engine_matches_quadrature` |

Run it yourself:

```
python -m pytest -q
```
