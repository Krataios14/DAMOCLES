"""Run the AC 33.14-1 Appendix 1 calibration test case and print the
qualification evidence: the computed event rates against the acceptance
bands the FAA published for exactly this purpose.

    python examples/ac3314_test_case.py
"""

from damocles.ac3314 import (
    ACCEPTANCE_NO_INSPECTION, ACCEPTANCE_WITH_INSPECTION, hoop_stress,
    run_test_case,
)


def verdict(value, band):
    lo, hi = band
    return "PASS" if lo <= value <= hi else "FAIL"


def main():
    print("AC 33.14-1 calibration test case: Ti-6Al-4V ring disk,")
    print("6,800 rpm, 50 MPa rim load, 20,000 cycle life, hard alpha")
    print(f"anomalies (#3/#3 FBH). Bore hoop stress {hoop_stress(0.3):.1f} MPa "
          "(AC quotes 572.4).")
    print()

    base = run_test_case(inspection=False, n_rings=18)
    insp = run_test_case(inspection=True, n_rings=18)

    lo, hi = ACCEPTANCE_NO_INSPECTION
    print(f"without inspection : {base.events_per_cycle:.3e} events/cycle")
    print(f"  acceptance band  : [{lo:.2e}, {hi:.2e}]  -> "
          f"{verdict(base.events_per_cycle, ACCEPTANCE_NO_INSPECTION)}")
    lo, hi = ACCEPTANCE_WITH_INSPECTION
    print(f"with UT at 10,000  : {insp.events_per_cycle:.3e} events/cycle")
    print(f"  acceptance band  : [{lo:.2e}, {hi:.2e}]  -> "
          f"{verdict(insp.events_per_cycle, ACCEPTANCE_WITH_INSPECTION)}")
    print()

    mc = run_test_case(inspection=True, n_rings=18, method="montecarlo",
                       n_samples=200_000, seed=3)
    drift = abs(mc.events_per_cycle / insp.events_per_cycle - 1.0)
    print(f"monte carlo cross-check of the quadrature: "
          f"{mc.events_per_cycle:.3e} ({drift:.2%} apart)")

    top = sorted(base.zone_risk.items(), key=lambda kv: -kv[1])[:5]
    print("\nlargest zone contributions (no inspection):")
    for name, risk in top:
        print(f"  {name:<14} {risk:.3e}  ({risk / base.pof_service:.1%})")


if __name__ == "__main__":
    main()
