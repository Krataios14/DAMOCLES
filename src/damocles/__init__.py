"""damocles: probabilistic damage tolerance analysis.

Monte Carlo fatigue crack growth with variance reduction, inspection
planning against POD curves, sensitivity analysis and material basis
values. Units throughout: stress in MPa, stress intensity in MPa*sqrt(m),
crack sizes in metres, life in cycles.
"""

from .ac3314 import (
    ExceedanceCurve,
    TabulatedPOD,
    hoop_stress as ac3314_hoop_stress,
    run_test_case as ac3314_test_case,
)
from .allowables import a_basis, b_basis, basis_value, tolerance_factor
from .fracture import (
    CenterCrack,
    CornerCrack,
    CustomGeometry,
    ParisLaw,
    SurfaceCrack,
    ThroughCrack,
    WalkerLaw,
    critical_size,
    grow,
    grow_spectrum,
    grow_spectrum_retarded,
)
from .spectrum import (
    MAX_ORDERED_CYCLES,
    CycleClass,
    OrderedCycle,
    Spectrum,
    SpectrumSequence,
    rainflow,
)
from .retardation import (
    MAX_RETARDED_BLOCKS,
    MAX_RETARDED_CYCLES,
    MAX_RETARDED_WORK,
    WillenborgConfig,
    WillenborgState,
    effective_kr,
    init_state,
    plastic_zone_radius,
    residual_stress_intensity,
)
from .inspection import InspectionPlan, PODCurve, apply_plan, sweep_intervals
from .materials import available as available_materials
from .materials import get as get_material
from .materials import growth_law as material_growth_law
from .nasgro import NasgroLaw, newman_opening_function
from .newman_raju import NewmanRajuCornerCrack, NewmanRajuSurfaceCrack
from .random_vars import (
    Deterministic,
    Gumbel,
    Lognormal,
    Normal,
    Uniform,
    Weibull,
    from_spec,
)
from .reliability import estimate_pof
from .sampling import map_to_physical, sample_unit
from .sensitivity import rank_drivers, sobol_indices
from .study import DamageToleranceStudy, build_study

__all__ = [
    "MAX_ORDERED_CYCLES",
    "MAX_RETARDED_BLOCKS",
    "MAX_RETARDED_CYCLES",
    "MAX_RETARDED_WORK",
    "CenterCrack",
    "CornerCrack",
    "CustomGeometry",
    "CycleClass",
    "DamageToleranceStudy",
    "Deterministic",
    "ExceedanceCurve",
    "Gumbel",
    "InspectionPlan",
    "Lognormal",
    "NasgroLaw",
    "NewmanRajuCornerCrack",
    "NewmanRajuSurfaceCrack",
    "Normal",
    "OrderedCycle",
    "PODCurve",
    "ParisLaw",
    "Spectrum",
    "SpectrumSequence",
    "SurfaceCrack",
    "TabulatedPOD",
    "ThroughCrack",
    "Uniform",
    "WalkerLaw",
    "Weibull",
    "WillenborgConfig",
    "WillenborgState",
    "a_basis",
    "ac3314_hoop_stress",
    "ac3314_test_case",
    "apply_plan",
    "available_materials",
    "b_basis",
    "basis_value",
    "build_study",
    "critical_size",
    "effective_kr",
    "estimate_pof",
    "from_spec",
    "get_material",
    "grow",
    "grow_spectrum",
    "grow_spectrum_retarded",
    "init_state",
    "map_to_physical",
    "material_growth_law",
    "newman_opening_function",
    "plastic_zone_radius",
    "rainflow",
    "rank_drivers",
    "residual_stress_intensity",
    "sample_unit",
    "sobol_indices",
    "sweep_intervals",
    "tolerance_factor",
]

__version__ = "0.3.0"
