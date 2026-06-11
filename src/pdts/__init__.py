"""pdts: probabilistic damage tolerance analysis.

Monte Carlo fatigue crack growth with variance reduction, inspection
planning against POD curves, sensitivity analysis and material basis
values. Units throughout: stress in MPa, stress intensity in MPa*sqrt(m),
crack sizes in metres, life in cycles.
"""

from .allowables import a_basis, b_basis, basis_value, tolerance_factor
from .fracture import (
    CenterCrack, CornerCrack, CustomGeometry, ParisLaw, SurfaceCrack,
    ThroughCrack, WalkerLaw, critical_size, grow,
)
from .inspection import InspectionPlan, PODCurve, apply_plan, sweep_intervals
from .random_vars import (
    Deterministic, Gumbel, Lognormal, Normal, Uniform, Weibull, from_spec,
)
from .reliability import estimate_pof
from .sampling import map_to_physical, sample_unit
from .sensitivity import rank_drivers, sobol_indices
from .study import DamageToleranceStudy, build_study

__version__ = "0.1.0"
