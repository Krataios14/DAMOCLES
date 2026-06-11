"""pdts: probabilistic damage tolerance analysis.

Monte Carlo fatigue crack growth with variance reduction, inspection
planning against POD curves, sensitivity analysis and material basis
values. Units throughout: stress in MPa, stress intensity in MPa*sqrt(m),
crack sizes in metres, life in cycles.
"""

from .allowables import a_basis, b_basis, basis_value, tolerance_factor
from .fracture import (
    CenterCrack, CornerCrack, CustomGeometry, ParisLaw, SurfaceCrack,
    ThroughCrack, WalkerLaw, critical_size, grow, grow_spectrum,
)
from .spectrum import CycleClass, Spectrum, rainflow
from .inspection import InspectionPlan, PODCurve, apply_plan, sweep_intervals
from .materials import available as available_materials
from .materials import get as get_material
from .materials import growth_law as material_growth_law
from .nasgro import NasgroLaw, newman_opening_function
from .newman_raju import NewmanRajuCornerCrack, NewmanRajuSurfaceCrack
from .random_vars import (
    Deterministic, Gumbel, Lognormal, Normal, Uniform, Weibull, from_spec,
)
from .reliability import estimate_pof
from .sampling import map_to_physical, sample_unit
from .sensitivity import rank_drivers, sobol_indices
from .study import DamageToleranceStudy, build_study

__version__ = "0.1.0"
