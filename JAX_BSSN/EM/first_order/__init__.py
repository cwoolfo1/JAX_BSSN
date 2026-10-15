"""PyPIC3D vacuum Maxwell evolution coupled to Cartesian BSSN."""

from PyPIC3D.relativity.core import D_FIELD_LOCATIONS, B_FIELD_LOCATIONS
from .grid import EMGrid, make_em_grid
from .variables import DensitizedMaxwellState
from JAX_BSSN.EM.variables import EinsteinMaxwellVariables
from .evolve import (
    common_densitized_fields,
    common_physical_fields,
    initialize_densitized_maxwell_state,
    initialize_first_order_einstein_maxwell_state,
    first_order_einstein_maxwell_step,
)
from .maxwell import maxwell_half_step
from .energy_momentum import (
    compute_densitized_electromagnetic_energy_momentum,
    physical_fields_at_centers,
)
from .diagnostics import (
    first_order_constraint_divergences,
    electromagnetic_output_fields,
)

__all__ = [
    "D_FIELD_LOCATIONS",
    "B_FIELD_LOCATIONS",
    "EMGrid",
    "make_em_grid",
    "DensitizedMaxwellState",
    "EinsteinMaxwellVariables",
    "common_densitized_fields",
    "common_physical_fields",
    "initialize_densitized_maxwell_state",
    "initialize_first_order_einstein_maxwell_state",
    "first_order_einstein_maxwell_step",
    "maxwell_half_step",
    "compute_densitized_electromagnetic_energy_momentum",
    "physical_fields_at_centers",
    "first_order_constraint_divergences",
    "electromagnetic_output_fields",
]
