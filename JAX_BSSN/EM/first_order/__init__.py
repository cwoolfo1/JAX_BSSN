"""Densitized first-order Maxwell fields coupled to BSSN."""

from JAX_BSSN.EM.first_order.diagnostics import (
    electromagnetic_output_fields,
    first_order_constraint_divergences,
)
from JAX_BSSN.EM.first_order.energy_momentum import (
    compute_densitized_electromagnetic_energy_momentum,
    physical_fields_at_centers,
)
from JAX_BSSN.EM.first_order.equations import (
    collocate_densitized_fields,
    compute_covariant_E,
    compute_covariant_H,
    curl_E_to_densitized_B,
    curl_H_to_densitized_D,
    densitized_displacement_divergence,
    densitized_magnetic_divergence,
    densitized_maxwell_rhs,
)
from JAX_BSSN.EM.first_order.evolve import (
    bootstrap_densitized_maxwell_state,
    common_densitized_fields,
    common_physical_fields,
    first_order_einstein_maxwell_step,
    initialize_densitized_maxwell_state,
    initialize_first_order_einstein_maxwell_state,
)
from JAX_BSSN.EM.first_order.staggering import (
    CENTER_LOCATION,
    DISPLACEMENT_FIELD_LOCATIONS,
    MAGNETIC_FIELD_LOCATIONS,
    interpolate_between_locations,
)
from JAX_BSSN.EM.first_order.variables import DensitizedMaxwellState
from JAX_BSSN.EM.first_order.pec import (
    PECBoundary,
    apply_pec_boundaries,
    enforce_pec_D,
    enforce_pec_B,
)
from JAX_BSSN.EM.variables import EinsteinMaxwellVariables


__all__ = [
    "PECBoundary",
    "apply_pec_boundaries",
    "enforce_pec_D",
    "enforce_pec_B",
    "CENTER_LOCATION",
    "DISPLACEMENT_FIELD_LOCATIONS",
    "DensitizedMaxwellState",
    "EinsteinMaxwellVariables",
    "MAGNETIC_FIELD_LOCATIONS",
    "bootstrap_densitized_maxwell_state",
    "collocate_densitized_fields",
    "common_densitized_fields",
    "common_physical_fields",
    "compute_covariant_E",
    "compute_covariant_H",
    "compute_densitized_electromagnetic_energy_momentum",
    "curl_E_to_densitized_B",
    "curl_H_to_densitized_D",
    "densitized_displacement_divergence",
    "densitized_magnetic_divergence",
    "densitized_maxwell_rhs",
    "electromagnetic_output_fields",
    "first_order_constraint_divergences",
    "first_order_einstein_maxwell_step",
    "initialize_densitized_maxwell_state",
    "initialize_first_order_einstein_maxwell_state",
    "interpolate_between_locations",
    "physical_fields_at_centers",
]
