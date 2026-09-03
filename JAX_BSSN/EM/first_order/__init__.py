"""Densitized first-order Maxwell fields coupled to BSSN."""

from JAX_BSSN.EM.first_order.cartoon import (
    axisymmetric_first_order_einstein_maxwell_step,
    axisymmetric_densitized_constraint_divergences,
    initialize_axisymmetric_first_order_state,
    initialize_spherical_first_order_state,
    spherical_first_order_einstein_maxwell_step,
    spherical_densitized_constraint_divergences,
)
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
from JAX_BSSN.EM.first_order.geometry import compute_bssn_yee_geometry
from JAX_BSSN.EM.first_order.staggering import (
    CENTER_LOCATION,
    DISPLACEMENT_FIELD_LOCATIONS,
    MAGNETIC_FIELD_LOCATIONS,
    interpolate_between_locations,
)
from JAX_BSSN.EM.first_order.variables import (
    BSSNYeeGeometry,
    ConformalMetricFields,
    DensitizedMaxwellState,
    FirstOrderEinsteinMaxwellState,
)


__all__ = [
    "BSSNYeeGeometry",
    "CENTER_LOCATION",
    "ConformalMetricFields",
    "DISPLACEMENT_FIELD_LOCATIONS",
    "DensitizedMaxwellState",
    "FirstOrderEinsteinMaxwellState",
    "MAGNETIC_FIELD_LOCATIONS",
    "axisymmetric_first_order_einstein_maxwell_step",
    "axisymmetric_densitized_constraint_divergences",
    "bootstrap_densitized_maxwell_state",
    "collocate_densitized_fields",
    "common_densitized_fields",
    "common_physical_fields",
    "compute_bssn_yee_geometry",
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
    "initialize_axisymmetric_first_order_state",
    "initialize_first_order_einstein_maxwell_state",
    "initialize_spherical_first_order_state",
    "interpolate_between_locations",
    "physical_fields_at_centers",
    "spherical_first_order_einstein_maxwell_step",
    "spherical_densitized_constraint_divergences",
]
