"""Second-order Maxwell fields coupled to the JAX BSSN spacetime solver."""

from JAX_BSSN.EM.second_order.cartoon import (
    axisymmetric_einstein_maxwell_rk4_step,
    compact_axisymmetric_wave,
    compact_cartoon_wave,
    compute_axisymmetric_constraint_divergences,
    compute_axisymmetric_einstein_maxwell_rhs,
    compute_cartoon_constraint_divergences,
    compute_spherical_einstein_maxwell_rhs,
    expand_axisymmetric_wave_plane,
    expand_cartoon_wave_axis,
    fill_axisymmetric_wave_ghosts,
    fill_cartoon_wave_ghosts,
    project_axisymmetric_wave_rhs,
    project_cartoon_wave_rhs,
    reconstruct_axisymmetric_wave_support,
    reconstruct_cartoon_wave_support,
    spherical_einstein_maxwell_rk4_step,
    validate_axisymmetric_wave_grid,
    validate_cartoon_wave_grid,
)
from JAX_BSSN.EM.second_order.diagnostics import electromagnetic_output_fields
from JAX_BSSN.EM.second_order.energy_momentum import (
    ElectromagneticStressEnergy,
    compute_electromagnetic_energy_momentum,
    compute_electromagnetic_stress_energy,
)
from JAX_BSSN.EM.second_order.equations import (
    compute_em_rhs,
    source_free_projected_field_dots,
)
from JAX_BSSN.EM.second_order.evolve import (
    compute_einstein_maxwell_rhs,
    einstein_maxwell_rk4_step
)
from JAX_BSSN.EM.second_order.geometry import compute_bssn_em_geometry
from JAX_BSSN.EM.second_order.initial_data import (
    conformal_vector_to_physical_covector,
    contract_conformal_electromagnetic_fields,
    electromagnetic_hamiltonian_residual,
    electromagnetic_hamiltonian_source,
    linearized_electromagnetic_hamiltonian_operator,
    linearized_electromagnetic_hamiltonian_source,
    off_centered_toroidal_electric_seed,
    solve_electromagnetic_conformal_factor,
)
from JAX_BSSN.EM.second_order.variables import (
    BSSNEMGeometry,
    EinsteinMaxwellVariables,
    EMVariables,
)


__all__ = [
    "BSSNEMGeometry",
    "EinsteinMaxwellVariables",
    "ElectromagneticStressEnergy",
    "EMVariables",
    "axisymmetric_einstein_maxwell_rk4_step",
    "compact_axisymmetric_wave",
    "compact_cartoon_wave",
    "compute_axisymmetric_constraint_divergences",
    "compute_axisymmetric_einstein_maxwell_rhs",
    "compute_bssn_em_geometry",
    "compute_cartoon_constraint_divergences",
    "compute_einstein_maxwell_rhs",
    "compute_electromagnetic_energy_momentum",
    "compute_electromagnetic_stress_energy",
    "compute_em_rhs",
    "compute_spherical_einstein_maxwell_rhs",
    "conformal_vector_to_physical_covector",
    "contract_conformal_electromagnetic_fields",
    "einstein_maxwell_rk4_step",
    "electromagnetic_hamiltonian_residual",
    "electromagnetic_hamiltonian_source",
    "electromagnetic_output_fields",
    "expand_axisymmetric_wave_plane",
    "expand_cartoon_wave_axis",
    "fill_axisymmetric_wave_ghosts",
    "fill_cartoon_wave_ghosts",
    "linearized_electromagnetic_hamiltonian_operator",
    "linearized_electromagnetic_hamiltonian_source",
    "off_centered_toroidal_electric_seed",
    "project_axisymmetric_wave_rhs",
    "project_cartoon_wave_rhs",
    "reconstruct_axisymmetric_wave_support",
    "reconstruct_cartoon_wave_support",
    "solve_electromagnetic_conformal_factor",
    "source_free_projected_field_dots",
    "spherical_einstein_maxwell_rk4_step",
    "validate_axisymmetric_wave_grid",
    "validate_cartoon_wave_grid",
]
