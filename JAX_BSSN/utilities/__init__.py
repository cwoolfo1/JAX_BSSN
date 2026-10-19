"""Reusable numerical-relativity utilities."""

from JAX_BSSN.utilities.apparent_horizon import build_geometry, find_apparent_horizon

from JAX_BSSN.utilities.bowen_york_solver import (
    axisymmetric_cell_centered_grid,
    axisymmetric_hamiltonian_residual,
    axisymmetric_laplacian,
    axisymmetric_linearized_hamiltonian_operator,
    bowen_york_extrinsic_curvature,
    brill_lindquist_conformal_factor,
    cell_centered_grid,
    contract_conformal_curvature,
    hamiltonian_residual,
    hamiltonian_source,
    laplacian,
    linearized_hamiltonian_operator,
    linearized_hamiltonian_source,
    solve_axisymmetric_conformal_factor,
    solve_conformal_factor,
)


def __getattr__(name):
    # Pure geometry diagnostics do not need the charged solver's PyPIC3D stack.
    if name == "single_charged_puncture_state":
        from JAX_BSSN.utilities.charged_puncture_solver import single_charged_puncture_state
        return single_charged_puncture_state
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "build_geometry",
    "find_apparent_horizon",
    "single_charged_puncture_state",
    "axisymmetric_cell_centered_grid",
    "axisymmetric_hamiltonian_residual",
    "axisymmetric_laplacian",
    "axisymmetric_linearized_hamiltonian_operator",
    "bowen_york_extrinsic_curvature",
    "brill_lindquist_conformal_factor",
    "cell_centered_grid",
    "contract_conformal_curvature",
    "hamiltonian_residual",
    "hamiltonian_source",
    "laplacian",
    "linearized_hamiltonian_operator",
    "linearized_hamiltonian_source",
    "solve_axisymmetric_conformal_factor",
    "solve_conformal_factor",
]
