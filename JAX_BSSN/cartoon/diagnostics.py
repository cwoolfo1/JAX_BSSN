"""Constraints and output mappings for compact spherical Cartoon states."""

from functools import partial

import jax.numpy as jnp
from jax import jit

from JAX_BSSN.bssn.constraints import ConstraintViolations, compute_all_constraints
from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables
from JAX_BSSN.cartoon.reconstruction import (
    cartoon_positive_radius,
    expand_cartoon_axis,
    expand_cartoon_scalar,
    expand_cartoon_vector,
    project_cartoon_scalar,
    project_cartoon_vector,
    reconstruct_cartoon_support,
)


@jit
def compute_cartoon_constraints(
    vars: BSSNVariables, params: BSSNParameters
) -> ConstraintViolations:
    """Compute constraints on Cartesian support and return compact fields."""

    support_vars = reconstruct_cartoon_support(vars, params)
    support = compute_all_constraints(support_vars, params)

    return ConstraintViolations(
        hamiltonian=project_cartoon_scalar(support.hamiltonian),
        momentum=project_cartoon_vector(support.momentum),
        det_gamma=project_cartoon_scalar(support.det_gamma),
        trace_A=project_cartoon_scalar(support.trace_A),
        gamma_condition=project_cartoon_scalar(support.gamma_condition),
    )


@partial(jit, static_argnames=("exclude_outer",))
def compute_cartoon_constraint_norms(
    violations: ConstraintViolations, exclude_outer: int = 4
) -> dict:
    """Compute norms over independent positive radii away from the outer edge."""

    def physical_radius(field):
        radius = cartoon_positive_radius(field)
        if exclude_outer:
            radius = radius[..., :-exclude_outer]
        return radius

    fields = {
        "hamiltonian": physical_radius(violations.hamiltonian),
        "momentum": physical_radius(violations.momentum),
        "det_gamma": physical_radius(violations.det_gamma),
        "trace_A": physical_radius(violations.trace_A),
        "gamma": physical_radius(violations.gamma_condition),
    }

    norms = {}
    for name, field in fields.items():
        norms[f"{name}_l2"] = jnp.sqrt(jnp.mean(field**2))
        norms[f"{name}_linf"] = jnp.max(jnp.abs(field))

    return norms


def cartoon_axis_output_fields(
    vars: BSSNVariables,
    hamiltonian: jnp.ndarray,
    momentum: jnp.ndarray,
) -> dict:
    """Return the complete BSSN state and constraints on the signed x-axis."""

    full_vars = expand_cartoon_axis(vars)
    full_hamiltonian = expand_cartoon_scalar(hamiltonian)
    full_momentum = expand_cartoon_vector(momentum)

    fields = {
        "W": full_vars.conformal_factor,
        "K": full_vars.trace_K,
        "lapse": full_vars.lapse,
        "shift": tuple(full_vars.shift[i] for i in range(3)),
        "conformal_connection": tuple(
            full_vars.conformal_connection[i] for i in range(3)
        ),
        "hamiltonian_constraint": full_hamiltonian,
        "momentum_constraint": tuple(full_momentum[i] for i in range(3)),
    }

    for name, tensor in (
        ("conformal_metric", full_vars.conformal_metric),
        ("traceless_K", full_vars.traceless_K),
    ):
        fields[f"{name}_xx"] = tensor[0, 0]
        fields[f"{name}_xy"] = tensor[0, 1]
        fields[f"{name}_xz"] = tensor[0, 2]
        fields[f"{name}_yy"] = tensor[1, 1]
        fields[f"{name}_yz"] = tensor[1, 2]
        fields[f"{name}_zz"] = tensor[2, 2]

    return fields
