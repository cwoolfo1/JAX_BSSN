"""Lapse and shift evolution equations."""

import jax
import jax.numpy as jnp
from jax import jit

from JAX_BSSN.derivatives import diff1_field, diff6_field
from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables, get_boundary_codes


@jit
def compute_shift_derivatives(
    shift: jnp.ndarray, params: BSSNParameters
) -> jnp.ndarray:
    """
    Compute spatial derivatives of the shift vector.

    The returned tensor is indexed as d_beta[i, j] = partial_j beta^i.
    Keeping the component and derivative axes separate is important for the
    weighted Lie derivative terms in the nonzero-shift BSSN equations.
    """

    d_beta = jnp.stack(
        [
            jnp.stack(
                [
                    diff1_field(
                        shift[i, ...],
                        j,
                        params.dx,
                        *get_boundary_codes(params, j), mad_q=params.mad_q,
                    )
                    for j in range(3)
                ],
                axis=0,
            )
            for i in range(3)
        ],
        axis=0,
    )

    return d_beta




@jit
def evolve_lapse(vars: BSSNVariables, params: BSSNParameters) -> jnp.ndarray:
    """
    Evolve lapse function.
    
    Args:
        vars: Current BSSN variables
        params: Evolution parameters
        
    Returns:
        Time derivative of lapse
    """
    dx = params.dx

    def harmonic_slicing(_):
        return -jnp.power(vars.lapse, 2) * vars.trace_K

    def one_plus_log_slicing(_):
        return -2.0 * vars.lapse * vars.trace_K

    slicing_term = jax.lax.cond(
        params.gauge == 0,
        harmonic_slicing,
        one_plus_log_slicing,
        operand=None,
    )
    # choose the lapse source term without leaving JIT-compatible control flow

    grad_alpha = jnp.stack(
        [
            diff1_field(
                vars.lapse, d, dx, *get_boundary_codes(params, d), mad_q=params.mad_q
            )
            for d in range(3)
        ],
        axis=0,
    )
    # first derivatives of the lapse

    advection_term = jnp.einsum('m...,m...->...', vars.shift, grad_alpha)
    # advect the lapse with the shift

    dalpha_dx1 = diff6_field(
        vars.lapse, 0, params.dx, *get_boundary_codes(params, 0)
    )
    dalpha_dx2 = diff6_field(
        vars.lapse, 1, params.dx, *get_boundary_codes(params, 1)
    )
    dalpha_dx3 = diff6_field(
        vars.lapse, 2, params.dx, *get_boundary_codes(params, 2)
    )
    # compute the 6th derivative in each direction

    dissipation_term = params.nu / 64 * params.dx**5 * (dalpha_dx1 + dalpha_dx2 + dalpha_dx3)
    # compute dissipation term
    
    return slicing_term + advection_term + dissipation_term


@jit
def evolve_shift(vars: BSSNVariables, params: BSSNParameters) -> jnp.ndarray:
    """
    Evolve shift vector β^i using Gamma driver.
    
    
    Args:
        vars: Current BSSN variables
        params: Evolution parameters
        
    Returns:
        Time derivative of shift
    """
    shift = vars.shift
    # unpack the shift vector

    grad_shift = compute_shift_derivatives(shift, params)
    # grad_shift[i, j] = partial_j beta^i

    advection_term = jnp.einsum('j...,ij...->i...', shift, grad_shift)
    # beta^j partial_j beta^i

    gamma_driver_term = params.g * vars.conformal_connection
    # single-variable Gamma-driver source for beta^i

    damping_term = -params.eta * shift
    # linear damping of the shift

    dt_beta = gamma_driver_term + advection_term + damping_term

    dbeta_dx1 = diff6_field(
        shift, 1, params.dx, *get_boundary_codes(params, 0)
    )
    dbeta_dx2 = diff6_field(
        shift, 2, params.dx, *get_boundary_codes(params, 1)
    )
    dbeta_dx3 = diff6_field(
        shift, 3, params.dx, *get_boundary_codes(params, 2)
    )
    # beta is shape (3, ni, nj, nk)
    # compute the 6th derivative in each direction

    dissipation_term = params.nu / 64 * params.dx**5 * (
        dbeta_dx1 + dbeta_dx2 + dbeta_dx3
    )
    # compute dissipation term
    
    return dt_beta + dissipation_term
