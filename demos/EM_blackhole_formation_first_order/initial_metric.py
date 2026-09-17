"""Solve the electromagnetic Hamiltonian constraint for initial BSSN data."""

import math
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import simulation_parameters as settings
from jax.scipy.sparse.linalg import cg

from JAX_BSSN.bssn.variables import BSSNVariables
from JAX_BSSN.cartoon.axisymmetry import compact_axisymmetric_state


@jax.jit
def electromagnetic_hamiltonian_source(
    psi: jnp.ndarray,
    conformal_field_squared: jnp.ndarray,
) -> jnp.ndarray:
    """Return ``S_EM(psi) = pi bar(E^2 + B^2) psi^-3``."""

    return jnp.pi * conformal_field_squared * psi**-3


@jax.jit
def linearized_electromagnetic_hamiltonian_source(
    delta_u: jnp.ndarray,
    psi_background: jnp.ndarray,
    conformal_field_squared: jnp.ndarray,
) -> jnp.ndarray:
    """Return ``delta S_EM = -3 pi bar(E^2+B^2) psi^-4 delta_u``."""

    return (
        -3.0
        * jnp.pi
        * conformal_field_squared
        * psi_background**-4
        * delta_u
    )


def _radial_weights(size, dx, dtype):
    centers = (jnp.arange(size, dtype=dtype) + 0.5) * dx
    lower = jnp.arange(size, dtype=dtype) * dx
    return centers, lower, lower + dx


@jax.jit
def multipole_outer_boundary(u, conformal_field_squared, dx):
    """Exterior l=0,2,4 Green-function boundary for localized axisymmetric data.

    Delta u=-S implies u=sum_l Q_l P_l(cos(theta))/r^(l+1), with
    Q_l=(1/2) integral S r^l P_l rho d rho dz. The Gaussian source must be
    negligible outside the smallest boundary radius. Odd multipoles vanish
    for the equatorially symmetric demo family.
    """
    rho=(jnp.arange(u.shape[0],dtype=u.dtype)+.5)[:,None]*dx
    z=(jnp.arange(u.shape[1],dtype=u.dtype)-(u.shape[1]-1)/2)[None,:]*dx
    radius=jnp.sqrt(rho**2+z**2)
    cosine=z/radius
    source=electromagnetic_hamiltonian_source(1+u,conformal_field_squared)
    boundary=jnp.zeros_like(u)
    for ell,legendre in ((0,jnp.ones_like(radius)),
                         (2,(3*cosine**2-1)/2),
                         (4,(35*cosine**4-30*cosine**2+3)/8)):
        moment=.5*dx**2*jnp.sum(source*radius**ell*legendre*rho)
        boundary=boundary+moment*legendre/radius**(ell+1)
    result=u.at[-1,:].set(boundary[-1,:])
    return result.at[:,0].set(boundary[:,0]).at[:,-1].set(boundary[:,-1])


@jax.jit
def _boundary_laplacian(u,dx):
    """Contribution of fixed outer values to the rho-weighted active operator."""
    rho,_,upper=_radial_weights(u.shape[0]-1,dx,u.dtype)
    values=jnp.zeros_like(u[:-1,1:-1])
    values=values.at[-1,:].add(upper[-1]*u[-1,1:-1]/dx**2)
    values=values.at[:,0].add(rho*u[:-1,0]/dx**2)
    return values.at[:,-1].add(rho*u[:-1,-1]/dx**2)


@jax.jit
def linearized_electromagnetic_hamiltonian_operator(
    delta_u: jnp.ndarray,
    psi_background: jnp.ndarray,
    conformal_field_squared: jnp.ndarray,
    dx: float,
) -> jnp.ndarray:
    """Apply the symmetric cylindrical Newton operator on active cells.

    ``delta_u`` contains ``rho=dx/2,...,rho_max-3dx/2`` and excludes the two
    fixed-z boundary rows.  Multiplication of the continuum equation by rho
    puts the radial Laplacian in flux form and makes this cell-centred
    discretization symmetric under the ordinary Euclidean inner product.
    """

    num_radial_active = delta_u.shape[0]
    volume_weight, radial_lower, radial_upper = _radial_weights(
        num_radial_active, dx, delta_u.dtype
    )

    full_delta_u = jnp.pad(delta_u, ((0, 1), (1, 1)))
    center = full_delta_u[:-1, 1:-1]
    radial_forward = full_delta_u[1:, 1:-1]
    radial_backward = jnp.concatenate((center[:1], center[:-1]), axis=0)

    radial_flux_divergence = (
        radial_upper[:, None] * (radial_forward - center)
        - radial_lower[:, None] * (center - radial_backward)
    ) / dx**2
    z_flux_divergence = volume_weight[:, None] * (
        (full_delta_u[:-1, 2:] - center)
        - (center - full_delta_u[:-1, :-2])
    ) / dx**2

    linearized_source = linearized_electromagnetic_hamiltonian_source(
        delta_u,
        psi_background,
        conformal_field_squared,
    )

    return (
        radial_flux_divergence
        + z_flux_divergence
        + volume_weight[:, None] * linearized_source
    )


@jax.jit
def electromagnetic_hamiltonian_residual(
    u: jnp.ndarray,
    conformal_field_squared: jnp.ndarray,
    dx: float,
) -> jnp.ndarray:
    """Evaluate the conservative cylindrical residual on active cells."""

    u_interior = u[:-1, 1:-1]
    psi_interior = 1.0 + u_interior
    field_squared_interior = conformal_field_squared[:-1, 1:-1]
    source = electromagnetic_hamiltonian_source(
        psi_interior, field_squared_interior
    )
    zero_field = jnp.zeros_like(field_squared_interior)
    differential_operator = linearized_electromagnetic_hamiltonian_operator(
        u_interior,
        jnp.ones_like(psi_interior),
        zero_field,
        dx,
    )
    volume_weight, _, _ = _radial_weights(u_interior.shape[0], dx, u.dtype)

    return differential_operator + _boundary_laplacian(u,dx) + volume_weight[:, None] * source


def solve_electromagnetic_conformal_factor(
    conformal_field_squared: jnp.ndarray,
    dx: float,
    newton_tolerance: float | None = None,
    max_newton_iterations: int | None = None,
    cg_tolerance: float | None = None,
    max_cg_iterations: int | None = None,
    verbose: bool = False,
    outer_boundary: str | None = None,
):
    """Solve the time-symmetric Einstein--Maxwell Hamiltonian constraint.

    The conformal metric is flat, ``K_ij=0``, and the physical Maxwell
    vectors obey ``E^i=psi^-6 bar(E)^i`` and likewise for ``B``.  In
    Lorentz--Heaviside units the constraint is

    ``Delta psi + pi (bar(E)^2 + bar(B)^2) psi^-3 = 0``.

    The input and returned fields live directly on the positive-rho,
    cell-centred Cartoon plane.  The numerical unknown is ``u = psi - 1``.
    By default its outer-rho row and both outer-z rows are fixed to zero.
    ``outer_boundary='multipole'`` instead updates these values using the
    exterior l=0,2,4 Green expansion of the current localized source. That
    update is a fixed-point boundary iteration within the interior Newton
    solve; the final residual includes the nonzero boundary values. The axis
    has regular zero flux. Each interior Newton correction uses CG on the
    negative, rho-weighted conservative operator.

    Returns
    -------
    psi : jax.Array
        The conformal factor on the positive-rho cylindrical grid.
    u : jax.Array
        The regular correction, including its zero-valued outer boundaries.
    residual_history : list[float]
        Interior RMS residual before each correction and after convergence.
    """

    newton_tolerance = (
        settings.NEWTON_TOLERANCE if newton_tolerance is None else newton_tolerance
    )
    max_newton_iterations = (
        settings.MAX_NEWTON_ITERATIONS
        if max_newton_iterations is None else max_newton_iterations
    )
    cg_tolerance = settings.CG_TOLERANCE if cg_tolerance is None else cg_tolerance
    max_cg_iterations = (
        settings.MAX_CG_ITERATIONS if max_cg_iterations is None else max_cg_iterations
    )
    outer_boundary = outer_boundary or getattr(settings, 'INITIAL_OUTER_BOUNDARY', 'dirichlet')
    if outer_boundary not in ('dirichlet','multipole'):
        raise ValueError('outer_boundary must be dirichlet or multipole')

    field_squared_interior = conformal_field_squared[:-1, 1:-1]
    u_n = jnp.zeros_like(field_squared_interior)
    residual_history = []
    volume_weight, _, _ = _radial_weights(u_n.shape[0], dx, u_n.dtype)

    for iteration in range(max_newton_iterations + 1):
        u_full=jnp.pad(u_n,((0,1),(1,1)))
        if outer_boundary == 'multipole':
            u_full=multipole_outer_boundary(u_full,conformal_field_squared,dx)
        psi_n = 1.0 + u_n
        source_n = electromagnetic_hamiltonian_source(
            psi_n, field_squared_interior
        )
        zero_field = jnp.zeros_like(field_squared_interior)
        differential_operator = linearized_electromagnetic_hamiltonian_operator(
            u_n,
            jnp.ones_like(psi_n),
            zero_field,
            dx,
        )
        residual_n = differential_operator + _boundary_laplacian(u_full,dx) + volume_weight[:, None] * source_n

        residual_rms = float(jnp.sqrt(jnp.mean(residual_n**2)))
        residual_history.append(residual_rms)

        if verbose:
            print(
                f"Newton iteration {iteration:2d}: "
                f"RMS residual = {residual_rms:.3e}"
            )

        if residual_rms <= newton_tolerance:
            break

        if iteration == max_newton_iterations:
            raise RuntimeError(
                "Einstein-Maxwell Hamiltonian solve did not converge: "
                f"RMS residual {residual_rms:.3e} exceeds "
                f"{newton_tolerance:.3e}."
            )

        def negative_linearized_operator(delta_u):
            return -linearized_electromagnetic_hamiltonian_operator(
                delta_u,
                psi_n,
                field_squared_interior,
                dx,
            )

        _, radial_lower, radial_upper = _radial_weights(u_n.shape[0], dx, u_n.dtype)
        diagonal = (
            (radial_lower + radial_upper)[:, None]
            + 2.0 * volume_weight[:, None]
        ) / dx**2 + volume_weight[:, None] * (
            3 * jnp.pi * field_squared_interior * psi_n**-4
        )
        # -delta(F) is SPD for psi > 0 and non-negative field energy.
        delta_u, _ = cg(
            negative_linearized_operator,
            residual_n,
            tol=cg_tolerance,
            atol=0.0,
            maxiter=max_cg_iterations,
            M=lambda value: value / diagonal,
        )
        linear_residual = negative_linearized_operator(delta_u) - residual_n
        relative_residual = float(jnp.linalg.norm(linear_residual) / jnp.maximum(jnp.linalg.norm(residual_n), 1e-300))
        if not math.isfinite(relative_residual) or relative_residual > max(10 * cg_tolerance, 1e-8):
            raise RuntimeError(f"Hamiltonian CG did not converge: relative residual {relative_residual:.3e}")
        u_n = u_n + delta_u
        jax.block_until_ready(u_n)

    u = u_full
    psi = 1.0 + u

    return psi, u, residual_history


def _flat_conformal_bssn_plane(psi_plane):
    """Map a solved conformal factor to time-symmetric BSSN variables."""

    W = psi_plane**-2
    shape = W.shape
    conformal_metric = (
        jnp.eye(3, dtype=W.dtype)[:, :, None, None, None]
        * jnp.ones((3, 3) + shape, dtype=W.dtype)
    )
    return BSSNVariables(
        conformal_metric=conformal_metric,
        conformal_factor=W,
        traceless_K=jnp.zeros_like(conformal_metric),
        trace_K=jnp.zeros(shape, dtype=W.dtype),
        conformal_connection=jnp.zeros((3,) + shape, dtype=W.dtype),
        lapse=W,
        shift=jnp.zeros((3,) + shape, dtype=W.dtype),
    )


def initial_metric(
    conformal_field_squared, dx, *, newton_tolerance=None,
    max_newton_iterations=None, cg_tolerance=None, max_cg_iterations=None,
    verbose=False,
):
    """Return ``(bssn, psi, u, residual_history)`` for positive-rho pulse data.

    ``psi`` and ``u`` have shape (num_rho, num_z); BSSN fields include the
    four compact radial ghost cells required by the evolution solver.
    """

    psi, u, residual_history = solve_electromagnetic_conformal_factor(
        conformal_field_squared, dx,
        newton_tolerance=newton_tolerance,
        max_newton_iterations=max_newton_iterations,
        cg_tolerance=cg_tolerance,
        max_cg_iterations=max_cg_iterations,
        verbose=verbose,
    )
    psi_plane = psi[:, None, :]
    signed_psi_plane = jnp.concatenate(
        (jnp.flip(psi_plane, axis=0), psi_plane), axis=0
    )
    bssn = compact_axisymmetric_state(
        _flat_conformal_bssn_plane(signed_psi_plane)
    )
    return bssn, psi, u, residual_history
