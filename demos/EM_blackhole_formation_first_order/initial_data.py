"""Conformally solved Einstein--Maxwell initial data."""

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg


@jax.jit
def off_centered_toroidal_electric_seed(
    grid: jnp.ndarray,
    amplitude: float = 0.08,
    width: float = 1.0,
    radial_center: float = 3.0,
) -> jnp.ndarray:
    """Return the conformal electric vector for the BGH dipole family.

    The literature field is the contravariant spherical-polar component

    ``bar(E_G)^phi = -4 eta G(r) / sigma^2``.

    Baumgarte, Gundlach, and Hilditch use Gaussian electromagnetic units.
    The wave solver uses Lorentz--Heaviside fields, so the returned field is
    divided by ``sqrt(4 pi)``.  This preserves the numerical stress-energy
    and lets ``amplitude`` retain the literature normalization.

    The Cartesian azimuthal basis is ``partial_phi = (-y, x, 0)``.  The
    returned component-leading array is therefore the conformal
    contravariant vector ``bar(E)^i`` in Cartesian coordinates.
    """

    radius = jnp.sqrt(jnp.einsum("...i,...i->...", grid, grid))
    gaussian = jnp.exp(-((radius - radial_center) / width) ** 2)
    gaussian = gaussian + jnp.exp(-((radius + radial_center) / width) ** 2)

    electric_phi = (
        -4.0
        * amplitude
        * gaussian
        / (jnp.sqrt(4.0 * jnp.pi) * width**2)
    )
    azimuthal_vector = jnp.stack(
        (-grid[..., 1], grid[..., 0], jnp.zeros_like(radius)), axis=0
    )

    return electric_phi[None, ...] * azimuthal_vector


@jax.jit
def contract_conformal_electromagnetic_fields(
    conformal_electric_field: jnp.ndarray,
    conformal_magnetic_field: jnp.ndarray,
) -> jnp.ndarray:
    """Return ``bar(E)_i bar(E)^i + bar(B)_i bar(B)^i`` for flat data."""

    electric_squared = jnp.einsum(
        "i...,i...->...", conformal_electric_field, conformal_electric_field
    )
    magnetic_squared = jnp.einsum(
        "i...,i...->...", conformal_magnetic_field, conformal_magnetic_field
    )

    return electric_squared + magnetic_squared


@jax.jit
def conformal_vector_to_physical_covector(
    conformal_vector: jnp.ndarray,
    psi: jnp.ndarray,
) -> jnp.ndarray:
    """Convert ``bar(V)^i`` to the stored physical covector ``V_i``.

    For ``gamma_ij = psi^4 delta_ij`` and the Maxwell conformal scaling
    ``V^i = psi^-6 bar(V)^i``, the physical covector is
    ``V_i = psi^-2 bar(V)_i``.  Flat conformal raising and lowering leaves
    the Cartesian component values unchanged.
    """

    return psi[None, ...] ** -2 * conformal_vector


@jax.jit
def conformal_vector_to_physical_contravariant(
    conformal_vector: jnp.ndarray,
    psi: jnp.ndarray,
) -> jnp.ndarray:
    """Convert ``bar(V)^i`` to the physical vector ``V^i``.

    The source-free Maxwell conformal scaling is
    ``V^i = psi^-6 bar(V)^i``.  This form is used to initialize the
    contravariant displacement and magnetic fields on the first-order Yee
    grid.
    """

    return psi[None, ...] ** -6 * conformal_vector


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
    rho = (jnp.arange(num_radial_active, dtype=delta_u.dtype) + 0.5) * dx
    rho_minus = jnp.arange(num_radial_active, dtype=delta_u.dtype) * dx
    rho_plus = rho_minus + dx

    full_delta_u = jnp.pad(delta_u, ((0, 1), (1, 1)))
    center = full_delta_u[:-1, 1:-1]
    radial_forward = full_delta_u[1:, 1:-1]
    radial_backward = jnp.concatenate((center[:1], center[:-1]), axis=0)

    radial_flux_divergence = (
        rho_plus[:, None] * (radial_forward - center)
        - rho_minus[:, None] * (center - radial_backward)
    ) / dx**2
    z_flux_divergence = rho[:, None] * (
        full_delta_u[:-1, 2:]
        - 2.0 * center
        + full_delta_u[:-1, :-2]
    ) / dx**2

    linearized_source = linearized_electromagnetic_hamiltonian_source(
        delta_u,
        psi_background,
        conformal_field_squared,
    )

    return (
        radial_flux_divergence
        + z_flux_divergence
        + rho[:, None] * linearized_source
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
    rho = (jnp.arange(u_interior.shape[0], dtype=u.dtype) + 0.5) * dx

    return differential_operator + rho[:, None] * source


def solve_electromagnetic_conformal_factor(
    conformal_field_squared: jnp.ndarray,
    dx: float,
    newton_tolerance: float = 1.0e-10,
    max_newton_iterations: int = 12,
    cg_tolerance: float = 1.0e-10,
    max_cg_iterations: int = 1000,
    verbose: bool = False,
):
    """Solve the time-symmetric Einstein--Maxwell Hamiltonian constraint.

    The conformal metric is flat, ``K_ij=0``, and the physical Maxwell
    vectors obey ``E^i=psi^-6 bar(E)^i`` and likewise for ``B``.  In
    Lorentz--Heaviside units the constraint is

    ``Delta psi + pi (bar(E)^2 + bar(B)^2) psi^-3 = 0``.

    The input and returned fields live directly on the positive-rho,
    cell-centred Cartoon plane.  The numerical unknown is ``u = psi - 1``.
    Its outer-rho row and both outer-z rows are fixed to zero, while the axis
    has the regular zero-flux condition.  Each Newton correction uses CG on
    the negative, rho-weighted conservative operator.

    Returns
    -------
    psi : jax.Array
        The conformal factor on the positive-rho cylindrical grid.
    u : jax.Array
        The regular correction, including its zero-valued outer boundaries.
    residual_history : list[float]
        Interior RMS residual before each correction and after convergence.
    """

    field_squared_interior = conformal_field_squared[:-1, 1:-1]
    u_n = jnp.zeros_like(field_squared_interior)
    residual_history = []
    rho = (jnp.arange(u_n.shape[0], dtype=u_n.dtype) + 0.5) * dx

    for iteration in range(max_newton_iterations + 1):
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
        residual_n = differential_operator + rho[:, None] * source_n

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

        # -delta(F) is SPD for psi > 0 and non-negative field energy.
        delta_u, _ = cg(
            negative_linearized_operator,
            residual_n,
            tol=cg_tolerance,
            atol=0.0,
            maxiter=max_cg_iterations,
        )
        u_n = u_n + delta_u
        jax.block_until_ready(u_n)

    u = jnp.pad(u_n, ((0, 1), (1, 1)))
    psi = 1.0 + u

    return psi, u, residual_history


__all__ = [
    "conformal_vector_to_physical_covector",
    "conformal_vector_to_physical_contravariant",
    "contract_conformal_electromagnetic_fields",
    "electromagnetic_hamiltonian_residual",
    "electromagnetic_hamiltonian_source",
    "linearized_electromagnetic_hamiltonian_operator",
    "linearized_electromagnetic_hamiltonian_source",
    "off_centered_toroidal_electric_seed",
    "solve_electromagnetic_conformal_factor",
]
