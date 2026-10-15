"""Bowen-York puncture initial data on Cartesian and axisymmetric grids."""

import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import bicgstab, cg


def cell_centered_grid(num_points=48, half_width=12.0):
    """Return a cubic cell-centered grid and its uniform grid spacing."""

    dx = 2.0 * half_width / num_points
    axis = -half_width + (jnp.arange(num_points, dtype=jnp.float64) + 0.5) * dx
    X, Y, Z = jnp.meshgrid(axis, axis, axis, indexing="ij")
    grid = jnp.stack((X, Y, Z), axis=-1)

    return grid, dx


def axisymmetric_cell_centered_grid(
    num_radial_points=48,
    radial_extent=12.0,
):
    """Return a positive-rho cell-centered meridional grid and its spacing.

    The returned grid has shape ``(Nrho, 1, 2 * Nrho, 3)``.  Its singleton
    dimension is the Cartesian ``y = 0`` plane used by the axisymmetric
    Cartoon evolution, while ``rho`` is stored as the nonnegative ``x``
    coordinate.
    """

    if not isinstance(num_radial_points, int) or num_radial_points < 2:
        raise ValueError("num_radial_points must be an integer of at least two")
    if not math.isfinite(radial_extent) or radial_extent <= 0.0:
        raise ValueError("radial_extent must be finite and positive")

    dx = radial_extent / num_radial_points
    dtype = jnp.float64
    rho = (jnp.arange(num_radial_points, dtype=dtype) + 0.5) * dx
    y = jnp.zeros(1, dtype=dtype)
    z = -radial_extent + (
        jnp.arange(2 * num_radial_points, dtype=dtype) + 0.5
    ) * dx
    R, Y, Z = jnp.meshgrid(rho, y, z, indexing="ij")
    grid = jnp.stack((R, Y, Z), axis=-1)

    return grid, dx


def brill_lindquist_conformal_factor(masses, positions, grid):
    """Compute ``psi_BL = 1 + sum_a m_a / (2 r_a)``."""

    masses = jnp.asarray(masses, dtype=grid.dtype)
    positions = jnp.asarray(positions, dtype=grid.dtype)

    displacement = grid[None, ...] - positions[:, None, None, None, :]
    radius_squared = jnp.einsum("a...i,a...i->a...", displacement, displacement)
    radius = jnp.sqrt(radius_squared)
    puncture_terms = masses[:, None, None, None] / (2.0 * radius)

    return 1.0 + jnp.einsum("a...->...", puncture_terms)


def bowen_york_extrinsic_curvature(momenta, positions, grid):
    """Compute the conformal Bowen-York curvature for boosted punctures."""

    momenta = jnp.asarray(momenta, dtype=grid.dtype)
    positions = jnp.asarray(positions, dtype=grid.dtype)

    displacement = grid[None, ...] - positions[:, None, None, None, :]
    radius_squared = jnp.einsum("a...i,a...i->a...", displacement, displacement)
    radius = jnp.sqrt(radius_squared)
    normal = displacement / radius[..., None]

    momentum_dot_normal = jnp.einsum("ai,a...i->a...", momenta, normal)
    momentum_normal = jnp.einsum("ai,a...j->a...ij", momenta, normal)
    normal_momentum = jnp.einsum("a...i,aj->a...ij", normal, momenta)
    normal_normal = jnp.einsum("a...i,a...j->a...ij", normal, normal)

    identity = jnp.eye(3, dtype=grid.dtype)
    transverse_metric = identity - normal_normal
    curvature = 3.0 / (2.0 * radius[..., None, None] ** 2) * (
        momentum_normal
        + normal_momentum
        - transverse_metric * momentum_dot_normal[..., None, None]
    )

    return jnp.einsum("a...ij->...ij", curvature)


@jax.jit
def contract_conformal_curvature(conformal_curvature):
    """Compute the flat-metric contraction ``A_tilde_ij A_tilde^ij``."""

    return jnp.einsum(
        "...ij,...ij->...",
        conformal_curvature,
        conformal_curvature,
    )


@jax.jit
def laplacian(interior_field, dx):
    """Apply the seven-point Laplacian with zero values outside the interior."""

    field = jnp.pad(interior_field, 1)

    return (
        field[2:, 1:-1, 1:-1]
        + field[:-2, 1:-1, 1:-1]
        + field[1:-1, 2:, 1:-1]
        + field[1:-1, :-2, 1:-1]
        + field[1:-1, 1:-1, 2:]
        + field[1:-1, 1:-1, :-2]
        - 6.0 * interior_field
    ) / dx**2


@jax.jit
def axisymmetric_laplacian(interior_field, dx):
    """Apply the centered axisymmetric Laplacian with zero outer values.

    ``interior_field`` contains all unknown positive-rho cells except the
    fixed outer radial cell, and excludes the two fixed outer z cells.  At
    the axis, even reflection supplies the regular ghost value
    ``u(-dx/2, z) = u(dx/2, z)``.
    """

    field = jnp.pad(interior_field, ((0, 1), (1, 1)))
    center = field[:-1, 1:-1]
    radial_plus = field[1:, 1:-1]
    radial_minus = jnp.concatenate(
        (field[:1, 1:-1], field[:-2, 1:-1]),
        axis=0,
    )
    rho = (jnp.arange(interior_field.shape[0], dtype=field.dtype) + 0.5) * dx

    radial_second = (radial_plus - 2.0 * center + radial_minus) / dx**2
    radial_first = (radial_plus - radial_minus) / (
        2.0 * rho[:, None] * dx
    )
    axial_second = (
        field[:-1, 2:] + field[:-1, :-2] - 2.0 * center
    ) / dx**2

    return radial_second + radial_first + axial_second


@jax.jit
def hamiltonian_source(psi, curvature_squared):
    """Compute ``S(psi) = A_tilde_ij A_tilde^ij psi^-7 / 8``."""

    return 1.0 / 8.0 * curvature_squared * psi**-7


@jax.jit
def linearized_hamiltonian_source(delta_u, psi_background, curvature_squared):
    """Compute ``delta S = -7 A_tilde^2 psi_background^-8 delta_u / 8``."""

    return -7.0 / 8.0 * curvature_squared * psi_background**-8 * delta_u


@jax.jit
def linearized_hamiltonian_operator(
    delta_u,
    psi_background,
    curvature_squared,
    dx,
):
    """Apply ``delta F = Laplacian(delta_u) + delta S``."""

    linearized_source = linearized_hamiltonian_source(
        delta_u,
        psi_background,
        curvature_squared,
    )

    return laplacian(delta_u, dx) + linearized_source


@jax.jit
def axisymmetric_linearized_hamiltonian_operator(
    delta_u,
    psi_background,
    curvature_squared,
    dx,
):
    """Apply the Newton Jacobian on the active axisymmetric cells."""

    linearized_source = linearized_hamiltonian_source(
        delta_u,
        psi_background,
        curvature_squared,
    )

    return axisymmetric_laplacian(delta_u, dx) + linearized_source


@jax.jit
def hamiltonian_residual(u, psi_bl, conformal_curvature, dx):
    """Evaluate the nonlinear Hamiltonian residual at interior grid points."""

    u_interior = u[1:-1, 1:-1, 1:-1]
    psi_interior = psi_bl[1:-1, 1:-1, 1:-1] + u_interior

    curvature_squared = contract_conformal_curvature(conformal_curvature)
    curvature_squared_interior = curvature_squared[1:-1, 1:-1, 1:-1]
    source = hamiltonian_source(psi_interior, curvature_squared_interior)

    return laplacian(u_interior, dx) + source


@jax.jit
def axisymmetric_hamiltonian_residual(
    u,
    psi_bl,
    conformal_curvature,
    dx,
):
    """Evaluate the Hamiltonian residual on active meridional-grid cells."""

    u_interior = u[:-1, 0, 1:-1]
    psi_interior = psi_bl[:-1, 0, 1:-1] + u_interior
    curvature_squared = contract_conformal_curvature(conformal_curvature)
    curvature_squared_interior = curvature_squared[:-1, 0, 1:-1]
    source = hamiltonian_source(psi_interior, curvature_squared_interior)

    return axisymmetric_laplacian(u_interior, dx) + source


def _validate_axisymmetric_solver_inputs(masses, positions, momenta, grid, dx):
    """Validate the geometry required by the axisymmetric puncture solve."""

    masses_np = np.asarray(masses)
    positions_np = np.asarray(positions)
    momenta_np = np.asarray(momenta)
    grid_np = np.asarray(grid)

    if masses_np.ndim != 1 or masses_np.size == 0:
        raise ValueError("masses must be a nonempty one-dimensional array")
    expected_shape = (masses_np.size, 3)
    if positions_np.shape != expected_shape:
        raise ValueError(f"positions must have shape {expected_shape}")
    if momenta_np.shape != expected_shape:
        raise ValueError(f"momenta must have shape {expected_shape}")
    if not np.all(np.isfinite(masses_np)) or np.any(masses_np <= 0.0):
        raise ValueError("masses must be finite and positive")
    if not np.all(np.isfinite(positions_np)):
        raise ValueError("positions must be finite")
    if not np.all(np.isfinite(momenta_np)):
        raise ValueError("momenta must be finite")
    if not np.allclose(positions_np[:, :2], 0.0, rtol=0.0, atol=1.0e-14):
        raise ValueError("axisymmetric punctures must lie on the z axis")
    if not np.allclose(momenta_np[:, :2], 0.0, rtol=0.0, atol=1.0e-14):
        raise ValueError("axisymmetric puncture momenta must be parallel to z")

    if grid_np.ndim != 4 or grid_np.shape[1] != 1 or grid_np.shape[-1] != 3:
        raise ValueError("grid must have shape (Nrho, 1, Nz, 3)")
    if grid_np.shape[0] < 2 or grid_np.shape[2] < 3:
        raise ValueError("grid requires at least two rho and three z cells")
    if not math.isfinite(float(dx)) or float(dx) <= 0.0:
        raise ValueError("dx must be finite and positive")
    if not np.all(np.isfinite(grid_np)):
        raise ValueError("grid coordinates must be finite")

    rho = grid_np[:, 0, 0, 0]
    z = grid_np[0, 0, :, 2]
    expected_rho = (np.arange(grid_np.shape[0]) + 0.5) * float(dx)
    if not np.allclose(rho, expected_rho):
        raise ValueError("grid rho coordinates must be positive half-cell centers")
    if not np.allclose(grid_np[..., 0], rho[:, None, None]):
        raise ValueError("grid rho coordinates must be constant along z")
    if not np.allclose(grid_np[..., 1], 0.0):
        raise ValueError("axisymmetric grid must lie on y=0")
    if not np.allclose(grid_np[..., 2], z[None, None, :]):
        raise ValueError("grid z coordinates must be constant along rho")
    if not np.allclose(np.diff(z), float(dx)):
        raise ValueError("grid z spacing must equal dx")


def solve_conformal_factor(
    masses,
    positions,
    momenta,
    grid,
    dx,
    newton_tolerance=1.0e-10,
    max_newton_iterations=12,
    cg_tolerance=1.0e-10,
    max_cg_iterations=1000,
    verbose=False,
):
    """Solve the puncture Hamiltonian constraint with Newton iterations.

    The singular Brill-Lindquist part is analytic. The numerical unknown is
    the regular correction ``u``, fixed to zero on all six outer grid layers.
    Each Newton correction is obtained with matrix-free conjugate gradients
    applied to the negative linearized Hamiltonian operator.

    Returns
    -------
    psi : jax.Array
        Total conformal factor ``psi_bl + u`` on the full grid.
    u : jax.Array
        Regular correction, including its zero-valued outer boundary.
    residual_history : list[float]
        Interior RMS residual before each Newton correction and after the
        final correction.
    """

    psi_bl = brill_lindquist_conformal_factor(masses, positions, grid)
    conformal_curvature = bowen_york_extrinsic_curvature(momenta, positions, grid)
    curvature_squared = contract_conformal_curvature(conformal_curvature)

    psi_bl_interior = psi_bl[1:-1, 1:-1, 1:-1]
    curvature_squared_interior = curvature_squared[1:-1, 1:-1, 1:-1]

    u_n = jnp.zeros_like(psi_bl_interior)
    residual_history = []

    for iteration in range(max_newton_iterations + 1):
        # Re-linearize about psi_n = psi_BL + u_n after every correction.
        psi_n = psi_bl_interior + u_n
        source_n = hamiltonian_source(psi_n, curvature_squared_interior)
        residual_n = laplacian(u_n, dx) + source_n

        residual_rms = float(jnp.sqrt(jnp.mean(residual_n**2)))
        residual_history.append(residual_rms)

        if verbose:
            print(f"Newton iteration {iteration:2d}: RMS residual = {residual_rms:.3e}")

        if residual_rms <= newton_tolerance:
            break

        if iteration == max_newton_iterations:
            raise RuntimeError(
                "Newton-Raphson solve did not converge: "
                f"RMS residual {residual_rms:.3e} exceeds {newton_tolerance:.3e}."
            )

        def negative_linearized_operator(delta_u):
            return -linearized_hamiltonian_operator(
                delta_u,
                psi_n,
                curvature_squared_interior,
                dx,
            )

        # -[Laplacian(delta_u) + delta S_n] = F(u_n) is SPD for psi_n > 0.
        delta_u, _ = cg(
            negative_linearized_operator,
            residual_n,
            tol=cg_tolerance,
            atol=0.0,
            maxiter=max_cg_iterations,
        )
        u_n = u_n + delta_u
        jax.block_until_ready(u_n)

    u = jnp.pad(u_n, 1)
    psi = psi_bl + u

    return psi, u, residual_history


def solve_axisymmetric_conformal_factor(
    masses,
    positions,
    momenta,
    grid,
    dx,
    newton_tolerance=1.0e-10,
    max_newton_iterations=12,
    linear_tolerance=1.0e-10,
    max_linear_iterations=1000,
    verbose=False,
):
    """Solve Bowen-York puncture data on a cell-centered meridional grid.

    Punctures must lie on the z axis and their momenta must be parallel to
    that axis.  The singular Brill-Lindquist factor is evaluated analytically;
    Newton-Raphson solves only for its regular correction ``u``.  The outermost
    rho cell and both outermost z cells are fixed to zero correction.

    Returns
    -------
    psi : jax.Array
        Total conformal factor on the ``(Nrho, 1, Nz)`` grid.
    u : jax.Array
        Regular correction, including its zero-valued outer boundary cells.
    residual_history : list[float]
        Active-cell RMS residual before each Newton correction and after the
        final correction.
    """

    _validate_axisymmetric_solver_inputs(
        masses,
        positions,
        momenta,
        grid,
        dx,
    )
    for name, value in (
        ("newton_tolerance", newton_tolerance),
        ("linear_tolerance", linear_tolerance),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not isinstance(max_newton_iterations, int) or max_newton_iterations < 0:
        raise ValueError("max_newton_iterations must be a nonnegative integer")
    if not isinstance(max_linear_iterations, int) or max_linear_iterations < 1:
        raise ValueError("max_linear_iterations must be a positive integer")

    psi_bl = brill_lindquist_conformal_factor(masses, positions, grid)
    conformal_curvature = bowen_york_extrinsic_curvature(
        momenta,
        positions,
        grid,
    )
    curvature_squared = contract_conformal_curvature(conformal_curvature)

    psi_bl_interior = psi_bl[:-1, 0, 1:-1]
    curvature_squared_interior = curvature_squared[:-1, 0, 1:-1]
    u_n = jnp.zeros_like(psi_bl_interior)
    residual_history = []

    for iteration in range(max_newton_iterations + 1):
        psi_n = psi_bl_interior + u_n
        source_n = hamiltonian_source(psi_n, curvature_squared_interior)
        residual_n = axisymmetric_laplacian(u_n, dx) + source_n
        residual_rms = float(jnp.sqrt(jnp.mean(residual_n**2)))
        residual_history.append(residual_rms)

        if verbose:
            print(
                f"Newton iteration {iteration:2d}: "
                f"RMS residual = {residual_rms:.3e}"
            )

        if not math.isfinite(residual_rms):
            raise RuntimeError("Newton-Raphson solve produced a nonfinite residual")
        if residual_rms <= newton_tolerance:
            break
        if iteration == max_newton_iterations:
            raise RuntimeError(
                "Newton-Raphson solve did not converge: "
                f"RMS residual {residual_rms:.3e} exceeds "
                f"{newton_tolerance:.3e}."
            )

        def negative_linearized_operator(delta_u):
            return -axisymmetric_linearized_hamiltonian_operator(
                delta_u,
                psi_n,
                curvature_squared_interior,
                dx,
            )

        delta_u, _ = bicgstab(
            negative_linearized_operator,
            residual_n,
            tol=linear_tolerance,
            atol=0.0,
            maxiter=max_linear_iterations,
        )
        u_n = u_n + delta_u
        jax.block_until_ready(u_n)
        if not bool(jnp.all(jnp.isfinite(u_n))):
            raise RuntimeError("Newton-Raphson correction is nonfinite")
        if not bool(jnp.all(psi_bl_interior + u_n > 0.0)):
            raise RuntimeError("Newton-Raphson correction made psi nonpositive")

    u = jnp.pad(u_n, ((0, 1), (1, 1)))[:, None, :]
    psi = psi_bl + u

    return psi, u, residual_history
