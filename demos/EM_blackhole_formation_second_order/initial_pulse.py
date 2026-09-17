"""Construct the conformal electromagnetic pulse and physical fields."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import simulation_parameters as settings


def electromagnetic_cylindrical_grid(
    num_radial_points: int,
    num_z_points: int,
    dx: float,
):
    """Return the positive-rho, full-z cell-centred elliptic grid."""

    rho = (jnp.arange(num_radial_points, dtype=jnp.float64) + 0.5) * dx
    z = (
        jnp.arange(num_z_points, dtype=jnp.float64)
        - (num_z_points - 1) / 2.0
    ) * dx
    RHO, Z = jnp.meshgrid(rho, z, indexing="ij")
    grid = jnp.stack((RHO, jnp.zeros_like(RHO), Z), axis=-1)
    return grid[:, None, :, :]


def off_centered_toroidal_electric_seed(
    grid: jnp.ndarray,
    amplitude: float | None = None,
    width: float | None = None,
    radial_center: float | None = None,
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

    amplitude = settings.AMPLITUDE if amplitude is None else amplitude
    width = settings.WIDTH if width is None else width
    radial_center = settings.RADIAL_CENTER if radial_center is None else radial_center
    return _toroidal_electric_seed(grid, amplitude, width, radial_center)


@jax.jit
def _toroidal_electric_seed(grid, amplitude, width, radial_center):
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


def initial_pulse(grid, amplitude=None, width=None, radial_center=None):
    """Return conformal electric and zero magnetic vectors on ``grid``."""

    electric = off_centered_toroidal_electric_seed(
        grid, amplitude, width, radial_center
    )
    return electric, jnp.zeros_like(electric)
