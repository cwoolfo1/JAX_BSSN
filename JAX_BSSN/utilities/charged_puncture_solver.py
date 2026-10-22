"""Analytic time-symmetric initial data for one charged puncture.

The construction follows Eqs. (2.8)--(2.13) of arXiv:1311.6483v2.  For a
single puncture the regular correction is exactly ``u = 1``.  Mass and charge
use the paper's geometric/Gaussian convention; the electric field is converted
to JAX_BSSN's rationalized convention by ``1 / sqrt(4 pi)``.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

from JAX_BSSN.bssn import BSSNVariables
from JAX_BSSN.EM.first_order import (
    D_FIELD_LOCATIONS,
    EinsteinMaxwellVariables,
    initialize_densitized_maxwell_state,
)


def _validate_inputs(mass, charge, position, params, em_grid):
    if not jax.config.x64_enabled:
        raise ValueError("charged puncture data require jax_enable_x64=True")
    try:
        mass = float(mass)
        charge = float(charge)
    except (TypeError, ValueError) as error:
        raise ValueError("mass and charge must be finite real scalars") from error
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError("mass must be finite and positive")
    if not math.isfinite(charge) or abs(charge) >= mass:
        raise ValueError("charge must be finite and subextremal: |charge| < mass")

    position = np.asarray(position, dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("position must contain three finite coordinates")
    if len(em_grid.shape) != 3 or any(size < 1 for size in em_grid.shape):
        raise ValueError("em_grid must describe a nonempty three-dimensional grid")
    if not math.isfinite(float(params.dx)) or params.dx <= 0.0:
        raise ValueError("params.dx must be finite and positive")
    if not math.isfinite(float(params.dt)) or params.dt <= 0.0:
        raise ValueError("params.dt must be finite and positive")
    origins = (params.x_min, params.y_min, params.z_min)
    if not all(math.isfinite(float(origin)) for origin in origins):
        raise ValueError("grid origins must be finite")
    return mass, charge, position


def _squared_radius(coordinates, position):
    return sum(
        (coordinate - center) ** 2
        for coordinate, center in zip(coordinates, position)
    )


def single_charged_puncture_state(mass, charge, position, params, em_grid, *,
                                 absorber_width=0.0, absorber_order=4,
                                 absorber_attenuation=1.0e-3):
    """Return analytic Reissner--Nordstrom puncture data and EM history.

    The BSSN variables live on the grid's C nodes.  Each native Maxwell
    component is sampled at its own Yee location.  The native displacement is
    initialized directly as the density ``sqrt(gamma) E^i``; for the conformal
    metric in the paper this is simply ``-partial^i(Q/r) / sqrt(4 pi)``.

    Args:
        mass: Positive bare mass parameter ``M``.
        charge: Subextremal charge parameter ``Q`` in Gaussian units.
        position: Cartesian puncture coordinates.
        params: BSSN parameters defining spacing, timestep, and C-node origin.
        em_grid: Cartesian :class:`~JAX_BSSN.EM.first_order.EMGrid`.

    Returns:
        An initialized :class:`~JAX_BSSN.EM.EinsteinMaxwellVariables` state.
    """
    mass, charge, position = _validate_inputs(
        mass, charge, position, params, em_grid
    )
    dtype = jnp.float64
    position = jnp.asarray(position, dtype=dtype)

    center_coordinates = em_grid.coordinates(params)
    radius_squared = _squared_radius(center_coordinates, position)
    if bool(jnp.any(radius_squared == 0.0)):
        raise ValueError("the puncture must not coincide with a BSSN C node")
    radius = jnp.sqrt(radius_squared)
    psi_squared = (1.0 + mass / (2.0 * radius)) ** 2 - (
        charge / (2.0 * radius)
    ) ** 2
    if not bool(jnp.all(jnp.isfinite(psi_squared) & (psi_squared > 0.0))):
        raise ValueError("charged puncture conformal factor is not positive")
    conformal_factor = 1.0 / psi_squared
    shape = em_grid.shape
    conformal_metric = jnp.broadcast_to(
        jnp.eye(3, dtype=dtype)[:, :, None, None, None], (3, 3) + shape
    )
    scalar_zero = jnp.zeros(shape, dtype=dtype)
    vector_zero = jnp.zeros((3,) + shape, dtype=dtype)
    bssn = BSSNVariables(
        conformal_metric=conformal_metric,
        conformal_factor=conformal_factor,
        traceless_K=jnp.zeros_like(conformal_metric),
        trace_K=scalar_zero,
        conformal_connection=vector_zero,
        lapse=conformal_factor,
        shift=vector_zero,
    )

    normalization = charge / math.sqrt(4.0 * math.pi)
    displacement = []
    for component, location in enumerate(D_FIELD_LOCATIONS):
        coordinates = em_grid.coordinates(params, location, tiled=absorber_width > 0)
        radius_squared = _squared_radius(coordinates, position)
        if bool(jnp.any(radius_squared == 0.0)):
            raise ValueError(
                "the puncture must not coincide with a native Maxwell node"
            )
        density = normalization * (
            coordinates[component] - position[component]
        ) / radius_squared ** 1.5
        native_shape = ((1, 1, 1) + tuple(n + 2 * em_grid.guard_cells for n in em_grid.tile_shape)
                        if absorber_width > 0 else shape)
        displacement.append(jnp.broadcast_to(density, native_shape))
    displacement = tuple(displacement)
    magnetic = tuple(jnp.zeros_like(component) for component in displacement)
    if absorber_width > 0:
        from JAX_BSSN.EM.first_order.absorber import make_reference_sponge
        from JAX_BSSN.EM.first_order.geometry import build_yee_metric
        from JAX_BSSN.EM.first_order.maxwell import initialize_native_state

        absorber = make_reference_sponge(
            em_grid, params, displacement, magnetic, width=absorber_width,
            order=absorber_order, target_attenuation=absorber_attenuation,
        )
        em = initialize_native_state(
            displacement, magnetic, build_yee_metric(bssn, em_grid), params, em_grid, absorber
        )
    else:
        em = initialize_densitized_maxwell_state(
            displacement, magnetic, bssn, params, em_grid
        )
    return EinsteinMaxwellVariables(bssn, em)


__all__ = ["single_charged_puncture_state"]
