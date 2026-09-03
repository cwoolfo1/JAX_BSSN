"""Spherical Cartoon evolution for radial staggered vector densities."""

import jax
import jax.numpy as jnp

from JAX_BSSN.bssn import BSSNParameters
from JAX_BSSN.cartoon.spherical_symmetry import (
    CARTOON_CENTER,
    CARTOON_GHOST_CELLS,
    CARTOON_OUTER_BUFFER_CELLS,
    CARTOON_SUPPORT_SIZE,
    fill_cartoon_ghosts,
    project_cartoon_rhs,
    reconstruct_cartoon_support,
)
from JAX_BSSN.cartoon.interpolation import lagrange6_nonperiodic
from JAX_BSSN.evolution.time_evolve import (
    compute_bssn_rhs_with_matter,
    enforce_algebraic_constraints,
)

from JAX_BSSN.EM.first_order.energy_momentum import (
    compute_densitized_electromagnetic_energy_momentum,
)
from JAX_BSSN.EM.first_order.evolve import common_densitized_fields
from JAX_BSSN.EM.first_order.equations import (
    densitized_displacement_divergence,
    densitized_magnetic_divergence,
    densitized_maxwell_rhs,
)
from JAX_BSSN.EM.first_order.geometry import (
    _conformal_factors_at_locations,
)
from JAX_BSSN.EM.first_order.staggering import (
    DISPLACEMENT_FIELD_LOCATIONS,
    MAGNETIC_FIELD_LOCATIONS,
)
from JAX_BSSN.EM.first_order.variables import DensitizedMaxwellState
from JAX_BSSN.EM.second_order.cartoon.spherical_symmetry import (
    compact_cartoon_wave,
    expand_cartoon_wave_axis,
)
from JAX_BSSN.EM.variables import EinsteinMaxwellVariables


def _native_offset(site):
    return -0.5 if site == "V" else 0.0


def _fill_vector_ghosts(field, locations):
    parity = jnp.asarray((-1.0, 1.0, 1.0), dtype=field.dtype)
    components = []
    for component, location in enumerate(locations):
        positive = field[component, CARTOON_GHOST_CELLS:, 0, 0]
        if location[0] == "V":
            if component == 0:
                positive = positive.at[0].set(0.0)
            ghosts = jnp.flip(
                positive[1 : CARTOON_GHOST_CELLS + 1], axis=0
            ) * parity[component]
        else:
            ghosts = jnp.flip(positive[:CARTOON_GHOST_CELLS], axis=0)
            ghosts *= parity[component]
        components.append(jnp.concatenate((ghosts, positive), axis=0))
    return jnp.stack(tuple(components), axis=0)[:, :, None, None]


def _outer_buffer(radial_axis, radial_site, params):
    num_radial_points = radial_axis.shape[-1] - CARTOON_GHOST_CELLS
    dx = jnp.asarray(params.dx, dtype=radial_axis.dtype)
    outer_radius = dx * (
        num_radial_points - 0.5 + _native_offset(radial_site)
    )
    buffer_radius = outer_radius + dx * jnp.arange(
        1, CARTOON_OUTER_BUFFER_CELLS + 1, dtype=radial_axis.dtype
    )
    buffer = radial_axis[-1:] * outer_radius / buffer_radius
    return jnp.concatenate((radial_axis, buffer), axis=-1)


def _target_coordinates(shape, location, params, dtype):
    axes = []
    for size, site, minimum in zip(
        shape,
        location,
        (params.x_min, params.y_min, params.z_min),
    ):
        axes.append(
            jnp.asarray(minimum, dtype=dtype)
            + params.dx
            * (
                jnp.arange(size, dtype=dtype)
                + _native_offset(site)
            )
        )
    return jnp.meshgrid(*axes, indexing="ij")


def _reconstruct_radial_vector(field, locations, params):
    radial_site = locations[0][0]
    radial_axis = field[0, :, 0, 0]
    source = _outer_buffer(radial_axis, radial_site, params)
    radial_origin = params.x_min + params.dx * _native_offset(radial_site)

    components = []
    for component, location in enumerate(locations):
        X, Y, Z = _target_coordinates(
            (field.shape[1], CARTOON_SUPPORT_SIZE, CARTOON_SUPPORT_SIZE),
            location,
            params,
            field.dtype,
        )
        radius = jnp.sqrt(X**2 + Y**2 + Z**2)
        radial = lagrange6_nonperiodic(
            source, (radius - radial_origin) / params.dx, axis=-1
        )
        safe_radius = jnp.where(radius > 0.0, radius, 1.0)
        direction = (X, Y, Z)[component] / safe_radius
        if component == 0:
            direction = jnp.where(radius > 0.0, direction, 1.0)
        components.append(radial * direction)
    return jnp.stack(tuple(components), axis=0)


def _center_transverse(component, location):
    if location[1] == "C":
        component = component[:, CARTOON_CENTER, :]
    else:
        component = 0.5 * (
            component[:, CARTOON_CENTER, :]
            + component[:, CARTOON_CENTER + 1, :]
        )
    if location[2] == "C":
        return component[:, CARTOON_CENTER]
    return 0.5 * (
        component[:, CARTOON_CENTER]
        + component[:, CARTOON_CENTER + 1]
    )


def _project_vector(field, locations):
    projected = jnp.stack(
        tuple(
            _center_transverse(field[i], locations[i]) for i in range(3)
        ),
        axis=0,
    )[:, CARTOON_GHOST_CELLS:]
    blank = jnp.zeros(
        (3, projected.shape[1] + CARTOON_GHOST_CELLS, 1, 1),
        dtype=field.dtype,
    ).at[:, CARTOON_GHOST_CELLS:, 0, 0].set(projected)
    return _fill_vector_ghosts(blank, locations)


def _project_scalar(field, location):
    reference = _center_transverse(field, location)
    positive = reference[CARTOON_GHOST_CELLS:]
    if location[0] == "V":
        ghosts = jnp.flip(positive[1 : CARTOON_GHOST_CELLS + 1])
    else:
        ghosts = jnp.flip(positive[:CARTOON_GHOST_CELLS])
    return jnp.concatenate((ghosts, positive))[:, None, None]


def compact_spherical_densitized_state(
    state: DensitizedMaxwellState,
) -> DensitizedMaxwellState:
    """Compact a complete signed radial axis to positive-radius storage."""

    return compact_cartoon_wave(state)


def fill_spherical_densitized_ghosts(
    state: DensitizedMaxwellState,
) -> DensitizedMaxwellState:
    """Refresh the four negative-radius parity cells."""

    return DensitizedMaxwellState(
        _fill_vector_ghosts(state.magnetic_previous, MAGNETIC_FIELD_LOCATIONS),
        _fill_vector_ghosts(state.magnetic_current, MAGNETIC_FIELD_LOCATIONS),
        _fill_vector_ghosts(
            state.displacement_left_half, DISPLACEMENT_FIELD_LOCATIONS
        ),
        _fill_vector_ghosts(
            state.displacement_right_half, DISPLACEMENT_FIELD_LOCATIONS
        ),
    )


def expand_spherical_densitized_state(
    state: DensitizedMaxwellState,
) -> DensitizedMaxwellState:
    """Expand compact radial densities onto a complete signed axis."""

    return expand_cartoon_wave_axis(state)


def reconstruct_spherical_densitized_support(
    state: DensitizedMaxwellState,
    params: BSSNParameters,
) -> DensitizedMaxwellState:
    """Reconstruct radial densities on the nine-by-nine Cartesian support."""

    state = fill_spherical_densitized_ghosts(state)
    return DensitizedMaxwellState(
        _reconstruct_radial_vector(
            state.magnetic_previous, MAGNETIC_FIELD_LOCATIONS, params
        ),
        _reconstruct_radial_vector(
            state.magnetic_current, MAGNETIC_FIELD_LOCATIONS, params
        ),
        _reconstruct_radial_vector(
            state.displacement_left_half,
            DISPLACEMENT_FIELD_LOCATIONS,
            params,
        ),
        _reconstruct_radial_vector(
            state.displacement_right_half,
            DISPLACEMENT_FIELD_LOCATIONS,
            params,
        ),
    )


def _project_spherical_densitized_support(
    state: DensitizedMaxwellState,
) -> DensitizedMaxwellState:
    return DensitizedMaxwellState(
        _project_vector(state.magnetic_previous, MAGNETIC_FIELD_LOCATIONS),
        _project_vector(state.magnetic_current, MAGNETIC_FIELD_LOCATIONS),
        _project_vector(
            state.displacement_left_half, DISPLACEMENT_FIELD_LOCATIONS
        ),
        _project_vector(
            state.displacement_right_half, DISPLACEMENT_FIELD_LOCATIONS
        ),
    )


def initialize_spherical_first_order_state(
    bssn,
    physical_displacement,
    physical_magnetic,
    params: BSSNParameters,
) -> EinsteinMaxwellVariables[DensitizedMaxwellState]:
    """Bootstrap compact radial fields through Cartesian support."""

    compact_bssn = fill_cartoon_ghosts(enforce_algebraic_constraints(bssn))
    support_bssn = enforce_algebraic_constraints(
        reconstruct_cartoon_support(compact_bssn, params)
    )
    physical_displacement = _fill_vector_ghosts(
        physical_displacement, DISPLACEMENT_FIELD_LOCATIONS
    )
    physical_magnetic = _fill_vector_ghosts(
        physical_magnetic, MAGNETIC_FIELD_LOCATIONS
    )
    support_displacement = _reconstruct_radial_vector(
        physical_displacement, DISPLACEMENT_FIELD_LOCATIONS, params
    )
    support_magnetic = _reconstruct_radial_vector(
        physical_magnetic, MAGNETIC_FIELD_LOCATIONS, params
    )
    displacement_W = _conformal_factors_at_locations(
        support_bssn, DISPLACEMENT_FIELD_LOCATIONS, params
    )
    magnetic_W = _conformal_factors_at_locations(
        support_bssn, MAGNETIC_FIELD_LOCATIONS, params
    )
    displacement_density = displacement_W**-3 * support_displacement
    magnetic_density = magnetic_W**-3 * support_magnetic
    displacement_rhs, magnetic_rhs = densitized_maxwell_rhs(
        displacement_density,
        magnetic_density,
        support_bssn,
        params,
    )
    half_step = 0.5 * params.dt
    support_em = DensitizedMaxwellState(
        magnetic_density - params.dt * magnetic_rhs,
        magnetic_density,
        displacement_density - half_step * displacement_rhs,
        displacement_density + half_step * displacement_rhs,
    )
    return EinsteinMaxwellVariables(
        bssn=compact_bssn,
        em=_project_spherical_densitized_support(support_em),
    )


def spherical_densitized_constraint_divergences(
    state: DensitizedMaxwellState,
    params: BSSNParameters,
):
    """Return compact spherical divergence constraints."""

    support = reconstruct_spherical_densitized_support(state, params)
    displacement, magnetic = common_densitized_fields(support)
    displacement_divergence = densitized_displacement_divergence(
        displacement, params
    )
    magnetic_divergence = densitized_magnetic_divergence(magnetic, params)
    return (
        _project_scalar(displacement_divergence, ("C", "C", "C")),
        _project_scalar(magnetic_divergence, ("V", "V", "V")),
    )


def _add_bssn_scaled(bssn, rhs, scale):
    return jax.tree_util.tree_map(
        lambda field, derivative: field + scale * derivative,
        bssn,
        rhs,
    )


def _prepare_bssn(bssn):
    return fill_cartoon_ghosts(enforce_algebraic_constraints(bssn))


def _support_bssn(bssn, params):
    return enforce_algebraic_constraints(
        reconstruct_cartoon_support(_prepare_bssn(bssn), params)
    )


def _support_vector(field, locations, params):
    return _reconstruct_radial_vector(
        _fill_vector_ghosts(field, locations), locations, params
    )


def _compact_bssn_rhs(bssn, displacement, magnetic, params):
    support_bssn = _support_bssn(bssn, params)
    support_displacement = _support_vector(
        displacement, DISPLACEMENT_FIELD_LOCATIONS, params
    )
    support_magnetic = _support_vector(
        magnetic, MAGNETIC_FIELD_LOCATIONS, params
    )
    sources = compute_densitized_electromagnetic_energy_momentum(
        support_displacement,
        support_magnetic,
        support_bssn,
        params,
    )
    support_rhs = compute_bssn_rhs_with_matter(
        support_bssn, params, *sources
    )
    return project_cartoon_rhs(support_rhs)


def _compact_maxwell_rhs(bssn, displacement, magnetic, params):
    support_bssn = _support_bssn(bssn, params)
    support_displacement = _support_vector(
        displacement, DISPLACEMENT_FIELD_LOCATIONS, params
    )
    support_magnetic = _support_vector(
        magnetic, MAGNETIC_FIELD_LOCATIONS, params
    )
    displacement_rhs, magnetic_rhs = densitized_maxwell_rhs(
        support_displacement,
        support_magnetic,
        support_bssn,
        params,
    )
    return (
        _project_vector(displacement_rhs, DISPLACEMENT_FIELD_LOCATIONS),
        _project_vector(magnetic_rhs, MAGNETIC_FIELD_LOCATIONS),
    )


@jax.jit
def spherical_first_order_einstein_maxwell_step(
    state: EinsteinMaxwellVariables[DensitizedMaxwellState],
    params: BSSNParameters,
) -> EinsteinMaxwellVariables[DensitizedMaxwellState]:
    """Advance compact spherical radial fields and BSSN by one timestep."""

    dt = params.dt
    bssn_n = _prepare_bssn(state.bssn)
    em = fill_spherical_densitized_ghosts(state.em)
    displacement_n, magnetic_n = common_densitized_fields(em)

    k1 = _compact_bssn_rhs(
        bssn_n, displacement_n, magnetic_n, params
    )
    _, magnetic_rhs_n = _compact_maxwell_rhs(
        bssn_n, displacement_n, magnetic_n, params
    )
    magnetic_left_half = 0.5 * (
        em.magnetic_previous + em.magnetic_current
    )
    magnetic_right_half = magnetic_left_half + dt * magnetic_rhs_n

    bssn_midpoint_1 = _prepare_bssn(
        _add_bssn_scaled(bssn_n, k1, 0.5 * dt)
    )
    k2 = _compact_bssn_rhs(
        bssn_midpoint_1,
        em.displacement_right_half,
        magnetic_right_half,
        params,
    )
    bssn_midpoint_2 = _prepare_bssn(
        _add_bssn_scaled(bssn_n, k2, 0.5 * dt)
    )
    k3 = _compact_bssn_rhs(
        bssn_midpoint_2,
        em.displacement_right_half,
        magnetic_right_half,
        params,
    )
    displacement_rhs_midpoint, magnetic_rhs_midpoint = _compact_maxwell_rhs(
        bssn_midpoint_2,
        em.displacement_right_half,
        magnetic_right_half,
        params,
    )
    displacement_next = displacement_n + dt * displacement_rhs_midpoint
    magnetic_next = magnetic_n + dt * magnetic_rhs_midpoint

    bssn_endpoint = _prepare_bssn(_add_bssn_scaled(bssn_n, k3, dt))
    k4 = _compact_bssn_rhs(
        bssn_endpoint, displacement_next, magnetic_next, params
    )
    bssn_increment = jax.tree_util.tree_map(
        lambda d1, d2, d3, d4: (d1 + 2.0 * d2 + 2.0 * d3 + d4)
        / 6.0,
        k1,
        k2,
        k3,
        k4,
    )
    bssn_next = _prepare_bssn(
        _add_bssn_scaled(bssn_n, bssn_increment, dt)
    )
    displacement_rhs_endpoint, _ = _compact_maxwell_rhs(
        bssn_next, displacement_next, magnetic_next, params
    )
    displacement_next_half = (
        em.displacement_right_half + dt * displacement_rhs_endpoint
    )

    return EinsteinMaxwellVariables(
        bssn=bssn_next,
        em=fill_spherical_densitized_ghosts(
            DensitizedMaxwellState(
                magnetic_n,
                magnetic_next,
                em.displacement_right_half,
                displacement_next_half,
            )
        ),
    )


__all__ = [
    "compact_spherical_densitized_state",
    "expand_spherical_densitized_state",
    "fill_spherical_densitized_ghosts",
    "initialize_spherical_first_order_state",
    "reconstruct_spherical_densitized_support",
    "spherical_first_order_einstein_maxwell_step",
    "spherical_densitized_constraint_divergences",
]
