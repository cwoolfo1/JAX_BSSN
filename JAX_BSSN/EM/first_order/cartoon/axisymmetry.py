"""Axisymmetric Cartoon evolution for staggered vector densities."""

import jax
import jax.numpy as jnp

from JAX_BSSN.bssn import BSSNParameters
from JAX_BSSN.cartoon.axisymmetry import (
    fill_axisymmetric_ghosts,
    project_axisymmetric_rhs,
    reconstruct_axisymmetric_support,
)
from JAX_BSSN.cartoon.axisymmetry.reconstruction import (
    AXISYMMETRIC_CENTER,
    AXISYMMETRIC_GHOST_CELLS,
    AXISYMMETRIC_OUTER_BUFFER_CELLS,
    AXISYMMETRIC_SUPPORT_SIZE,
)
from JAX_BSSN.cartoon.interpolation import lagrange6_nonperiodic
from JAX_BSSN.evolution.boundaries import PERIODIC_BC
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
from JAX_BSSN.EM.second_order.cartoon.axisymmetry import (
    compact_axisymmetric_wave,
    expand_axisymmetric_wave_plane,
)
from JAX_BSSN.EM.variables import EinsteinMaxwellVariables


def _native_offset(site):
    return -0.5 if site == "V" else 0.0


def _fill_vector_ghosts(field, locations):
    parity = jnp.asarray((-1.0, -1.0, 1.0), dtype=field.dtype)
    components = []
    for component, location in enumerate(locations):
        positive = field[component, AXISYMMETRIC_GHOST_CELLS:, 0, :]
        if location[0] == "V":
            if component < 2:
                positive = positive.at[0].set(0.0)
            ghosts = jnp.flip(
                positive[1 : AXISYMMETRIC_GHOST_CELLS + 1], axis=0
            ) * parity[component]
        else:
            ghosts = jnp.flip(
                positive[:AXISYMMETRIC_GHOST_CELLS], axis=0
            ) * parity[component]
        components.append(jnp.concatenate((ghosts, positive), axis=0))
    return jnp.stack(tuple(components), axis=0)[:, :, None, :]


def _pad_z(source, params):
    def nonperiodic_left(values):
        slope = values[..., 1:2] - values[..., :1]
        return jnp.concatenate(
            tuple(values[..., :1] - multiple * slope for multiple in (3, 2, 1)),
            axis=-1,
        )

    def nonperiodic_right(values):
        slope = values[..., -1:] - values[..., -2:-1]
        return jnp.concatenate(
            tuple(values[..., -1:] + multiple * slope for multiple in (1, 2, 3)),
            axis=-1,
        )

    left = jax.lax.cond(
        params.zl_bc == PERIODIC_BC,
        lambda values: values[..., -3:],
        nonperiodic_left,
        source,
    )
    right = jax.lax.cond(
        params.zr_bc == PERIODIC_BC,
        lambda values: values[..., :3],
        nonperiodic_right,
        source,
    )
    return jnp.concatenate((left, source, right), axis=-1)


def _outer_buffer(reference, source_location, params):
    num_radial_points = reference.shape[0] - AXISYMMETRIC_GHOST_CELLS
    dx = jnp.asarray(params.dx, dtype=reference.dtype)
    rho_edge = dx * (
        num_radial_points - 0.5 + _native_offset(source_location[0])
    )
    z = jnp.asarray(params.z_min, dtype=reference.dtype) + dx * (
        jnp.arange(reference.shape[-1], dtype=reference.dtype)
        + _native_offset(source_location[2])
    )
    radius_edge = jnp.sqrt(rho_edge**2 + z**2)
    rho_buffer = rho_edge + dx * jnp.arange(
        1, AXISYMMETRIC_OUTER_BUFFER_CELLS + 1, dtype=reference.dtype
    )
    radius_buffer = jnp.sqrt(rho_buffer[:, None] ** 2 + z[None, :] ** 2)
    buffer = reference[-1:, :] * radius_edge[None, :] / radius_buffer
    return jnp.concatenate((reference, buffer), axis=0)


def _target_coordinates(shape, location, params, dtype):
    nx, _, nz = shape
    dx = jnp.asarray(params.dx, dtype=dtype)
    x = jnp.asarray(params.x_min, dtype=dtype) + dx * (
        jnp.arange(nx, dtype=dtype) + _native_offset(location[0])
    )
    y = jnp.asarray(params.y_min, dtype=dtype) + dx * (
        jnp.arange(AXISYMMETRIC_SUPPORT_SIZE, dtype=dtype)
        + _native_offset(location[1])
    )
    z = jnp.asarray(params.z_min, dtype=dtype) + dx * (
        jnp.arange(nz, dtype=dtype) + _native_offset(location[2])
    )
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    rho = jnp.sqrt(X**2 + Y**2)
    safe_rho = jnp.where(rho > 0.0, rho, 1.0)
    return X, Y, Z, rho, X / safe_rho, Y / safe_rho


def _interpolate_component(reference, source_location, target_data, params):
    _, _, target_z, rho, _, _ = target_data
    source = _outer_buffer(reference, source_location, params)
    dx = jnp.asarray(params.dx, dtype=reference.dtype)
    radial_origin = jnp.asarray(params.x_min, dtype=reference.dtype) + (
        dx * _native_offset(source_location[0])
    )
    radial_q = (rho[:, :, 0] - radial_origin) / dx
    radial_values = lagrange6_nonperiodic(source, radial_q, axis=0)

    radial_values = _pad_z(radial_values, params)
    z_origin = jnp.asarray(params.z_min, dtype=reference.dtype) + (
        dx * _native_offset(source_location[2])
    )
    z_q = (target_z[0, 0, :] - z_origin) / dx + 3.0
    return lagrange6_nonperiodic(radial_values, z_q, axis=-1)


def _reconstruct_vector(field, locations, params):
    references = tuple(field[i, :, 0, :] for i in range(3))
    components = []
    for target_component, target_location in enumerate(locations):
        target_data = _target_coordinates(
            field.shape[-3:], target_location, params, field.dtype
        )
        _, _, _, _, cosine, sine = target_data
        cylindrical = jnp.stack(
            tuple(
                _interpolate_component(
                    references[source_component],
                    locations[source_component],
                    target_data,
                    params,
                )
                for source_component in range(3)
            ),
            axis=0,
        )
        if target_component == 0:
            component = cosine * cylindrical[0] - sine * cylindrical[1]
        elif target_component == 1:
            component = sine * cylindrical[0] + cosine * cylindrical[1]
        else:
            component = cylindrical[2]
        components.append(component)
    return jnp.stack(tuple(components), axis=0)


def _center_y_component(component, y_site):
    if y_site == "C":
        return component[:, AXISYMMETRIC_CENTER, :]
    return 0.5 * (
        component[:, AXISYMMETRIC_CENTER, :]
        + component[:, AXISYMMETRIC_CENTER + 1, :]
    )


def _project_vector(field, locations):
    projected = jnp.stack(
        tuple(
            _center_y_component(field[i], locations[i][1])
            for i in range(3)
        ),
        axis=0,
    )[:, AXISYMMETRIC_GHOST_CELLS:, :]
    blank = jnp.zeros(
        (
            3,
            projected.shape[1] + AXISYMMETRIC_GHOST_CELLS,
            1,
            projected.shape[2],
        ),
        dtype=field.dtype,
    ).at[:, AXISYMMETRIC_GHOST_CELLS:, 0, :].set(projected)
    return _fill_vector_ghosts(blank, locations)


def _project_scalar(field, location):
    reference = _center_y_component(field, location[1])
    positive = reference[AXISYMMETRIC_GHOST_CELLS:, :]
    if location[0] == "V":
        ghosts = jnp.flip(
            positive[1 : AXISYMMETRIC_GHOST_CELLS + 1], axis=0
        )
    else:
        ghosts = jnp.flip(positive[:AXISYMMETRIC_GHOST_CELLS], axis=0)
    return jnp.concatenate((ghosts, positive), axis=0)[:, None, :]


def compact_axisymmetric_densitized_state(
    state: DensitizedMaxwellState,
) -> DensitizedMaxwellState:
    """Compact a complete signed rho-z plane to positive-rho storage."""

    # A weight-one vector density transforms as a vector under the proper
    # rotations used by Cartoon, so the existing vector-state helper applies.
    return compact_axisymmetric_wave(state)


def fill_axisymmetric_densitized_ghosts(
    state: DensitizedMaxwellState,
) -> DensitizedMaxwellState:
    """Refresh the four negative-rho parity cells on all six Yee grids."""

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


def expand_axisymmetric_densitized_state(
    state: DensitizedMaxwellState,
) -> DensitizedMaxwellState:
    """Expand compact densities onto the complete signed reference plane."""

    return expand_axisymmetric_wave_plane(state)


def reconstruct_axisymmetric_densitized_support(
    state: DensitizedMaxwellState,
    params: BSSNParameters,
) -> DensitizedMaxwellState:
    """Rotate/interpolate every history vector to nine Cartesian y planes."""

    state = fill_axisymmetric_densitized_ghosts(state)
    return DensitizedMaxwellState(
        _reconstruct_vector(
            state.magnetic_previous, MAGNETIC_FIELD_LOCATIONS, params
        ),
        _reconstruct_vector(
            state.magnetic_current, MAGNETIC_FIELD_LOCATIONS, params
        ),
        _reconstruct_vector(
            state.displacement_left_half,
            DISPLACEMENT_FIELD_LOCATIONS,
            params,
        ),
        _reconstruct_vector(
            state.displacement_right_half,
            DISPLACEMENT_FIELD_LOCATIONS,
            params,
        ),
    )


def _project_axisymmetric_densitized_support(
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


def initialize_axisymmetric_first_order_state(
    bssn,
    physical_displacement,
    physical_magnetic,
    params: BSSNParameters,
) -> EinsteinMaxwellVariables[DensitizedMaxwellState]:
    """Bootstrap compact axisymmetric fields through Cartesian support."""

    compact_bssn = fill_axisymmetric_ghosts(
        enforce_algebraic_constraints(bssn)
    )
    support_bssn = enforce_algebraic_constraints(
        reconstruct_axisymmetric_support(compact_bssn, params)
    )
    physical_displacement = _fill_vector_ghosts(
        physical_displacement, DISPLACEMENT_FIELD_LOCATIONS
    )
    physical_magnetic = _fill_vector_ghosts(
        physical_magnetic, MAGNETIC_FIELD_LOCATIONS
    )
    support_displacement = _reconstruct_vector(
        physical_displacement, DISPLACEMENT_FIELD_LOCATIONS, params
    )
    support_magnetic = _reconstruct_vector(
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
        em=_project_axisymmetric_densitized_support(support_em),
    )


def axisymmetric_densitized_constraint_divergences(
    state: DensitizedMaxwellState,
    params: BSSNParameters,
):
    """Return compact axisymmetric divergence constraints."""

    support = reconstruct_axisymmetric_densitized_support(state, params)
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
    return fill_axisymmetric_ghosts(enforce_algebraic_constraints(bssn))


def _support_bssn(bssn, params):
    return enforce_algebraic_constraints(
        reconstruct_axisymmetric_support(_prepare_bssn(bssn), params)
    )


def _support_vector(field, locations, params):
    return _reconstruct_vector(
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
    return project_axisymmetric_rhs(support_rhs)


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
def axisymmetric_first_order_einstein_maxwell_step(
    state: EinsteinMaxwellVariables[DensitizedMaxwellState],
    params: BSSNParameters,
) -> EinsteinMaxwellVariables[DensitizedMaxwellState]:
    """Advance compact axisymmetric BSSN and densitized Maxwell data."""

    dt = params.dt
    bssn_n = _prepare_bssn(state.bssn)
    em = fill_axisymmetric_densitized_ghosts(state.em)
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
        em=fill_axisymmetric_densitized_ghosts(
            DensitizedMaxwellState(
                magnetic_n,
                magnetic_next,
                em.displacement_right_half,
                displacement_next_half,
            )
        ),
    )


__all__ = [
    "axisymmetric_first_order_einstein_maxwell_step",
    "axisymmetric_densitized_constraint_divergences",
    "compact_axisymmetric_densitized_state",
    "expand_axisymmetric_densitized_state",
    "fill_axisymmetric_densitized_ghosts",
    "initialize_axisymmetric_first_order_state",
    "reconstruct_axisymmetric_densitized_support",
]
