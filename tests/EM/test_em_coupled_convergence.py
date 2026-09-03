import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from JAX_BSSN.bssn import BSSNParameters
from JAX_BSSN.bssn.constraints import compute_all_constraints_with_matter
from JAX_BSSN.cartoon.axisymmetry import reconstruct_axisymmetric_support
from JAX_BSSN.cartoon.spherical_symmetry import reconstruct_cartoon_support
from JAX_BSSN.evolution.boundaries import PERIODIC_BC, SOMMERFELD_BC
from JAX_BSSN.evolution.time_evolve import compute_bssn_rhs_with_matter

from JAX_BSSN.EM.cartoon.axisymmetry import (
    axisymmetric_einstein_maxwell_rk4_step,
    compute_axisymmetric_constraint_divergences,
    fill_axisymmetric_wave_ghosts,
    project_axisymmetric_wave_rhs,
    reconstruct_axisymmetric_wave_support,
)
from JAX_BSSN.EM.cartoon.spherical_symmetry import (
    compute_cartoon_constraint_divergences,
    fill_cartoon_wave_ghosts,
    reconstruct_cartoon_wave_support,
    spherical_einstein_maxwell_rk4_step,
)
from JAX_BSSN.EM.energy_momentum import (
    compute_electromagnetic_energy_momentum,
)
from JAX_BSSN.EM.equations import source_free_projected_field_dots
from JAX_BSSN.EM.geometry import compute_bssn_em_geometry
from JAX_BSSN.EM.evolve import (
    compute_einstein_maxwell_rhs,
    einstein_maxwell_rk4_step,
)
from JAX_BSSN.EM.variables import EMVariables, EinsteinMaxwellVariables
from tests.EM.em_helpers import flat_bssn_variables


def _tree_l2_difference(left, right):
    squared_error = sum(
        jnp.mean((left_field - right_field) ** 2)
        for left_field, right_field in zip(left, right)
    )
    return float(jnp.sqrt(squared_error))


def _self_convergence_order(coarse, medium, fine):
    coarse_medium = _tree_l2_difference(coarse, medium)
    medium_fine = _tree_l2_difference(medium, fine)
    return math.log2(coarse_medium / medium_fine)


def _constraint_self_convergence(coarse, medium, fine):
    coarse_medium = _tree_l2_difference(coarse, medium)
    medium_fine = _tree_l2_difference(medium, fine)
    if min(coarse_medium, medium_fine) < 1.0e-12:
        return None, coarse_medium, medium_fine
    return (
        math.log2(coarse_medium / medium_fine),
        coarse_medium,
        medium_fine,
    )


def _cartesian_params(grid_size, dt):
    return BSSNParameters(
        dx=2.0 * math.pi / grid_size,
        dt=dt,
        nu=0.0,
        kappa=0.0,
        eta=0.0,
        g=0.0,
        zero_shift=1,
    )


def _cartesian_initial_state(grid_size, params):
    shape = (grid_size, 1, 1)
    x = params.dx * (
        jnp.arange(grid_size, dtype=jnp.float64) + 0.5
    )
    sine = jnp.sin(2.0 * x)[:, None, None]
    cosine = jnp.cos(2.0 * x)[:, None, None]

    electric_field = jnp.zeros((3,) + shape, dtype=jnp.float64)
    electric_field = electric_field.at[1].set(0.1 * sine)
    electric_field_dot = jnp.zeros_like(electric_field)
    electric_field_dot = electric_field_dot.at[1].set(-0.2 * cosine)
    magnetic_field = jnp.zeros_like(electric_field)
    magnetic_field = magnetic_field.at[2].set(0.1 * sine)
    magnetic_field_dot = jnp.zeros_like(electric_field)
    magnetic_field_dot = magnetic_field_dot.at[2].set(-0.2 * cosine)

    return EinsteinMaxwellVariables(
        bssn=flat_bssn_variables(shape),
        em=EMVariables(
            electric_field,
            electric_field_dot,
            magnetic_field,
            magnetic_field_dot,
        ),
    )


def _axisymmetric_params(num_rho, dt):
    num_z = 2 * num_rho
    dx = 2.0 / num_rho
    return BSSNParameters(
        dx=dx,
        dt=dt,
        nu=0.0,
        kappa=0.0,
        eta=0.0,
        g=0.0,
        zero_shift=1,
        gauge=0,
        xl_bc=PERIODIC_BC,
        xr_bc=SOMMERFELD_BC,
        yl_bc=PERIODIC_BC,
        yr_bc=PERIODIC_BC,
        zl_bc=SOMMERFELD_BC,
        zr_bc=SOMMERFELD_BC,
        x_min=-3.5 * dx,
        y_min=-4.0 * dx,
        z_min=-(num_z - 1) * dx / 2.0,
        mad_q=1.0,
    )


def _axisymmetric_initial_state(num_rho, params):
    num_z = 2 * num_rho
    shape = (num_rho + 4, 1, num_z)
    rho = (jnp.arange(num_rho, dtype=jnp.float64) + 0.5) * params.dx
    z = params.z_min + params.dx * jnp.arange(num_z, dtype=jnp.float64)
    R, Z = jnp.meshgrid(rho, z, indexing="ij")
    envelope = jnp.exp(-(R**2 + Z**2) / 0.8**2)
    azimuthal_profile = R * envelope

    electric_field = jnp.zeros((3,) + shape, dtype=jnp.float64)
    electric_field = electric_field.at[1, 4:, 0, :].set(
        0.2 * azimuthal_profile
    )
    magnetic_field = jnp.zeros_like(electric_field)
    magnetic_field = magnetic_field.at[1, 4:, 0, :].set(
        -0.15 * azimuthal_profile
    )

    bssn = flat_bssn_variables(shape)
    zero_vector = jnp.zeros_like(electric_field)
    em = fill_axisymmetric_wave_ghosts(
        EMVariables(
            electric_field,
            zero_vector,
            magnetic_field,
            zero_vector,
        )
    )

    # The second-order wave state must also satisfy the first-order Maxwell
    # propagation equations on the same coupled BSSN stage.
    support_bssn = reconstruct_axisymmetric_support(bssn, params)
    support_em = reconstruct_axisymmetric_wave_support(em, params)
    backreaction_sources = compute_electromagnetic_energy_momentum(
        support_em.electric_field,
        support_em.magnetic_field,
        support_bssn,
    )
    support_bssn_rhs = compute_bssn_rhs_with_matter(
        support_bssn,
        params,
        *backreaction_sources,
    )
    geometry = compute_bssn_em_geometry(
        support_bssn,
        support_bssn_rhs,
        params,
        backreaction_sources=backreaction_sources,
    )
    electric_field_dot, magnetic_field_dot = (
        source_free_projected_field_dots(
            support_em.electric_field,
            support_em.magnetic_field,
            support_bssn,
            geometry,
            params,
        )
    )
    compact_dots = project_axisymmetric_wave_rhs(
        EMVariables(
            electric_field_dot,
            jnp.zeros_like(electric_field_dot),
            magnetic_field_dot,
            jnp.zeros_like(magnetic_field_dot),
        )
    )
    em = fill_axisymmetric_wave_ghosts(
        EMVariables(
            electric_field,
            compact_dots.electric_field,
            magnetic_field,
            compact_dots.magnetic_field,
        )
    )

    return EinsteinMaxwellVariables(
        bssn=bssn,
        em=em,
    )


def _spherical_params(num_radial_points, dt):
    dx = 2.0 / num_radial_points
    return BSSNParameters(
        dx=dx,
        dt=dt,
        nu=0.0,
        kappa=0.0,
        eta=0.0,
        g=0.0,
        zero_shift=1,
        gauge=0,
        xl_bc=PERIODIC_BC,
        xr_bc=SOMMERFELD_BC,
        yl_bc=PERIODIC_BC,
        yr_bc=PERIODIC_BC,
        zl_bc=PERIODIC_BC,
        zr_bc=PERIODIC_BC,
        x_min=-3.5 * dx,
        y_min=-4.0 * dx,
        z_min=-4.0 * dx,
        mad_q=1.0,
    )


def _spherical_initial_state(num_radial_points, params):
    shape = (num_radial_points + 4, 1, 1)
    radius = (
        jnp.arange(num_radial_points, dtype=jnp.float64) + 0.5
    ) * params.dx
    radial_profile = radius * jnp.exp(-(radius / 0.6) ** 2)

    electric_field = jnp.zeros((3,) + shape, dtype=jnp.float64)
    electric_field = electric_field.at[0, 4:, 0, 0].set(
        0.2 * radial_profile
    )
    electric_field_dot = jnp.zeros_like(electric_field)
    electric_field_dot = electric_field_dot.at[0, 4:, 0, 0].set(
        -0.1 * radial_profile
    )
    magnetic_field = jnp.zeros_like(electric_field)
    magnetic_field = magnetic_field.at[0, 4:, 0, 0].set(
        -0.15 * radial_profile
    )
    magnetic_field_dot = jnp.zeros_like(electric_field)
    magnetic_field_dot = magnetic_field_dot.at[0, 4:, 0, 0].set(
        0.08 * radial_profile
    )
    em = fill_cartoon_wave_ghosts(
        EMVariables(
            electric_field,
            electric_field_dot,
            magnetic_field,
            magnetic_field_dot,
        )
    )

    return EinsteinMaxwellVariables(
        bssn=flat_bssn_variables(shape),
        em=em,
    )


@jax.jit
def _evolve_axisymmetric(state, params, num_steps):
    return jax.lax.fori_loop(
        0,
        num_steps,
        lambda _, current: axisymmetric_einstein_maxwell_rk4_step(
            current, params
        ),
        state,
    )


@jax.jit
def _evolve_spherical(state, params, num_steps):
    return jax.lax.fori_loop(
        0,
        num_steps,
        lambda _, current: spherical_einstein_maxwell_rk4_step(
            current, params
        ),
        state,
    )


@jax.jit
def _cartesian_matter_constraints(state, params):
    energy_density, momentum_density, _ = (
        compute_electromagnetic_energy_momentum(
            state.em.electric_field,
            state.em.magnetic_field,
            state.bssn,
        )
    )
    constraints = compute_all_constraints_with_matter(
        state.bssn,
        params,
        energy_density,
        momentum_density,
    )
    return constraints.hamiltonian, constraints.momentum


@jax.jit
def _axisymmetric_matter_constraints(state, params):
    support_bssn = reconstruct_axisymmetric_support(state.bssn, params)
    support_em = reconstruct_axisymmetric_wave_support(state.em, params)
    energy_density, momentum_density, _ = (
        compute_electromagnetic_energy_momentum(
            support_em.electric_field,
            support_em.magnetic_field,
            support_bssn,
        )
    )
    constraints = compute_all_constraints_with_matter(
        support_bssn,
        params,
        energy_density,
        momentum_density,
    )
    div_E, div_B = compute_axisymmetric_constraint_divergences(
        state.em, state.bssn, params
    )
    return (
        constraints.hamiltonian[4:, 4, :],
        constraints.momentum[:, 4:, 4, :],
        div_E[4:, 0, :],
        div_B[4:, 0, :],
    )


@jax.jit
def _spherical_matter_constraints(state, params):
    support_bssn = reconstruct_cartoon_support(state.bssn, params)
    support_em = reconstruct_cartoon_wave_support(state.em, params)
    energy_density, momentum_density, _ = (
        compute_electromagnetic_energy_momentum(
            support_em.electric_field,
            support_em.magnetic_field,
            support_bssn,
        )
    )
    constraints = compute_all_constraints_with_matter(
        support_bssn,
        params,
        energy_density,
        momentum_density,
    )
    div_E, div_B = compute_cartoon_constraint_divergences(
        state.em, state.bssn, params
    )
    return (
        constraints.hamiltonian[4:, 4, 4],
        constraints.momentum[:, 4:, 4, 4],
        div_E[4:, 0, 0],
        div_B[4:, 0, 0],
    )


def _temporal_state_orders(evolved):
    bssn_order = _self_convergence_order(
        evolved[0].bssn, evolved[1].bssn, evolved[2].bssn
    )
    em_order = _self_convergence_order(
        evolved[0].em, evolved[1].em, evolved[2].em
    )
    return bssn_order, em_order


def test_cartesian_coupled_rk4_temporal_self_convergence():
    grid_size = 8
    final_time = 0.32
    timesteps = (0.08, 0.04, 0.02)
    evolved = []
    constraints = []
    for dt in timesteps:
        params = _cartesian_params(grid_size, dt)
        initial = _cartesian_initial_state(grid_size, params)

        state = jax.lax.fori_loop(
            0,
            round(final_time / dt),
            lambda _, current: einstein_maxwell_rk4_step(current, params),
            initial,
        )
        # evolve the bssn and em variables for the given number of steps

        evolved.append(state)
        constraints.append(_cartesian_matter_constraints(state, params))
    jax.block_until_ready((evolved, constraints))

    state_orders = _temporal_state_orders(evolved)
    constraint_order, _, _ = _constraint_self_convergence(*constraints)
    print(
        "Cartesian coupled temporal orders: "
        f"BSSN={state_orders[0]:.6f}, EM={state_orders[1]:.6f}, "
        f"Einstein constraints={constraint_order:.6f}"
    )
    assert min(*state_orders, constraint_order) > 3.5


def test_axisymmetric_coupled_rk4_temporal_self_convergence():
    num_rho = 6
    final_time = 0.16
    timesteps = (0.04, 0.02, 0.01)
    evolved = []
    constraints = []
    for dt in timesteps:
        params = _axisymmetric_params(num_rho, dt)
        initial = _axisymmetric_initial_state(num_rho, params)
        state = _evolve_axisymmetric(
            initial, params, round(final_time / dt)
        )
        evolved.append(state)
        constraints.append(_axisymmetric_matter_constraints(state, params))
    jax.block_until_ready((evolved, constraints))

    state_orders = _temporal_state_orders(evolved)
    einstein_constraints = [value[:2] for value in constraints]
    maxwell_constraints = [value[2:] for value in constraints]
    einstein_order, _, _ = _constraint_self_convergence(
        *einstein_constraints
    )
    maxwell_order, maxwell_coarse_medium, maxwell_medium_fine = (
        _constraint_self_convergence(*maxwell_constraints)
    )
    maxwell_report = (
        f"{maxwell_order:.6f}"
        if maxwell_order is not None
        else (
            "roundoff-only "
            f"({maxwell_coarse_medium:.3e}, {maxwell_medium_fine:.3e})"
        )
    )
    print(
        "Axisymmetric coupled temporal orders: "
        f"BSSN={state_orders[0]:.6f}, EM={state_orders[1]:.6f}, "
        f"Einstein constraints={einstein_order:.6f}, "
        f"Maxwell constraints={maxwell_report}"
    )
    assert min(*state_orders, einstein_order) > 3.5
    if maxwell_order is not None:
        assert maxwell_order > 3.5


def test_spherical_coupled_rk4_temporal_self_convergence():
    num_radial_points = 8
    final_time = 0.16
    timesteps = (0.04, 0.02, 0.01)
    evolved = []
    constraints = []
    for dt in timesteps:
        params = _spherical_params(num_radial_points, dt)
        initial = _spherical_initial_state(num_radial_points, params)
        state = _evolve_spherical(
            initial, params, round(final_time / dt)
        )
        evolved.append(state)
        constraints.append(_spherical_matter_constraints(state, params))
    jax.block_until_ready((evolved, constraints))

    state_orders = _temporal_state_orders(evolved)
    einstein_constraints = [value[:2] for value in constraints]
    maxwell_constraints = [value[2:] for value in constraints]
    einstein_order, _, _ = _constraint_self_convergence(
        *einstein_constraints
    )
    maxwell_order, maxwell_coarse_medium, maxwell_medium_fine = (
        _constraint_self_convergence(*maxwell_constraints)
    )
    maxwell_report = (
        f"{maxwell_order:.6f}"
        if maxwell_order is not None
        else (
            "roundoff-only "
            f"({maxwell_coarse_medium:.3e}, {maxwell_medium_fine:.3e})"
        )
    )
    print(
        "Spherical coupled temporal orders: "
        f"BSSN={state_orders[0]:.6f}, EM={state_orders[1]:.6f}, "
        f"Einstein constraints={einstein_order:.6f}, "
        f"Maxwell constraints={maxwell_report}"
    )
    assert min(*state_orders, einstein_order) > 3.5
    if maxwell_order is not None:
        assert maxwell_order > 3.5


def _restrict_cartesian(field):
    num_coarse = field.shape[-3] // 2
    center = 2 * jnp.arange(num_coarse)

    def sample(offset):
        indices = (center + offset) % field.shape[-3]
        return jnp.take(field, indices, axis=-3)

    # Fourth-order interpolation from the four fine cell centers surrounding
    # each coarse cell center on the periodic grid.
    return (
        -sample(-1) + 9.0 * sample(0) + 9.0 * sample(1) - sample(2)
    ) / 16.0


def _cartesian_spatial_difference(coarse, fine):
    return _tree_l2_difference(
        coarse,
        jax.tree_util.tree_map(_restrict_cartesian, fine),
    )


def _cartesian_spatial_order(coarse, medium, fine):
    coarse_medium = _cartesian_spatial_difference(coarse, medium)
    medium_fine = _cartesian_spatial_difference(medium, fine)
    return math.log2(coarse_medium / medium_fine)


def test_cartesian_backreacting_spatial_self_convergence():
    final_time = 0.16
    resolutions = (8, 16, 32)
    evolved = []
    constraints = []
    for grid_size in resolutions:
        num_steps = grid_size // 2
        params = _cartesian_params(grid_size, final_time / num_steps)
        initial = _cartesian_initial_state(grid_size, params)

        state = jax.lax.fori_loop(
            0,
            num_steps,
            lambda _, current: einstein_maxwell_rk4_step(current, params),
            initial,
        )
        # advance the bssn and em variables for the given number of steps

        evolved.append(state)
        constraints.append(_cartesian_matter_constraints(state, params))
    jax.block_until_ready((evolved, constraints))

    bssn_order = _cartesian_spatial_order(
        evolved[0].bssn, evolved[1].bssn, evolved[2].bssn
    )
    em_order = _cartesian_spatial_order(
        evolved[0].em, evolved[1].em, evolved[2].em
    )
    constraint_order = _cartesian_spatial_order(*constraints)
    print(
        "Cartesian coupled spatial orders: "
        f"BSSN={bssn_order:.6f}, EM={em_order:.6f}, "
        f"Einstein constraints={constraint_order:.6f}"
    )
    assert min(bssn_order, em_order, constraint_order) >= 1.8


def _axisymmetric_physical(field):
    return field[..., 4:, 0, :]


_AXISYMMETRIC_BASE_CROP = 2


def _interpolate_physical_fine(field, crop):
    num_coarse_rho = field.shape[-2] // 2
    rho_center = 2 * jnp.arange(crop, num_coarse_rho - crop)

    def rho_sample(offset):
        return jnp.take(field, rho_center + offset, axis=-2)

    rho_interpolated = (
        -rho_sample(-1)
        + 9.0 * rho_sample(0)
        + 9.0 * rho_sample(1)
        - rho_sample(2)
    ) / 16.0

    num_coarse_z = field.shape[-1] // 2
    z_center = 2 * jnp.arange(crop, num_coarse_z - crop)

    def z_sample(offset):
        return jnp.take(rho_interpolated, z_center + offset, axis=-1)

    return (
        -z_sample(-1)
        + 9.0 * z_sample(0)
        + 9.0 * z_sample(1)
        - z_sample(2)
    ) / 16.0


def _axisymmetric_spatial_difference(
    coarse, fine, crop, physical=False
):
    coarse_fields = jax.tree_util.tree_map(
        lambda field: (
            field if physical else _axisymmetric_physical(field)
        )[..., crop:-crop, crop:-crop],
        coarse,
    )
    fine_fields = jax.tree_util.tree_map(
        lambda field: _interpolate_physical_fine(
            field if physical else _axisymmetric_physical(field), crop
        ),
        fine,
    )

    squared_error = 0.0
    for coarse_field, fine_field in zip(coarse_fields, fine_fields):
        difference = coarse_field - fine_field
        num_rho, num_z = difference.shape[-2:]
        full_num_rho = num_rho + 2 * crop
        dx = 2.0 / full_num_rho
        rho = dx * (
            jnp.arange(crop, full_num_rho - crop, dtype=difference.dtype)
            + 0.5
        )
        weight = rho.reshape((1,) * (difference.ndim - 2) + (num_rho, 1))
        component_count = math.prod(difference.shape[:-2])
        normalization = jnp.sum(rho) * num_z * component_count
        squared_error += jnp.sum(weight * difference**2) / normalization
    return float(jnp.sqrt(squared_error))


def _axisymmetric_spatial_order(coarse, medium, fine, physical=False):
    coarse_medium = _axisymmetric_spatial_difference(
        coarse,
        medium,
        crop=_AXISYMMETRIC_BASE_CROP,
        physical=physical,
    )
    medium_fine = _axisymmetric_spatial_difference(
        medium,
        fine,
        crop=2 * _AXISYMMETRIC_BASE_CROP,
        physical=physical,
    )
    return math.log2(coarse_medium / medium_fine)


def test_axisymmetric_backreacting_spatial_self_convergence():
    final_time = 0.08
    resolutions = (8, 16, 32)
    evolved = []
    constraints = []
    for num_rho in resolutions:
        num_steps = num_rho // 2
        params = _axisymmetric_params(num_rho, final_time / num_steps)
        initial = _axisymmetric_initial_state(num_rho, params)
        state = _evolve_axisymmetric(initial, params, num_steps)
        evolved.append(state)
        constraints.append(_axisymmetric_matter_constraints(state, params))
    jax.block_until_ready((evolved, constraints))

    bssn_order = _axisymmetric_spatial_order(
        evolved[0].bssn, evolved[1].bssn, evolved[2].bssn
    )
    em_order = _axisymmetric_spatial_order(
        evolved[0].em, evolved[1].em, evolved[2].em
    )
    einstein_constraints = [value[:2] for value in constraints]
    maxwell_constraints = [value[2:] for value in constraints]
    einstein_order = _axisymmetric_spatial_order(
        *einstein_constraints, physical=True
    )
    maxwell_coarse_medium = _axisymmetric_spatial_difference(
        maxwell_constraints[0],
        maxwell_constraints[1],
        crop=_AXISYMMETRIC_BASE_CROP,
        physical=True,
    )
    maxwell_medium_fine = _axisymmetric_spatial_difference(
        maxwell_constraints[1],
        maxwell_constraints[2],
        crop=2 * _AXISYMMETRIC_BASE_CROP,
        physical=True,
    )
    maxwell_order = None
    if min(maxwell_coarse_medium, maxwell_medium_fine) >= 1.0e-12:
        maxwell_order = math.log2(
            maxwell_coarse_medium / maxwell_medium_fine
        )
    maxwell_report = (
        f"{maxwell_order:.6f}"
        if maxwell_order is not None
        else (
            "roundoff-only "
            f"({maxwell_coarse_medium:.3e}, {maxwell_medium_fine:.3e})"
        )
    )
    print(
        "Axisymmetric coupled spatial orders: "
        f"BSSN={bssn_order:.6f}, EM={em_order:.6f}, "
        f"Einstein constraints={einstein_order:.6f}, "
        f"Maxwell constraints={maxwell_report} "
        f"(differences={maxwell_coarse_medium:.3e}, "
        f"{maxwell_medium_fine:.3e})"
    )
    assert min(bssn_order, em_order, einstein_order) >= 1.8
    if maxwell_order is not None:
        assert maxwell_order >= 1.8
