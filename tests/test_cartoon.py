import jax
import jax.numpy as jnp
import numpy as np
import pytest

from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables
from JAX_BSSN.cartoon import (
    CARTOON_CENTER,
    CARTOON_GHOST_CELLS,
    CARTOON_SUPPORT_SIZE,
    cartoon_centerline,
    cartoon_positive_radius,
    cartoon_rk4_step,
    compact_cartoon_state,
    compute_cartoon_constraint_norms,
    compute_cartoon_constraints,
    compute_cartoon_rhs,
    expand_cartoon_axis,
    fill_cartoon_ghosts,
    reconstruct_cartoon_support,
    validate_cartoon_grid,
)
from JAX_BSSN.evolution.boundaries import PERIODIC_BC, SOMMERFELD_BC
from JAX_BSSN.evolution.derivatives import diff1_field, diff2_field, divergence_3d
from JAX_BSSN.evolution.time_evolve import compute_bssn_rhs


jax.config.update("jax_enable_x64", True)


def _params(num_radial_points, dx):
    del num_radial_points
    return BSSNParameters(
        dx=dx,
        dt=0.1 * dx,
        xl_bc=PERIODIC_BC,
        xr_bc=SOMMERFELD_BC,
        yl_bc=PERIODIC_BC,
        yr_bc=PERIODIC_BC,
        zl_bc=PERIODIC_BC,
        zr_bc=PERIODIC_BC,
        x_min=-(CARTOON_GHOST_CELLS - 0.5) * dx,
        y_min=-CARTOON_GHOST_CELLS * dx,
        z_min=-CARTOON_GHOST_CELLS * dx,
        mad_q=1.0,
    )


def _empty_variables(num_radial_points, dtype=jnp.float64):
    shape = (num_radial_points + CARTOON_GHOST_CELLS, 1, 1)
    scalar = jnp.zeros(shape, dtype=dtype)
    vector = jnp.zeros((3,) + shape, dtype=dtype)
    tensor = jnp.zeros((3, 3) + shape, dtype=dtype)
    return BSSNVariables(tensor, scalar, tensor, scalar, vector, scalar, vector)


def _positive_radius(num_radial_points, dx):
    return (jnp.arange(num_radial_points) + 0.5) * dx


def _support_coordinates(num_radial_points, dx):
    x = (
        jnp.arange(num_radial_points + CARTOON_GHOST_CELLS)
        - CARTOON_GHOST_CELLS
        + 0.5
    ) * dx
    transverse = (jnp.arange(CARTOON_SUPPORT_SIZE) - CARTOON_CENTER) * dx
    return jnp.meshgrid(x, transverse, transverse, indexing="ij")


def _set_positive(field, values):
    return field.at[..., CARTOON_GHOST_CELLS:, 0, 0].set(values)


def test_cartoon_grid_validation_and_axis_helpers():
    num_radial_points, dx = 12, 0.2
    vars = _empty_variables(num_radial_points)

    validate_cartoon_grid(vars, _params(num_radial_points, dx))
    assert cartoon_centerline(vars.lapse).shape == (num_radial_points + 4,)
    assert cartoon_positive_radius(vars.lapse).shape == (num_radial_points,)


def test_cartoon_grid_validation_rejects_unsupported_contracts():
    num_radial_points, dx = 12, 0.2
    vars = _empty_variables(num_radial_points)
    params = _params(num_radial_points, dx)

    shape = (num_radial_points + 4, 9, 9)
    scalar = jnp.zeros(shape)
    vector = jnp.zeros((3,) + shape)
    tensor = jnp.zeros((3, 3) + shape)
    slab = BSSNVariables(tensor, scalar, tensor, scalar, vector, scalar, vector)

    with pytest.raises(ValueError, match="Nr.*1 x 1"):
        validate_cartoon_grid(slab, params)
    with pytest.raises(ValueError, match="at least six"):
        validate_cartoon_grid(_empty_variables(5), params)
    with pytest.raises(ValueError, match="parity ghost"):
        validate_cartoon_grid(vars, params._replace(xl_bc=SOMMERFELD_BC))
    with pytest.raises(ValueError, match="x-right"):
        validate_cartoon_grid(vars, params._replace(xr_bc=PERIODIC_BC))
    with pytest.raises(ValueError, match="y/z"):
        validate_cartoon_grid(vars, params._replace(yl_bc=SOMMERFELD_BC))
    with pytest.raises(ValueError, match="mad_q=1"):
        validate_cartoon_grid(vars, params._replace(mad_q=0.5))
    with pytest.raises(ValueError, match="coordinate minima"):
        validate_cartoon_grid(vars, params._replace(x_min=0.0))


def test_compact_cartoon_state_reflects_all_component_parities():
    num_radial_points = 8
    full_nx = 2 * num_radial_points
    shape = (full_nx, 1, 1)
    positive = jnp.arange(num_radial_points, dtype=jnp.float64) + 1.0

    scalar = jnp.zeros(shape).at[num_radial_points:, 0, 0].set(positive)
    vector = jnp.zeros((3,) + shape)
    tensor = jnp.zeros((3, 3) + shape)
    for component in range(3):
        vector = vector.at[component, num_radial_points:, 0, 0].set(
            (component + 1.0) * positive
        )
        for second_component in range(3):
            tensor = tensor.at[
                component, second_component, num_radial_points:, 0, 0
            ].set((3.0 * component + second_component + 1.0) * positive)

    full = BSSNVariables(
        tensor,
        scalar,
        2.0 * tensor,
        2.0 * scalar,
        vector,
        3.0 * scalar,
        2.0 * vector,
    )
    compact = compact_cartoon_state(full)

    np.testing.assert_array_equal(
        compact.conformal_factor[:4, 0, 0], positive[:4][::-1]
    )

    vector_parity = np.array([-1.0, 1.0, 1.0])
    for component in range(3):
        expected = (
            vector_parity[component]
            * (component + 1.0)
            * np.asarray(positive[:4][::-1])
        )
        np.testing.assert_array_equal(
            compact.conformal_connection[component, :4, 0, 0], expected
        )

    tensor_parity = vector_parity[:, None] * vector_parity[None, :]
    for i in range(3):
        for j in range(3):
            expected = (
                tensor_parity[i, j]
                * (3.0 * i + j + 1.0)
                * np.asarray(positive[:4][::-1])
            )
            np.testing.assert_array_equal(
                compact.conformal_metric[i, j, :4, 0, 0], expected
            )


def test_fill_cartoon_ghosts_overwrites_corrupted_values():
    num_radial_points = 8
    vars = _empty_variables(num_radial_points)
    radius = _positive_radius(num_radial_points, 0.2)
    lapse = _set_positive(vars.lapse, 1.0 + radius)
    shift = _set_positive(
        vars.shift, jnp.stack((radius, 2.0 * radius, 3.0 * radius))
    )
    vars = vars._replace(lapse=lapse.at[:4, 0, 0].set(-99.0), shift=shift)

    filled = fill_cartoon_ghosts(vars)
    np.testing.assert_array_equal(
        filled.lapse[:4, 0, 0], np.asarray((1.0 + radius[:4])[::-1])
    )
    np.testing.assert_array_equal(
        filled.shift[0, :4, 0, 0], -np.asarray(radius[:4][::-1])
    )
    np.testing.assert_array_equal(
        filled.shift[1, :4, 0, 0], 2.0 * np.asarray(radius[:4][::-1])
    )


def test_scalar_cartoon_support_reconstruction():
    num_radial_points, dx = 12, 0.2
    vars = _empty_variables(num_radial_points)
    radius = _positive_radius(num_radial_points, dx)
    W = _set_positive(vars.conformal_factor, 1.0 + radius**2)

    support = reconstruct_cartoon_support(
        vars._replace(conformal_factor=W), _params(num_radial_points, dx)
    )
    X, Y, Z = _support_coordinates(num_radial_points, dx)
    np.testing.assert_allclose(
        support.conformal_factor, 1.0 + X**2 + Y**2 + Z**2, atol=3.0e-13
    )


def test_vector_and_tensor_cartoon_support_reconstruction():
    num_radial_points, dx = 12, 0.2
    vars = _empty_variables(num_radial_points)
    radius = _positive_radius(num_radial_points, dx)
    shift = _set_positive(
        vars.shift, jnp.stack((radius, 0.0 * radius, 0.0 * radius))
    )

    radial = 1.0 + radius**2
    tangential = 2.0 + 0.5 * radius**2
    metric = _set_positive(
        vars.conformal_metric,
        jnp.zeros((3, 3, num_radial_points))
        .at[0, 0].set(radial)
        .at[1, 1].set(tangential)
        .at[2, 2].set(tangential),
    )
    support = reconstruct_cartoon_support(
        vars._replace(shift=shift, conformal_metric=metric),
        _params(num_radial_points, dx),
    )

    X, Y, Z = _support_coordinates(num_radial_points, dx)
    coordinates = jnp.stack((X, Y, Z))
    r = jnp.sqrt(X**2 + Y**2 + Z**2)
    direction = coordinates / r
    np.testing.assert_allclose(support.shift, coordinates, atol=3.0e-13)

    expected_radial = 1.0 + r**2
    expected_tangential = 2.0 + 0.5 * r**2
    expected_metric = (
        expected_tangential[None, None]
        * jnp.eye(3)[:, :, None, None, None]
        + (expected_radial - expected_tangential)[None, None]
        * jnp.einsum("i...,j...->ij...", direction, direction)
    )
    np.testing.assert_allclose(
        support.conformal_metric, expected_metric, atol=5.0e-13
    )


def test_cartesian_derivatives_recover_transverse_geometry():
    num_radial_points, dx = 12, 0.2
    vars = _empty_variables(num_radial_points)
    radius = _positive_radius(num_radial_points, dx)
    scalar = _set_positive(vars.conformal_factor, radius**2)
    shift = _set_positive(
        vars.shift, jnp.stack((radius, 0.0 * radius, 0.0 * radius))
    )
    support = reconstruct_cartoon_support(
        vars._replace(conformal_factor=scalar, shift=shift),
        _params(num_radial_points, dx),
    )

    dfdy = diff1_field(support.conformal_factor, 1, dx)
    dfdz = diff1_field(support.conformal_factor, 2, dx)
    d2fdy2 = diff1_field(dfdy, 1, dx)
    d2fdz2 = diff1_field(dfdz, 2, dx)
    center = (slice(CARTOON_GHOST_CELLS, -4), CARTOON_CENTER, CARTOON_CENTER)

    np.testing.assert_allclose(dfdy[center], 0.0, atol=2.0e-13)
    np.testing.assert_allclose(dfdz[center], 0.0, atol=2.0e-13)
    np.testing.assert_allclose(d2fdy2[center], 2.0, atol=2.0e-12)
    np.testing.assert_allclose(d2fdz2[center], 2.0, atol=2.0e-12)

    divergence = divergence_3d(support.shift, dx)
    np.testing.assert_allclose(
        divergence[4:-4, CARTOON_CENTER, CARTOON_CENTER],
        3.0,
        atol=2.0e-12,
    )


def test_transverse_second_derivative_converges_at_fourth_order():
    errors = []
    for num_radial_points in (16, 32, 64):
        dx = 2.0 / num_radial_points
        vars = _empty_variables(num_radial_points)
        radius = _positive_radius(num_radial_points, dx)
        scalar = _set_positive(vars.conformal_factor, jnp.exp(-radius**2))
        support = reconstruct_cartoon_support(
            vars._replace(conformal_factor=scalar),
            _params(num_radial_points, dx),
        )

        d2fdy2 = diff2_field(
            support.conformal_factor, 1, dx
        )[:, CARTOON_CENTER, CARTOON_CENTER]
        x = (
            jnp.arange(num_radial_points + CARTOON_GHOST_CELLS)
            - CARTOON_GHOST_CELLS
            + 0.5
        ) * dx
        interior = (x > 0.4) & (x < 1.2)
        exact = -2.0 * jnp.exp(-x[interior] ** 2)
        errors.append(
            float(jnp.sqrt(jnp.mean((d2fdy2[interior] - exact) ** 2)))
        )

    orders = [
        np.log(errors[index] / errors[index + 1]) / np.log(2.0)
        for index in range(2)
    ]
    assert min(orders) > 3.8


def test_cartoon_rk4_evolves_compact_axis_and_refreshes_parity():
    num_radial_points, dx = 12, 0.2
    vars = _empty_variables(num_radial_points)
    radius = _positive_radius(num_radial_points, dx)
    identity = jnp.eye(3)[:, :, None]
    metric = _set_positive(
        vars.conformal_metric,
        jnp.broadcast_to(identity, (3, 3, num_radial_points)),
    )
    W = _set_positive(
        vars.conformal_factor, 1.0 + 0.01 * jnp.exp(-radius**2)
    )
    lapse = _set_positive(
        vars.lapse, 1.0 - 0.01 * jnp.exp(-radius**2)
    )
    vars = fill_cartoon_ghosts(
        vars._replace(conformal_metric=metric, conformal_factor=W, lapse=lapse)
    )
    params = _params(num_radial_points, dx)._replace(
        dt=1.0e-4, nu=0.0, zero_shift=1
    )

    evolved = cartoon_rk4_step(vars, params)
    assert evolved.conformal_factor.shape == (num_radial_points + 4, 1, 1)

    expanded = expand_cartoon_axis(evolved)
    for scalar in (
        expanded.conformal_factor,
        expanded.trace_K,
        expanded.lapse,
    ):
        np.testing.assert_allclose(
            scalar[:, 0, 0], scalar[::-1, 0, 0], atol=2.0e-13
        )
    for vector in (expanded.shift, expanded.conformal_connection):
        np.testing.assert_allclose(
            vector[0, :, 0, 0], -vector[0, ::-1, 0, 0], atol=2.0e-13
        )
        np.testing.assert_allclose(
            vector[1:, :, 0, 0], vector[1:, ::-1, 0, 0], atol=2.0e-13
        )


def _smooth_spherical_state(X, Y, Z):
    shape = X.shape
    coordinates = jnp.stack((X, Y, Z))
    r2 = X**2 + Y**2 + Z**2
    r = jnp.sqrt(r2)
    direction = coordinates / r
    radial_outer_product = jnp.einsum(
        "i...,j...->ij...", direction, direction
    )
    identity = jnp.eye(3)[:, :, None, None, None]
    radial_envelope = jnp.exp(-r2)

    metric_shape = 0.01 * r2 * radial_envelope
    metric_radial = jnp.exp(2.0 * metric_shape)
    metric_tangential = jnp.exp(-metric_shape)
    conformal_metric = (
        metric_tangential[None, None] * identity
        + (metric_radial - metric_tangential)[None, None]
        * radial_outer_product
    )

    A_shape = 0.002 * r2 * radial_envelope
    A_radial = 2.0 * A_shape * metric_radial
    A_tangential = -A_shape * metric_tangential
    traceless_K = (
        A_tangential[None, None] * identity
        + (A_radial - A_tangential)[None, None] * radial_outer_product
    )
    connection_radial = 0.005 * r * radial_envelope
    shift_radial = 0.01 * r * radial_envelope

    return BSSNVariables(
        conformal_metric=conformal_metric,
        conformal_factor=1.0 + 0.01 * radial_envelope,
        traceless_K=traceless_K,
        trace_K=0.005 * radial_envelope,
        conformal_connection=connection_radial[None] * direction,
        lapse=1.0 - 0.02 * radial_envelope,
        shift=shift_radial[None] * direction,
    )


def _full_and_compact_states(full_nx, dx):
    x = (jnp.arange(full_nx) - (full_nx - 1) / 2.0) * dx
    transverse = (jnp.arange(full_nx + 1) - full_nx / 2.0) * dx
    X, Y, Z = jnp.meshgrid(x, transverse, transverse, indexing="ij")
    full_vars = _smooth_spherical_state(X, Y, Z)
    full_params = BSSNParameters(
        eta=2.0,
        kappa=0.0,
        nu=0.0,
        g=0.75,
        dx=dx,
        dt=0.01 * dx,
        gauge=1,
        x_min=float(x[0]),
        y_min=float(transverse[0]),
        z_min=float(transverse[0]),
    )

    X_axis = x[:, None, None]
    axis_vars = _smooth_spherical_state(
        X_axis, jnp.zeros_like(X_axis), jnp.zeros_like(X_axis)
    )
    compact_vars = compact_cartoon_state(axis_vars)
    compact_params = _params(full_nx // 2, dx)._replace(
        eta=2.0,
        kappa=0.0,
        nu=0.0,
        g=0.75,
        dt=0.01 * dx,
        gauge=1,
    )
    return full_vars, full_params, compact_vars, compact_params


def test_full_3d_and_compact_cartoon_rhs_converge_on_positive_axis():
    resolution_errors = []

    for full_nx in (16, 24):
        dx = 4.0 / full_nx
        full_vars, full_params, compact_vars, compact_params = (
            _full_and_compact_states(full_nx, dx)
        )
        full_rhs = compute_bssn_rhs(full_vars, full_params)
        compact_rhs = compute_cartoon_rhs(compact_vars, compact_params)
        field_errors = []

        for full_field, compact_field in zip(full_rhs, compact_rhs):
            full_axis = full_field[
                (..., slice(full_nx // 2, None), full_nx // 2, full_nx // 2)
            ]
            compact_axis = cartoon_positive_radius(compact_field)
            field_errors.append(
                float(
                    jnp.max(
                        jnp.abs(full_axis[..., :-4] - compact_axis[..., :-4])
                    )
                )
            )

        resolution_errors.append(field_errors)

    coarse_errors, fine_errors = resolution_errors
    for coarse_error, fine_error in zip(coarse_errors, fine_errors):
        if coarse_error > 1.0e-12:
            assert fine_error < coarse_error
        else:
            assert fine_error < 1.0e-12

    coarse_max = max(coarse_errors)
    fine_max = max(fine_errors)
    observed_order = np.log(coarse_max / fine_max) / np.log(24.0 / 16.0)
    assert observed_order > 3.4


def test_cartoon_constraints_have_compact_shape_parity_and_radial_norms():
    _, _, compact_vars, compact_params = _full_and_compact_states(16, 0.25)

    violations = compute_cartoon_constraints(compact_vars, compact_params)
    assert violations.hamiltonian.shape == (12, 1, 1)
    assert violations.momentum.shape == (3, 12, 1, 1)

    expanded = expand_cartoon_axis(
        compact_vars._replace(
            conformal_factor=violations.hamiltonian,
            trace_K=violations.hamiltonian,
            lapse=violations.hamiltonian,
        )
    ).conformal_factor
    np.testing.assert_allclose(
        expanded[:, 0, 0], expanded[::-1, 0, 0], atol=2.0e-13
    )

    norms = compute_cartoon_constraint_norms(violations)
    assert set(norms) == {
        "hamiltonian_l2",
        "hamiltonian_linf",
        "momentum_l2",
        "momentum_linf",
        "det_gamma_l2",
        "det_gamma_linf",
        "trace_A_l2",
        "trace_A_linf",
        "gamma_l2",
        "gamma_linf",
    }
    assert all(bool(jnp.isfinite(value)) for value in norms.values())
