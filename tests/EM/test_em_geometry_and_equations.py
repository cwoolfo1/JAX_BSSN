import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from JAX_BSSN.bssn import BSSNParameters, BSSNVariables

from JAX_BSSN.EM.derivatives import (
    covariant_derivative_covector,
    covariant_vector_laplacian,
)
from JAX_BSSN.EM.equations import curved_source_terms
from JAX_BSSN.EM.geometry import compute_bssn_em_geometry
from tests.EM.em_helpers import (
    flat_bssn_variables,
    zero_bssn_rhs,
    zero_em_variables,
)
from JAX_BSSN.EM.variables import EMVariables


def test_minkowski_geometry_and_sources_vanish():
    shape = (12, 3, 2)
    bssn = flat_bssn_variables(shape)
    params = BSSNParameters(dx=0.2, dt=0.01, nu=0.0)
    geometry = compute_bssn_em_geometry(
        bssn, zero_bssn_rhs(bssn), params
    )
    electric_source, magnetic_source = curved_source_terms(
        zero_em_variables(shape), bssn, geometry, params
    )

    zero_geometry_fields = (
        geometry.christoffel,
        geometry.acceleration,
        geometry.acceleration_dot,
        geometry.spatial_ricci,
        geometry.electric_weyl,
        geometry.magnetic_weyl,
    )
    for field in zero_geometry_fields:
        np.testing.assert_allclose(field, 0.0, rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(electric_source, 0.0, rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(magnetic_source, 0.0, rtol=0.0, atol=1.0e-13)


def stretched_flat_residual(grid_size):
    length = 2.0 * np.pi
    dx = length / grid_size
    shape = (grid_size, 1, 1)
    x = dx * jnp.arange(grid_size, dtype=jnp.float64)
    scale = 1.0 + 0.1 * jnp.sin(x)
    scale = jnp.broadcast_to(scale[:, None, None], shape)

    physical_metric = jnp.zeros((3, 3) + shape, dtype=jnp.float64)
    physical_metric = physical_metric.at[0, 0].set(scale**2)
    physical_metric = physical_metric.at[1, 1].set(1.0)
    physical_metric = physical_metric.at[2, 2].set(1.0)
    W = scale ** (-1.0 / 3.0)
    conformal_metric = physical_metric * W**2

    zero = jnp.zeros(shape, dtype=jnp.float64)
    vector_zero = jnp.zeros((3,) + shape, dtype=jnp.float64)
    bssn = BSSNVariables(
        conformal_metric=conformal_metric,
        conformal_factor=W,
        traceless_K=jnp.zeros_like(physical_metric),
        trace_K=zero,
        conformal_connection=vector_zero,
        lapse=jnp.ones(shape, dtype=jnp.float64),
        shift=vector_zero,
    )
    params = BSSNParameters(dx=dx, dt=0.1 * dx, nu=0.0)
    geometry = compute_bssn_em_geometry(
        bssn, zero_bssn_rhs(bssn), params
    )

    covariantly_constant = jnp.zeros((3,) + shape, dtype=jnp.float64)
    covariantly_constant = covariantly_constant.at[0].set(scale)
    derivative = covariant_derivative_covector(
        covariantly_constant, geometry.christoffel, params
    )
    laplacian = covariant_vector_laplacian(
        covariantly_constant,
        geometry.inverse_metric,
        geometry.christoffel,
        params,
    )

    return (
        float(jnp.max(jnp.abs(geometry.spatial_ricci))),
        float(jnp.max(jnp.abs(derivative))),
        float(jnp.max(jnp.abs(laplacian))),
    )


def test_stretched_coordinate_flat_geometry_converges_at_second_order():
    coarse = stretched_flat_residual(32)
    fine = stretched_flat_residual(64)

    assert coarse[0] < 1.0e-12
    assert fine[0] < 1.0e-12
    assert coarse[1] / fine[1] > 3.8
    assert coarse[2] / fine[2] > 3.8


def test_curved_sources_obey_electric_magnetic_duality():
    n = 12
    shape = (n, 3, 3)
    dx = 2.0 * np.pi / n
    x = dx * jnp.arange(n, dtype=jnp.float64)[:, None, None]
    sine = jnp.broadcast_to(jnp.sin(x), shape)
    cosine = jnp.broadcast_to(jnp.cos(x), shape)
    one = jnp.ones(shape, dtype=jnp.float64)

    conformal_metric = jnp.zeros((3, 3) + shape, dtype=jnp.float64)
    conformal_metric = conformal_metric.at[0, 0].set(jnp.exp(0.1 * sine))
    conformal_metric = conformal_metric.at[1, 1].set(jnp.exp(-0.1 * sine))
    conformal_metric = conformal_metric.at[2, 2].set(one)
    traceless_K = jnp.zeros_like(conformal_metric)
    traceless_K = traceless_K.at[0, 0].set(0.01 * cosine)
    traceless_K = traceless_K.at[1, 1].set(-0.01 * cosine)
    vector = jnp.stack((0.01 * cosine, 0.02 * sine, 0.01 * sine))

    bssn = BSSNVariables(
        conformal_metric=conformal_metric,
        conformal_factor=1.0 + 0.02 * sine,
        traceless_K=traceless_K,
        trace_K=0.02 * sine,
        conformal_connection=vector,
        lapse=1.0 + 0.05 * cosine,
        shift=0.01 * vector,
    )
    bssn_rhs = zero_bssn_rhs(bssn)._replace(lapse=0.03 * sine)
    params = BSSNParameters(dx=dx, dt=0.01, nu=0.0)
    geometry = compute_bssn_em_geometry(bssn, bssn_rhs, params)

    E = jnp.stack((sine, cosine, sine * cosine))
    P = jnp.stack((cosine, -sine, cosine**2))
    H = jnp.stack((cosine, sine, sine**2))
    Q = jnp.stack((-sine, cosine, sine + cosine))
    wave = EMVariables(E, P, H, Q)
    dual_wave = EMVariables(H, Q, -E, -P)

    electric_source, magnetic_source = curved_source_terms(
        wave, bssn, geometry, params
    )
    dual_electric, dual_magnetic = curved_source_terms(
        dual_wave, bssn, geometry, params
    )

    np.testing.assert_allclose(
        dual_electric, magnetic_source, rtol=1.0e-13, atol=1.0e-13
    )
    np.testing.assert_allclose(
        dual_magnetic, -electric_source, rtol=1.0e-13, atol=1.0e-13
    )


def test_spatial_lapse_profile_produces_acceleration_and_ricci_source():
    n = 20
    shape = (n, 1, 1)
    dx = 0.05
    x = dx * jnp.arange(n, dtype=jnp.float64)
    bssn = flat_bssn_variables(shape)._replace(
        lapse=jnp.exp(0.2 * x)[:, None, None]
    )
    params = BSSNParameters(dx=dx, dt=0.01, nu=0.0)
    geometry = compute_bssn_em_geometry(
        bssn, zero_bssn_rhs(bssn), params
    )

    E = jnp.zeros((3,) + shape, dtype=jnp.float64).at[1].set(1.0)
    zero_vector = jnp.zeros_like(E)
    wave = EMVariables(E, zero_vector, zero_vector, zero_vector)
    electric_source, _ = curved_source_terms(wave, bssn, geometry, params)

    interior = electric_source[1, 2:-2, 0, 0]
    expected = (1.0 + 2.0 / 3.0) * 0.2**2
    np.testing.assert_allclose(interior, expected, rtol=5.0e-4, atol=5.0e-6)


def test_constant_K_and_A_match_independent_algebraic_transcription():
    shape = (8, 1, 1)
    K_value = 0.12
    A_value = 0.04
    bssn = flat_bssn_variables(shape)
    traceless_K = jnp.zeros_like(bssn.traceless_K)
    traceless_K = traceless_K.at[0, 0].set(A_value)
    traceless_K = traceless_K.at[1, 1].set(-A_value)
    bssn = bssn._replace(
        traceless_K=traceless_K,
        trace_K=K_value * jnp.ones(shape),
    )
    params = BSSNParameters(dx=0.1, dt=0.01, nu=0.0)
    geometry = compute_bssn_em_geometry(
        bssn, zero_bssn_rhs(bssn), params
    )

    E = jnp.broadcast_to(
        jnp.asarray([0.3, -0.2, 0.5])[:, None, None, None],
        (3,) + shape,
    )
    P = jnp.broadcast_to(
        jnp.asarray([-0.1, 0.4, 0.2])[:, None, None, None],
        (3,) + shape,
    )
    H = jnp.broadcast_to(
        jnp.asarray([0.6, 0.1, -0.3])[:, None, None, None],
        (3,) + shape,
    )
    Q = jnp.broadcast_to(
        jnp.asarray([0.2, -0.5, 0.7])[:, None, None, None],
        (3,) + shape,
    )
    wave = EMVariables(E, P, H, Q)
    electric_source, magnetic_source = curved_source_terms(
        wave, bssn, geometry, params
    )

    A = traceless_K
    A_squared = jnp.einsum("ij...,ij...->...", A, A)
    K = bssn.trace_K
    electric_weyl = geometry.electric_weyl
    normal_normal_ricci = geometry.normal_normal_ricci

    def expected(field, field_dot):
        return (
            (-4.0 * K**2 / 9.0 + A_squared) * field
            + 2.0 * normal_normal_ricci * field / 3.0
            + K * jnp.einsum("ij...,j...->i...", A, field) / 3.0
            - jnp.einsum("ik...,bk...,b...->i...", A, A, field)
            + 5.0 * K * field_dot / 3.0
            - jnp.einsum("ij...,j...->i...", A, field_dot)
            - jnp.einsum("ij...,j...->i...", electric_weyl, field)
        )

    np.testing.assert_allclose(
        electric_source, expected(E, P), rtol=1.0e-13, atol=1.0e-13
    )
    np.testing.assert_allclose(
        magnetic_source, expected(H, Q), rtol=1.0e-13, atol=1.0e-13
    )
