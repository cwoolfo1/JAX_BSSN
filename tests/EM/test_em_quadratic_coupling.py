"""Gravity-source normalization, positivity, and native quadratic sampling."""

import math

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from JAX_BSSN.bssn.variables import BSSNParameters
from JAX_BSSN.EM.first_order.coupling import (
    quadratic_moments,
    sources_from_moments,
)
from JAX_BSSN.EM.first_order.energy_momentum import (
    compute_densitized_electromagnetic_energy_momentum as _sources,
)
from JAX_BSSN.EM.first_order import (
    make_em_grid,
    initialize_densitized_maxwell_state,
    common_densitized_fields,
)
from PyPIC3D.relativity.core import D_FIELD_LOCATIONS as DL, B_FIELD_LOCATIONS as BL
from tests.EM.em_helpers import flat_bssn_variables


def compute_densitized_electromagnetic_energy_momentum(d, b, state, params):
    grid = make_em_grid(state.lapse.shape, params)
    em = initialize_densitized_maxwell_state(d, b, state, params, grid)
    return _sources(*common_densitized_fields(em), state, grid)


def test_native_checkerboard_keeps_energy_and_stress():
    shape = (16, 1, 1)
    zero = jnp.zeros((3,) + shape)
    magnetic = zero.at[2, :, 0, 0].set((-1.0) ** jnp.arange(16))
    rho, momentum, stress = compute_densitized_electromagnetic_energy_momentum(
        zero, magnetic, flat_bssn_variables(shape), BSSNParameters(dx=0.1)
    )
    np.testing.assert_allclose(rho, 0.5, atol=1.0e-14)
    np.testing.assert_allclose(momentum, 0.0, atol=1.0e-14)
    expected = np.broadcast_to(
        np.diag([0.5, 0.5, -0.5])[:, :, None, None, None], stress.shape
    )
    np.testing.assert_allclose(stress, expected, atol=1.0e-14)


def test_joint_moments_obey_stress_trace_and_energy_bound():
    shape = (8, 5, 7)
    rng = np.random.default_rng(19)
    d, b = (jnp.asarray(rng.normal(size=(3,) + shape)) for _ in range(2))
    raw = jnp.array([[1.5, 0.3, -0.1], [0.3, 1.0, 0.2], [-0.1, 0.2, 0.8]])
    g = raw / jnp.linalg.det(raw) ** (1 / 3)
    state = flat_bssn_variables(shape)._replace(
        conformal_metric=jnp.broadcast_to(g[:, :, None, None, None], (3, 3) + shape),
        conformal_factor=jnp.full(shape, 0.7),
    )
    rho, momentum, stress = compute_densitized_electromagnetic_energy_momentum(
        d, b, state, BSSNParameters()
    )
    inverse = 0.7**2 * jnp.linalg.inv(g)
    np.testing.assert_allclose(
        jnp.einsum("ij,ij...->...", inverse, stress), rho, atol=1.0e-12
    )
    momentum_norm = jnp.sqrt(
        jnp.einsum("ij,i...,j...->...", inverse, momentum, momentum)
    )
    assert np.all(rho >= 0)
    assert np.all(momentum_norm <= rho + 1.0e-12)


def test_smooth_quadratic_sources_converge_at_second_order():
    errors = []
    for n in (16, 32, 64):
        dx = 2 * math.pi / n
        p = BSSNParameters(dx=dx)
        x = dx * (jnp.arange(n) + 0.5)
        X, Z = jnp.meshgrid(x, x, indexing="ij")
        X, Z = X[:, None, :], Z[:, None, :]

        def fields(x, z):
            return (
                jnp.stack((0.2 * jnp.sin(x), 0.3 * jnp.cos(z), 0.1 * jnp.sin(x + z))),
                jnp.stack(
                    (0.1 * jnp.cos(2 * x), 0.2 * jnp.sin(z), 0.15 * jnp.cos(x - z))
                ),
            )

        native = []
        for which, locations in enumerate((DL, BL)):
            native.append(
                jnp.stack(
                    tuple(
                        fields(
                            X + (dx / 2 if loc[0] == "V" else 0),
                            Z + (dx / 2 if loc[2] == "V" else 0),
                        )[which][i]
                        for i, loc in enumerate(locations)
                    )
                )
            )
        state = flat_bssn_variables((n, 1, n))
        shear = 0.4 * jnp.sin(X) * jnp.cos(Z)
        state = state._replace(
            conformal_metric=state.conformal_metric.at[0, 2]
            .set(shear)
            .at[2, 0]
            .set(shear)
            .at[2, 2]
            .set(1 + shear**2)
        )
        dc, bc = fields(X, Z)
        exact_moments = (
            jnp.einsum("i...,j...->ij...", dc, dc),
            jnp.einsum("i...,j...->ij...", bc, bc),
            jnp.einsum("i...,j...->ij...", dc, bc),
        )
        exact = sources_from_moments(exact_moments, state)
        measured = compute_densitized_electromagnetic_energy_momentum(*native, state, p)
        errors.append(
            [float(jnp.sqrt(jnp.mean((a - b) ** 2))) for a, b in zip(measured, exact)]
        )
    assert np.all(np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:])) > 1.8)
