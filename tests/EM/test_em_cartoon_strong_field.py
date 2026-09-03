import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from JAX_BSSN.bssn import BSSNParameters, BSSNVariables
from JAX_BSSN.evolution.boundaries import PERIODIC_BC, SOMMERFELD_BC
from JAX_BSSN.EM.second_order.cartoon.axisymmetry import (
    compute_axisymmetric_constraint_divergences,
)
from JAX_BSSN.EM.second_order.cartoon.spherical_symmetry import (
    compute_cartoon_constraint_divergences,
)
from JAX_BSSN.EM.second_order.variables import EMVariables


def _zero_W_state(shape):
    scalar_zero = jnp.zeros(shape, dtype=jnp.float64)
    vector_zero = jnp.zeros((3,) + shape, dtype=jnp.float64)
    identity = jnp.eye(3, dtype=jnp.float64)[:, :, None, None, None]
    conformal_metric = jnp.broadcast_to(identity, (3, 3) + shape)
    bssn = BSSNVariables(
        conformal_metric=conformal_metric,
        conformal_factor=scalar_zero,
        traceless_K=jnp.zeros_like(conformal_metric),
        trace_K=scalar_zero,
        conformal_connection=vector_zero,
        lapse=jnp.ones(shape, dtype=jnp.float64),
        shift=vector_zero,
    )
    em = EMVariables(
        vector_zero,
        vector_zero,
        vector_zero,
        vector_zero,
    )
    return bssn, em


def _cartoon_parameters(dx, z_min):
    return BSSNParameters(
        dx=dx,
        dt=0.05 * dx,
        nu=0.0,
        xl_bc=PERIODIC_BC,
        xr_bc=SOMMERFELD_BC,
        yl_bc=PERIODIC_BC,
        yr_bc=PERIODIC_BC,
        zl_bc=PERIODIC_BC,
        zr_bc=PERIODIC_BC,
        x_min=-3.5 * dx,
        y_min=-4.0 * dx,
        z_min=z_min,
        mad_q=1.0,
    )


def test_cartoon_maxwell_divergences_use_the_active_W_floor():
    dx = 0.2

    axisymmetric_bssn, axisymmetric_em = _zero_W_state((10, 1, 9))
    axisymmetric = compute_axisymmetric_constraint_divergences(
        axisymmetric_em,
        axisymmetric_bssn,
        _cartoon_parameters(dx, -4.0 * dx),
    )

    spherical_bssn, spherical_em = _zero_W_state((10, 1, 1))
    spherical = compute_cartoon_constraint_divergences(
        spherical_em,
        spherical_bssn,
        _cartoon_parameters(dx, -4.0 * dx),
    )

    for divergence in axisymmetric + spherical:
        assert bool(jnp.all(jnp.isfinite(divergence)))
        np.testing.assert_array_equal(divergence, jnp.zeros_like(divergence))
