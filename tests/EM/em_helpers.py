"""Shared flat-space initial data for wave-solver tests."""

import jax.numpy as jnp

from JAX_BSSN.bssn import BSSNVariables

from JAX_BSSN.EM.second_order.variables import EMVariables


def flat_bssn_variables(shape, dtype=jnp.float64):
    metric = jnp.eye(3, dtype=dtype)[:, :, None, None, None]
    metric = jnp.broadcast_to(metric, (3, 3) + shape)
    zero = jnp.zeros(shape, dtype=dtype)
    vector_zero = jnp.zeros((3,) + shape, dtype=dtype)

    return BSSNVariables(
        conformal_metric=metric,
        conformal_factor=jnp.ones(shape, dtype=dtype),
        traceless_K=jnp.zeros_like(metric),
        trace_K=zero,
        conformal_connection=vector_zero,
        lapse=jnp.ones(shape, dtype=dtype),
        shift=vector_zero,
    )


def zero_bssn_rhs(bssn):
    return BSSNVariables(*(jnp.zeros_like(field) for field in bssn))


def zero_em_variables(shape, dtype=jnp.float64):
    vector_zero = jnp.zeros((3,) + shape, dtype=dtype)
    return EMVariables(
        vector_zero,
        vector_zero,
        vector_zero,
        vector_zero,
    )

