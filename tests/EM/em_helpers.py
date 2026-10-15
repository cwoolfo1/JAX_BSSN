"""Shared flat-space initial data for Cartesian Maxwell tests."""

import jax.numpy as jnp

from JAX_BSSN.bssn import BSSNVariables


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
