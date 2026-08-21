"""Six-point interpolation for spherical Cartoon reconstruction."""

from functools import partial

import jax.numpy as jnp
from jax import jit


def lagrange6_weights(q: jnp.ndarray, nodes: jnp.ndarray) -> jnp.ndarray:
    """Return degree-five Lagrange weights for six source nodes."""

    q = jnp.asarray(q)
    nodes = jnp.asarray(nodes)
    weights = jnp.ones(nodes.shape, dtype=jnp.result_type(q, nodes, 1.0))

    for k in range(6):
        for m in range(6):
            if k != m:
                weights = weights.at[..., k].multiply(
                    (q - nodes[..., m]) / (nodes[..., k] - nodes[..., m])
                )

    return weights


def _interpolate_nodes(array, nodes, weights, axis):
    """Gather and contract one interpolation axis."""

    axis = axis % array.ndim
    targets_ndim = nodes.ndim - 1
    moved = jnp.moveaxis(array, axis, -1)
    gathered = jnp.take(moved, nodes, axis=-1)

    leading_shape = moved.shape[:-1]
    weight_shape = (1,) * len(leading_shape) + weights.shape
    interpolated = jnp.sum(gathered * weights.reshape(weight_shape), axis=-1)

    # jnp.take appends target axes after the non-interpolated axes. Put the
    # target geometry where the source interpolation axis was located.
    axes_before = tuple(range(axis))
    axes_after = tuple(range(axis, array.ndim - 1))
    target_start = array.ndim - 1
    target_axes = tuple(range(target_start, target_start + targets_ndim))
    permutation = axes_before + target_axes + axes_after

    return jnp.transpose(interpolated, permutation)


@partial(jit, static_argnames=("axis",))
def lagrange6_nonperiodic(
    array: jnp.ndarray,
    q: jnp.ndarray,
    axis: int = -1,
) -> jnp.ndarray:
    """Interpolate or extrapolate from the nearest six valid source nodes."""

    q = jnp.asarray(q)
    source_size = array.shape[axis]
    base = jnp.floor(q).astype(jnp.int32)
    start = jnp.clip(base - 2, 0, source_size - 6)
    nodes = start[..., None] + jnp.arange(6, dtype=jnp.int32)
    weights = lagrange6_weights(q, nodes)

    return _interpolate_nodes(array, nodes, weights, axis)
