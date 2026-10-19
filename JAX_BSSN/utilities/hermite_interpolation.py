"""C1 Hermite interpolation of metric and extrinsic-curvature components.

Use tensor-product cubic polynomials with fourth-order centered nodal slopes
on uniform radial, planar, or Cartesian grids. Tensor reconstruction and
spatial differentiation belong to the apparent-horizon module.
"""

import jax.numpy as jnp


def hermite_weights(index, size):
    """Six nodal weights for cubic Hermite interpolation with fourth-order slopes."""
    cell = jnp.clip(jnp.floor(index).astype(int), 2, size - 4)
    t = index - cell
    h00 = 2*t**3 - 3*t**2 + 1
    h01 = -2*t**3 + 3*t**2
    h10 = t**3 - 2*t**2 + t
    h11 = t**3 - t**2
    # The endpoint slopes use the same centered stencil in adjacent cells.
    left_slope = jnp.array([1., -8., 0., 8., -1., 0.])/12
    right_slope = jnp.array([0., 1., -8., 0., 8., -1.])/12
    weights = h10*left_slope + h11*right_slope
    weights = weights.at[2].add(h00).at[3].add(h01)
    return cell + jnp.arange(-2, 4), weights


def interpolate_point(table, indices):
    """Tensor-product Hermite interpolation; table has shape (fields, *grid)."""
    nodes, weights = zip(*(hermite_weights(q, n)
                          for q, n in zip(indices, table.shape[1:])))
    if len(indices) == 1:
        return table[:, nodes[0]] @ weights[0]
    if len(indices) == 2:
        block = table[:, nodes[0][:, None], nodes[1][None, :]]
        return jnp.einsum('fab,a,b->f', block, *weights)
    block = table[:, nodes[0][:, None, None], nodes[1][None, :, None],
                  nodes[2][None, None, :]]
    return jnp.einsum('fabc,a,b,c->f', block, *weights)
