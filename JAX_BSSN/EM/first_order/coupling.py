"""Gravity-specific quadratic moments of PyPIC3D's native densities.

Square native samples before averaging, retaining unresolved field energy.
The Maxwell constitutive operator is owned exclusively by PyPIC3D.
"""

from itertools import product
import jax
import jax.numpy as jnp
from PyPIC3D.relativity.core import D_FIELD_LOCATIONS, B_FIELD_LOCATIONS
from JAX_BSSN.bssn.geometry import W_FLOOR_VALUE
from JAX_BSSN.bssn.tensor_algebra import determinant_3x3_metric


def quadratic_moments(displacement, magnetic, grid):
    """Return <DD>, <BB>, <DB> on BSSN C nodes from refreshed native tiles.

    A V sample lies above C. Each C node gathers V from its current and
    lower neighbouring index, sharing corner choices between components.
    """

    def gather(vector, locations, corner):
        return jnp.stack(
            tuple(
                grid.from_tile(
                    jnp.roll(
                        value,
                        tuple(
                            c if loc == "V" else 0 for c, loc in zip(corner, location)
                        ),
                        axis=(3, 4, 5),
                    )
                )
                for value, location in zip(vector, locations)
            )
        )

    dd = bb = db = jnp.zeros((3, 3) + grid.shape, dtype=displacement[0].dtype)
    for corner in product((0, 1), repeat=3):
        d = gather(displacement, D_FIELD_LOCATIONS, corner)
        b = gather(magnetic, B_FIELD_LOCATIONS, corner)
        dd += jnp.einsum("i...,j...->ij...", d, d) / 8
        bb += jnp.einsum("i...,j...->ij...", b, b) / 8
        db += jnp.einsum("i...,j...->ij...", d, b) / 8
    return dd, bb, db


def cell_metric(bssn):
    metric = bssn.conformal_metric
    return metric * determinant_3x3_metric(metric) ** (-1 / 3)


@jax.jit
def sources_from_moments(moments, bssn):
    """Eulerian rho, S_i, S_ij; retain JAX_BSSN's rationalized EM units."""
    dd, bb, db = moments
    q = dd + bb
    W = jnp.maximum(bssn.conformal_factor, W_FLOOR_VALUE)
    metric = cell_metric(bssn)
    contraction = jnp.einsum("ij...,ij...->...", metric, q)
    rho = 0.5 * W**4 * contraction
    momentum = W**3 * jnp.stack(
        (db[1, 2] - db[2, 1], db[2, 0] - db[0, 2], db[0, 1] - db[1, 0])
    )
    stress = W**2 * (
        0.5 * metric * contraction
        - jnp.einsum("ik...,jl...,kl...->ij...", metric, metric, q)
    )
    return rho, momentum, stress
