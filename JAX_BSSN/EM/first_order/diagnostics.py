"""Synchronized physical output and imported native Gauss constraints."""

from functools import partial
import jax
from PyPIC3D.diagnostics.static_metric import divergence
from .evolve import common_densitized_fields, common_physical_fields


@partial(jax.jit, static_argnames=("grid",))
def first_order_constraint_divergences(state, params, grid):
    d, b = common_densitized_fields(state.em)
    dynamic = grid.dynamic(params, state.em.half_dt)
    return (
        grid.from_tile(divergence(d, dynamic)),
        grid.from_tile(divergence(b, dynamic, forward=True)),
    )


def electromagnetic_output_fields(state, grid):
    d, b = common_physical_fields(state, grid)
    return {"D": tuple(d), "B": tuple(b)}
