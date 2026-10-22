"""Original arithmetic weights, native placement, and metric guard contracts."""

from dataclasses import replace
from itertools import product
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from JAX_BSSN.bssn import BSSNParameters
from JAX_BSSN.EM.first_order import make_em_grid
from JAX_BSSN.EM.first_order.geometry import build_yee_metric
from tests.EM.em_helpers import flat_bssn_variables


def test_native_metric_uses_equal_neighbor_weights_through_outer_guards():
    p = BSSNParameters(dx=.2)
    grid = make_em_grid((7,8,9), p, boundary_conditions=("constant",)*3)
    padded = replace(grid, static=grid.static._replace(guard_cells=grid.guard_cells+1))
    values = jax.random.normal(jax.random.key(371), grid.shape)
    field = padded.to_tile(values)[0,0,0]
    bssn = flat_bssn_variables(grid.shape)._replace(lapse=values)
    native = jax.jit(lambda q:build_yee_metric(q,grid))(bssn)
    # Includes lower/upper physical faces and the outermost retained ghosts.
    indices = jnp.array(list(product((0,3,12), (0,3,13), (0,3,14))))
    locations = (("V","C","C"),("C","V","C"),("C","C","V"),
                 ("C","V","V"),("V","C","V"),("V","V","C"),
                 ("C","C","C"),("V","V","V"))
    for location, metric in zip(locations, (*native.D,*native.B,native.center,native.vertex)):
        # Independent direct gather of the 1, 2, 4, or 8 neighboring centers;
        # one temporary guard provides the upper neighbor at the last site.
        offsets = jnp.array(list(product(*(
            (0,1) if loc == "V" else (0,) for loc in location))))
        neighbors = indices[:,None,:] + 1 + offsets[None,:,:]
        expected = jnp.mean(field[tuple(jnp.moveaxis(neighbors,-1,0))], axis=1)
        actual = metric.lapse[0,0,0][tuple(indices.T)]
        assert bool(jnp.allclose(actual,expected,atol=2e-13,rtol=2e-13))


def test_affine_geometry_has_exact_native_coordinates_including_ghosts():
    p = BSSNParameters(dx=.25,x_min=-1.,y_min=-2.,z_min=-3.)
    grid = make_em_grid((7,8,9),p,boundary_conditions=("constant",)*3)
    x,y,z = grid.coordinates(p)
    field = jnp.broadcast_to(1+.01*x+.02*y-.03*z,grid.shape)
    bssn = flat_bssn_variables(grid.shape)._replace(lapse=field,shift=jnp.stack((field,2*field,-field)))
    metric = jax.jit(lambda q:build_yee_metric(q,grid))(bssn)
    locations = (("V","C","C"),("C","V","C"),("C","C","V"),
                 ("C","V","V"),("V","C","V"),("V","V","C"),("C","C","C"),("V","V","V"))
    for loc,m in zip(locations,(*metric.D,*metric.B,metric.center,metric.vertex)):
        x,y,z = grid.coordinates(p,loc,tiled=True)
        expected = jnp.broadcast_to(1+.01*x+.02*y-.03*z,m.lapse.shape)
        assert jnp.allclose(m.lapse,expected,atol=1e-14,rtol=1e-14)
        assert jnp.allclose(m.shift[...,1],2*expected,atol=2e-14,rtol=1e-14)
        assert jnp.allclose(m.sqrt_gamma,1.,atol=2e-14,rtol=1e-14)

