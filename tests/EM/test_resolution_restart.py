"""Interpolation placement and equivalence of the memory-limited runner."""

from dataclasses import replace
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from demos.single_charged_blackhole.run_single_charged_blackhole import SingleChargedBlackHoleConfig,build_single_charged_blackhole,state_is_valid
from demos.single_charged_blackhole.resolution_restart import interpolate_uniform,make_staged_step,save_native_checkpoint,interpolated_restart,staged_state_is_valid
from JAX_BSSN.EM.first_order.evolve import first_order_einstein_maxwell_step,common_densitized_fields


def test_cubic_transfer_at_center_and_native_coordinates():
    dx=.5
    for location in (('C','C','C'),('V','C','C'),('C','V','V')):
        origins=tuple(-2+.5*dx*(l=='V') for l in location)
        axes=[o+dx*jnp.arange(9) for o in origins]
        x,y,z=jnp.meshgrid(*axes,indexing='ij')
        f=lambda x,y,z:1+x**3+.2*x*y-2*y*y+z**3
        values=f(x,y,z)
        target=[jnp.linspace(o-.1,o+4.1,18) for o in origins]
        actual=interpolate_uniform(values,origins,dx,target)
        assert bool(jnp.allclose(actual,f(*jnp.meshgrid(*target,indexing='ij')),rtol=1e-12,atol=1e-12))
        assert bool(jnp.array_equal(interpolate_uniform(values,origins,dx,axes),values))


def test_staged_step_and_checkpoint_restart(tmp_path):
    config=SingleChargedBlackHoleConfig(num_points=12,half_width=3.,absorber_width=.75,
                                       final_time=.02,platform='cpu')
    state,params,grid,_=build_single_charged_blackhole(config)
    staged=make_staged_step(params,grid)
    a,b=state,state
    for _ in range(2):
        a=staged(a)
        b=first_order_einstein_maxwell_step(b,params,grid)
    for actual,expected in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):
        assert bool(jnp.allclose(actual,expected,rtol=3e-12,atol=3e-13))
    path=tmp_path/'native.npz'
    save_native_checkpoint(path,a,params,config,.04)
    restarted,_,restarted_grid,time=interpolated_restart(path,replace(config,final_time=.1))
    assert time==.04
    for actual,expected in zip(jax.tree_util.tree_leaves(common_densitized_fields(restarted.em)),jax.tree_util.tree_leaves(common_densitized_fields(a.em))):
        assert bool(jnp.allclose(actual,expected,rtol=1e-13,atol=1e-14))
    fine,p,fine_grid,_=interpolated_restart(path,replace(config,num_points=24,final_time=.1))
    assert bool(state_is_valid(fine))
    assert fine.em.half_dt==p.dt/2


def test_staged_validity_matches_monolithic_predicate():
    from typing import NamedTuple
    import jax.numpy as jnp
    class Geometry(NamedTuple):
        conformal_metric: object
        conformal_factor: object
    class State(NamedTuple):
        bssn: object
        fields: object
    metric=jnp.broadcast_to(jnp.eye(3)[:,:,None,None,None],(3,3,3,3,3))
    geometry=Geometry(metric,jnp.ones((3,3,3)))
    state=State(geometry,jnp.ones((3,3,3)))
    cases=(state,state._replace(fields=state.fields.at[0,0,0].set(jnp.nan)),
           state._replace(bssn=geometry._replace(conformal_factor=-geometry.conformal_factor)),
           state._replace(bssn=geometry._replace(conformal_metric=metric.at[2,2,0,0,0].set(-1.))))
    for candidate in cases:
        assert staged_state_is_valid(candidate)==bool(state_is_valid(candidate))
