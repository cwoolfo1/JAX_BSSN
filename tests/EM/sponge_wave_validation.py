"""Outgoing packet reflection against a larger-domain Yee reference.

Run with ``python -m tests.EM.sponge_wave_validation OUTPUT_DIRECTORY``.
The 1D cases cover both directions of all axes; 2D cases hit faces obliquely
and a corner. All initial packets have zero native magnetic divergence.
"""

import json
import math
from pathlib import Path
import sys
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from PyPIC3D.relativity.core import D_FIELD_LOCATIONS, B_FIELD_LOCATIONS
from PyPIC3D.diagnostics.static_metric import divergence
from JAX_BSSN.bssn import BSSNParameters
from JAX_BSSN.EM.first_order.grid import make_em_grid
from JAX_BSSN.EM.first_order.absorber import make_reference_sponge
from JAX_BSSN.EM.first_order.geometry import build_yee_metric
from JAX_BSSN.EM.first_order.maxwell import initialize_native_state, maxwell_half_step
from JAX_BSSN.EM.first_order.evolve import common_densitized_fields
from tests.EM.em_helpers import flat_bssn_variables


def packet_run(direction, mode, *, n=96, duration=28.):
    active = tuple(abs(x)>0 for x in direction)
    extent = 36. if mode == 'large' else 12.
    dx = 24./n
    shape = tuple((3*n if mode == 'large' else n) if on else 1 for on in active)
    p = BSSNParameters(dx=dx,dt=.05,
                       x_min=-extent+dx/2 if active[0] else 0.,
                       y_min=-extent+dx/2 if active[1] else 0.,
                       z_min=-extent+dx/2 if active[2] else 0.)
    grid = make_em_grid(shape,p,boundary_conditions=tuple('constant' if on else 'periodic' for on in active))
    native_shape = (1,1,1)+tuple(s+2*grid.guard_cells for s in grid.tile_shape)
    d,b = [],[]
    axis = next(i for i,on in enumerate(active) if on)
    one_dimensional = sum(active)==1
    electric_axis = (axis+1)%3
    magnetic_axis = (axis+2)%3

    def profile(x):
        return jnp.exp(-.5*(x/1.2)**2)*jnp.cos(2*jnp.pi*x/3.)

    def potential(x,y):
        s = direction[0]*x+direction[1]*y
        u = -direction[1]*x+direction[0]*y
        k = 2*jnp.pi/4.5
        return jnp.exp(-.5*(s/1.5)**2-.5*(u/2.5)**2)*jnp.sin(k*s)/k

    for kind,locs,values in (('D',D_FIELD_LOCATIONS,d),('B',B_FIELD_LOCATIONS,b)):
        for component,loc in enumerate(locs):
            xyz = grid.coordinates(p,loc,tiled=True)
            if one_dimensional:
                live = electric_axis if kind=='D' else magnetic_axis
                value = profile(xyz[axis])*(1 if kind=='D' else direction[axis]) if component==live else 0.
            else:
                x,y,_ = xyz
                if kind=='D' and component==2:
                    s = direction[0]*x+direction[1]*y
                    u = -direction[1]*x+direction[0]*y
                    k = 2*jnp.pi/4.5
                    value = jnp.exp(-.5*(s/1.5)**2-.5*(u/2.5)**2)*(jnp.cos(k*s)-s/(1.5**2*k)*jnp.sin(k*s))
                elif kind=='B' and component==0:
                    value = (potential(x,y+dx/2)-potential(x,y-dx/2))/dx
                elif kind=='B' and component==1:
                    value = -(potential(x+dx/2,y)-potential(x-dx/2,y))/dx
                else:
                    value = 0.
            values.append(jnp.broadcast_to(jnp.asarray(value),native_shape))
    d,b=tuple(d),tuple(b)
    absorber = make_reference_sponge(grid,p,d,b,width=3.) if mode=='sponge' else None
    metric = build_yee_metric(flat_bssn_variables(shape),grid)
    em = initialize_native_state(d,b,metric,p,grid,absorber)
    initial_energy = sum(float(jnp.sum(grid.from_tile(q)**2)) for v in (d,b) for q in v)
    initial_div = max(float(jnp.max(jnp.abs(grid.from_tile(divergence(v,grid.dynamic(p,p.dt/2),forward=f)))))
                      for v,f in ((d,False),(b,True)))
    steps = round(2*duration/p.dt)
    em = jax.jit(lambda q: jax.lax.fori_loop(0,steps,lambda i,s:maxwell_half_step(s,p,grid),q))(em)
    fields = common_densitized_fields(em)
    final = np.stack([np.asarray(grid.from_tile(q)) for v in fields for q in v])
    coordinates = grid.coordinates(p)
    core = jnp.ones(shape,dtype=bool)
    for x,on in zip(coordinates,active):
        if on:
            core = core & (jnp.abs(x)<9.)
    constraints = {}
    for label,v,forward in (('electric',fields[0],False),('magnetic',fields[1],True)):
        div = grid.from_tile(divergence(v,grid.dynamic(p,p.dt/2),forward=forward))
        for region,mask in (('core',core),('layer',~core)):
            constraints[f'{region}_{label}_div_rms'] = float(jnp.sqrt(jnp.sum(jnp.where(mask,div**2,0))/jnp.sum(mask)))
    return final,np.asarray(core),initial_energy,initial_div,constraints


def main(output):
    output.mkdir(parents=True,exist_ok=True)
    records=[]
    directions=[]
    for axis in range(3):
        for sign in (-1,1):
            direction=[0.,0.,0.]
            direction[axis]=float(sign)
            directions.append((direction,96))
    directions += [([2/math.sqrt(5),1/math.sqrt(5),0.],64),([-1/math.sqrt(2),1/math.sqrt(2),0.],64)]
    for direction,n in directions:
        large,_,_,_,_=packet_run(direction,'large',n=n)
        slices=tuple(slice(n,2*n) if v else slice(None) for v in direction)
        reference=large[(slice(None),)+slices]
        entry={'direction':direction,'n':n,'dx':24/n,'duration':28.}
        for mode in ('constant','sponge'):
            final,core,energy,initial_div,constraints=packet_run(direction,mode,n=n)
            entry[mode]={'reflected_energy_fraction':float(np.sum((final[:,core]-reference[:,core])**2)/energy),
                         'initial_div_max':initial_div,**constraints}
        entry['passed']=(entry['sponge']['reflected_energy_fraction']<.01 if sum(bool(v) for v in direction)==1
                         else entry['sponge']['reflected_energy_fraction']<entry['constant']['reflected_energy_fraction'])
        records.append(entry)
        (output/'wave_reflection.json').write_text(json.dumps(records,indent=2)+'\n')
        print(json.dumps(entry),flush=True)
        jax.clear_caches()
    assert all(r['passed'] for r in records), 'Reflection acceptance failed; see wave_reflection.json'


if __name__=='__main__':
    main(Path(sys.argv[1]))
