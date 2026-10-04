"""Evolve an exact Rosen electromagnetic packet with production RK4/leapfrog."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time as clock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from JAX_BSSN.bssn.variables import BSSNParameters
from JAX_BSSN.bssn.constraints import compute_all_constraints_with_matter
from JAX_BSSN.EM.first_order.evolve import first_order_einstein_maxwell_step, common_densitized_fields
from JAX_BSSN.EM.first_order.energy_momentum import compute_densitized_electromagnetic_energy_momentum, physical_fields_at_centers
from JAX_BSSN.EM.first_order.equations import densitized_displacement_divergence, densitized_magnetic_divergence
from demos.EM_waves.solution import Packet, RosenSolution, GHOSTS, prescribed_gauge, profile


def physical(q):
    return np.asarray(q)[...,0,0,GHOSTS:-GHOSTS]


def rms(q):
    return float(np.sqrt(np.mean(np.asarray(q)**2)))


def component_errors(q, reference, baseline=0):
    q, reference = physical(q), physical(reference)
    delta = q-reference
    absolute = np.sqrt(np.mean(delta**2,axis=-1))
    scale = np.sqrt(np.mean((reference-baseline)**2,axis=-1))
    relative = np.where(scale > 1e-12, absolute/np.maximum(scale,1e-30), np.nan)
    return dict(rms=rms(delta), max=float(np.max(abs(delta))),
                component_rms=absolute.tolist(),
                component_relative=np.where(np.isfinite(relative),relative,None).tolist(),
                relative=rms(delta)/max(rms(reference-baseline),1e-30))


def make_diagnostic(params):
    @jax.jit
    def diagnostic(state):
        d,b = common_densitized_fields(state.em)
        rho,momentum,_ = compute_densitized_electromagnetic_energy_momentum(d,b,state.bssn,params)
        constraints = compute_all_constraints_with_matter(state.bssn,params,rho,momentum)
        e,mag = physical_fields_at_centers(d,b,state.bssn,params)
        return d,b,rho,e,mag,constraints,densitized_displacement_divergence(d,params),densitized_magnetic_divergence(b,params)
    return diagnostic


def snapshot(state, t, solution, diagnostic):
    d,b,rho,e,mag,constraints,divd,divb = diagnostic(state)
    ref = solution.bssn(t)
    rd,rb = solution.fields(t)
    w = t-solution.z-solution.packet.offset
    a,_ = solution.scale(w)
    f = profile(w,solution.packet)
    exact_e = -f/(a*jnp.sqrt(4*jnp.pi))
    metric = state.bssn.conformal_metric/state.bssn.conformal_factor**2
    errors = {}
    for name,actual,expected in zip(ref._fields,state.bssn,ref):
        baseline = np.eye(3)[:,:,None] if name=='conformal_metric' else (1 if name in ('lapse','conformal_factor') else 0)
        errors[name] = component_errors(actual,expected,baseline)
    errors['D'] = component_errors(d,rd)
    errors['B'] = component_errors(b,rb)
    errors['rho'] = component_errors(rho,f*f/(4*jnp.pi))
    norms = {name:rms(physical(value)) for name,value in zip(constraints._fields,constraints)}
    norms.update(div_D=rms(physical(divd)),div_B=rms(physical(divb)))
    arrays = dict(time=np.asarray(t), E=physical(e[0]), B=physical(mag[1]),
                  E_exact=physical(exact_e),B_exact=physical(exact_e),
                  rho=physical(rho),rho_exact=physical(f*f/(4*jnp.pi)),
                  gamma_xx=physical(metric[0,0]),gamma_xx_exact=physical(a*a),
                  W=physical(state.bssn.conformal_factor),W_exact=physical(ref.conformal_factor),
                  K=physical(state.bssn.trace_K),K_exact=physical(ref.trace_K))
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise FloatingPointError(f'Nonfinite solution at t={t}')
    # Store every evolved component and both native fields for independent reuse.
    for name,q,r in zip(ref._fields,state.bssn,ref):
        arrays['bssn_'+name],arrays['exact_'+name] = physical(q),physical(r)
    arrays.update(D_native=physical(d),B_native=physical(b),D_native_exact=physical(rd),B_native_exact=physical(rb))
    return arrays,dict(time=float(t),errors=errors,constraints=norms)


def run(output, *, n=256, duration=4., cfl=.2, packet=Packet(), frames=201, device='cpu'):
    output=Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    if (not all(np.isfinite(v) for v in (duration,cfl,*asdict(packet).values()))
            or n<16 or duration<=0 or not 0<cfl<=.2 or frames<2):
        raise ValueError('Require n>=16, duration>0, 0<cfl<=0.2, frames>=2')
    devices=jax.devices(device.split(':')[0])
    selected=devices[int(device.split(':')[1]) if ':' in device else 0]
    with jax.default_device(selected):
        dx=8/n
        steps=int(np.ceil(duration/(cfl*dx)))
        dt=duration/steps
        z=-4+(np.arange(n+2*GHOSTS)-GHOSTS+.5)*dx
        solution=RosenSolution(z,dx,dt,duration,packet)
        params=BSSNParameters(dx=dx,dt=dt,nu=0.,kappa=0.,zero_shift=1,z_min=float(z[0]))
        state=solution.state(0.,dt)
        boundary=solution.boundary
        @jax.jit
        def step(state,t):
            return first_order_einstein_maxwell_step(state,params,time=t,
                         prescribed_gauge=prescribed_gauge,stage_boundary=boundary)
        diagnostic=make_diagnostic(params)
        selected_steps=set(np.rint(np.linspace(0,steps,min(frames,steps+1))).astype(int))
        output.mkdir(parents=True)
        config=dict(n=n,duration=duration,cfl=cfl,dx=dx,dt=dt,steps=steps,packet=asdict(packet),
                    device=str(selected),ghost_cells=GHOSTS,reference=solution.metadata,
                    source='arXiv:1202.0540v3, Eqs. 2.10-2.19',status='running')
        (output/'configuration.json').write_text(json.dumps(config,indent=2)+'\n')
        rows, snapshots=[],[]
        start=clock.monotonic()
        for i in range(steps+1):
            if i in selected_steps:
                arrays,row=snapshot(state,i*dt,solution,diagnostic)
                snapshots.append(arrays); rows.append(row)
            if i==steps: break
            state=step(state,jnp.asarray(i*dt))
            if i%max(1,steps//8)==0:
                jax.block_until_ready(state)
                print(f'n={n} step={i+1}/{steps} t={(i+1)*dt:.3f}',flush=True)
        data={key:np.stack([s[key] for s in snapshots]) for key in snapshots[0]}
        data.update(z=z[GHOSTS:-GHOSTS],z_B=z[GHOSTS:-GHOSTS]-dx/2)
        np.savez_compressed(output/'snapshots.npz',**data)
        final={name:np.asarray(q) for name,q in zip(state.bssn._fields,state.bssn)}
        final.update({name:np.asarray(q) for name,q in zip(state.em._fields,state.em)})
        np.savez_compressed(output/'final_state.npz',**final,z=z,time=duration)
        (output/'diagnostics.json').write_text(json.dumps(rows,indent=2)+'\n')
        config.update(status='complete',elapsed_seconds=clock.monotonic()-start,frames=len(rows))
        (output/'configuration.json').write_text(json.dumps(config,indent=2)+'\n')
        print(json.dumps(dict(output=str(output),seconds=config['elapsed_seconds'],
                             final_D_error=rows[-1]['errors']['D']['relative'],
                             final_B_error=rows[-1]['errors']['B']['relative'])),flush=True)
        return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path(__file__).parent/'output')
    p.add_argument('--n',type=int,default=256)
    p.add_argument('--duration',type=float,default=4.)
    p.add_argument('--cfl',type=float,default=.2)
    p.add_argument('--frames',type=int,default=201)
    p.add_argument('--device',default='cpu',help='cpu, gpu:0, or gpu:1')
    p.add_argument('--amplitude',type=float,default=.1)
    p.add_argument('--width',type=float,default=.6)
    p.add_argument('--wavelength',type=float,default=1.5)
    args=p.parse_args()
    run(args.output,n=args.n,duration=args.duration,cfl=args.cfl,frames=args.frames,device=args.device,
        packet=Packet(args.amplitude,args.width,args.wavelength))

if __name__=='__main__': main()
