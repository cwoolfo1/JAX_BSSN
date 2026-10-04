"""Independent continuum audit and space/time convergence for Rosen EM waves."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from demos.EM_waves.run import run, rms, physical
import jax
import jax.numpy as jnp
import numpy as np
from JAX_BSSN.bssn.variables import BSSNParameters
from JAX_BSSN.evolution.time_evolve import compute_bssn_rhs_with_matter
from JAX_BSSN.EM.first_order.energy_momentum import compute_densitized_electromagnetic_energy_momentum
from demos.EM_waves.solution import RosenSolution, GHOSTS


def symbolic_audit():
    """Build curvature and Maxwell tensors independently of the BSSN code."""
    import sympy as s
    w=s.symbols('w',real=True)
    a=s.Function('a')(w); f=s.Function('f')(w)
    g=s.diag(-1,a*a,a*a,1); inv=g.inv()
    du=(1,0,0,-1)
    derivative=lambda q,i: du[i]*s.diff(q,w)
    simp=lambda q: s.simplify(q.subs(s.diff(a,w,2),-f*f*a))
    connection=[[[s.simplify(sum(inv[i,l]*(derivative(g[l,j],k)+derivative(g[l,k],j)-derivative(g[j,k],l))/2 for l in range(4))) for k in range(4)] for j in range(4)] for i in range(4)]
    ricci=s.Matrix(4,4,lambda i,j: s.simplify(sum(
        derivative(connection[k][i][j],k)-derivative(connection[k][i][k],j)
        +sum(connection[k][i][j]*connection[l][k][l]-connection[l][i][k]*connection[k][j][l] for l in range(4)) for k in range(4))))
    scalar=s.simplify(sum(inv[i,j]*ricci[i,j] for i in range(4) for j in range(4)))
    F=s.zeros(4); F[0,1]=a*f/s.sqrt(4*s.pi); F[3,1]=-F[0,1]; F=F-F.T
    raised=inv*F*inv
    invariant=s.simplify(sum(F[i,j]*raised[i,j] for i in range(4) for j in range(4)))
    stress=F*inv*F.T-g*invariant/4
    einstein=[simp(q) for q in ricci-g*scalar/2-8*s.pi*stress]
    divergence=[simp(sum(derivative(a*a*raised[i,j],i) for i in range(4))) for j in range(4)]
    closure=[simp(derivative(F[i,j],k)+derivative(F[j,k],i)+derivative(F[k,i],j)) for i in range(4) for j in range(4) for k in range(4)]
    dual=sum(s.LeviCivita(i,j,k,l)*F[i,j]*F[k,l] for i in range(4) for j in range(4) for k in range(4) for l in range(4))
    checks=einstein+divergence+closure+[invariant,simp(dual),simp(stress[0,0]-f*f/(4*s.pi))]
    return dict(passed=all(q==0 for q in checks),nonzero_residuals=[str(q) for q in checks if q!=0],
                method='Independent 4D Christoffel/Ricci/Einstein tensor, Maxwell divergence and closure, both null invariants',
                rho=str(stress[0,0]),ode="a''=-f^2*a")


def negative_control(n=512):
    dx=8/n; dt=.2*dx
    z=-4+(np.arange(n+2*GHOSTS)-GHOSTS+.5)*dx
    sol=RosenSolution(z,dx,dt,4.)
    p=BSSNParameters(dx=dx,dt=dt,nu=0,kappa=0,zero_shift=1)
    state=sol.bssn(1.)
    d,b=sol.fields(1.)
    rho,momentum,stress=compute_densitized_electromagnetic_energy_momentum(d,b,state,p)
    coupled=compute_bssn_rhs_with_matter(state,p,rho,momentum,stress)
    omitted=compute_bssn_rhs_with_matter(state,p,0*rho,0*momentum,0*stress)
    # Exact K_t = -2(a''/a - (a'/a)^2), using the ODE directly.
    from demos.EM_waves.solution import profile
    w=1.-sol.z-sol.packet.offset
    a,ap=sol.scale(w)
    exact=2*(profile(w,sol.packet)**2+(ap/a)**2)
    errors={name:rms(physical(rhs.trace_K-exact)) for name,rhs in [('coupled',coupled),('omitted_matter',omitted)]}
    errors['passed']=errors['omitted_matter']>20*errors['coupled']
    return errors


def pair_orders(values):
    values=np.asarray(values)
    return np.log2(values[:-1]/values[1:]).tolist()


def validate(output,device='cpu'):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'validation.json').exists(): raise FileExistsError(output/'validation.json')
    audit=symbolic_audit()
    if not audit['passed']: raise AssertionError(audit)
    print('Independent continuum audit passed',flush=True)
    spatial=[]
    for n in (128,256,512):
        rows=run(output/f'n{n}',n=n,device=device,frames=201)
        spatial.append(rows[-1])
    orders={k:pair_orders([r['errors'][k]['rms'] for r in spatial])
            for k in spatial[0]['errors'] if min(r['errors'][k]['rms'] for r in spatial)>1e-11}
    constraints={k:pair_orders([r['constraints'][k] for r in spatial])
                 for k in spatial[0]['constraints'] if min(r['constraints'][k] for r in spatial)>1e-11}
    # Hold space fixed: compare full numerical final states, not errors against
    # the continuum, which plateau at the fixed spatial truncation error.
    time_paths=[output/'n256']
    for cfl,label in ((.1,'dt_half'),(.05,'dt_quarter')):
        path=output/label
        run(path,n=256,cfl=cfl,device=device,frames=2)
        time_paths.append(path)
    time_states=[np.load(p/'snapshots.npz') for p in time_paths]
    time_orders={}
    for key in ('D_native','B_native','bssn_conformal_factor','bssn_trace_K','bssn_conformal_metric','bssn_traceless_K','bssn_conformal_connection'):
        differences=[rms(time_states[i][key][-1]-time_states[i+1][key][-1]) for i in (0,1)]
        time_orders[key]=dict(differences=differences,order=float(np.log2(differences[0]/differences[1])))
    control=negative_control()
    checks=dict(continuum=audit['passed'],
                spatial=all(min(v)>1.7 for v in orders.values()),
                constraints=all(min(v)>1.7 for v in constraints.values()),
                temporal=all(v['order']>1.7 for v in time_orders.values()),
                em_accuracy=max(spatial[-1]['errors'][k]['relative'] for k in ('D','B'))<.02,
                negative_control=control['passed'])
    report=dict(passed=all(checks.values()),checks=checks,continuum=audit,spatial_final=spatial,
                spatial_orders=orders,constraint_orders=constraints,time_refinement=time_orders,negative_control=control)
    (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    write_report(output,report)
    print(json.dumps(checks),flush=True)
    if not report['passed']: raise AssertionError('Validation failed; see validation.json')
    return report


def write_report(output,report):
    spatial=report['spatial_final']
    lines=['# Rosen EM-wave validation','',f"Passed: {report['passed']}",'',
           'Independent 4D Einstein-Maxwell audit: '+str(report['continuum']['passed']), '',
           '| Field | N=128 RMS | N=256 RMS | N=512 RMS | Orders |','|---|---:|---:|---:|---|']
    for key,order in report['spatial_orders'].items():
        values=' | '.join(f"{r['errors'][key]['rms']:.6g}" for r in spatial)
        lines.append(f"| {key} | {values} | {', '.join(f'{o:.3f}' for o in order)} |")
    lines+=['','## Temporal self-convergence','','| Field | Order |','|---|---:|']
    lines += [f"| {key} | {value['order']:.4f} |" for key,value in report['time_refinement'].items()]
    lines+=['','## Constraint convergence','','| Constraint | Spatial orders |','|---|---|']
    lines += [f"| {key} | {', '.join(f'{p:.4f}' for p in value)} |" for key,value in report['constraint_orders'].items()]
    control=report['negative_control']
    lines+=['','## Accuracy and coupling control','',
            f"Finest final relative RMS D error: {100*spatial[-1]['errors']['D']['relative']:.4f}%.",
            f"Finest final relative RMS B error: {100*spatial[-1]['errors']['B']['relative']:.4f}%.",'',
            f"Instantaneous K RHS RMS error: {control['coupled']:.6g} with matter; "
            f"{control['omitted_matter']:.6g} with the matter source omitted.",'',
            'The reference and gauge are prescribed only at boundaries and in lapse/shift; all physical interior fields evolve.']
    (Path(output)/'validation.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path(__file__).parent/'output_validation')
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    validate(args.output,args.device)
