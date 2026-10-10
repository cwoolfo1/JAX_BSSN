"""Rosen form of Harte & Drivas, arXiv:1202.0540v3, Eqs. 2.10-2.19."""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
import jax
import jax.numpy as jnp
from JAX_BSSN.bssn.variables import BSSNVariables
from JAX_BSSN.EM.first_order import initialize_densitized_maxwell_state
from JAX_BSSN.EM.variables import EinsteinMaxwellVariables

GHOSTS = 8

@dataclass(frozen=True)
class Packet:
    amplitude: float = .1
    width: float = .6
    wavelength: float = 1.5
    offset: float = 2.


def profile(w, packet, xp=jnp):
    return packet.amplitude * xp.exp(-.5 * (w / packet.width)**2) * xp.cos(2 * xp.pi * w / packet.wavelength)


class RosenSolution:
    """High-accuracy ODE reference with differentiable cubic Hermite sampling.

    The table is much finer than the evolution grid. Its independently checked
    interpolation error is recorded, so reference accuracy is not assumed.
    """
    def __init__(self, z, dx, dt, duration, packet=Packet()):
        if packet.width <= 0 or packet.wavelength <= 0:
            raise ValueError('width and wavelength must be positive')
        self.packet, self.dx = packet, dx
        self.z = jnp.asarray(z)[None, None, :]
        lower = -dt - np.max(z) - packet.offset
        upper = duration + 1.5 * dt - np.min(z) + dx/2 - packet.offset
        start = min(-12 * packet.width, lower - packet.width)
        spacing = min(dx/16, packet.width/128, packet.wavelength/256)
        count = int(np.ceil((upper-start)/spacing)) + 1
        grid = np.linspace(start, upper, count)
        ode = solve_ivp(lambda w,y: (y[1], -profile(w,packet,np)**2*y[0]),
                        (start, upper), (1., 0.), method='DOP853',
                        rtol=1e-12, atol=1e-14, dense_output=True,
                        max_step=min(packet.width,packet.wavelength)/16)
        if not ode.success:
            raise RuntimeError(ode.message)
        a, ap = ode.sol(grid)
        # Stop before the first crossing, even if a later branch becomes positive.
        if np.min(a) < .5:
            raise ValueError('Requested interval approaches Rosen coordinate focusing (a < 0.5)')
        self.start, self.spacing = float(start), float(grid[1]-grid[0])
        self.a, self.ap = jnp.asarray(a), jnp.asarray(ap)
        check = (grid[:-1] + grid[1:])/2
        ai, api = self.scale(jnp.asarray(check))
        exact = ode.sol(check)
        self.metadata = dict(reference_start=start, reference_end=upper,
                             minimum_a=float(np.min(a)), table_spacing=self.spacing,
                             interpolation_error_a=float(np.max(abs(np.asarray(ai)-exact[0]))),
                             interpolation_error_ap=float(np.max(abs(np.asarray(api)-exact[1]))))
        if max(self.metadata['interpolation_error_a'], self.metadata['interpolation_error_ap']) > 1e-9:
            raise ValueError('Reference interpolation is insufficiently accurate')

    def scale(self, w):
        position = (w-self.start)/self.spacing
        index = jnp.clip(jnp.floor(position).astype(int), 0, self.a.size-2)
        s = position-index
        a0, a1 = self.a[index], self.a[index+1]
        p0, p1 = self.ap[index], self.ap[index+1]
        h = self.spacing
        a = (2*s**3-3*s**2+1)*a0 + (s**3-2*s**2+s)*h*p0 + (-2*s**3+3*s**2)*a1 + (s**3-s**2)*h*p1
        ap = ((6*s*s-6*s)*a0 + (-6*s*s+6*s)*a1)/h + (3*s*s-4*s+1)*p0 + (3*s*s-2*s)*p1
        return a, ap

    def bssn(self, time):
        w = time-self.z-self.packet.offset
        a, ap = self.scale(w)
        q = ap/a
        zero = jnp.zeros_like(a)
        metric = jnp.stack((jnp.stack((a**(2/3),zero,zero)),
                            jnp.stack((zero,a**(2/3),zero)),
                            jnp.stack((zero,zero,a**(-4/3)))))
        traceless = metric*jnp.stack((-q/3,-q/3,2*q/3))[:,None]
        connection = jnp.stack((zero,zero,4/3*q*a**(4/3)))
        return BSSNVariables(metric,a**(-2/3),traceless,-2*q,connection,
                             jnp.ones_like(a),jnp.stack((zero,zero,zero)))

    def fields(self, time):
        w = time-self.z-self.packet.offset
        wb = w-self.dx/2  # PyPIC3D B_y is on the upper z vertex.
        d = -self.scale(w)[0]*profile(w,self.packet)/jnp.sqrt(4*jnp.pi)
        b = -self.scale(wb)[0]*profile(wb,self.packet)/jnp.sqrt(4*jnp.pi)
        zero = jnp.zeros_like(d)
        return jnp.stack((d,zero,zero)), jnp.stack((zero,b,zero))

    def state(self, time, params, grid):
        bssn = self.bssn(time)
        em = initialize_densitized_maxwell_state(*self.fields(time), bssn, params, grid)
        return EinsteinMaxwellVariables(bssn, em)

    def boundary(self, bssn, time):
        def fill(value, exact):
            return value.at[...,:GHOSTS].set(exact[...,:GHOSTS]).at[...,-GHOSTS:].set(exact[...,-GHOSTS:])
        return jax.tree_util.tree_map(fill, bssn, self.bssn(time))


def prescribed_gauge(bssn, time):
    return (jnp.ones_like(bssn.lapse), jnp.zeros_like(bssn.shift),
            jnp.zeros_like(bssn.lapse), jnp.zeros_like(bssn.shift))


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

