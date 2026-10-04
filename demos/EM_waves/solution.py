"""Rosen form of Harte & Drivas, arXiv:1202.0540v3, Eqs. 2.10-2.19."""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
import jax
import jax.numpy as jnp
from JAX_BSSN.bssn.variables import BSSNVariables
from JAX_BSSN.EM.first_order.variables import DensitizedMaxwellState
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
        wb = w+self.dx/2  # B_y is on the lower z vertex, D_x at the center.
        d = -self.scale(w)[0]*profile(w,self.packet)/jnp.sqrt(4*jnp.pi)
        b = -self.scale(wb)[0]*profile(wb,self.packet)/jnp.sqrt(4*jnp.pi)
        zero = jnp.zeros_like(d)
        return jnp.stack((d,zero,zero)), jnp.stack((zero,b,zero))

    def state(self, time, dt):
        return EinsteinMaxwellVariables(self.bssn(time), DensitizedMaxwellState(
            self.fields(time-dt)[1], self.fields(time)[1],
            self.fields(time-dt/2)[0], self.fields(time+dt/2)[0]))

    def boundary(self, bssn, d, b, time):
        reference = (self.bssn(time), *self.fields(time))
        def fill(value, exact):
            return value.at[...,:GHOSTS].set(exact[...,:GHOSTS]).at[...,-GHOSTS:].set(exact[...,-GHOSTS:])
        return jax.tree_util.tree_map(fill, (bssn,d,b), reference)


def prescribed_gauge(bssn, time):
    return (jnp.ones_like(bssn.lapse), jnp.zeros_like(bssn.shift),
            jnp.zeros_like(bssn.lapse), jnp.zeros_like(bssn.shift))
