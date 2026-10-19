"""One dense-Newton finder for star-shaped marginally outer trapped surfaces.

Solve D_i s^i + K_ij s^i s^j - K = 0 using physical covariant spatial tensors,
with the BSSN sign convention dt(gamma_ij) = -2 alpha K_ij + Lie_beta(gamma_ij).
The center and initial guess are supplied by the caller and never searched or
recentered. A converged root is not a proof of global outermostness.

The full h(theta,phi) grid is used for every storage layout. Following
Thornburg (2004), https://arxiv.org/abs/gr-qc/0306056, angular derivatives are
fourth-order finite differences. JAX differentiates the entire discrete
residual, including interpolation and motion through the spatial geometry.
"""

import numpy as np
import jax
import jax.numpy as jnp

from JAX_BSSN.cartoon.interpolation import lagrange6_nonperiodic
from JAX_BSSN.utilities.hermite_interpolation import interpolate_point


def build_geometry(coordinates, gamma, Kij, margin=4):
    """Return (table, sample), where sample(table, points) evaluates the geometry.

    Coordinates are (r,), (x,z), or (x,y,z); tensors have shape (3,3,*grid).
    sample returns gamma[n,i,j], d_gamma[n,k,i,j], Kij[n,i,j], valid[n].
    The coordinate origin/axis of reduced data is independent of the supplied
    horizon center. No translation of the geometry is implied by that center.
    """
    if isinstance(coordinates, np.ndarray) and coordinates.ndim == 1:
        coordinates = (coordinates,)
    axes = tuple(np.asarray(axis, dtype=float) for axis in coordinates)
    if len(axes) not in (1, 2, 3):
        raise ValueError('Pass (r,), (x,z), or (x,y,z) coordinates')
    if margin < 2:
        raise ValueError('Hermite interpolation needs at least two margin cells')
    for axis in axes:
        if (axis.ndim != 1 or len(axis) < max(6, 2*margin + 2)
                or axis[1] <= axis[0]
                or not np.allclose(np.diff(axis), axis[1] - axis[0], rtol=1.e-10, atol=1.e-14)):
            raise ValueError('Coordinates must be increasing uniform axes with a resolved interior')
    shape = (3, 3) + tuple(len(axis) for axis in axes)
    gamma, Kij = np.asarray(gamma), np.asarray(Kij)
    if gamma.shape != shape or Kij.shape != shape:
        raise ValueError(f'Physical gamma and Kij must have shape {shape}')
    for tensor in (gamma, Kij):
        if not np.allclose(tensor, tensor.swapaxes(0, 1), rtol=1.e-12, atol=1.e-14):
            raise ValueError('Physical gamma and Kij must be symmetric')
    if len(axes) == 1:
        if axes[0][0] <= 0:
            raise ValueError('The radial line must contain positive radii')
        for tensor in (gamma, Kij):
            scale = np.max(np.abs(tensor), axis=(0, 1))
            error = np.maximum.reduce([np.abs(tensor[1, 1] - tensor[2, 2]),
                                       np.abs(tensor[0, 1]), np.abs(tensor[0, 2]),
                                       np.abs(tensor[1, 2])])
            # Evolved Cartoon data can carry roundoff in nominally zero terms.
            if np.any(error > 1.e-10*scale + 1.e-14):
                raise ValueError('Radial tensors must be diagonal with equal transverse components')
        table = jnp.asarray(np.stack([gamma[0, 0], gamma[1, 1], Kij[0, 0], Kij[1, 1]]))
    else:
        if len(axes) == 2 and not axes[0][margin] < 0 < axes[0][-margin - 1]:
            raise ValueError('Axisymmetric input needs a signed x-z plane')
        table = jnp.asarray(np.concatenate((gamma.reshape((9,) + shape[2:]),
                                            Kij.reshape((9,) + shape[2:]))))
    origin = jnp.asarray([axis[0] for axis in axes])
    spacing = jnp.asarray([axis[1] - axis[0] for axis in axes])
    sizes = jnp.asarray([len(axis) for axis in axes])

    def stored_coordinates(point):
        if len(axes) == 1:
            return jnp.linalg.norm(point)[None]
        if len(axes) == 2:
            return jnp.array([jnp.linalg.norm(point[:2]), point[2]])
        return point

    def tensors(values, point):
        coordinate = stored_coordinates(point)
        data = interpolate_point(values, (coordinate - origin)/spacing)
        if len(axes) == 1:
            normal = point/jnp.maximum(coordinate[0], 1.e-30)
            radial = normal[:, None]*normal[None, :]
            transverse = jnp.eye(3) - radial
            return data[0]*radial + data[1]*transverse, data[2]*radial + data[3]*transverse
        metric, curvature = data[:9].reshape(3, 3), data[9:].reshape(3, 3)
        if len(axes) == 2:
            c, s = point[:2]/jnp.maximum(coordinate[0], 1.e-30)
            rotation = jnp.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
            metric = rotation @ metric @ rotation.T
            curvature = rotation @ curvature @ rotation.T
        return metric, curvature

    def resolved(point):
        coordinate = stored_coordinates(point)
        index = (coordinate - origin)/spacing
        inside = jnp.all((index >= margin) & (index <= sizes - 1 - margin))
        if len(axes) < 3:
            inside &= coordinate[0] > 1.e-10*spacing[0]
        return inside

    def sample(values, points):
        metric, curvature = jax.vmap(tensors, in_axes=(None, 0))(values, points)
        derivative = jax.vmap(jax.jacfwd(lambda p: tensors(values, p)[0]))(points)
        return metric, jnp.moveaxis(derivative, -1, 1), curvature, jax.vmap(resolved)(points)

    return table, sample


def angular_grid(n_theta, n_phi):
    """Cell-centered polar nodes, periodic azimuthal nodes, and polar area weights."""
    if n_theta < 6 or n_phi < 6 or n_phi % 2:
        raise ValueError('Use at least six nodes per angle and an even n_phi')
    theta = (np.arange(n_theta) + 0.5)*np.pi/n_theta
    phi = np.arange(n_phi)*2*np.pi/n_phi
    # Integrate the regular area factor against sin(theta), exactly for the
    # resolved cosine modes. A plain midpoint rule would cap area at order two.
    modes = np.arange(n_theta)
    integrals = np.zeros(n_theta)
    even = modes[modes % 2 == 0]
    integrals[even] = 2/(1 - even**2)
    integrals[1:] *= 2
    weights = integrals @ np.cos(np.outer(modes, theta))/n_theta
    return theta, phi, weights


def theta_ghosts(field, width):
    """A scalar crossing a pole reappears half a turn around the azimuth."""
    shift = field.shape[1]//2
    low = jnp.flip(jnp.roll(field[:width], shift, axis=1), axis=0)
    high = jnp.flip(jnp.roll(field[-width:], shift, axis=1), axis=0)
    return jnp.concatenate((low, field, high), axis=0)


def angular_derivatives(field):
    """h_theta, h_thetatheta, h_phi, h_phiphi, h_thetaphi on the full sphere."""
    n_theta, n_phi = field.shape
    first = (1/12, -2/3, 0., 2/3, -1/12)
    second = (-1/12, 4/3, -2.5, 4/3, -1/12)

    def polar(values, stencil, power):
        extended = theta_ghosts(values, 2)
        return sum(w*extended[k:k + n_theta] for k, w in enumerate(stencil) if w)/(np.pi/n_theta)**power

    def azimuthal(values, stencil, power):
        return sum(w*jnp.roll(values, 2 - k, axis=1) for k, w in enumerate(stencil) if w)/(2*np.pi/n_phi)**power

    h_phi = azimuthal(field, first, 1)
    return polar(field, first, 1), polar(field, second, 2), h_phi, azimuthal(field, second, 2), polar(h_phi, first, 1)


def interpolate_angular(radius, theta, phi):
    """Interpolate a surface onto new angular nodes with the same pole extension."""
    n_theta, n_phi = radius.shape
    field = theta_ghosts(radius, 3)
    field = lagrange6_nonperiodic(field, jnp.asarray(theta)*n_theta/np.pi - 0.5 + 3, axis=0)
    field = jnp.concatenate((field[:, -3:], field, field[:, :3]), axis=1)
    return lagrange6_nonperiodic(field, jnp.asarray(phi)*n_phi/(2*np.pi) + 3, axis=1)


def surface(radius, center, table, sample):
    """Expansion, regular area density, and validity on a full angular grid."""
    theta, phi, _ = angular_grid(*radius.shape)
    theta, phi = theta[:, None], phi[None, :]
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    sin_p, cos_p = np.sin(phi), np.cos(phi)
    one = np.ones(radius.shape)
    radial = jnp.asarray(np.stack(np.broadcast_arrays(sin_t*cos_p, sin_t*sin_p, cos_t*one), axis=-1))
    polar = jnp.asarray(np.stack(np.broadcast_arrays(cos_t*cos_p, cos_t*sin_p, -sin_t*one), axis=-1))
    azimuthal = jnp.asarray(np.stack(np.broadcast_arrays(-sin_p*one, cos_p*one, np.zeros(radius.shape)), axis=-1))
    h_t, h_tt, h_p, h_pp, h_tp = angular_derivatives(radius)
    sine = jnp.asarray(sin_t)
    cotangent = jnp.asarray(cos_t/sin_t)
    points = center + radius[..., None]*radial
    gamma, d_gamma, curvature, resolved = sample(table, points.reshape(-1, 3))

    flat = (lambda values: jnp.broadcast_to(values, radius.shape).reshape(-1))
    h, h_t, h_tt = flat(radius), flat(h_t), flat(h_tt)
    h_p, h_pp, h_tp = flat(h_p), flat(h_pp), flat(h_tp)
    sine, cotangent = flat(sine), flat(cotangent)
    radial = radial.reshape(-1, 3)
    polar = polar.reshape(-1, 3)
    azimuthal = azimuthal.reshape(-1, 3)

    # Flat Cartesian gradient and Hessian of F=r-h(theta,phi), evaluated at F=0.
    gradient = (radial - (h_t/h)[:, None]*polar
                - (h_p/(h*sine))[:, None]*azimuthal)

    def symmetrize(first, second):
        return first[:, :, None]*second[:, None, :] + second[:, :, None]*first[:, None, :]

    hessian = (h_t/h**2)[:, None, None]*symmetrize(radial, polar)
    hessian += (h_p/(h**2*sine))[:, None, None]*symmetrize(radial, azimuthal)
    hessian += (1/h - h_tt/h**2)[:, None, None]*symmetrize(polar, polar)/2
    hessian += ((cotangent*h_p - h_tp)/(h**2*sine))[:, None, None]*symmetrize(polar, azimuthal)
    hessian += (1/h - cotangent*h_t/h**2
                - h_pp/(h**2*sine**2))[:, None, None]*symmetrize(azimuthal, azimuthal)/2

    # A nonphysical interpolated metric shows up as a nonfinite Cholesky factor.
    physical = jnp.all(jnp.isfinite(jnp.linalg.cholesky(gamma)))
    inverse = jnp.linalg.inv(gamma)
    gradient_up = jnp.einsum("nij,nj->ni", inverse, gradient)
    norm = jnp.sqrt(jnp.sum(gradient*gradient_up, axis=1))
    normal = gradient_up/norm[:, None]
    projector = inverse - normal[:, :, None]*normal[:, None, :]

    # Gamma^k_ij F_k = F^l (d_i gamma_lj + d_j gamma_li - d_l gamma_ij)/2.
    connection_gradient = 0.5*jnp.einsum(
        "nl,nijl->nij", gradient_up,
        d_gamma.transpose(0, 1, 3, 2) + d_gamma.transpose(0, 3, 1, 2)
        - d_gamma.transpose(0, 2, 3, 1),
    )
    divergence = jnp.einsum("nij,nij->n", projector, hessian - connection_gradient)/norm
    expansion = (divergence + jnp.einsum("nij,ni,nj->n", curvature, normal, normal)
                 - jnp.einsum("nij,nij->n", inverse, curvature))

    # Factor sin(theta) out of the phi tangent; the polar weights carry it back.
    tangent_theta = h_t[:, None]*radial + h[:, None]*polar
    tangent_phi = (h_p/sine)[:, None]*radial + h[:, None]*azimuthal
    metric_tt = jnp.einsum("ni,nij,nj->n", tangent_theta, gamma, tangent_theta)
    metric_pp = jnp.einsum("ni,nij,nj->n", tangent_phi, gamma, tangent_phi)
    metric_tp = jnp.einsum("ni,nij,nj->n", tangent_theta, gamma, tangent_phi)
    density = jnp.sqrt(metric_tt*metric_pp - metric_tp**2)

    usable = (jnp.all(resolved) & physical & jnp.all(jnp.isfinite(expansion))
              & jnp.all(jnp.isfinite(density)) & jnp.all(h > 0))
    return expansion.reshape(radius.shape), density.reshape(radius.shape), usable


def dense_jacobian(residual, radius):
    """Differentiate all residuals with respect to all radii, 16 columns at a time.

    linearize retains the primal intermediates once. Each batch propagates
    exact AD directional derivatives, not finite perturbations of the surface.
    """
    _, tangent = jax.linearize(residual, radius)
    size = radius.size

    def columns(start):
        basis = jax.nn.one_hot(start + jnp.arange(16), size, dtype=radius.dtype)
        return jax.vmap(tangent)(basis.reshape((16,) + radius.shape)).reshape(16, size)

    blocks = jax.lax.map(columns, jnp.arange(0, size, 16))
    return blocks.reshape(-1, size)[:size].T


def find_apparent_horizon(coordinates, gamma, Kij, *, center, initial_radius,
                         n_theta=12, n_phi=24, max_angular=48,
                         tolerance=1.e-8, audit_tolerance=1.e-4,
                         mass_tolerance=1.e-3, max_iterations=50, margin=4):
    """Find the marginal surface reached from the supplied center and guess.

    initial_radius is a scalar or a positive (n_theta,n_phi) array. An array
    defines the starting grid, overriding n_theta and n_phi. The center stays
    fixed in Cartesian coordinates. There is no automatic seed search.

    Every accepted result passes a separate angular audit and area-mass
    convergence between two resolutions. Residuals are mean(h)*max(abs(Theta)).
    Returns a plain dictionary; raises RuntimeError on numerical failure.
    Enable jax_enable_x64 before calling. Dense Jacobian storage is O(N**2)
    and factorization O(N**3), where N=n_theta*n_phi.
    """
    if not jax.config.x64_enabled:
        raise ValueError('Horizon finding requires jax_enable_x64')
    table, sample = build_geometry(coordinates, gamma, Kij, margin=margin)
    center = np.asarray(center, dtype=float)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError('center must be a finite Cartesian triple')
    initial = np.asarray(initial_radius, dtype=float)
    if initial.ndim == 0:
        initial = np.full((n_theta, n_phi), initial)
    elif initial.ndim != 2:
        raise ValueError('initial_radius must be a scalar or a two-dimensional angular surface')
    angular_grid(*initial.shape)
    if not np.all(np.isfinite(initial)) or np.any(initial <= 0):
        raise ValueError('The initial surface must have finite positive radii')
    if max_angular < 2*initial.shape[0]:
        raise ValueError('max_angular must allow at least one angular refinement')
    radius = jnp.asarray(initial)
    center_jax = jnp.asarray(center)
    history = []
    previous_mass = None
    measure = jax.jit(lambda h, data: surface(h, center_jax, data, sample))
    # Passing table as an argument keeps large grid data out of XLA constants.
    jacobian_fn = jax.jit(lambda h, data: dense_jacobian(
        lambda r: measure(r, data)[0].reshape(-1), h))
    solve = jax.jit(jnp.linalg.solve)

    while True:
        for iteration in range(max_iterations + 1):
            expansion, density, usable = measure(radius, table)
            residual_norm = float(jnp.mean(radius)*jnp.max(jnp.abs(expansion)))
            if not bool(usable):
                raise RuntimeError('Surface leaves the resolved domain or has nonphysical geometry')
            if residual_norm < tolerance:
                break
            if iteration == max_iterations:
                raise RuntimeError(f'Newton did not converge after {max_iterations} iterations: {residual_norm:.3e}')
            jacobian = jacobian_fn(radius, table)
            delta = solve(jacobian, -expansion.reshape(-1)).reshape(radius.shape)
            if not bool(jnp.all(jnp.isfinite(delta))):
                raise RuntimeError('Newton Jacobian solve is singular or nonfinite')
            step = min(1., float(0.1*jnp.mean(radius)/jnp.max(jnp.abs(delta))))
            # The physical domain and positive-definite metric are checked on
            # every trial, including those rejected by the line search.
            for backtrack in range(20):
                trial = radius + step*delta
                trial_expansion, _, trial_usable = measure(trial, table)
                trial_norm = float(jnp.mean(trial)*jnp.max(jnp.abs(trial_expansion)))
                if bool(trial_usable) and trial_norm < residual_norm*(1 - 1.e-4*step):
                    radius = trial
                    break
                step *= 0.5
            else:
                raise RuntimeError(f'Newton line search stalled at residual {residual_norm:.3e}')

        n_theta, n_phi = radius.shape
        theta, phi, weights = angular_grid(n_theta, n_phi)
        area = float(jnp.sum(jnp.asarray(weights)[:, None]*density)*(2*np.pi/n_phi))
        mass = np.sqrt(area/(16*np.pi))
        audit_theta, audit_phi, _ = angular_grid(2*n_theta + 2, 2*n_phi + 2)
        audit_radius = interpolate_angular(radius, audit_theta, audit_phi)
        audit_expansion, _, audit_usable = measure(audit_radius, table)
        if not bool(audit_usable):
            raise RuntimeError('Angular audit leaves the resolved domain or has nonphysical geometry')
        audit = float(jnp.mean(audit_radius)*jnp.max(jnp.abs(audit_expansion)))
        mass_change = np.inf if previous_mass is None else abs(mass/previous_mass - 1)
        history.append((n_theta, n_phi, mass, residual_norm, audit, iteration))
        if previous_mass is not None and audit < audit_tolerance and mass_change < mass_tolerance:
            return dict(theta=theta, phi=phi, radius=np.asarray(radius), center=center,
                        expansion=np.asarray(expansion), area=area, mass=float(mass),
                        residual=residual_norm, audit_residual=audit,
                        mass_relative_change=mass_change, convergence=np.asarray(history),
                        iterations=sum(int(row[5]) for row in history))
        if 2*n_theta > max_angular:
            raise RuntimeError(f'Angular convergence failed: audit={audit:.3e}, mass change={mass_change:.3e}')
        theta, phi, _ = angular_grid(2*n_theta, 2*n_phi)
        radius = interpolate_angular(radius, theta, phi)
        previous_mass = mass
