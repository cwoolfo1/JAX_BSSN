"""Apparent horizons in full 3D, axisymmetry and spherical symmetry.

This is a JAX postprocessing solver for physical gamma_ij and K_ij, with the
BSSN convention dt(gamma_ij) = -2 alpha K_ij + Lie_beta(gamma_ij).  It solves
D_i s^i + K_ij s^i s^j - K = 0 on a star-shaped surface r = h(theta, phi) about
a center, following Thornburg, Class. Quantum Grav. 21 (2004) 743,
https://arxiv.org/abs/gr-qc/0306056.

The unknowns are the radii themselves on a cell-centered finite-difference
angular grid, not the coefficients of any angular basis.  Symmetry therefore
only collapses the shape of that grid -- h(theta, phi) in 3D, h(theta) in
axisymmetry, a single radius in spherical symmetry -- while the residual, the
area integral and the Newton solve are shared by all three.  The poles are
never sampled and the theta ghost rows are the exact index shift
h(-theta, phi) = h(theta, phi + pi), so no basis function or pole regularity
condition enters anywhere.

No stationarity or spherical symmetry of the data is assumed.  The search
selects the outermost nested candidate it finds; it does not establish global
outermostness outside this star-shaped family.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jacfwd, jit, jvp
from jax.scipy.sparse.linalg import gmres
from scipy.optimize import brentq

from JAX_BSSN.cartoon.interpolation import lagrange6_nonperiodic
from JAX_BSSN.utilities.horizon_geometry import build_geometry

# Below this many unknowns an exact dense Jacobian beats a Krylov solve.
DENSE_JACOBIAN_LIMIT = 256
# Centered fourth-order stencils on offsets -2..2.
FIRST_DERIVATIVE_STENCIL = np.array([1.0, -8.0, 0.0, 8.0, -1.0])/12.0
SECOND_DERIVATIVE_STENCIL = np.array([-1.0, 16.0, -30.0, 16.0, -1.0])/12.0
GHOST_WIDTH = 3  # Theta ghost rows needed by six-point angular interpolation.


def _polar_weights(n_theta, theta):
    """Quadrature weights for integral(f(theta) sin(theta) dtheta) on cell centers.

    The area element carries a sin(theta) whose periodic extension has a kink at
    the poles, so the plain midpoint rule would cap the area at second order.
    Absorbing sin(theta) into the weights instead integrates the regular factor
    exactly through cosine mode n_theta-1, which is the same spectral accuracy
    the uniform phi rule already has.
    """

    modes = np.arange(n_theta)
    integrals = np.zeros(n_theta)
    even = modes[modes % 2 == 0]
    integrals[even] = 2.0/(1.0 - even.astype(float)**2)
    integrals[1:] *= 2.0  # Orthogonality of cos(k theta) on cell centers.
    return integrals @ np.cos(np.outer(modes, theta))/n_theta


def _theta_ghosts(field, width):
    """Extend across both poles with the exact shift h(-theta,phi)=h(theta,phi+pi)."""

    shift = field.shape[1]//2
    low = jnp.flip(jnp.roll(field[:width], shift, axis=1), axis=0)
    high = jnp.flip(jnp.roll(field[-width:], shift, axis=1), axis=0)
    return jnp.concatenate((low, field, high), axis=0)


class AngularGrid:
    """Cell-centered theta nodes and uniform periodic phi nodes.

    A collapsed axis carries a single node and contributes zero derivatives,
    which is exactly what axisymmetry (n_phi=1) and spherical symmetry
    (n_theta=n_phi=1) require of the surface.
    """

    def __init__(self, n_theta, n_phi):
        if n_theta != 1 and n_theta < 6:
            raise ValueError("A resolved theta grid needs at least six nodes")
        if n_phi != 1 and (n_phi < 6 or n_phi % 2):
            raise ValueError("A resolved phi grid needs an even count of at least six")
        self.shape = (int(n_theta), int(n_phi))
        self.size = int(n_theta*n_phi)
        self.d_theta = np.pi/n_theta
        self.d_phi = 2*np.pi/n_phi
        self.theta = (np.arange(n_theta) + 0.5)*self.d_theta
        self.phi = np.arange(n_phi)*self.d_phi
        self.theta_weights = _polar_weights(n_theta, self.theta)

    def refined(self, max_theta):
        """Next grid of the resolution ladder, or None when none exists."""

        n_theta, n_phi = self.shape
        if n_theta == 1:
            return None  # A single radius has no angular resolution to double.
        if 2*n_theta > max_theta:
            return None
        return AngularGrid(2*n_theta, n_phi if n_phi == 1 else 2*n_phi)

    def audit(self):
        """An independent grid whose nodes never coincide with this one's."""

        n_theta, n_phi = self.shape
        if n_theta == 1:
            # A constant radius is only tested by sampling genuinely new angles.
            return AngularGrid(9, 8)
        return AngularGrid(2*n_theta + 2, n_phi if n_phi == 1 else 2*n_phi)

    def derivatives(self, field):
        """Fourth-order h_theta, h_thetatheta, h_phi, h_phiphi, h_thetaphi."""

        def along_theta(values, stencil, power):
            if self.shape[0] == 1:
                return jnp.zeros_like(values)
            extended = _theta_ghosts(values, 2)
            total = sum(weight*extended[k:k + self.shape[0]]
                        for k, weight in enumerate(stencil) if weight)
            return total/self.d_theta**power

        def along_phi(values, stencil, power):
            if self.shape[1] == 1:
                return jnp.zeros_like(values)
            total = sum(weight*jnp.roll(values, 2 - k, axis=1)
                        for k, weight in enumerate(stencil) if weight)
            return total/self.d_phi**power

        d_phi = along_phi(field, FIRST_DERIVATIVE_STENCIL, 1)
        return (along_theta(field, FIRST_DERIVATIVE_STENCIL, 1),
                along_theta(field, SECOND_DERIVATIVE_STENCIL, 2),
                d_phi,
                along_phi(field, SECOND_DERIVATIVE_STENCIL, 2),
                along_theta(d_phi, FIRST_DERIVATIVE_STENCIL, 1))

    def triad(self):
        """Unit radial, polar and azimuthal vectors of shape (n_theta,n_phi,3)."""

        theta, phi = self.theta[:, None], self.phi[None, :]
        sin_t, cos_t = np.sin(theta), np.cos(theta)
        sin_p, cos_p = np.sin(phi), np.cos(phi)
        zero, one = np.zeros(self.shape), np.ones(self.shape)
        radial = np.stack(np.broadcast_arrays(sin_t*cos_p, sin_t*sin_p, cos_t*one), axis=-1)
        polar = np.stack(np.broadcast_arrays(cos_t*cos_p, cos_t*sin_p, -sin_t*one), axis=-1)
        azimuthal = np.stack(np.broadcast_arrays(-sin_p*one, cos_p*one, zero), axis=-1)
        return jnp.asarray(radial), jnp.asarray(polar), jnp.asarray(azimuthal)


def interpolate_angular(field, grid, target):
    """Evaluate ``field`` given on ``grid`` at the nodes of ``target``."""

    if grid.shape == target.shape:
        return field
    n_theta, n_phi = grid.shape
    if n_theta == 1:
        field = jnp.repeat(field, target.shape[0], axis=0)
    else:
        extended = _theta_ghosts(field, GHOST_WIDTH)
        index = target.theta/grid.d_theta - 0.5 + GHOST_WIDTH
        field = lagrange6_nonperiodic(extended, jnp.asarray(index), axis=0)
    if n_phi == 1:
        return jnp.repeat(field, target.shape[1], axis=1)
    extended = jnp.concatenate(
        (field[:, -GHOST_WIDTH:], field, field[:, :GHOST_WIDTH]), axis=1)
    index = target.phi/grid.d_phi + GHOST_WIDTH
    return lagrange6_nonperiodic(extended, jnp.asarray(index), axis=1)


def _surface(radius, center, geometry, tables, grid):
    """Outgoing expansion and area density of the trial surface r=h(theta,phi).

    Returns the expansion and the area density on the grid, plus a flag that
    is false when any node left the resolved region or produced a nonphysical
    interpolated metric.  Out-of-domain samples stay finite so that a Newton
    line search can back out of them.
    """

    radial, polar, azimuthal = grid.triad()
    h_t, h_tt, h_p, h_pp, h_tp = grid.derivatives(radius)
    sine = jnp.asarray(np.sin(grid.theta))[:, None]
    cotangent = jnp.asarray(np.cos(grid.theta))[:, None]/sine

    points = center + radius[..., None]*radial
    gamma, d_gamma, curvature, resolved = geometry.sample(
        tables, points.reshape(-1, 3))

    flat = (lambda values: jnp.broadcast_to(values, grid.shape).reshape(-1))
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
              & jnp.all(jnp.isfinite(density)) & jnp.all(h >= geometry.inner_radius))
    return expansion.reshape(grid.shape), density.reshape(grid.shape), usable


class _Kernels:
    """Compiled trial-surface evaluations for one geometry and angular grid."""

    def __init__(self, geometry, grid):
        self.geometry, self.grid = geometry, grid

        def surface(tables, flat, center):
            return _surface(flat.reshape(grid.shape), center, geometry, tables, grid)

        def expansion(tables, flat, center):
            return surface(tables, flat, center)[0].reshape(-1)

        self._expansion = expansion
        self._surface = jit(surface)
        self._jacobian = None
        self._action = None

    def surface(self, radius, center):
        """Expansion, area density and the resolved flag of a trial surface."""

        return self._surface(self.geometry.tables, radius.reshape(-1), center)

    def measure(self, radius, center):
        """Expansion, area and the resolved flag of a trial surface."""

        expansion, density, usable = self.surface(radius, center)
        weights = jnp.asarray(self.grid.theta_weights)[:, None]
        return expansion, float(self.grid.d_phi*jnp.sum(weights*density)), bool(usable)

    def mean_expansion(self, radius, center):
        """Area-weighted mean expansion, the quantity the seed search brackets."""

        expansion, _, usable = self.surface(radius, center)
        weights = jnp.asarray(self.grid.theta_weights)[:, None]
        return float(jnp.sum(weights*expansion)/(2*self.grid.shape[1])), bool(usable)

    def update(self, radius, center, residual, preconditioner):
        """Newton update solving J du = -residual."""

        flat = radius.reshape(-1)
        if self.grid.size <= DENSE_JACOBIAN_LIMIT:
            if self._jacobian is None:
                self._jacobian = jit(jacfwd(self._expansion, argnums=1))
            jacobian = self._jacobian(self.geometry.tables, flat, center)
            return jnp.linalg.solve(jacobian, -residual)
        if self._action is None:
            self._action = jit(lambda tables, values, origin, vector: jvp(
                lambda trial: self._expansion(tables, trial, origin),
                (values,), (vector,))[1])
        action = lambda vector: self._action(self.geometry.tables, flat, center, vector)
        update, _ = gmres(action, -residual, M=preconditioner, tol=1.e-8, atol=0.0,
                          restart=min(flat.size, 40), maxiter=20)
        return update


class _KernelCache:
    """One compiled kernel set per angular grid shape, reused across the solve."""

    def __init__(self, geometry):
        self.geometry, self._cache = geometry, {}

    def __call__(self, grid):
        if grid.shape not in self._cache:
            self._cache[grid.shape] = _Kernels(self.geometry, grid)
        return self._cache[grid.shape]


def _dimensionless_residual(radius, expansion):
    """Mean coordinate radius times the largest outgoing expansion."""

    return float(np.mean(np.asarray(radius))*np.max(np.abs(np.asarray(expansion))))


def helmholtz_preconditioner(grid, radius):
    """Approximate inverse Jacobian from the flat-space linearization of Theta.

    Linearizing Theta about a sphere of radius R gives -(2 + Laplacian)/R^2, so
    (2 - Laplacian)/R^2 shares its sign and its large-l growth while staying
    invertible.  A uniform phi grid decouples the azimuthal index, leaving one
    dense theta system per index; the pole reflection enters those systems as
    the sign (-1)^m.  This only reshapes the Newton update: the unknowns stay
    grid radii and the solve converges without it, more slowly.
    """

    n_theta, n_phi = grid.shape
    theta = grid.theta
    operators = []
    for parity in (1.0, -1.0):
        first = np.zeros((n_theta, n_theta))
        second = np.zeros((n_theta, n_theta))
        for row in range(n_theta):
            for offset in range(-2, 3):
                index, sign = row + offset, 1.0
                if index < 0:
                    index, sign = -1 - index, parity
                elif index >= n_theta:
                    index, sign = 2*n_theta - 1 - index, parity
                if 0 <= index < n_theta:
                    weight = sign*FIRST_DERIVATIVE_STENCIL[offset + 2]
                    first[row, index] += weight/grid.d_theta
                    weight = sign*SECOND_DERIVATIVE_STENCIL[offset + 2]
                    second[row, index] += weight/grid.d_theta**2
        operators.append(-(second + np.cos(theta)[:, None]/np.sin(theta)[:, None]*first))

    modes = np.arange(n_phi//2 + 1)
    matrices = np.stack([operators[mode % 2]
                         + np.diag(mode**2/np.sin(theta)**2 + 2.0) for mode in modes])
    matrices = jnp.asarray(matrices/radius**2, dtype=complex)

    def apply(vector):
        spectrum = jnp.fft.rfft(vector.reshape(grid.shape), axis=1)
        solution = jnp.linalg.solve(matrices, spectrum.T[..., None])[..., 0]
        return jnp.fft.irfft(solution.T, n=n_phi, axis=1).reshape(-1)

    return apply


def _newton(radius, center, kernels, tolerance, max_iterations, preconditioned):
    """Damped Newton solve of the expansion on the angular grid."""

    grid = kernels.grid
    radius = jnp.asarray(radius)
    for iteration in range(max_iterations + 1):
        expansion, _, usable = kernels.measure(radius, center)
        if not usable:
            raise ValueError("Trial horizon leaves the resolved interpolation domain")
        residual = _dimensionless_residual(radius, expansion)
        if residual < tolerance:
            return radius, residual, iteration
        if iteration == max_iterations:
            break

        preconditioner = None
        if preconditioned and grid.size > DENSE_JACOBIAN_LIMIT:
            preconditioner = helmholtz_preconditioner(grid, float(jnp.mean(radius)))
        update = kernels.update(radius, center, expansion.reshape(-1), preconditioner)
        if not np.all(np.isfinite(np.asarray(update))):
            raise RuntimeError("Horizon Newton update is not finite")

        previous = float(jnp.linalg.norm(expansion))
        damping = 1.0
        for _ in range(20):
            trial = radius + damping*update.reshape(grid.shape)
            attempt, _, trial_usable = kernels.measure(trial, center)
            if trial_usable and float(jnp.linalg.norm(attempt)) < (1 - 1.e-4*damping)*previous:
                radius = trial
                break
            damping *= 0.5  # Back out of out-of-domain or nonphysical trial surfaces.
        else:
            raise RuntimeError(f"Horizon Newton line search failed; residual={residual:.3e}")
    raise RuntimeError(f"Horizon Newton solve did not converge; residual={residual:.3e}")


def _centroid_shift(radius, center, kernels):
    """Area-weighted offset of the surface from its star-shape center."""

    _, density, usable = kernels.surface(radius, center)
    if not bool(usable):
        raise ValueError("Trial horizon leaves the resolved interpolation domain")
    weights = jnp.asarray(kernels.grid.theta_weights)[:, None]*density
    radial = kernels.grid.triad()[0]
    return np.asarray(jnp.einsum("tp,tpi->i", weights*radius, radial)/jnp.sum(weights))


def _recenter(radius, center, kernels, free_axes, tolerance, max_passes,
              newton_arguments):
    """Move the star-shape center onto the surface centroid and re-solve."""

    shift_total = np.zeros(3)
    residual = iterations = None
    for _ in range(max_passes):
        shift = _centroid_shift(radius, center, kernels)*free_axes
        if np.linalg.norm(shift) < tolerance*float(jnp.mean(radius)):
            break
        center = center + jnp.asarray(shift)
        shift_total = shift_total + shift
        # First-order reparametrization about the new center; exact for a sphere.
        radial = kernels.grid.triad()[0]
        radius = radius - jnp.einsum("i,tpi->tp", jnp.asarray(shift), radial)
        radius, residual, iterations = _newton(radius, center, kernels, *newton_arguments)
    return radius, center, shift_total, residual, iterations


def _candidate(seed, center, cache, settings):
    """Converge one seed through the angular-resolution ladder."""

    grid = AngularGrid(*settings["grid_shape"])
    center = jnp.asarray(center, dtype=float)
    radius = jnp.full(grid.shape, float(seed))
    free_axes = np.asarray(settings["free_axes"], dtype=float)
    newton_arguments = (settings["tolerance"], settings["max_iterations"],
                        settings["preconditioned"])
    history, shift_total = [], np.zeros(3)
    mass = mass_change = None

    while True:
        kernels = cache(grid)
        radius, residual, iterations = _newton(radius, center, kernels, *newton_arguments)
        if np.any(free_axes):
            radius, center, shift, moved_residual, moved_iterations = _recenter(
                radius, center, kernels, free_axes, settings["recenter_tolerance"],
                settings["max_recenter"], newton_arguments)
            shift_total = shift_total + shift
            if moved_residual is not None:
                residual, iterations = moved_residual, moved_iterations

        expansion, area, _ = kernels.measure(radius, center)
        previous_mass, mass = mass, float(np.sqrt(area/(16*np.pi)))

        audit_grid = grid.audit()
        audit_radius = interpolate_angular(radius, grid, audit_grid)
        audit_expansion, _, audit_usable = cache(audit_grid).measure(
            audit_radius, center)
        if not audit_usable:
            raise ValueError("Converged horizon leaves the resolved domain between nodes")
        audit_residual = _dimensionless_residual(audit_radius, audit_expansion)

        # The first pass has nothing to compare against, so it never converges.
        mass_change = abs(mass/previous_mass - 1) if previous_mass else np.inf
        if grid.size == 1:
            mass_change = 0.0  # A single radius has no angular resolution to refine.
        history.append((grid.shape[0], grid.shape[1], mass, residual,
                        audit_residual, iterations))
        if (audit_residual < settings["audit_tolerance"]
                and mass_change < settings["mass_tolerance"]):
            return {
                "theta": np.asarray(grid.theta), "phi": np.asarray(grid.phi),
                "radius": np.asarray(radius), "area": area, "mass": mass,
                "center": np.asarray(center), "expansion": np.asarray(expansion),
                "residual": residual, "audit_residual": audit_residual,
                "mass_relative_change": mass_change, "recenter_shift": shift_total,
                "convergence": np.asarray(history), "seed_radius": float(seed),
                "symmetry": cache.geometry.symmetry, "grid": grid,
            }
        finer = grid.refined(settings["max_angular"])
        if finer is None:
            raise RuntimeError("Horizon failed angular residual or mass convergence: "
                               f"audit={audit_residual:.3e}, mass change={mass_change:.3e}")
        radius, grid = interpolate_angular(radius, grid, finer), finer


def _seed_radii(cache, center, grid_shape, initial_radii, failures):
    """Trial-sphere radii where the area-weighted mean expansion changes sign."""

    inner, outer = cache.geometry.radial_bounds(center)
    if outer <= 1.1*inner:
        raise ValueError("Insufficient resolved domain around the horizon center")
    kernels = cache(AngularGrid(*grid_shape))
    center = jnp.asarray(center, dtype=float)

    def mean_expansion(value):
        mean, usable = kernels.mean_expansion(
            jnp.full(kernels.grid.shape, float(value)), center)
        if not usable:
            raise ValueError(f"trial sphere r={value:.6g} leaves the resolved domain")
        return mean

    radii = np.geomspace(1.01*inner, 0.99*outer, 48)
    values = np.full_like(radii, np.nan)
    for index, value in enumerate(radii):
        try:
            values[index] = mean_expansion(value)
        except ValueError:
            continue
    seeds = [] if initial_radii is None else [float(r) for r in initial_radii]
    for index in range(len(radii) - 1):
        if values[index]*values[index + 1] <= 0:
            try:
                seeds.append(brentq(mean_expansion, radii[index], radii[index + 1]))
            except ValueError as error:
                # A sign change across an interpolation singularity is not a seed.
                failures.append(f"sphere bracket [{radii[index]:.6g}, "
                                f"{radii[index + 1]:.6g}]: {error}")
    if not seeds:
        raise RuntimeError("No horizon seed: no resolved trial-sphere expansion sign "
                           "change about the search center; pass center or initial_radii")
    return sorted(set(seeds), reverse=True)


def _select_outermost(candidates, tolerance):
    """Keep the candidate that encloses every other one."""

    reference = candidates[0]["center"]
    comparison = max((candidate["grid"] for candidate in candidates),
                     key=lambda grid: grid.size).audit()
    profiles = []
    for candidate in candidates:
        offset = candidate["center"] - reference
        scale = float(np.mean(candidate["radius"]))
        if np.linalg.norm(offset) > 0.25*scale:
            raise RuntimeError("Converged horizon candidates have different centers; "
                               "outermost is ambiguous")
        radial = comparison.triad()[0]
        shape = interpolate_angular(jnp.asarray(candidate["radius"]),
                                    candidate["grid"], comparison)
        profiles.append(np.asarray(shape + jnp.einsum("i,tpi->tp",
                                                      jnp.asarray(offset), radial)))

    selected = 0
    for index in range(1, len(candidates)):
        margin = 10*tolerance*max(profiles[selected].mean(), profiles[index].mean())
        if np.all(profiles[index] >= profiles[selected] - margin):
            selected = index
        elif not np.all(profiles[selected] >= profiles[index] - margin):
            raise RuntimeError("Converged horizon candidates are not nested; "
                               "outermost is ambiguous")
    return candidates[selected]


def find_apparent_horizon(coordinates, gamma, Kij, symmetry=None, center=(0.0, 0.0, 0.0),
                          initial_radii=None, n_theta=24, n_phi=48, max_angular=96,
                          tolerance=1.e-6, audit_tolerance=1.e-4, mass_tolerance=1.e-3,
                          max_iterations=30, max_recenter=3, recenter_tolerance=1.e-3,
                          preconditioned=True, margin=4):
    """Find the outermost resolved, nested apparent horizon candidate.

    ``coordinates`` is the tuple of coordinate axes, and selects the symmetry:
    ``(r,)`` with gamma and Kij of shape (3,3,nr) for spherical symmetry,
    ``(x, z)`` with shape (3,3,nx,nz) on the signed y=0 plane for axisymmetry,
    and ``(x, y, z)`` with shape (3,3,nx,ny,nz) in full 3D.  Each axis must be
    increasing and uniform; gamma and Kij are physical covariant tensors.
    ``margin`` cells are dropped at every outer face.

    The surface starts on an ``n_theta`` by ``n_phi`` angular grid that is
    doubled until the area mass settles or ``max_angular`` theta nodes are
    reached.  Up to ``max_recenter`` passes move the star-shape center onto the
    surface centroid along the axes the symmetry leaves free, stopping once the
    shift falls below ``recenter_tolerance`` times the mean radius.  Optional
    ``initial_radii`` add spherical seeds to the automatic sign-change search,
    which is what to reach for when a strongly distorted surface has no trial
    sphere bracketing it.  ``preconditioned`` only affects how fast the Krylov
    path converges, never what it converges to.

    Returns a dictionary with the angular grid, the coordinate radii, area,
    mass (irreducible), the center and the distance it moved, the outgoing
    expansion, and the angular-convergence history.  Residuals are
    dimensionless: mean coordinate radius times max |Theta|.  Raises
    RuntimeError if no candidate passes the independent residual and
    mass-convergence checks.  There is no fixed-mass or spherical fallback.
    """

    if not jax.config.read("jax_enable_x64"):
        raise ValueError("Horizon finding needs float64; enable jax_enable_x64")
    geometry = build_geometry(coordinates, gamma, Kij, symmetry=symmetry, margin=margin)
    center = np.asarray(center, dtype=float)
    if center.shape != (3,):
        raise ValueError("center must be a Cartesian triple")

    grid_shape, free_axes = {
        "spherical": ((1, 1), (0.0, 0.0, 0.0)),
        "axisymmetric": ((n_theta, 1), (0.0, 0.0, 1.0)),
        "3d": ((n_theta, n_phi), (1.0, 1.0, 1.0)),
    }[geometry.symmetry]
    if geometry.symmetry != "spherical" and (n_theta < 6 or max_angular < 2*n_theta):
        raise ValueError("Angular convergence requires at least two theta resolutions")
    settings = {
        "grid_shape": grid_shape, "free_axes": free_axes, "tolerance": tolerance,
        "audit_tolerance": audit_tolerance, "mass_tolerance": mass_tolerance,
        "max_iterations": max_iterations, "max_angular": max_angular,
        "max_recenter": max_recenter, "recenter_tolerance": recenter_tolerance,
        "preconditioned": preconditioned,
    }

    failures = []
    cache = _KernelCache(geometry)
    seeds = _seed_radii(cache, center, grid_shape, initial_radii, failures)
    candidates = []
    for seed in seeds:
        try:
            candidates.append(_candidate(seed, center, cache, settings))
        except (RuntimeError, ValueError) as error:
            failures.append(f"seed {seed:.6g}: {error}")
    if not candidates:
        raise RuntimeError("No converged apparent horizon. " + "; ".join(failures))

    selected = _select_outermost(candidates, tolerance)
    selected.pop("grid")
    selected["candidate_count"] = len(candidates)
    selected["search_failures"] = np.asarray(failures, dtype=str)
    return selected
