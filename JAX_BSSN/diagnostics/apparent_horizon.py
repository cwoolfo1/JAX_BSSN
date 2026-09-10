"""Axisymmetric star-shaped apparent-horizon diagnostics.

The surface is represented as ``r=h(theta)`` in even Legendre modes.  Its
outgoing expansion follows Gundlach's sign convention,

``Theta = D_i s^i + K_ij s^i s^j - K``.

The evolved Cartoon data remain on their native grid.  Only this diagnostic
copies the physical metric, its derivatives, and the extrinsic curvature to
the host for interpolation and the small nonlinear surface solve.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial import legendre
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares

from JAX_BSSN.bssn.geometry import W_FLOOR_VALUE
from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables, get_boundary_codes
from JAX_BSSN.cartoon.axisymmetry import reconstruct_axisymmetric_support
from JAX_BSSN.cartoon.axisymmetry.reconstruction import AXISYMMETRIC_CENTER
from JAX_BSSN.evolution.derivatives import diff1_field


class AxisymmetricHorizon(NamedTuple):
    """A converged marginal surface and its intrinsic diagnostics."""

    coefficients: np.ndarray
    expansion_l2: float
    expansion_linf: float
    area: float
    irreducible_mass: float
    circumference_ratio: float
    polar_radius: float
    equatorial_radius: float


def _even_legendre_values(coefficients, mu):
    full_coefficients = np.zeros(2 * len(coefficients) - 1)
    full_coefficients[::2] = coefficients
    return legendre.legval(mu, full_coefficients)


def _level_set(point, coefficients):
    radius = jnp.sqrt(jnp.dot(point, point))
    mu = point[2] / radius

    p_previous = jnp.ones_like(mu)
    value = coefficients[0] * p_previous
    if coefficients.shape[0] == 1:
        return radius - value

    p_current = mu
    for degree in range(2, 2 * coefficients.shape[0] - 1):
        p_next = (
            (2 * degree - 1) * mu * p_current
            - (degree - 1) * p_previous
        ) / degree
        if degree % 2 == 0:
            value = value + coefficients[degree // 2] * p_next
        p_previous, p_current = p_current, p_next

    return radius - value


_level_set_gradient = jax.grad(_level_set, argnums=0)
_level_set_hessian = jax.jacfwd(_level_set_gradient, argnums=0)
_surface_level_set_derivatives = jax.jit(
    jax.vmap(
        lambda point, coefficients: (
            _level_set_gradient(point, coefficients),
            _level_set_hessian(point, coefficients),
        ),
        in_axes=(0, None),
    )
)


def _plane_interpolator(field, x, z):
    values = np.moveaxis(np.asarray(field), (-2, -1), (0, 1))
    return RegularGridInterpolator(
        (x, z),
        values,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )


def _horizon_geometry(vars: BSSNVariables, params: BSSNParameters):
    support = reconstruct_axisymmetric_support(vars, params)
    W = jnp.maximum(support.conformal_factor, W_FLOOR_VALUE)
    physical_metric = support.conformal_metric / W**2
    physical_extrinsic_curvature = (
        support.traceless_K
        + support.conformal_metric * support.trace_K / 3.0
    ) / W**2
    metric_derivatives = jnp.stack(
        tuple(
            diff1_field(
                physical_metric,
                direction + 2,
                params.dx,
                *get_boundary_codes(params, direction),
                mad_q=params.mad_q,
            )
            for direction in range(3)
        ),
        axis=0,
    )
    jax.block_until_ready(
        (physical_metric, physical_extrinsic_curvature, metric_derivatives)
    )

    dx = float(params.dx)
    nx, _, nz = support.conformal_factor.shape
    x = float(params.x_min) + dx * np.arange(nx)
    z = float(params.z_min) + dx * np.arange(nz)
    center = AXISYMMETRIC_CENTER

    return {
        "metric": _plane_interpolator(
            np.asarray(jax.device_get(physical_metric[:, :, :, center, :])),
            x,
            z,
        ),
        "metric_derivatives": _plane_interpolator(
            np.asarray(jax.device_get(metric_derivatives[:, :, :, :, center, :])),
            x,
            z,
        ),
        "extrinsic_curvature": _plane_interpolator(
            np.asarray(
                jax.device_get(physical_extrinsic_curvature[:, :, :, center, :])
            ),
            x,
            z,
        ),
        "trace_K": _plane_interpolator(
            np.asarray(jax.device_get(support.trace_K[:, center, :])),
            x,
            z,
        ),
        "dx": dx,
        "maximum_radius": 0.65 * min(
            (vars.conformal_factor.shape[0] - 4) * dx,
            max(abs(z[0]), abs(z[-1])),
        ),
    }


def _surface_points(coefficients, mu):
    radius = _even_legendre_values(coefficients, mu)
    cylindrical_radius = radius * np.sqrt(1.0 - mu**2)
    return radius, np.stack(
        (cylindrical_radius, np.zeros_like(radius), radius * mu), axis=-1
    )


def _christoffel_symbols(metric, metric_derivatives):
    inverse_metric = np.linalg.inv(metric)
    christoffel = np.zeros(metric.shape[:-2] + (3, 3, 3))
    for upper in range(3):
        for first in range(3):
            for second in range(3):
                for contracted in range(3):
                    christoffel[..., upper, first, second] += 0.5 * inverse_metric[
                        ..., upper, contracted
                    ] * (
                        metric_derivatives[..., first, contracted, second]
                        + metric_derivatives[..., second, contracted, first]
                        - metric_derivatives[..., contracted, first, second]
                    )
    return inverse_metric, christoffel


def _expansion(coefficients, mu, geometry):
    radius, points = _surface_points(coefficients, mu)
    if (
        np.any(radius <= 1.5 * geometry["dx"])
        or np.any(radius >= geometry["maximum_radius"])
    ):
        return np.full(mu.shape, 1.0e3)

    interpolation_points = points[:, (0, 2)]
    metric = geometry["metric"](interpolation_points)
    metric_derivatives = geometry["metric_derivatives"](interpolation_points)
    extrinsic_curvature = geometry["extrinsic_curvature"](interpolation_points)
    trace_K = geometry["trace_K"](interpolation_points)
    if not all(
        np.all(np.isfinite(field))
        for field in (metric, metric_derivatives, extrinsic_curvature, trace_K)
    ):
        return np.full(mu.shape, 1.0e3)

    gradient, hessian = _surface_level_set_derivatives(
        jnp.asarray(points), jnp.asarray(coefficients)
    )
    gradient = np.asarray(gradient)
    hessian = np.asarray(hessian)

    inverse_metric, christoffel = _christoffel_symbols(
        metric, metric_derivatives
    )
    gradient_norm = np.sqrt(
        np.einsum("...ij,...i,...j->...", inverse_metric, gradient, gradient)
    )
    normal = np.einsum("...ij,...j->...i", inverse_metric, gradient)
    normal /= gradient_norm[..., None]
    projector = inverse_metric - np.einsum("...i,...j->...ij", normal, normal)
    covariant_hessian = hessian - np.einsum(
        "...kij,...k->...ij", christoffel, gradient
    )
    divergence = np.einsum(
        "...ij,...ij->...", projector, covariant_hessian
    ) / gradient_norm
    normal_extrinsic_curvature = np.einsum(
        "...ij,...i,...j->...", extrinsic_curvature, normal, normal
    )
    return divergence + normal_extrinsic_curvature - trace_K


def _intrinsic_diagnostics(coefficients, geometry, quadrature_order=64):
    quadrature_nodes, quadrature_weights = np.polynomial.legendre.leggauss(
        quadrature_order
    )
    theta = 0.5 * np.pi * (quadrature_nodes + 1.0)
    weights = 0.5 * np.pi * quadrature_weights
    mu = np.cos(theta)
    radius, points = _surface_points(coefficients, mu)
    full_coefficients = np.zeros(2 * len(coefficients) - 1)
    full_coefficients[::2] = coefficients
    dh_dmu = legendre.legval(mu, legendre.legder(full_coefficients))
    sine = np.sqrt(1.0 - mu**2)
    dh_dtheta = -sine * dh_dmu

    tangent_theta = np.stack(
        (
            dh_dtheta * sine + radius * mu,
            np.zeros_like(radius),
            dh_dtheta * mu - radius * sine,
        ),
        axis=-1,
    )
    tangent_phi = np.stack(
        (np.zeros_like(radius), radius * sine, np.zeros_like(radius)), axis=-1
    )
    metric = geometry["metric"](points[:, (0, 2)])
    q_theta_theta = np.einsum(
        "...ij,...i,...j->...", metric, tangent_theta, tangent_theta
    )
    q_phi_phi = np.einsum(
        "...ij,...i,...j->...", metric, tangent_phi, tangent_phi
    )
    q_theta_phi = np.einsum(
        "...ij,...i,...j->...", metric, tangent_theta, tangent_phi
    )
    surface_measure = np.sqrt(
        np.maximum(q_theta_theta * q_phi_phi - q_theta_phi**2, 0.0)
    )
    area = 2.0 * np.pi * np.sum(weights * surface_measure)

    polar_circumference = 2.0 * np.sum(
        weights * np.sqrt(np.maximum(q_theta_theta, 0.0))
    )
    equatorial_radius = float(_even_legendre_values(coefficients, 0.0))
    equatorial_metric = geometry["metric"](
        np.asarray([[equatorial_radius, 0.0]])
    )[0]
    equatorial_circumference = (
        2.0 * np.pi * equatorial_radius * np.sqrt(equatorial_metric[1, 1])
    )

    return (
        float(area),
        float(np.sqrt(area / (16.0 * np.pi))),
        float(polar_circumference / equatorial_circumference),
        float(_even_legendre_values(coefficients, 1.0)),
        equatorial_radius,
    )


def find_axisymmetric_apparent_horizon(
    vars: BSSNVariables,
    params: BSSNParameters,
    initial_coefficients=None,
    maximum_legendre_degree: int = 6,
    num_collocation_points: int = 24,
    residual_tolerance: float = 5.0e-2,
):
    """Return the outermost converged even-Legendre marginal surface.

    ``residual_tolerance`` applies to ``h_mean * max(abs(Theta))`` and is
    therefore dimensionless.  Several spherical starts are used at first
    formation; later calls should pass the previous accepted coefficients.
    """

    if maximum_legendre_degree % 2:
        raise ValueError("maximum_legendre_degree must be even")

    geometry = _horizon_geometry(vars, params)
    mu, _ = np.polynomial.legendre.leggauss(num_collocation_points)
    num_coefficients = maximum_legendre_degree // 2 + 1
    starts = []
    if initial_coefficients is not None:
        previous = np.asarray(initial_coefficients, dtype=float)
        if previous.shape == (num_coefficients,):
            starts.append(previous)

    minimum_radius = 2.0 * geometry["dx"]
    maximum_radius = geometry["maximum_radius"]
    for radius in np.geomspace(minimum_radius, 0.95 * maximum_radius, 10):
        coefficients = np.zeros(num_coefficients)
        coefficients[0] = radius
        starts.append(coefficients)

    lower = np.full(num_coefficients, -0.25 * maximum_radius)
    upper = np.full(num_coefficients, 0.25 * maximum_radius)
    lower[0] = minimum_radius
    upper[0] = maximum_radius
    accepted = []

    for start in starts:
        start = np.clip(start, lower + 1.0e-12, upper - 1.0e-12)

        def residual(coefficients):
            expansion = _expansion(coefficients, mu, geometry)
            return coefficients[0] * expansion

        solution = least_squares(
            residual,
            start,
            bounds=(lower, upper),
            xtol=1.0e-10,
            ftol=1.0e-10,
            gtol=1.0e-10,
            max_nfev=100,
        )
        coefficients = solution.x
        radius = _even_legendre_values(coefficients, mu)
        expansion = _expansion(coefficients, mu, geometry)
        dimensionless_linf = float(
            np.mean(radius) * np.max(np.abs(expansion))
        )
        if (
            not solution.success
            or dimensionless_linf > residual_tolerance
            or np.any(radius <= 1.5 * geometry["dx"])
            or np.any(radius >= geometry["maximum_radius"])
        ):
            continue
        if any(
            abs(coefficients[0] - other[0]) < 0.25 * geometry["dx"]
            for other in accepted
        ):
            continue
        accepted.append(coefficients)

    if not accepted:
        return None

    coefficients = max(accepted, key=lambda candidate: candidate[0])
    expansion = _expansion(coefficients, mu, geometry)
    area, mass, ratio, polar_radius, equatorial_radius = _intrinsic_diagnostics(
        coefficients, geometry
    )
    return AxisymmetricHorizon(
        coefficients=np.asarray(coefficients),
        expansion_l2=float(np.sqrt(np.mean(expansion**2))),
        expansion_linf=float(np.max(np.abs(expansion))),
        area=area,
        irreducible_mass=mass,
        circumference_ratio=ratio,
        polar_radius=polar_radius,
        equatorial_radius=equatorial_radius,
    )


__all__ = ["AxisymmetricHorizon", "find_axisymmetric_apparent_horizon"]
