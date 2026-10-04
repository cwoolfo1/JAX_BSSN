"""Compatible native Yee curls and quadratic sources on the rho-z plane."""

import jax
import jax.numpy as jnp

from JAX_BSSN.bssn.geometry import W_FLOOR_VALUE
from JAX_BSSN.cartoon.axisymmetry.reconstruction import (
    _compact_field, _tensor_parity, _support_geometry, _rotate_tensor,
)
from JAX_BSSN.EM.first_order.coupling import (
    AXIS_DISPLACEMENT_LOCATIONS as DL,
    AXIS_MAGNETIC_LOCATIONS as BL,
    constitutive_fields, quadratic_moments, sources_from_moments,
)
from JAX_BSSN.EM.first_order.equations import _forward_difference, _backward_difference
from JAX_BSSN.EM.first_order.geometry import _metric_fields_at_location
from JAX_BSSN.EM.first_order.boundaries import _apply_component_boundary


def positive_bssn(bssn):
    return jax.tree_util.tree_map(lambda q: q[..., 4:, :, :], bssn)


def _radii(shape, dx, dtype):
    i = jnp.arange(shape[0], dtype=dtype)
    return (i - 3.5)[:, None, None] * dx, (i - 4.0)[:, None, None] * dx


def _vertex_radial_divergence(radial, params):
    rc, rv = _radii(radial.shape, params.dx, radial.dtype)
    result = _backward_difference(rc * radial, 0, params) / jnp.where(rv == 0, 1.0, rv)
    return result.at[4].set(4.0 * radial[4] / params.dx)


@jax.jit
def cylindrical_curls(electric, magnetic_covector, params):
    """Return (curl H, -curl E) with annular native dual volumes."""
    er, ep, ez = electric
    hr, hp, hz = magnetic_covector
    rc, rv = _radii(er.shape, params.dx, er.dtype)
    dd = jnp.stack((
        -_forward_difference(hp, 2, params),
        _forward_difference(hr, 2, params) - _forward_difference(hz, 0, params),
        _forward_difference(rv * hp, 0, params) / rc,
    ))
    db = jnp.stack((
        _backward_difference(ep, 2, params),
        _backward_difference(ez, 0, params) - _backward_difference(er, 2, params),
        -_vertex_radial_divergence(ep, params),
    ))
    return dd.at[0, 4].set(0.0), db.at[1, 4].set(0.0)


@jax.jit
def cylindrical_divergences(displacement, magnetic, params):
    rc, rv = _radii(displacement.shape[-3:], params.dx, displacement.dtype)
    dd = _forward_difference(rv * displacement[0], 0, params) / rc + _forward_difference(displacement[2], 2, params)
    db = _vertex_radial_divergence(magnetic[0], params) + _backward_difference(magnetic[2], 2, params)
    # Diagnostics retain compact scalar parity ghosts.
    return tuple(jnp.concatenate((jnp.flip(q[4:8], axis=0), q[4:]), axis=0) for q in (dd, db))


def _vector_ghosts(vector, locations):
    # Import lazily to keep the existing public parity helper as one source of truth.
    from JAX_BSSN.EM.first_order.cartoon.axisymmetry import _fill_vector_ghosts
    blank = jnp.pad(vector, ((0, 0), (4, 0), (0, 0), (0, 0)))
    return _fill_vector_ghosts(blank, locations)


@jax.jit
def cylindrical_maxwell_rhs(bssn, displacement, magnetic, params):
    from JAX_BSSN.EM.first_order.cartoon.axisymmetry import _fill_vector_ghosts
    d = _fill_vector_ghosts(displacement, DL)
    b = _fill_vector_ghosts(magnetic, BL)
    electric, h = constitutive_fields(d[:, 4:], b[:, 4:], positive_bssn(bssn), params, axisymmetric=True)
    electric, h = _vector_ghosts(electric, DL), _vector_ghosts(h, BL)
    dd, db = cylindrical_curls(electric, h, params)
    # Native cylindrical sites all lie at y=0, including the azimuthal fields.
    native_params = params._replace(y_min=0.0)
    results = []
    for field, rhs, locations in ((d, dd, DL), (b, db, BL)):
        geometry = tuple(_metric_fields_at_location(bssn, loc, native_params) for loc in locations)
        result = jnp.stack(tuple(_apply_component_boundary(field[i], rhs[i], geometry[i], locations[i], native_params) for i in range(3)))
        results.append(_fill_vector_ghosts(result, locations))
    return tuple(results)


def compact_moments(displacement, magnetic, params):
    moments = quadratic_moments(displacement[:, 4:], magnetic[:, 4:], params, axisymmetric=True)
    return tuple(_compact_field(q[..., 0, :], _tensor_parity(q.dtype)) for q in moments)


def _moment_support(moment, params):
    """Convex radial interpolation and proper rotations preserve joint PSD."""
    reference = moment[..., 0, :]
    nx, nz = reference.shape[-2:]
    edge = (nx - 4.5) * params.dx
    z = params.z_min + params.dx * jnp.arange(nz, dtype=moment.dtype)
    rb = edge + params.dx * jnp.arange(1, 4, dtype=moment.dtype)
    ratio_squared = (edge**2 + z**2)[None, :] / (rb[:, None]**2 + z[None, :]**2)
    source = jnp.concatenate((reference, reference[..., -1:, :] * ratio_squared), axis=-2)
    q, cosine, sine = _support_geometry(reference, params)
    lower = jnp.floor(q).astype(jnp.int32)
    fraction = (q - lower)[..., None]
    low = jnp.take(source, lower, axis=-2)
    high = jnp.take(source, lower + 1, axis=-2)
    values = (1.0 - fraction) * low + fraction * high
    return _rotate_tensor(values, cosine, sine, moment.dtype)


@jax.jit
def axisymmetric_electromagnetic_sources(displacement, magnetic, bssn_support, params):
    """Form native quadratic moments before reconstructing BSSN matter support."""
    moments = tuple(_moment_support(q, params) for q in compact_moments(displacement, magnetic, params))
    return sources_from_moments(moments, bssn_support)


@jax.jit
def axisymmetric_physical_fields(displacement, magnetic, bssn, params):
    """Cell-centered physical vectors for output (not quadratic sources)."""
    from JAX_BSSN.EM.first_order.staggering import vector_at_location, CENTER_LOCATION
    W = jnp.maximum(bssn.conformal_factor, W_FLOOR_VALUE)
    return (W**3 * vector_at_location(displacement, DL, CENTER_LOCATION, params),
            W**3 * vector_at_location(magnetic, BL, CENTER_LOCATION, params))
