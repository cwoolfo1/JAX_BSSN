"""Local quadratic Yee energy and its transpose constitutive transfer.

Products are formed from native samples, not from an averaged vector.  The
same corner gathers define the stress moments and the discrete Hamiltonian.
Constitutive fields use the transpose gathers, with native volume weights.
"""

from itertools import product

import jax
import jax.numpy as jnp

from JAX_BSSN.bssn.geometry import W_FLOOR_VALUE
from JAX_BSSN.bssn.tensor_algebra import determinant_3x3_metric
from JAX_BSSN.bssn.variables import get_boundary_codes
from JAX_BSSN.evolution.boundaries import SOMMERFELD_BC
from JAX_BSSN.EM.first_order.staggering import (
    DISPLACEMENT_FIELD_LOCATIONS, MAGNETIC_FIELD_LOCATIONS,
)


def _halo(field, params):
    """One-cell halos; extrapolated ghosts are boundary data for the scatter."""
    for direction in range(3):
        axis = field.ndim - 3 + direction
        if field.shape[axis] == 1:
            continue
        first = jnp.take(field, jnp.array([0]), axis=axis)
        last = jnp.take(field, jnp.array([-1]), axis=axis)
        second = jnp.take(field, jnp.array([1]), axis=axis)
        previous = jnp.take(field, jnp.array([-2]), axis=axis)
        left_bc, right_bc = get_boundary_codes(params, direction)
        left = jnp.where(left_bc == SOMMERFELD_BC, 2 * first - second, last)
        right = jnp.where(right_bc == SOMMERFELD_BC, 2 * last - previous, first)
        field = jnp.concatenate((left, field, right), axis=axis)
    return field


def _trim(field, shape):
    return field[(Ellipsis,) + tuple(slice(1, -1) if n > 1 else slice(None) for n in shape)]


def _shift(field, location, corner, transpose=False):
    offsets = tuple((1 if transpose else -1) * corner[d] if location[d] == "V" else 0 for d in range(3))
    return jnp.roll(field, offsets, axis=(-3, -2, -1))


def _gather(vector, locations, corner):
    return jnp.stack(tuple(_shift(vector[i], locations[i], corner) for i in range(3)))


def _scatter(vector, locations, corner):
    return jnp.stack(tuple(_shift(vector[i], locations[i], corner, True) for i in range(3)))


def _quadrature(dtype):
    for corner in product((0, 1), repeat=3):
        yield corner, jnp.asarray(0.125, dtype=dtype)


@jax.jit
def quadratic_moments(displacement, magnetic, params):
    """Return cell-centered quadratic moments <dd>, <bb>, <db>."""
    shape = displacement.shape[-3:]
    dl = DISPLACEMENT_FIELD_LOCATIONS
    bl = MAGNETIC_FIELD_LOCATIONS
    d = _halo(displacement, params)
    b = _halo(magnetic, params)
    dd = bb = db = jnp.zeros((3, 3) + d.shape[-3:], dtype=d.dtype)
    for corner, weight in _quadrature(d.dtype):
        dc, bc = _gather(d, dl, corner), _gather(b, bl, corner)
        dd = dd + weight * jnp.einsum("i...,j...->ij...", dc, dc)
        bb = bb + weight * jnp.einsum("i...,j...->ij...", bc, bc)
        db = db + weight * jnp.einsum("i...,j...->ij...", dc, bc)
    moments = tuple(_trim(q, shape) for q in (dd, bb, db))
    return moments


def cell_metric(bssn):
    metric = bssn.conformal_metric
    return metric * determinant_3x3_metric(metric)**(-1.0 / 3.0)


@jax.jit
def sources_from_moments(moments, bssn):
    dd, bb, db = moments
    q = dd + bb
    W = jnp.maximum(bssn.conformal_factor, W_FLOOR_VALUE)
    metric = cell_metric(bssn)
    contraction = jnp.einsum("ij...,ij...->...", metric, q)
    rho = 0.5 * W**4 * contraction
    momentum = W**3 * jnp.stack((db[1, 2] - db[2, 1], db[2, 0] - db[0, 2], db[0, 1] - db[1, 0]))
    stress = W**2 * (0.5 * metric * contraction - jnp.einsum("ik...,jl...,kl...->ij...", metric, metric, q))
    return rho, momentum, stress


@jax.jit
def constitutive_fields(displacement, magnetic, bssn, params):
    """Hamiltonian derivatives on native sites, using explicit transpose sums."""
    shape = displacement.shape[-3:]
    dl = DISPLACEMENT_FIELD_LOCATIONS
    bl = MAGNETIC_FIELD_LOCATIONS
    d = _halo(displacement, params)
    b = _halo(magnetic, params)
    W = jnp.maximum(bssn.conformal_factor, W_FLOOR_VALUE)
    coefficient = _halo(bssn.lapse * W * cell_metric(bssn), params)
    shift = _halo(bssn.shift, params)
    electric = magnetic_covector = jnp.zeros_like(d)
    for corner, weight in _quadrature(d.dtype):
        dc, bc = _gather(d, dl, corner), _gather(b, bl, corner)
        ed = jnp.einsum("ij...,j...->i...", coefficient, dc) + jnp.cross(shift, bc, axisa=0, axisb=0, axisc=0)
        hb = jnp.einsum("ij...,j...->i...", coefficient, bc) - jnp.cross(shift, dc, axisa=0, axisb=0, axisc=0)
        electric = electric + _scatter(weight * ed, dl, corner)
        magnetic_covector = magnetic_covector + _scatter(weight * hb, bl, corner)
    electric, magnetic_covector = _trim(electric, shape), _trim(magnetic_covector, shape)
    return electric, magnetic_covector
