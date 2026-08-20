"""A minimal, stage-synchronous fixed mesh refinement hierarchy.

This module implements one vertex-centred 2:1 patch with four guard cells.
Six-point (degree-five) spatial prolongation supplies every RK stage, and
coincident fine nodes are injected back after a complete step.  Both levels use
the same fine-CFL timestep: Berger--Oliger subcycling is intentionally deferred.
The ordinary BSSN finite differences and KO operator are used unchanged; no
mesh-adapted stencil is introduced.
"""

from typing import NamedTuple

import jax.numpy as jnp

from JAX_BSSN.bssn import BSSNParameters, BSSNVariables
from JAX_BSSN.evolve import (compute_bssn_rhs,
    eliminate_trace_A, enforce_unit_determinant_conformal_metric)


class FMRPatchSpec(NamedTuple):
    """Inclusive coarse-index bounds for one immutable vertex-centred patch."""
    coarse_lo: tuple[int, int, int]
    coarse_hi: tuple[int, int, int]
    refinement_ratio: int = 2
    ghost_width: int = 4



def _validate_spec(spec: FMRPatchSpec) -> None:
    if spec.refinement_ratio != 2:
        raise ValueError("the first-pass FMR ratio is fixed at 2")
    if spec.ghost_width != 4:
        raise ValueError("the BSSN composed stencil requires four guard cells")
    if any(h < l for l, h in zip(spec.coarse_lo, spec.coarse_hi)):
        raise ValueError("coarse_hi must not precede coarse_lo")


def fine_active_shape(spec: FMRPatchSpec) -> tuple[int, int, int]:
    _validate_spec(spec)
    return tuple(2 * (h - l) + 1 for l, h in zip(spec.coarse_lo, spec.coarse_hi))


def fine_active_slice(spec: FMRPatchSpec) -> tuple[slice, slice, slice]:
    g = spec.ghost_width
    return tuple(slice(g, g + n) for n in fine_active_shape(spec))


def fine_active_view(array: jnp.ndarray, spec: FMRPatchSpec) -> jnp.ndarray:
    return array[(...,) + fine_active_slice(spec)]


def fine_coordinates(spec: FMRPatchSpec, dx_coarse: float,
                     origin=(0.0, 0.0, 0.0)):
    """Return coordinates for the padded fine allocation, including guards."""
    g = spec.ghost_width
    axes = [origin[d] + (spec.coarse_lo[d] +
            (jnp.arange(fine_active_shape(spec)[d] + 2*g) - g) / 2.0) * dx_coarse
            for d in range(3)]
    return jnp.meshgrid(*axes, indexing="ij")


def _interpolate_axis(a, targets, axis):
    """Periodic six-node Lagrange interpolation at coarse-index coordinates."""
    base = jnp.floor(targets).astype(jnp.int32)
    offsets = jnp.arange(-2, 4, dtype=jnp.int32)
    nodes = base[:, None] + offsets[None, :]
    x = targets[:, None]
    weights = jnp.ones_like(x + offsets, dtype=a.dtype)
    for k in range(6):
        for m in range(6):
            if k != m:
                weights = weights.at[:, k].multiply(
                    (x[:, 0] - nodes[:, m]) / (nodes[:, k] - nodes[:, m]))
    gathered = jnp.take(a, nodes % a.shape[axis], axis=axis)
    # jnp.take inserts (target, stencil) at axis.
    shape = [1] * gathered.ndim
    shape[axis], shape[axis + 1] = targets.size, 6
    return jnp.sum(gathered * weights.reshape(shape), axis=axis + 1)


def prolongate_to_fine(coarse: jnp.ndarray, spec: FMRPatchSpec) -> jnp.ndarray:
    """Tensor-product quintic prolongation for arbitrary leading dimensions."""
    g = spec.ghost_width
    out = coarse
    for d in range(3):
        n = fine_active_shape(spec)[d] + 2*g
        q = spec.coarse_lo[d] + (jnp.arange(n) - g) / 2.0
        out = _interpolate_axis(out, q, out.ndim - 3 + d)
    return out


def _map_vars(fn, variables):
    return BSSNVariables(*(fn(x) for x in variables))


def fill_fine_ghosts(coarse_vars, fine_vars, spec):
    """Refresh faces, edges, and corners without overwriting active fine data."""
    active = fine_active_slice(spec)
    fields = []
    for coarse, fine in zip(coarse_vars, fine_vars):
        supplied = prolongate_to_fine(coarse, spec)
        supplied = supplied.at[(...,) + active].set(fine[(...,) + active])
        fields.append(supplied)
    return BSSNVariables(*fields)


def restrict_to_coarse(coarse_vars, fine_vars, spec):
    """Inject coincident active fine nodes into the inclusive covered box."""
    covered = tuple(slice(l, h + 1) for l, h in zip(spec.coarse_lo, spec.coarse_hi))
    fields = []
    for coarse, fine in zip(coarse_vars, fine_vars):
        injected = fine_active_view(fine, spec)[(..., slice(None, None, 2),
                                               slice(None, None, 2), slice(None, None, 2))]
        fields.append(coarse.at[(...,) + covered].set(injected))
    return BSSNVariables(*fields)


def _project(v):
    return eliminate_trace_A(enforce_unit_determinant_conformal_metric(v))


def _add(v, rhs, scale):
    return BSSNVariables(*(x + scale*y for x, y in zip(v, rhs)))


def _stage(coarse, fine, cp, fp, spec):
    coarse, fine = _project(coarse), _project(fine)
    fine = fill_fine_ghosts(coarse, fine, spec)
    return coarse, fine, compute_bssn_rhs(coarse, cp), compute_bssn_rhs(fine, fp)


def fmr_rk4_step(coarse, fine, coarse_params: BSSNParameters,
                 fine_params: BSSNParameters, spec: FMRPatchSpec):
    """Advance matching coarse/fine RK stages with one shared timestep."""
    if coarse_params.dt != fine_params.dt:
        raise ValueError("stage-synchronous FMR requires identical timesteps")
    if fine_params.dx * 2 != coarse_params.dx:
        raise ValueError("fine dx must be coarse dx / 2")
    dt = coarse_params.dt
    c0, f0, k1c, k1f = _stage(coarse, fine, coarse_params, fine_params, spec)
    _, _, k2c, k2f = _stage(_add(c0, k1c, dt/2), _add(f0, k1f, dt/2), coarse_params, fine_params, spec)
    _, _, k3c, k3f = _stage(_add(c0, k2c, dt/2), _add(f0, k2f, dt/2), coarse_params, fine_params, spec)
    _, _, k4c, k4f = _stage(_add(c0, k3c, dt), _add(f0, k3f, dt), coarse_params, fine_params, spec)
    c = BSSNVariables(*(x + dt*(a + 2*b + 2*d + e)/6
                        for x, a, b, d, e in zip(c0, k1c, k2c, k3c, k4c)))
    f = BSSNVariables(*(x + dt*(a + 2*b + 2*d + e)/6
                        for x, a, b, d, e in zip(f0, k1f, k2f, k3f, k4f)))
    f = _project(f)
    c = _project(restrict_to_coarse(c, f, spec))
    return c, fill_fine_ghosts(c, f, spec)
