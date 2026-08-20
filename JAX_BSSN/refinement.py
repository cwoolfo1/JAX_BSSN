"""A minimal, stage-synchronous fixed mesh refinement hierarchy.

This module implements one vertex-centred 2:1 patch with four guard cells.
Six-point (degree-five) spatial prolongation supplies every RK stage, and
coincident fine nodes are injected back after a complete step.  Both levels use
the same fine-CFL timestep: Berger--Oliger subcycling is intentionally deferred.
The ordinary BSSN finite differences and KO operator are used unchanged; no
mesh-adapted stencil is introduced.
"""

from functools import lru_cache
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

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
    dtype = jnp.result_type(dx_coarse, 1.0)
    spacing = jnp.asarray(dx_coarse, dtype=dtype)
    half = jnp.asarray(2.0, dtype=dtype)
    axes = [
        jnp.asarray(origin[d], dtype=dtype)
        + (
            jnp.asarray(spec.coarse_lo[d], dtype=dtype)
            + (
                jnp.arange(
                    fine_active_shape(spec)[d] + 2*g,
                    dtype=dtype,
                )
                - jnp.asarray(g, dtype=dtype)
            ) / half
        ) * spacing
        for d in range(3)
    ]
    return jnp.meshgrid(*axes, indexing="ij")


@lru_cache(maxsize=None)
def _cached_interpolation_matrices(coarse_shape, spec, dtype_name):
    """Build periodic quintic interpolation matrices for one static patch."""
    dtype = np.dtype(dtype_name)
    offsets = np.arange(-2, 4, dtype=np.int32)
    matrices = []

    for direction in range(3):
        target_count = (
            fine_active_shape(spec)[direction] + 2 * spec.ghost_width
        )
        targets = (
            spec.coarse_lo[direction]
            + (
                np.arange(target_count, dtype=dtype)
                - dtype.type(spec.ghost_width)
            ) / dtype.type(2.0)
        )
        base = np.floor(targets).astype(np.int32)
        nodes = base[:, None] + offsets[None, :]
        weights = np.ones(nodes.shape, dtype=dtype)

        for k in range(6):
            for m in range(6):
                if k != m:
                    weights[:, k] *= (
                        (targets - nodes[:, m])
                        / (nodes[:, k] - nodes[:, m])
                    )

        matrix = np.zeros(
            (target_count, coarse_shape[direction]), dtype=dtype
        )
        periodic_nodes = nodes % coarse_shape[direction]
        # Small periodic grids can wrap multiple stencil nodes onto one point.
        for target_index in range(target_count):
            for stencil_index in range(6):
                matrix[
                    target_index,
                    periodic_nodes[target_index, stencil_index],
                ] += weights[target_index, stencil_index]
        matrices.append(matrix)

    return tuple(matrices)


def _interpolation_matrices(coarse, spec):
    host_matrices = _cached_interpolation_matrices(
        tuple(coarse.shape[-3:]), spec, str(coarse.dtype)
    )
    return tuple(
        jnp.asarray(matrix, dtype=coarse.dtype)
        for matrix in host_matrices
    )


def _prolongate_with_matrices(coarse, matrices):
    """Apply the separable three-dimensional interpolation operator."""
    matrix_x, matrix_y, matrix_z = matrices
    fine = jnp.einsum("ia,...abc->...ibc", matrix_x, coarse)
    fine = jnp.einsum("jb,...ibc->...ijc", matrix_y, fine)
    return jnp.einsum("kc,...ijc->...ijk", matrix_z, fine)


def prolongate_to_fine(coarse: jnp.ndarray, spec: FMRPatchSpec) -> jnp.ndarray:
    """Tensor-product quintic prolongation for arbitrary leading dimensions."""
    matrices = _interpolation_matrices(coarse, spec)
    return _prolongate_with_matrices(coarse, matrices)


def _map_vars(fn, variables):
    return BSSNVariables(*(fn(x) for x in variables))


def _fill_fine_ghosts_with_matrices(coarse_vars, fine_vars, spec, matrices):
    active = fine_active_slice(spec)
    fields = []
    for coarse, fine in zip(coarse_vars, fine_vars):
        supplied = _prolongate_with_matrices(coarse, matrices)
        supplied = supplied.at[(...,) + active].set(fine[(...,) + active])
        fields.append(supplied)
    return BSSNVariables(*fields)


def fill_fine_ghosts(coarse_vars, fine_vars, spec):
    """Refresh faces, edges, and corners without overwriting active fine data."""
    matrices = _interpolation_matrices(coarse_vars[0], spec)
    return _fill_fine_ghosts_with_matrices(
        coarse_vars, fine_vars, spec, matrices
    )


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


def _stage(coarse, fine, cp, fp, spec, matrices):
    coarse, fine = _project(coarse), _project(fine)
    fine = _fill_fine_ghosts_with_matrices(coarse, fine, spec, matrices)
    return coarse, fine, compute_bssn_rhs(coarse, cp), compute_bssn_rhs(fine, fp)


def fmr_rk4_step(coarse, fine, coarse_params: BSSNParameters,
                 fine_params: BSSNParameters, spec: FMRPatchSpec):
    """Advance matching coarse/fine RK stages with one shared timestep."""
    if coarse_params.dt != fine_params.dt:
        raise ValueError("stage-synchronous FMR requires identical timesteps")
    if fine_params.dx * 2 != coarse_params.dx:
        raise ValueError("fine dx must be coarse dx / 2")
    dt = coarse_params.dt
    matrices = _interpolation_matrices(coarse[0], spec)
    c0, f0, k1c, k1f = _stage(
        coarse, fine, coarse_params, fine_params, spec, matrices
    )
    _, _, k2c, k2f = _stage(
        _add(c0, k1c, dt/2), _add(f0, k1f, dt/2),
        coarse_params, fine_params, spec, matrices,
    )
    _, _, k3c, k3f = _stage(
        _add(c0, k2c, dt/2), _add(f0, k2f, dt/2),
        coarse_params, fine_params, spec, matrices,
    )
    _, _, k4c, k4f = _stage(
        _add(c0, k3c, dt), _add(f0, k3f, dt),
        coarse_params, fine_params, spec, matrices,
    )
    c = BSSNVariables(*(x + dt*(a + 2*b + 2*d + e)/6
                        for x, a, b, d, e in zip(c0, k1c, k2c, k3c, k4c)))
    f = BSSNVariables(*(x + dt*(a + 2*b + 2*d + e)/6
                        for x, a, b, d, e in zip(f0, k1f, k2f, k3f, k4f)))
    f = _project(f)
    c = _project(restrict_to_coarse(c, f, spec))
    return c, _fill_fine_ghosts_with_matrices(c, f, spec, matrices)
