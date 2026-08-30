"""Periodic and planar Sommerfeld boundaries for the Maxwell wave state."""

import jax
import jax.numpy as jnp

from JAX_BSSN.bssn import BSSNParameters, BSSNVariables
from JAX_BSSN.evolution.boundaries import SOMMERFELD_BC

from JAX_BSSN.EM.derivatives import spatial_derivatives
from JAX_BSSN.EM.variables import BSSNEMGeometry, EMVariables


def _outward_boundary_covector(
    shape: tuple[int, int, int],
    params: BSSNParameters,
    dtype,
) -> jnp.ndarray:
    """Return the sum of active outward face covectors at each grid point."""

    nx, ny, nz = shape
    i = jnp.arange(nx)[:, None, None]
    j = jnp.arange(ny)[None, :, None]
    k = jnp.arange(nz)[None, None, :]

    qx = jnp.zeros(shape, dtype=dtype)
    qx = qx + jnp.where(
        (params.xl_bc == SOMMERFELD_BC) & (i == 0), -1.0, 0.0
    )
    qx = qx + jnp.where(
        (params.xr_bc == SOMMERFELD_BC) & (i == nx - 1), 1.0, 0.0
    )

    qy = jnp.zeros(shape, dtype=dtype)
    qy = qy + jnp.where(
        (params.yl_bc == SOMMERFELD_BC) & (j == 0), -1.0, 0.0
    )
    qy = qy + jnp.where(
        (params.yr_bc == SOMMERFELD_BC) & (j == ny - 1), 1.0, 0.0
    )

    qz = jnp.zeros(shape, dtype=dtype)
    qz = qz + jnp.where(
        (params.zl_bc == SOMMERFELD_BC) & (k == 0), -1.0, 0.0
    )
    qz = qz + jnp.where(
        (params.zr_bc == SOMMERFELD_BC) & (k == nz - 1), 1.0, 0.0
    )

    return jnp.stack((qx, qy, qz), axis=0)


def _apply_planar_sommerfeld_field(
    field: jnp.ndarray,
    rhs: jnp.ndarray,
    outward_normal: jnp.ndarray,
    mask: jnp.ndarray,
    bssn: BSSNVariables,
    params: BSSNParameters,
) -> jnp.ndarray:
    """Replace one covector RHS by its outgoing local characteristic."""

    gradient = spatial_derivatives(field, params)
    characteristic_velocity = bssn.shift - bssn.lapse * outward_normal
    boundary_rhs = jnp.einsum(
        "j...,ji...->i...", characteristic_velocity, gradient
    )
    return jnp.where(mask[None, ...], boundary_rhs, rhs)


@jax.jit
def apply_planar_sommerfeld_boundaries(
    wave: EMVariables,
    rhs: EMVariables,
    bssn: BSSNVariables,
    geometry: BSSNEMGeometry,
    params: BSSNParameters,
) -> EMVariables:
    """Apply one face-union planar Sommerfeld replacement to the wave RHS."""

    shape = wave.electric_field.shape[-3:]
    boundary_covector = _outward_boundary_covector(
        shape, params, wave.electric_field.dtype
    )
    norm_squared = jnp.einsum(
        "ij...,i...,j...->...",
        geometry.inverse_metric,
        boundary_covector,
        boundary_covector,
    )
    mask = norm_squared > 0.0
    safe_norm = jnp.sqrt(jnp.where(mask, norm_squared, 1.0))
    outward_normal = jnp.einsum(
        "ij...,j...->i...", geometry.inverse_metric, boundary_covector
    ) / safe_norm

    return EMVariables(
        electric_field=_apply_planar_sommerfeld_field(
            wave.electric_field,
            rhs.electric_field,
            outward_normal,
            mask,
            bssn,
            params,
        ),
        electric_field_dot=_apply_planar_sommerfeld_field(
            wave.electric_field_dot,
            rhs.electric_field_dot,
            outward_normal,
            mask,
            bssn,
            params,
        ),
        magnetic_field=_apply_planar_sommerfeld_field(
            wave.magnetic_field,
            rhs.magnetic_field,
            outward_normal,
            mask,
            bssn,
            params,
        ),
        magnetic_field_dot=_apply_planar_sommerfeld_field(
            wave.magnetic_field_dot,
            rhs.magnetic_field_dot,
            outward_normal,
            mask,
            bssn,
            params,
        ),
    )
