"""Reconstruct the BSSN geometry used by the Maxwell wave equations."""

import jax
import jax.numpy as jnp

from JAX_BSSN.bssn import BSSNParameters, BSSNVariables
from JAX_BSSN.bssn.geometry import W_FLOOR_VALUE
from JAX_BSSN.bssn.tensor_algebra import (
    christoffel_symbols_second_kind,
    determinant_3x3_metric,
    invert_3x3_metric,
)

from JAX_BSSN.EM.derivatives import (
    covariant_derivative_covector,
    covariant_derivative_tensor2,
    lie_derivative_covector,
    spatial_derivatives,
)
from JAX_BSSN.EM.variables import BSSNEMGeometry


LAPSE_FLOOR_VALUE = 1.0e-12
EINSTEIN_COUPLING = 8.0 * jnp.pi

LEVI_CIVITA_SYMBOL = jnp.asarray(
    [
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        [[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ]
)


def tracefree(
    tensor: jnp.ndarray,
    metric: jnp.ndarray,
    inverse_metric: jnp.ndarray,
) -> jnp.ndarray:
    """Return the symmetric trace-free part of a covariant rank-two tensor."""

    symmetric = 0.5 * (tensor + jnp.swapaxes(tensor, 0, 1))
    trace = jnp.einsum("ij...,ij...->...", inverse_metric, symmetric)
    return symmetric - metric * trace / 3.0


def spatial_ricci_tensor(
    christoffel: jnp.ndarray, params: BSSNParameters
) -> jnp.ndarray:
    """Compute the physical spatial Ricci tensor from Christoffel symbols."""

    derivative = spatial_derivatives(christoffel, params)

    derivative_trace = jnp.einsum("kkij...->ij...", derivative)
    trace_derivative = jnp.einsum("jkik...->ij...", derivative)
    connection_trace = jnp.einsum("lkl...->k...", christoffel)
    quadratic_trace = jnp.einsum(
        "kij...,k...->ij...", christoffel, connection_trace
    )
    quadratic = jnp.einsum(
        "kil...,ljk...->ij...", christoffel, christoffel
    )

    ricci = derivative_trace - trace_derivative + quadratic_trace - quadratic
    return 0.5 * (ricci + jnp.swapaxes(ricci, 0, 1))


def curl_symmetric_tensor(
    tensor: jnp.ndarray,
    metric: jnp.ndarray,
    inverse_metric: jnp.ndarray,
    levi_civita: jnp.ndarray,
    christoffel: jnp.ndarray,
    params: BSSNParameters,
) -> jnp.ndarray:
    """Return the projected symmetric trace-free curl of ``tensor_ij``."""

    derivative = covariant_derivative_tensor2(tensor, christoffel, params)
    raw = jnp.einsum(
        "klx...,kp...,lq...,pyq...->xy...",
        levi_civita,
        inverse_metric,
        inverse_metric,
        derivative,
    )
    return tracefree(raw, metric, inverse_metric)


def _signed_floor(value: jnp.ndarray, floor: float) -> jnp.ndarray:
    """Keep a denominator finite while preserving its nonzero sign."""

    return jnp.where(
        value >= 0.0,
        jnp.maximum(value, floor),
        jnp.minimum(value, -floor),
    )


def _physical_christoffel(
    metric: jnp.ndarray,
    inverse_metric: jnp.ndarray,
    params: BSSNParameters,
) -> jnp.ndarray:
    """Return physical Christoffel symbols from the floored physical metric."""

    metric_derivatives = spatial_derivatives(metric, params)
    return christoffel_symbols_second_kind(
        inverse_metric, metric_derivatives
    )


def _matter_projections(
    backreaction_sources,
    metric: jnp.ndarray,
    inverse_metric: jnp.ndarray,
):
    """Return the EM ``rho``, stress trace, and anisotropic stress."""

    if backreaction_sources is None:
        shape = metric.shape[2:]
        rho = jnp.zeros(shape, dtype=metric.dtype)
        stress_trace = jnp.zeros_like(rho)
        anisotropic_stress = jnp.zeros_like(metric)
        return rho, stress_trace, anisotropic_stress

    if hasattr(backreaction_sources, "energy_density"):
        return (
            backreaction_sources.energy_density,
            backreaction_sources.stress_trace,
            backreaction_sources.anisotropic_stress,
        )

    rho, _, spatial_stress = backreaction_sources
    stress_trace = jnp.einsum(
        "ij...,ij...->...", inverse_metric, spatial_stress
    )
    anisotropic_stress = spatial_stress - metric * stress_trace / 3.0
    return rho, stress_trace, anisotropic_stress


@jax.jit
def compute_bssn_em_geometry(
    bssn: BSSNVariables,
    bssn_rhs: BSSNVariables,
    params: BSSNParameters,
    backreaction_sources=None,
) -> BSSNEMGeometry:
    """Build curved-space source tensors from one synchronized BSSN stage.

    ``backreaction_sources`` is the electromagnetic ``(rho, S_i, S_ij)``
    tuple used in the BSSN stage.  Removing its Ricci projections here avoids
    counting the Maxwell field once through the evolved geometry and a second
    time through the explicit second-order Maxwell source equation.
    """

    W = jnp.maximum(bssn.conformal_factor, W_FLOOR_VALUE)
    inverse_conformal_metric = invert_3x3_metric(bssn.conformal_metric)
    metric = bssn.conformal_metric / W**2
    inverse_metric = W**2 * inverse_conformal_metric

    christoffel = _physical_christoffel(metric, inverse_metric, params)
    spatial_ricci = spatial_ricci_tensor(christoffel, params)

    traceless_extrinsic_curvature = bssn.traceless_K / W**2
    extrinsic_curvature = (
        traceless_extrinsic_curvature + metric * bssn.trace_K / 3.0
    )
    mixed_extrinsic_curvature = jnp.einsum(
        "ik...,kj...->ij...", extrinsic_curvature, inverse_metric
    )

    sqrt_conformal_determinant = jnp.sqrt(
        determinant_3x3_metric(bssn.conformal_metric)
    )
    sqrt_metric_determinant = sqrt_conformal_determinant / W**3
    levi_civita = (
        LEVI_CIVITA_SYMBOL.astype(metric.dtype)[:, :, :, None, None, None]
        * sqrt_metric_determinant
    )

    lapse = _signed_floor(bssn.lapse, LAPSE_FLOOR_VALUE)
    acceleration = spatial_derivatives(bssn.lapse, params) / lapse
    coordinate_acceleration_dot = spatial_derivatives(
        bssn_rhs.lapse / lapse, params
    )
    acceleration_lie = lie_derivative_covector(
        acceleration, bssn.shift, params
    )
    acceleration_dot = (
        coordinate_acceleration_dot - acceleration_lie
    ) / lapse + jnp.einsum(
        "ij...,j...->i...", mixed_extrinsic_curvature, acceleration
    )

    # Raychaudhuri with Theta=-K.  The BSSN stage already contains the EM
    # stress, so subtract its R_nn=4 pi (rho+S) contribution below.
    trace_K_derivative = spatial_derivatives(bssn.trace_K, params)
    trace_K_lie = jnp.einsum(
        "i...,i...->...", bssn.shift, trace_K_derivative
    )
    trace_K_dot = (bssn_rhs.trace_K - trace_K_lie) / lapse

    acceleration_derivative = covariant_derivative_covector(
        acceleration, christoffel, params
    )
    acceleration_divergence = jnp.einsum(
        "ij...,ij...->...", inverse_metric, acceleration_derivative
    )
    acceleration_squared = jnp.einsum(
        "ij...,i...,j...->...",
        inverse_metric,
        acceleration,
        acceleration,
    )
    traceless_extrinsic_curvature_squared = jnp.einsum(
        "ij...,ik...,jl...,kl...->...",
        traceless_extrinsic_curvature,
        inverse_metric,
        inverse_metric,
        traceless_extrinsic_curvature,
    )
    total_normal_normal_ricci = (
        trace_K_dot
        - bssn.trace_K**2 / 3.0
        - traceless_extrinsic_curvature_squared
        + acceleration_divergence
        + acceleration_squared
    )

    rho, stress_trace, anisotropic_stress = _matter_projections(
        backreaction_sources, metric, inverse_metric
    )
    normal_normal_ricci = total_normal_normal_ricci - 0.5 * (
        EINSTEIN_COUPLING * (rho + stress_trace)
    )

    # The Gauss seed contains matter Ricci curvature.  Subtract 4 pi pi_ij
    # before taking its STF part to obtain the electric Weyl tensor itself.
    electric_weyl_seed = (
        spatial_ricci
        + bssn.trace_K * extrinsic_curvature
        - jnp.einsum(
            "ik...,kj...->ij...",
            mixed_extrinsic_curvature,
            extrinsic_curvature,
        )
        - 0.5 * EINSTEIN_COUPLING * anisotropic_stress
    )
    electric_weyl = tracefree(
        electric_weyl_seed, metric, inverse_metric
    )

    # The symmetric trace-free curl of K_ij is the magnetic Weyl tensor also
    # with matter present.  Symmetrization removes the momentum-constraint
    # antisymmetric part rather than assuming a vacuum constraint.
    magnetic_weyl = -curl_symmetric_tensor(
        traceless_extrinsic_curvature,
        metric,
        inverse_metric,
        levi_civita,
        christoffel,
        params,
    )

    return BSSNEMGeometry(
        metric=metric,
        inverse_metric=inverse_metric,
        traceless_extrinsic_curvature=traceless_extrinsic_curvature,
        extrinsic_curvature=extrinsic_curvature,
        christoffel=christoffel,
        levi_civita=levi_civita,
        acceleration=acceleration,
        acceleration_dot=acceleration_dot,
        normal_normal_ricci=normal_normal_ricci,
        spatial_ricci=spatial_ricci,
        electric_weyl=electric_weyl,
        magnetic_weyl=magnetic_weyl,
    )


__all__ = [
    "EINSTEIN_COUPLING",
    "LAPSE_FLOOR_VALUE",
    "LEVI_CIVITA_SYMBOL",
    "compute_bssn_em_geometry",
    "curl_symmetric_tensor",
    "spatial_ricci_tensor",
    "tracefree",
]
