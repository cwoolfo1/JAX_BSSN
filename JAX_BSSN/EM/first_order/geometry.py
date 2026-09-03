"""Interpolate projected BSSN fields onto the FPIC Yee locations."""

import jax
import jax.numpy as jnp

from JAX_BSSN.bssn import BSSNParameters, BSSNVariables
from JAX_BSSN.bssn.geometry import W_FLOOR_VALUE
from JAX_BSSN.bssn.tensor_algebra import invert_3x3_metric

from JAX_BSSN.EM.first_order.staggering import (
    CENTER_LOCATION,
    DISPLACEMENT_FIELD_LOCATIONS,
    MAGNETIC_FIELD_LOCATIONS,
    interpolate_between_locations,
)
from JAX_BSSN.EM.first_order.variables import (
    BSSNYeeGeometry,
    ConformalMetricFields,
)


def _metric_fields_at_location(
    bssn: BSSNVariables,
    location,
    params: BSSNParameters,
) -> ConformalMetricFields:
    W = interpolate_between_locations(
        bssn.conformal_factor, CENTER_LOCATION, location, params
    )
    lapse = interpolate_between_locations(
        bssn.lapse, CENTER_LOCATION, location, params
    )
    shift = interpolate_between_locations(
        bssn.shift, CENTER_LOCATION, location, params
    )
    conformal_metric = interpolate_between_locations(
        bssn.conformal_metric, CENTER_LOCATION, location, params
    )

    W = jnp.maximum(W, W_FLOOR_VALUE)
    inverse_conformal_metric = invert_3x3_metric(conformal_metric)
    return ConformalMetricFields(
        W=W,
        lapse=lapse,
        shift=shift,
        conformal_metric=conformal_metric,
        inverse_conformal_metric=inverse_conformal_metric,
    )


@jax.jit
def compute_bssn_yee_geometry(
    bssn: BSSNVariables, params: BSSNParameters
) -> BSSNYeeGeometry:
    """Return BSSN geometry without numerically evaluating a determinant."""

    displacement = tuple(
        _metric_fields_at_location(bssn, location, params)
        for location in DISPLACEMENT_FIELD_LOCATIONS
    )
    magnetic = tuple(
        _metric_fields_at_location(bssn, location, params)
        for location in MAGNETIC_FIELD_LOCATIONS
    )
    center = _metric_fields_at_location(bssn, CENTER_LOCATION, params)
    return BSSNYeeGeometry(
        displacement=displacement,
        magnetic=magnetic,
        center=center,
    )


__all__ = ["compute_bssn_yee_geometry"]
