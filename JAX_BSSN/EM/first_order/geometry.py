"""Interpolate existing BSSN variables onto the FPIC Yee locations."""

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


def _metric_fields_at_location(
    bssn: BSSNVariables,
    location,
    params: BSSNParameters,
):
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
    return (
        W,
        lapse,
        shift,
        conformal_metric,
        inverse_conformal_metric,
    )


def _metric_fields_on_yee_sites(
    bssn: BSSNVariables, params: BSSNParameters
) -> tuple[tuple, tuple]:
    """Return short-lived metric tuples on the six native field sites."""

    displacement = tuple(
        _metric_fields_at_location(bssn, location, params)
        for location in DISPLACEMENT_FIELD_LOCATIONS
    )
    magnetic = tuple(
        _metric_fields_at_location(bssn, location, params)
        for location in MAGNETIC_FIELD_LOCATIONS
    )
    return displacement, magnetic


def _conformal_factors_at_locations(
    bssn: BSSNVariables,
    locations,
    params: BSSNParameters,
) -> jnp.ndarray:
    """Return floored ``W`` at three native Yee locations."""

    return jnp.stack(
        tuple(
            jnp.maximum(
                interpolate_between_locations(
                    bssn.conformal_factor,
                    CENTER_LOCATION,
                    location,
                    params,
                ),
                W_FLOOR_VALUE,
            )
            for location in locations
        ),
        axis=0,
    )


__all__ = []
