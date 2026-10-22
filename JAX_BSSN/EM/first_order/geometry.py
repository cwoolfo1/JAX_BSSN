"""Convert BSSN geometry to PyPIC3D's native metric samples."""

import jax.numpy as jnp
from dataclasses import replace
from PyPIC3D.relativity.core import (
    Metric,
    YeeMetric,
    D_FIELD_LOCATIONS,
    B_FIELD_LOCATIONS,
)
from PyPIC3D.relativity.field_interpolation import location_interpolate
from JAX_BSSN.bssn.geometry import W_FLOOR_VALUE
from .grid import CENTER_LOCATION


def build_yee_metric(bssn, grid):
    """Average conformal geometry, then derive its physical volume/inverse.

    Each C-to-V transfer averages two adjacent samples with equal weights.
    Apply one transfer per staggered axis to lapse, shift, W, and conformal
    metric before the determinant projection and physical-metric conversion.
    """
    # One extra temporary halo covers the upper neighbor of every retained
    # V node, including the outermost metric guard. Trim it after averaging.
    padded = replace(
        grid, static=grid.static._replace(guard_cells=grid.guard_cells + 1)
    )
    W = padded.to_tile(bssn.conformal_factor)
    lapse = padded.to_tile(bssn.lapse)
    shift = tuple(padded.to_tile(v) for v in bssn.shift)
    metric = tuple(
        tuple(padded.to_tile(v) for v in row) for row in bssn.conformal_metric
    )

    def sample(location):
        def transfer(a):
            sampled = location_interpolate(a, CENTER_LOCATION, location)
            return sampled[..., 1:-1, 1:-1, 1:-1]

        w = jnp.maximum(transfer(W), W_FLOOR_VALUE)
        conformal = jnp.stack(
            tuple(jnp.stack(tuple(transfer(v) for v in row)) for row in metric)
        )
        conformal = jnp.moveaxis(conformal, (0, 1), (-2, -1))
        conformal *= jnp.linalg.det(conformal)[..., None, None] ** (-1 / 3)
        inverse = jnp.linalg.inv(conformal) * w[..., None, None] ** 2
        physical = conformal / w[..., None, None] ** 2
        return Metric(
            transfer(lapse),
            jnp.stack(tuple(transfer(v) for v in shift), axis=-1),
            physical,
            inverse,
            w**-3,
        )

    return YeeMetric(
        tuple(sample(loc) for loc in D_FIELD_LOCATIONS),
        tuple(sample(loc) for loc in B_FIELD_LOCATIONS),
        sample(CENTER_LOCATION),
        sample(("V", "V", "V")),
    )
