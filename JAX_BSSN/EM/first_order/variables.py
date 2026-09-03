"""State containers for the densitized first-order Maxwell system."""

from typing import NamedTuple

import jax.numpy as jnp

from JAX_BSSN.bssn import BSSNVariables


class DensitizedMaxwellState(NamedTuple):
    """Doubled Entity leapfrog state centered on one integer time."""

    magnetic_previous: jnp.ndarray
    magnetic_current: jnp.ndarray
    displacement_left_half: jnp.ndarray
    displacement_right_half: jnp.ndarray


class FirstOrderEinsteinMaxwellState(NamedTuple):
    """BSSN variables and a synchronized densitized Maxwell state."""

    bssn: BSSNVariables
    em: DensitizedMaxwellState


class ConformalMetricFields(NamedTuple):
    """BSSN metric quantities evaluated at one logical Yee location."""

    W: jnp.ndarray
    lapse: jnp.ndarray
    shift: jnp.ndarray
    conformal_metric: jnp.ndarray
    inverse_conformal_metric: jnp.ndarray


class BSSNYeeGeometry(NamedTuple):
    """Projected BSSN geometry on the six Maxwell sites and at cell centers."""

    displacement: tuple[ConformalMetricFields, ...]
    magnetic: tuple[ConformalMetricFields, ...]
    center: ConformalMetricFields


__all__ = [
    "BSSNYeeGeometry",
    "ConformalMetricFields",
    "DensitizedMaxwellState",
    "FirstOrderEinsteinMaxwellState",
]
