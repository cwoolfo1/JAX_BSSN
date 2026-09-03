"""State containers for the densitized first-order Maxwell system."""

from typing import NamedTuple

import jax.numpy as jnp


class DensitizedMaxwellState(NamedTuple):
    """Doubled Entity leapfrog state centered on one integer time."""

    magnetic_previous: jnp.ndarray
    magnetic_current: jnp.ndarray
    displacement_left_half: jnp.ndarray
    displacement_right_half: jnp.ndarray


__all__ = ["DensitizedMaxwellState"]
