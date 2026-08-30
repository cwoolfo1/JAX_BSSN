"""State containers for the second-order Maxwell wave system."""

from typing import NamedTuple

import jax.numpy as jnp

from JAX_BSSN.bssn import BSSNVariables


class EMVariables(NamedTuple):
    """Electromagnetic covectors and their Eulerian projected derivatives."""

    electric_field: jnp.ndarray
    electric_field_dot: jnp.ndarray
    magnetic_field: jnp.ndarray
    magnetic_field_dot: jnp.ndarray


class BSSNEMGeometry(NamedTuple):
    """BSSN geometry needed by the curved Maxwell source equations."""

    metric: jnp.ndarray
    inverse_metric: jnp.ndarray
    traceless_extrinsic_curvature: jnp.ndarray
    extrinsic_curvature: jnp.ndarray
    christoffel: jnp.ndarray
    levi_civita: jnp.ndarray
    acceleration: jnp.ndarray
    acceleration_dot: jnp.ndarray
    normal_normal_ricci: jnp.ndarray
    spatial_ricci: jnp.ndarray
    electric_weyl: jnp.ndarray
    magnetic_weyl: jnp.ndarray


class EinsteinMaxwellVariables(NamedTuple):
    """Coupled BSSN and electromagnetic state."""

    bssn: BSSNVariables
    em: EMVariables
