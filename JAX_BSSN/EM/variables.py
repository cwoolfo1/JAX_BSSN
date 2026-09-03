"""State containers shared by the electromagnetic formulations."""

from typing import Generic, NamedTuple, TypeVar

from JAX_BSSN.bssn import BSSNVariables


EMState = TypeVar("EMState")


class EinsteinMaxwellVariables(NamedTuple, Generic[EMState]):
    """Coupled BSSN and electromagnetic state."""

    bssn: BSSNVariables
    em: EMState


__all__ = ["EinsteinMaxwellVariables"]
