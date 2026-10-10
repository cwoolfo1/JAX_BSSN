"""PyPIC3D runtime state and a derived synchronized magnetic field."""

from typing import NamedTuple


class DensitizedMaxwellState(NamedTuple):
    """Full PyPIC3D nine-slot field tuple plus common-time output B.

    fields[0] is D at t, fields[1] is B at t-h/2 and fields[7] stores
    (D at t-h, B at t-3h/2), where h is half the coupled BSSN timestep.
    synchronized_magnetic is a diagnostic/source view at t, never history.
    half_dt records the history spacing; changing dt requires reinitializing.
    """

    fields: tuple
    synchronized_magnetic: tuple
    half_dt: object
