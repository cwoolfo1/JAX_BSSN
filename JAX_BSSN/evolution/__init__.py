"""Spatial discretization, boundaries, and time integration."""

__all__ = ["compute_bssn_rhs", "compute_bssn_rhs_with_matter", "rk4_step"]


def __getattr__(name):
    """Load time integrators lazily to keep derivative imports acyclic."""

    if name in __all__:
        from JAX_BSSN.evolution import time_evolve

        return getattr(time_evolve, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
