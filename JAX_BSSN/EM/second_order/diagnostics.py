"""Output helpers for Einstein--Maxwell states."""

from JAX_BSSN.EM.second_order.variables import EMVariables


def electromagnetic_output_fields(em: EMVariables) -> dict:
    """Return physical electric and magnetic vectors for mesh writers.

    The projected time derivatives are evolution auxiliaries and are
    intentionally not part of the output field map.
    """

    return {
        "E": tuple(em.electric_field[i] for i in range(3)),
        "B": tuple(em.magnetic_field[i] for i in range(3)),
    }


__all__ = ["electromagnetic_output_fields"]
