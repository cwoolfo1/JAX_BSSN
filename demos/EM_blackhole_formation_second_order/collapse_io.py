"""Local checkpoint and summary helpers for the EM formation demo."""

import json
import os
from pathlib import Path
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables
from JAX_BSSN.EM.first_order.variables import DensitizedMaxwellState
from JAX_BSSN.EM.second_order.variables import EMVariables
from JAX_BSSN.EM.variables import EinsteinMaxwellVariables


def _json_value(value):
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def parameters_to_dict(params: BSSNParameters):
    return {name: _json_value(value) for name, value in params._asdict().items()}


def atomic_write_json(path, data):
    """Replace a JSON file only after its complete new contents are durable."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        json.dump(data, temporary_file, indent=2, sort_keys=True)
        temporary_file.write("\n")
        temporary_path = Path(temporary_file.name)
    os.replace(temporary_path, path)


def write_collapse_checkpoint(
    path,
    state,
    u,
    residual_history,
    formulation,
    step,
    time,
    params,
    configuration,
):
    """Atomically write the complete formulation-specific evolution state."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        f"bssn_{name}": np.asarray(jax.device_get(field))
        for name, field in zip(BSSNVariables._fields, state.bssn)
    }
    arrays.update(
        {
            f"em_{name}": np.asarray(jax.device_get(field))
            for name, field in zip(state.em._fields, state.em)
        }
    )
    arrays["u"] = np.asarray(jax.device_get(u))
    arrays["residual_history"] = np.asarray(residual_history, dtype=float)
    metadata = {
        "format_version": 1,
        "formulation": formulation,
        "step": int(step),
        "time": float(time),
        "parameters": parameters_to_dict(params),
        "configuration": configuration,
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))

    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        np.savez(temporary_file, **arrays)
        temporary_path = Path(temporary_file.name)
    os.replace(temporary_path, path)


def load_collapse_checkpoint(path, expected_formulation=None):
    """Load a checkpoint written by :func:`write_collapse_checkpoint`."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as checkpoint:
        metadata = json.loads(str(checkpoint["metadata_json"]))
        formulation = metadata["formulation"]
        if expected_formulation is not None and formulation != expected_formulation:
            raise ValueError(
                f"checkpoint formulation is {formulation!r}, not "
                f"{expected_formulation!r}"
            )

        bssn = BSSNVariables(
            *(jnp.asarray(checkpoint[f"bssn_{name}"]) for name in BSSNVariables._fields)
        )
        if formulation == "first_order":
            em_type = DensitizedMaxwellState
        elif formulation == "second_order":
            em_type = EMVariables
        else:
            raise ValueError(f"unsupported checkpoint formulation {formulation!r}")
        em = em_type(
            *(jnp.asarray(checkpoint[f"em_{name}"]) for name in em_type._fields)
        )
        state = EinsteinMaxwellVariables(bssn=bssn, em=em)
        u = jnp.asarray(checkpoint["u"])
        residual_history = checkpoint["residual_history"].astype(float).tolist()

    return state, u, residual_history, metadata


def state_is_finite(state):
    return all(
        np.all(np.isfinite(np.asarray(jax.device_get(field))))
        for field in jax.tree_util.tree_leaves(state)
    )


__all__ = [
    "parameters_to_dict",
    "atomic_write_json",
    "write_collapse_checkpoint",
    "load_collapse_checkpoint",
    "state_is_finite",
]
