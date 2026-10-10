import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import openpmd_api as io

from JAX_BSSN.diagnostics.openpmd import OpenPMDWriter
from JAX_BSSN.bssn import BSSNParameters
from JAX_BSSN.EM.first_order import (
    initialize_densitized_maxwell_state,
    make_em_grid,
    EinsteinMaxwellVariables,
    electromagnetic_output_fields,
)
from tests.EM.em_helpers import flat_bssn_variables


def test_openpmd_writes_common_time_physical_cartesian_fields(tmp_path):
    shape = (8, 6, 9)
    W = 0.8
    bssn = flat_bssn_variables(shape)._replace(conformal_factor=jnp.full(shape, W))
    displacement = jnp.broadcast_to(
        jnp.array([1.0, 2.0, 3.0])[:, None, None, None], (3,) + shape
    )
    magnetic = -2.0 * displacement
    params = BSSNParameters(dx=0.25)
    grid = make_em_grid(shape, params)
    em = initialize_densitized_maxwell_state(displacement, magnetic, bssn, params, grid)
    fields = electromagnetic_output_fields(EinsteinMaxwellVariables(bssn, em), grid)

    dx = 0.25
    z_min = -1.0
    filename = tmp_path / "einstein_maxwell.h5"
    with OpenPMDWriter(
        filename,
        grid_spacing=(dx, dx, dx),
        grid_global_offset=(
            -1.0,
            0.0,
            z_min,
        ),
        grid_position=(0.0, 0.0, 0.0),
        dt=0.02,
        ghost_cells=0,
    ) as writer:
        writer.write(fields, step=3, time=0.06)

    series = io.Series(str(filename), io.Access.read_only)
    iteration = series.iterations[3]
    pending = {
        mesh_name: {
            component_name: iteration.meshes[mesh_name][component_name].load_chunk()
            for component_name in iteration.meshes[mesh_name]
        }
        for mesh_name in iteration.meshes
    }
    metadata = {
        mesh_name: (
            tuple(iteration.meshes[mesh_name].grid_spacing),
            tuple(iteration.meshes[mesh_name].grid_global_offset),
        )
        for mesh_name in iteration.meshes
    }
    time = float(iteration.time)
    series.flush()
    arrays = {
        mesh_name: {
            component_name: np.array(array, copy=True)
            for component_name, array in components.items()
        }
        for mesh_name, components in pending.items()
    }
    iteration.close()
    series.close()

    assert set(arrays) == {"D", "B"}
    assert set(arrays["D"]) == {"x", "y", "z"}
    assert set(arrays["B"]) == {"x", "y", "z"}
    assert time == 0.06

    expected_offset = (-1.0, 0.0, z_min)
    for mesh_name in ("D", "B"):
        assert metadata[mesh_name] == ((dx, dx, dx), expected_offset)
        for array in arrays[mesh_name].values():
            assert array.shape == shape

    for field_name, expected in (("D", W**3 * displacement), ("B", W**3 * magnetic)):
        for component, component_name in enumerate(("x", "y", "z")):
            np.testing.assert_allclose(
                arrays[field_name][component_name], expected[component]
            )
