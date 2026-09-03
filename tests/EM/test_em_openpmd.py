import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import openpmd_api as io

from JAX_BSSN.diagnostics.openpmd import OpenPMDWriter
from JAX_BSSN.EM.second_order.cartoon.axisymmetry import (
    compact_axisymmetric_wave,
    expand_axisymmetric_wave_plane,
)
from JAX_BSSN.EM.second_order.diagnostics import electromagnetic_output_fields
from JAX_BSSN.EM.second_order.variables import EMVariables


def test_openpmd_writes_only_E_and_B_with_axisymmetric_vector_parity(tmp_path):
    num_radial_points = 6
    num_z_points = 9
    full_nx = 2 * num_radial_points
    shape = (3, full_nx, 1, num_z_points)
    radial = jnp.arange(num_radial_points, dtype=jnp.float64)[:, None] + 1.0
    z_factor = jnp.arange(num_z_points, dtype=jnp.float64)[None, :] + 1.0
    profile = radial * z_factor

    electric = jnp.zeros(shape, dtype=jnp.float64)
    magnetic = jnp.zeros(shape, dtype=jnp.float64)
    for component in range(3):
        electric = electric.at[
            component, num_radial_points:, 0, :
        ].set((component + 1.0) * profile)
        magnetic = magnetic.at[
            component, num_radial_points:, 0, :
        ].set(-(component + 4.0) * profile)

    full_em = EMVariables(
        electric_field=electric,
        electric_field_dot=7.0 * electric,
        magnetic_field=magnetic,
        magnetic_field_dot=11.0 * magnetic,
    )
    compact_em = compact_axisymmetric_wave(full_em)
    expanded_em = expand_axisymmetric_wave_plane(compact_em)
    fields = electromagnetic_output_fields(expanded_em)

    dx = 0.25
    z_min = -1.0
    filename = tmp_path / "einstein_maxwell.h5"
    with OpenPMDWriter(
        filename,
        grid_spacing=(dx, dx, dx),
        grid_global_offset=(
            -(num_radial_points - 0.5) * dx,
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

    assert set(arrays) == {"E", "B"}
    assert set(arrays["E"]) == {"x", "y", "z"}
    assert set(arrays["B"]) == {"x", "y", "z"}
    assert time == 0.06

    expected_offset = (-(num_radial_points - 0.5) * dx, 0.0, z_min)
    for mesh_name in ("E", "B"):
        assert metadata[mesh_name] == ((dx, dx, dx), expected_offset)
        for array in arrays[mesh_name].values():
            assert array.shape == (full_nx, 1, num_z_points)

    parity = (-1.0, -1.0, 1.0)
    for field_name, positive_components in (
        ("E", electric[:, num_radial_points:, 0, :]),
        ("B", magnetic[:, num_radial_points:, 0, :]),
    ):
        for component, component_name in enumerate(("x", "y", "z")):
            output = arrays[field_name][component_name][:, 0, :]
            positive = np.asarray(positive_components[component])
            np.testing.assert_array_equal(output[num_radial_points:], positive)
            np.testing.assert_array_equal(
                output[:num_radial_points], parity[component] * positive[::-1]
            )
