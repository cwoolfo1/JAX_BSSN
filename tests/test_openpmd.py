import numpy as np
import openpmd_api as io
import pytest

from JAX_BSSN.diagnostics.openpmd import (
    OpenPMDWriter,
    _prepare_composite_levels,
    _repeat_vertex_centered,
)


def _scalar_levels(coarse, fine, fine_offset=(1.0, 1.0, 1.0), ghost_cells=0):
    return {
        "coarse": {
            "fields": {"chi": coarse},
            "grid_spacing": 1.0,
            "grid_global_offset": (0.0, 0.0, 0.0),
        },
        "fine": {
            "fields": {"chi": fine},
            "grid_spacing": 0.5,
            "grid_global_offset": fine_offset,
            "ghost_cells": ghost_cells,
        },
    }


def _read_meshes(filename):
    series = io.Series(str(filename), io.Access.read_only)
    iteration = series.iterations[0]
    meshes = {}

    for mesh_name in iteration.meshes:
        mesh = iteration.meshes[mesh_name]
        component_names = list(mesh)
        arrays = {}
        for component_name in component_names:
            arrays[component_name] = mesh[component_name].load_chunk()

        series.flush()
        meshes[mesh_name] = {
            "grid_spacing": tuple(mesh.grid_spacing),
            "grid_global_offset": tuple(mesh.grid_global_offset),
            "arrays": {
                name: np.array(array, copy=True)
                for name, array in arrays.items()
            },
        }

    iteration.close()
    series.close()
    return meshes


def test_vertex_centered_replication_in_one_dimension():
    np.testing.assert_array_equal(
        _repeat_vertex_centered(np.array([10, 20, 30]), 2),
        [10, 10, 20, 20, 30],
    )
    np.testing.assert_array_equal(
        _repeat_vertex_centered(np.array([10, 20]), 4),
        [10, 10, 10, 10, 20],
    )


def test_vertex_centered_replication_in_three_dimensions():
    coarse = np.arange(8).reshape(2, 2, 2)
    expanded = _repeat_vertex_centered(coarse, (2, 2, 2))

    assert expanded.shape == (3, 3, 3)
    expected = np.empty((3, 3, 3), dtype=coarse.dtype)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                expected[i, j, k] = coarse[min(i // 2, 1),
                                           min(j // 2, 1),
                                           min(k // 2, 1)]
    np.testing.assert_array_equal(expanded, expected)


def test_composite_scalar_mesh_uses_finest_spacing_and_fine_overwrite(tmp_path):
    coarse = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    fine = 100.0 + np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    filename = tmp_path / "scalar_composite.h5"
    levels = _scalar_levels(coarse, fine)
    # Resolution, rather than mapping order, determines overwrite order.
    levels = {"fine": levels["fine"], "coarse": levels["coarse"]}

    with OpenPMDWriter(filename, 1.0, 0.0, dt=0.1) as writer:
        writer.write_levels(levels, step=0, time=0.0)

    meshes = _read_meshes(filename)
    assert set(meshes) == {"chi"}
    assert meshes["chi"]["grid_spacing"] == (0.5, 0.5, 0.5)
    assert meshes["chi"]["grid_global_offset"] == (0.0, 0.0, 0.0)

    data = meshes["chi"]["arrays"][io.Mesh_Record_Component.SCALAR]
    expected = _repeat_vertex_centered(coarse, (2, 2, 2))
    expected[2:5, 2:5, 2:5] = fine
    assert data.shape == (5, 5, 5)
    np.testing.assert_array_equal(data, expected)


def test_composite_vector_components_share_geometry_and_overwrite(tmp_path):
    coarse_base = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    fine_base = 100.0 + np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    coarse = tuple(coarse_base + 10.0 * component for component in range(3))
    fine = tuple(fine_base + 10.0 * component for component in range(3))
    levels = {
        "coarse": {
            "fields": {"beta": coarse},
            "grid_spacing": (1.0, 1.0, 1.0),
            "grid_global_offset": (0.0, 0.0, 0.0),
        },
        "fine": {
            "fields": {"beta": fine},
            "grid_spacing": (0.5, 0.5, 0.5),
            "grid_global_offset": (1.0, 1.0, 1.0),
        },
    }
    filename = tmp_path / "vector_composite.h5"

    with OpenPMDWriter(filename, 1.0, 0.0, dt=0.1) as writer:
        writer.write_levels(levels, step=0, time=0.0)

    mesh = _read_meshes(filename)["beta"]
    assert mesh["grid_spacing"] == (0.5, 0.5, 0.5)
    assert set(mesh["arrays"]) == {"x", "y", "z"}
    for component, name in enumerate(("x", "y", "z")):
        expected = _repeat_vertex_centered(coarse[component], (2, 2, 2))
        expected[2:5, 2:5, 2:5] = fine[component]
        assert mesh["arrays"][name].shape == (5, 5, 5)
        np.testing.assert_array_equal(mesh["arrays"][name], expected)


def test_composite_hierarchy_rejects_noninteger_spacing_ratio():
    field = np.zeros((3, 3, 3))
    levels = _scalar_levels(field, field)
    levels["fine"]["grid_spacing"] = 0.6

    with pytest.raises(ValueError, match="grid_spacing ratio"):
        _prepare_composite_levels(levels)


def test_composite_hierarchy_rejects_misaligned_physical_offset():
    field = np.zeros((3, 3, 3))
    levels = _scalar_levels(field, field, fine_offset=(1.25, 1.0, 1.0))

    with pytest.raises(ValueError, match="grid_global_offset"):
        _prepare_composite_levels(levels)


def test_composite_ghost_cells_are_removed_before_geometry_and_write(tmp_path):
    coarse = np.ones((3, 3, 3), dtype=np.float64)
    padded_fine = np.full((5, 5, 5), -99.0)
    padded_fine[1:-1, 1:-1, 1:-1] = 7.0
    filename = tmp_path / "ghost_composite.h5"

    with OpenPMDWriter(filename, 1.0, 0.0, dt=0.1) as writer:
        writer.write_levels(
            _scalar_levels(coarse, padded_fine, ghost_cells=1),
            step=0,
            time=0.0,
        )

    data = _read_meshes(filename)["chi"]["arrays"][io.Mesh_Record_Component.SCALAR]
    assert data.shape == (5, 5, 5)
    np.testing.assert_array_equal(data[2:5, 2:5, 2:5], 7.0)
    assert not np.any(data == -99.0)


def test_composite_levels_require_matching_field_sets_and_kinds():
    field = np.zeros((3, 3, 3))
    levels = _scalar_levels(field, field)
    levels["fine"]["fields"] = {"alpha": field}
    with pytest.raises(ValueError, match="incompatible field set"):
        _prepare_composite_levels(levels)

    levels = _scalar_levels(field, field)
    levels["fine"]["fields"]["chi"] = (field, field, field)
    with pytest.raises(ValueError, match="changes from scalar to vector"):
        _prepare_composite_levels(levels)


def test_uniform_write_keeps_existing_single_grid_behavior(tmp_path):
    field = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    filename = tmp_path / "uniform.h5"

    with OpenPMDWriter(
        filename,
        grid_spacing=(0.5, 0.25, 0.125),
        grid_global_offset=(-1.0, -2.0, -3.0),
        dt=0.1,
    ) as writer:
        writer.write({"chi": field}, step=0, time=0.0)

    mesh = _read_meshes(filename)["chi"]
    assert mesh["grid_spacing"] == (0.5, 0.25, 0.125)
    assert mesh["grid_global_offset"] == (-1.0, -2.0, -3.0)
    np.testing.assert_array_equal(
        mesh["arrays"][io.Mesh_Record_Component.SCALAR],
        field,
    )
