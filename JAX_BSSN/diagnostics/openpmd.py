"""Synchronous openPMD output for global Cartesian mesh fields."""

from pathlib import Path

import jax
import numpy as np
import openpmd_api as io


def _as_3tuple(value):
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value, value, value)


def _ensure_openpmd_array(data):
    """Return host-owned floating-point data suitable for openPMD."""

    array = np.asarray(jax.device_get(data))
    if not np.issubdtype(array.dtype, np.floating):
        array = np.asarray(array, dtype=np.float64)
    if not array.flags.c_contiguous or not array.flags.writeable:
        array = np.array(array, copy=True, order="C")

    return array


def _strip_ghost_cells(field_map, ghost_cells):
    """Remove the configured outer ghost width from scalar and vector fields."""

    ghost_cells = tuple(int(width) for width in _as_3tuple(ghost_cells))
    interior = tuple(
        slice(width, -width) if width > 0 else slice(None)
        for width in ghost_cells
    )

    interior_fields = {}
    for name, field in field_map.items():
        is_vector = isinstance(field, (tuple, list)) and len(field) == 3
        if is_vector:
            interior_fields[name] = tuple(component[interior] for component in field)
        else:
            interior_fields[name] = field[interior]

    return interior_fields


def _configure_mesh(mesh, grid_spacing, grid_global_offset):
    mesh.geometry = io.Geometry.cartesian
    mesh.data_order = io.Data_Order.C if hasattr(io, "Data_Order") else "C"
    mesh.axis_labels = ["x", "y", "z"]
    mesh.grid_spacing = list(grid_spacing)
    mesh.grid_global_offset = list(grid_global_offset)
    mesh.grid_unit_SI = 1.0
    mesh.unit_SI = 1.0


def _write_scalar_mesh(
    iteration,
    name,
    data,
    grid_spacing,
    grid_global_offset,
    grid_position,
):
    mesh = iteration.meshes[name]
    _configure_mesh(mesh, grid_spacing, grid_global_offset)

    array = _ensure_openpmd_array(data)
    component = mesh[io.Mesh_Record_Component.SCALAR]
    component.position = list(grid_position)
    component.reset_dataset(io.Dataset(array.dtype, array.shape))
    component.store_chunk(array, [0, 0, 0], array.shape)
    component.unit_SI = 1.0


def _write_vector_mesh(
    iteration,
    name,
    components,
    grid_spacing,
    grid_global_offset,
    grid_position,
):
    mesh = iteration.meshes[name]
    _configure_mesh(mesh, grid_spacing, grid_global_offset)

    for component_name, data in zip(("x", "y", "z"), components):
        array = _ensure_openpmd_array(data)
        component = mesh[component_name]
        component.position = list(grid_position)
        component.reset_dataset(io.Dataset(array.dtype, array.shape))
        component.store_chunk(array, [0, 0, 0], array.shape)
        component.unit_SI = 1.0


class OpenPMDWriter:
    """Own one synchronous openPMD series for Cartesian mesh diagnostics."""

    def __init__(
        self,
        filename,
        grid_spacing,
        grid_global_offset,
        dt,
        ghost_cells=0,
        grid_position=(0.0, 0.0, 0.0),
    ):
        output_path = Path(filename)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".h5")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.filename = output_path
        self.grid_spacing = tuple(float(value) for value in _as_3tuple(grid_spacing))
        self.grid_global_offset = tuple(
            float(value) for value in _as_3tuple(grid_global_offset)
        )
        self.grid_position = tuple(
            float(value) for value in _as_3tuple(grid_position)
        )
        self.dt = float(dt)
        self.ghost_cells = ghost_cells

        self.series = io.Series(str(output_path), io.Access.create)
        self.series.set_attribute("software", "JAX_BSSN")

    def write(self, field_map, step, time):
        """Write one iteration after removing ghost cells from each field."""

        interior_fields = _strip_ghost_cells(field_map, self.ghost_cells)

        iteration = self.series.iterations[int(step)]
        iteration.time = float(time)
        iteration.dt = self.dt
        iteration.time_unit_SI = 1.0

        for name, field in interior_fields.items():
            is_vector = isinstance(field, (tuple, list)) and len(field) == 3
            if is_vector:
                _write_vector_mesh(
                    iteration,
                    name,
                    field,
                    self.grid_spacing,
                    self.grid_global_offset,
                    self.grid_position,
                )
            else:
                _write_scalar_mesh(
                    iteration,
                    name,
                    field,
                    self.grid_spacing,
                    self.grid_global_offset,
                    self.grid_position,
                )

        self.series.flush()
        iteration.close()

    def close(self):
        if self.series is not None:
            self.series.close()
            self.series = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
