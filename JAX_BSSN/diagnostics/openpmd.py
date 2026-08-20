"""Synchronous openPMD output for Cartesian mesh fields."""

from pathlib import Path

import jax
import numpy as np
import openpmd_api as io


_ALIGNMENT_RTOL = 1.0e-9
_ALIGNMENT_ATOL = 1.0e-10


def _as_3tuple(value):
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value, value, value)


def _float_3tuple(value, name):
    values = tuple(float(item) for item in _as_3tuple(value))
    if len(values) != 3:
        raise ValueError(f"{name} must be a scalar or a 3-tuple")
    return values


def _integer_aligned(value, description):
    nearest = int(np.rint(value))
    if not np.isclose(
        value,
        nearest,
        rtol=_ALIGNMENT_RTOL,
        atol=_ALIGNMENT_ATOL,
    ):
        raise ValueError(
            f"{description} must map to an integer finest-grid index; "
            f"got {value:.16g}"
        )
    return nearest


def _integer_refinement_ratio(level_spacing, finest_spacing):
    """Return the integer level-to-finest spacing ratio along each axis."""

    level_spacing = _float_3tuple(level_spacing, "grid_spacing")
    finest_spacing = _float_3tuple(finest_spacing, "finest grid_spacing")
    if any(spacing <= 0.0 for spacing in level_spacing + finest_spacing):
        raise ValueError("grid_spacing values must be positive")

    return tuple(
        _integer_aligned(
            level / finest,
            f"grid_spacing ratio on axis {axis}",
        )
        for axis, (level, finest) in enumerate(
            zip(level_spacing, finest_spacing)
        )
    )


def _repeat_vertex_centered(array, ratio):
    """Repeat vertex values piecewise-constantly at a finer index spacing."""

    expanded = np.asarray(array)
    if isinstance(ratio, (tuple, list)):
        ratio = tuple(int(value) for value in ratio)
    else:
        ratio = (int(ratio),) * expanded.ndim
    if len(ratio) != expanded.ndim or any(value < 1 for value in ratio):
        raise ValueError(
            "refinement ratio must contain one positive integer per array axis"
        )

    for axis, refinement in enumerate(ratio):
        if refinement == 1:
            continue

        output_size = (expanded.shape[axis] - 1) * refinement + 1
        expanded = np.repeat(expanded, refinement, axis=axis)
        active = [slice(None)] * expanded.ndim
        active[axis] = slice(output_size)
        expanded = expanded[tuple(active)]

    return np.ascontiguousarray(expanded)


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
    if len(ghost_cells) != 3 or any(width < 0 for width in ghost_cells):
        raise ValueError("ghost_cells must be a non-negative scalar or 3-tuple")
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


def _field_kind(field):
    is_vector = isinstance(field, (tuple, list)) and len(field) == 3
    return "vector" if is_vector else "scalar"


def _active_field_shape(field, field_name, level_name):
    components = field if _field_kind(field) == "vector" else (field,)
    shapes = [tuple(component.shape) for component in components]
    if any(len(shape) != 3 for shape in shapes):
        raise ValueError(
            f"field {field_name!r} on level {level_name!r} must have "
            "three-dimensional components"
        )
    if any(shape != shapes[0] for shape in shapes[1:]):
        raise ValueError(
            f"vector field {field_name!r} on level {level_name!r} has "
            "components with different active shapes"
        )
    if any(size < 1 for size in shapes[0]):
        raise ValueError(
            f"ghost stripping leaves field {field_name!r} on level "
            f"{level_name!r} with an empty axis"
        )
    return shapes[0]


def _prepare_composite_levels(levels):
    """Validate level geometry and express it in the finest index space."""

    if not levels:
        raise ValueError("write_levels requires at least one refinement level")

    prepared = []
    reference_fields = None
    field_kinds = None
    reference_grid_position = None

    for input_order, (level_name, level) in enumerate(levels.items()):
        spacing = _float_3tuple(level["grid_spacing"], "grid_spacing")
        offset = _float_3tuple(
            level["grid_global_offset"],
            "grid_global_offset",
        )
        grid_position = _float_3tuple(
            level.get("grid_position", (0.0, 0.0, 0.0)),
            "grid_position",
        )
        fields = _strip_ghost_cells(
            level["fields"],
            level.get("ghost_cells", 0),
        )

        names = set(fields)
        kinds = {name: _field_kind(field) for name, field in fields.items()}
        if reference_fields is None:
            reference_fields = names
            field_kinds = kinds
            reference_grid_position = grid_position
        else:
            if names != reference_fields:
                missing = sorted(reference_fields - names)
                extra = sorted(names - reference_fields)
                raise ValueError(
                    f"level {level_name!r} has an incompatible field set; "
                    f"missing={missing}, extra={extra}"
                )
            for field_name in reference_fields:
                if kinds[field_name] != field_kinds[field_name]:
                    raise ValueError(
                        f"field {field_name!r} changes from "
                        f"{field_kinds[field_name]} to {kinds[field_name]} "
                        f"on level {level_name!r}"
                    )
            if not np.allclose(
                grid_position,
                reference_grid_position,
                rtol=_ALIGNMENT_RTOL,
                atol=_ALIGNMENT_ATOL,
            ):
                raise ValueError(
                    "all composite levels must use the same grid_position"
                )

        active_shape = None
        for field_name, field in fields.items():
            field_shape = _active_field_shape(field, field_name, level_name)
            if active_shape is None:
                active_shape = field_shape
            elif field_shape != active_shape:
                raise ValueError(
                    f"fields on level {level_name!r} do not share one "
                    "active mesh shape"
                )

        if active_shape is None:
            raise ValueError(f"level {level_name!r} contains no fields")

        prepared.append(
            {
                "name": level_name,
                "input_order": input_order,
                "fields": fields,
                "spacing": spacing,
                "offset": offset,
                "grid_position": grid_position,
                "active_shape": active_shape,
            }
        )

    finest_spacing = tuple(
        min(level["spacing"][axis] for level in prepared)
        for axis in range(3)
    )
    for level in prepared:
        level["ratio"] = _integer_refinement_ratio(
            level["spacing"],
            finest_spacing,
        )

    # A hierarchy must have a consistent coarse-to-fine ordering on all axes.
    prepared.sort(
        key=lambda level: (-np.prod(level["ratio"]), level["input_order"])
    )
    for coarse, fine in zip(prepared, prepared[1:]):
        if any(
            coarse_spacing < fine_spacing
            and not np.isclose(
                coarse_spacing,
                fine_spacing,
                rtol=_ALIGNMENT_RTOL,
                atol=_ALIGNMENT_ATOL,
            )
            for coarse_spacing, fine_spacing in zip(
                coarse["spacing"], fine["spacing"]
            )
        ):
            raise ValueError(
                "refinement levels do not have a consistent "
                "coarse-to-fine spacing order"
            )

    global_offset = tuple(
        min(level["offset"][axis] for level in prepared)
        for axis in range(3)
    )
    for level in prepared:
        level["chunk_offset"] = tuple(
            _integer_aligned(
                (offset - origin) / spacing,
                f"grid_global_offset for level {level['name']!r} "
                f"on axis {axis}",
            )
            for axis, (offset, origin, spacing) in enumerate(
                zip(level["offset"], global_offset, finest_spacing)
            )
        )

    global_upper = tuple(
        max(
            level["offset"][axis]
            + (level["active_shape"][axis] - 1) * level["spacing"][axis]
            for level in prepared
        )
        for axis in range(3)
    )
    global_shape = tuple(
        _integer_aligned(
            (upper - origin) / spacing,
            f"global physical extent on axis {axis}",
        )
        + 1
        for axis, (origin, upper, spacing) in enumerate(
            zip(global_offset, global_upper, finest_spacing)
        )
    )

    for level in prepared:
        level["expanded_shape"] = tuple(
            (size - 1) * ratio + 1
            for size, ratio in zip(level["active_shape"], level["ratio"])
        )
        if any(
            offset < 0 or offset + extent > domain
            for offset, extent, domain in zip(
                level["chunk_offset"],
                level["expanded_shape"],
                global_shape,
            )
        ):
            raise ValueError(
                f"expanded chunk for level {level['name']!r} lies outside "
                "the global finest-grid dataset"
            )

    return (
        prepared,
        finest_spacing,
        global_offset,
        global_shape,
        reference_grid_position,
        field_kinds,
    )


def _configure_mesh(
    mesh,
    grid_spacing,
    grid_global_offset,
    refinement_level=None,
    field_name=None,
):
    mesh.geometry = io.Geometry.cartesian
    mesh.data_order = io.Data_Order.C if hasattr(io, "Data_Order") else "C"
    mesh.axis_labels = ["x", "y", "z"]
    mesh.grid_spacing = list(grid_spacing)
    mesh.grid_global_offset = list(grid_global_offset)
    mesh.grid_unit_SI = 1.0
    mesh.unit_SI = 1.0
    if refinement_level is not None:
        mesh.set_attribute("refinementLevel", str(refinement_level))
        mesh.set_attribute("fieldName", str(field_name))


def _write_scalar_mesh(
    iteration,
    name,
    data,
    grid_spacing,
    grid_global_offset,
    grid_position,
    refinement_level=None,
    field_name=None,
):
    mesh = iteration.meshes[name]
    _configure_mesh(
        mesh,
        grid_spacing,
        grid_global_offset,
        refinement_level,
        field_name,
    )

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
    refinement_level=None,
    field_name=None,
):
    mesh = iteration.meshes[name]
    _configure_mesh(
        mesh,
        grid_spacing,
        grid_global_offset,
        refinement_level,
        field_name,
    )

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
        self.grid_spacing = _float_3tuple(grid_spacing, "grid_spacing")
        self.grid_global_offset = _float_3tuple(
            grid_global_offset,
            "grid_global_offset",
        )
        self.grid_position = _float_3tuple(grid_position, "grid_position")
        self.dt = float(dt)
        self.ghost_cells = ghost_cells

        self.series = io.Series(str(output_path), io.Access.create)
        self.series.set_attribute("software", "JAX_BSSN")

    def _write_field_map(
        self,
        iteration,
        field_map,
        grid_spacing,
        grid_global_offset,
        grid_position,
        ghost_cells,
        prefix="",
        refinement_level=None,
    ):
        interior_fields = _strip_ghost_cells(field_map, ghost_cells)

        for field_name, field in interior_fields.items():
            name = f"{prefix}{field_name}"
            is_vector = isinstance(field, (tuple, list)) and len(field) == 3
            if is_vector:
                _write_vector_mesh(
                    iteration,
                    name,
                    field,
                    grid_spacing,
                    grid_global_offset,
                    grid_position,
                    refinement_level,
                    field_name,
                )
            else:
                _write_scalar_mesh(
                    iteration,
                    name,
                    field,
                    grid_spacing,
                    grid_global_offset,
                    grid_position,
                    refinement_level,
                    field_name,
                )

    def _iteration(self, step, time):
        iteration = self.series.iterations[int(step)]
        iteration.time = float(time)
        iteration.dt = self.dt
        iteration.time_unit_SI = 1.0
        return iteration

    def write(self, field_map, step, time):
        """Write one iteration after removing ghost cells from each field."""

        iteration = self._iteration(step, time)
        self._write_field_map(
            iteration,
            field_map,
            self.grid_spacing,
            self.grid_global_offset,
            self.grid_position,
            self.ghost_cells,
        )

        self.series.flush()
        iteration.close()

    def _write_separate_levels(self, iteration, levels):
        for level_name, level in levels.items():
            grid_spacing = _float_3tuple(level["grid_spacing"], "grid_spacing")
            grid_global_offset = _float_3tuple(
                level["grid_global_offset"],
                "grid_global_offset",
            )
            grid_position = _float_3tuple(
                level.get("grid_position", (0.0, 0.0, 0.0)),
                "grid_position",
            )

            self._write_field_map(
                iteration,
                level["fields"],
                grid_spacing,
                grid_global_offset,
                grid_position,
                level.get("ghost_cells", 0),
                prefix=f"{level_name}_",
                refinement_level=level_name,
            )

    def _write_composite_field(
        self,
        iteration,
        field_name,
        field_kind,
        levels,
        finest_spacing,
        global_offset,
        global_shape,
        grid_position,
    ):
        mesh = iteration.meshes[field_name]
        _configure_mesh(
            mesh,
            finest_spacing,
            global_offset,
        )

        component_names = (
            ("x", "y", "z")
            if field_kind == "vector"
            else (io.Mesh_Record_Component.SCALAR,)
        )
        pending_arrays = []

        for component_index, component_name in enumerate(component_names):
            component_data = []
            for level in levels:
                field = level["fields"][field_name]
                data = field[component_index] if field_kind == "vector" else field
                component_data.append(data)

            dtypes = []
            for data in component_data:
                dtype = np.dtype(data.dtype)
                dtypes.append(
                    dtype if np.issubdtype(dtype, np.floating) else np.float64
                )
            dataset_dtype = np.result_type(*dtypes)

            component = mesh[component_name]
            component.position = list(grid_position)
            component.reset_dataset(io.Dataset(dataset_dtype, global_shape))
            component.unit_SI = 1.0

            for level, data in zip(levels, component_data):
                array = _ensure_openpmd_array(data).astype(
                    dataset_dtype,
                    copy=False,
                )
                array = _repeat_vertex_centered(array, level["ratio"])
                if array.shape != level["expanded_shape"]:
                    raise ValueError(
                        f"expanded field {field_name!r} on level "
                        f"{level['name']!r} has shape {array.shape}, expected "
                        f"{level['expanded_shape']}"
                    )

                component.store_chunk(
                    array,
                    list(level["chunk_offset"]),
                    array.shape,
                )
                pending_arrays.append(array)

        # openPMD writes are deferred. Keep every chunk buffer alive through
        # this flush, while limiting temporary storage to one physical field.
        self.series.flush()

    def write_levels(self, levels, step, time, composite=True):
        """Write named Cartesian refinement levels into one iteration.

        By default, all levels are composed into one finest-resolution mesh per
        physical field. Fields and scalar/vector kinds must match on every
        level. Active data are obtained by stripping each level's ghost cells,
        coarse vertices are repeated piecewise-constantly, and finer chunks
        overwrite coarser chunks. Set ``composite=False`` to retain the legacy
        level-prefixed mesh records.
        """

        if not composite:
            iteration = self._iteration(step, time)
            self._write_separate_levels(iteration, levels)
            self.series.flush()
            iteration.close()
            return

        (
            prepared_levels,
            finest_spacing,
            global_offset,
            global_shape,
            grid_position,
            field_kinds,
        ) = _prepare_composite_levels(levels)

        iteration = self._iteration(step, time)
        first_level = next(iter(levels.values()))
        for field_name in first_level["fields"]:
            self._write_composite_field(
                iteration,
                field_name,
                field_kinds[field_name],
                prepared_levels,
                finest_spacing,
                global_offset,
                global_shape,
                grid_position,
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
