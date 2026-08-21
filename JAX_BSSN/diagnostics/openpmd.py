"""Synchronous openPMD output for Cartesian mesh fields."""

from collections.abc import Mapping
import os
from pathlib import Path
import re
import tempfile

import jax
import numpy as np
import openpmd_api as io


_ALIGNMENT_RTOL = 1.0e-9
_ALIGNMENT_ATOL = 1.0e-10
_VECTOR_COMPONENTS = ("x", "y", "z")


def _as_3tuple(value):
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value, value, value)


def _float_3tuple(value, name):
    values = tuple(float(item) for item in _as_3tuple(value))
    if len(values) != 3:
        raise ValueError(f"{name} must be a scalar or a 3-tuple")
    return values


def _integer_3tuple(value, name):
    values = tuple(int(item) for item in _as_3tuple(value))
    if len(values) != 3:
        raise ValueError(f"{name} must be a scalar or a 3-tuple")
    return values


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

    ghost_cells = _integer_3tuple(ghost_cells, "ghost_cells")
    if any(width < 0 for width in ghost_cells):
        raise ValueError("ghost_cells must be a non-negative scalar or 3-tuple")
    interior = tuple(
        slice(width, -width) if width > 0 else slice(None)
        for width in ghost_cells
    )

    interior_fields = {}
    for name, field in field_map.items():
        if _field_kind(field) == "vector":
            interior_fields[name] = tuple(
                component[interior] for component in field
            )
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
            f"field {field_name!r} on {level_name} must have "
            "three-dimensional components"
        )
    if any(shape != shapes[0] for shape in shapes[1:]):
        raise ValueError(
            f"vector field {field_name!r} on {level_name} has components "
            "with different active shapes"
        )
    if any(size < 1 for size in shapes[0]):
        raise ValueError(
            f"ghost stripping leaves field {field_name!r} on {level_name} "
            "with an empty axis"
        )
    return shapes[0]


def _validate_field_maps(root_fields, fine_fields):
    if not root_fields or not fine_fields:
        raise ValueError("root and fine patches must both contain fields")

    root_names = set(root_fields)
    fine_names = set(fine_fields)
    if root_names != fine_names:
        missing = sorted(root_names - fine_names)
        extra = sorted(fine_names - root_names)
        raise ValueError(
            "root and fine patches have incompatible field sets; "
            f"missing={missing}, extra={extra}"
        )

    patch_shapes = []
    for level_name, fields in (("root patch", root_fields),
                               ("fine patch", fine_fields)):
        active_shape = None
        for field_name, field in fields.items():
            shape = _active_field_shape(field, field_name, level_name)
            if active_shape is None:
                active_shape = shape
            elif shape != active_shape:
                raise ValueError(
                    f"fields on {level_name} do not share one active mesh shape"
                )
        patch_shapes.append(active_shape)

    for name in root_names:
        root_kind = _field_kind(root_fields[name])
        fine_kind = _field_kind(fine_fields[name])
        if root_kind != fine_kind:
            raise ValueError(
                f"field {name!r} changes from {root_kind} on the root patch "
                f"to {fine_kind} on the fine patch"
            )

    return tuple(patch_shapes)


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
    return array


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

    arrays = []
    for component_name, data in zip(_VECTOR_COMPONENTS, components):
        array = _ensure_openpmd_array(data)
        component = mesh[component_name]
        component.position = list(grid_position)
        component.reset_dataset(io.Dataset(array.dtype, array.shape))
        component.store_chunk(array, [0, 0, 0], array.shape)
        component.unit_SI = 1.0
        arrays.append(array)
    return arrays


def _write_field_map(
    series,
    iteration,
    field_map,
    grid_spacing,
    grid_global_offset,
    grid_position,
):
    # openPMD stores chunks asynchronously. Retain every host buffer until the
    # flush that makes this operation synchronous.
    pending_arrays = []
    for field_name, field in field_map.items():
        if _field_kind(field) == "vector":
            pending_arrays.extend(
                _write_vector_mesh(
                    iteration,
                    field_name,
                    field,
                    grid_spacing,
                    grid_global_offset,
                    grid_position,
                )
            )
        else:
            pending_arrays.append(
                _write_scalar_mesh(
                    iteration,
                    field_name,
                    field,
                    grid_spacing,
                    grid_global_offset,
                    grid_position,
                )
            )
    series.flush()


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

    def _iteration(self, step, time):
        iteration = self.series.iterations[int(step)]
        iteration.time = float(time)
        iteration.dt = self.dt
        iteration.time_unit_SI = 1.0
        return iteration

    def write(self, field_map, step, time):
        """Write one iteration after removing ghost cells from each field."""

        iteration = self._iteration(step, time)
        interior_fields = _strip_ghost_cells(field_map, self.ghost_cells)
        _write_field_map(
            self.series,
            iteration,
            interior_fields,
            self.grid_spacing,
            self.grid_global_offset,
            self.grid_position,
        )
        iteration.close()

    def close(self):
        if self.series is not None:
            self.series.close()
            self.series = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class FMRPatchSeriesWriter:
    """Write one root patch and one ratio-2 fine patch as separate series.

    Each call is synchronous and closes both file-based openPMD series before
    updating the relative VisIt aliases and restart-safe manifest.
    """

    _PATCHES = ((0, 0), (1, 0))

    def __init__(self, base_name, dt):
        base_path = Path(base_name)
        if base_path.suffix in (".h5", ".visit", ".pmd", ".opmd"):
            base_path = base_path.with_suffix("")
        base_path.parent.mkdir(parents=True, exist_ok=True)

        self.base_path = base_path
        self.dt = float(dt)
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        self.visit_path = base_path.with_suffix(".visit")
        self._topology = None

    def _pattern(self, level, patch, suffix="h5"):
        return self.base_path.parent / (
            f"{self.base_path.name}_level_{level:02d}_patch_{patch:03d}_"
            f"%08T.{suffix}"
        )

    def _output_path(self, level, patch, output_index, suffix="h5"):
        return Path(
            str(self._pattern(level, patch, suffix)).replace(
                "%08T", f"{output_index:08d}"
            )
        )

    def _helper_path(self, level, patch):
        return self.base_path.parent / (
            f"{self.base_path.name}_level_{level:02d}_patch_{patch:03d}.pmd"
        )

    @staticmethod
    def _iteration_attributes(level, active_shape, patch_spec, simulation_step):
        if level == 0:
            start = (0, 0, 0)
            stop = tuple(int(size) for size in active_shape)
            parent = -1
            ratio = 1
        else:
            start = tuple(int(index) for index in patch_spec.coarse_lo)
            stop = tuple(int(index) + 1 for index in patch_spec.coarse_hi)
            parent = 0
            ratio = 2

        return {
            "fmrLevel": np.int64(level),
            "fmrPatch": np.int64(0),
            "fmrParent": np.int64(parent),
            "refinementRatio": np.int64(ratio),
            "coarseStart": np.asarray(start, dtype=np.int64),
            "coarseStop": np.asarray(stop, dtype=np.int64),
            "simulationStep": np.int64(simulation_step),
        }

    def _write_patch(
        self,
        level,
        field_map,
        active_shape,
        spacing,
        origin,
        grid_position,
        patch_spec,
        output_index,
        simulation_step,
        time,
    ):
        pattern = self._pattern(level, 0)
        series = io.Series(str(pattern), io.Access.create)
        try:
            series.set_attribute("software", "JAX_BSSN")
            iteration = series.iterations[output_index]
            iteration.time = float(time)
            iteration.dt = self.dt
            iteration.time_unit_SI = 1.0
            for name, value in self._iteration_attributes(
                level, active_shape, patch_spec, simulation_step
            ).items():
                iteration.set_attribute(name, value)

            _write_field_map(
                series,
                iteration,
                field_map,
                spacing,
                origin,
                grid_position,
            )
            iteration.close()
        finally:
            series.close()

    @staticmethod
    def _topology_signature(
        root_shape,
        fine_shape,
        root_spacing,
        fine_spacing,
        root_origin,
        fine_origin,
        root_position,
        fine_position,
        patch_spec,
        field_kinds,
    ):
        return (
            root_shape,
            fine_shape,
            root_spacing,
            fine_spacing,
            root_origin,
            fine_origin,
            root_position,
            fine_position,
            tuple(int(value) for value in patch_spec.coarse_lo),
            tuple(int(value) for value in patch_spec.coarse_hi),
            int(patch_spec.refinement_ratio),
            tuple(sorted(field_kinds.items())),
        )

    def _validate_topology(
        self,
        root_fields,
        fine_fields,
        root_spacing,
        fine_spacing,
        root_origin,
        fine_origin,
        root_position,
        fine_position,
        patch_spec,
    ):
        if int(patch_spec.refinement_ratio) != 2:
            raise ValueError("FMR patch-series output supports only 2:1 refinement")
        if len(patch_spec.coarse_lo) != 3 or len(patch_spec.coarse_hi) != 3:
            raise ValueError("FMR patch bounds must be three-dimensional")
        if any(hi < lo for lo, hi in
               zip(patch_spec.coarse_lo, patch_spec.coarse_hi)):
            raise ValueError("fine patch coarse bounds are invalid")
        if any(value <= 0.0 for value in root_spacing + fine_spacing):
            raise ValueError("patch grid spacings must be positive")
        if not np.allclose(
            np.asarray(fine_spacing) * 2.0,
            root_spacing,
            rtol=_ALIGNMENT_RTOL,
            atol=_ALIGNMENT_ATOL,
        ):
            raise ValueError("fine spacing must equal root spacing divided by two")

        root_shape, fine_shape = _validate_field_maps(root_fields, fine_fields)
        expected_fine_shape = tuple(
            2 * (int(hi) - int(lo)) + 1
            for lo, hi in zip(patch_spec.coarse_lo, patch_spec.coarse_hi)
        )
        if fine_shape != expected_fine_shape:
            raise ValueError(
                f"fine active shape {fine_shape} does not match patch bounds; "
                f"expected {expected_fine_shape}"
            )
        if any(lo < 0 or hi >= size for lo, hi, size in zip(
            patch_spec.coarse_lo, patch_spec.coarse_hi, root_shape
        )):
            raise ValueError("fine patch lies outside the active root patch")
        if any(lo == 0 or hi == size - 1 for lo, hi, size in zip(
            patch_spec.coarse_lo, patch_spec.coarse_hi, root_shape
        )):
            raise ValueError("the supported fine patch must be interior to the root")

        expected_fine_origin = tuple(
            origin + int(start) * spacing
            for origin, start, spacing in zip(
                root_origin, patch_spec.coarse_lo, root_spacing
            )
        )
        if not np.allclose(
            fine_origin,
            expected_fine_origin,
            rtol=_ALIGNMENT_RTOL,
            atol=_ALIGNMENT_ATOL,
        ):
            raise ValueError(
                "fine origin must equal root_origin + coarseStart * root_spacing"
            )

        field_kinds = {name: _field_kind(field)
                       for name, field in root_fields.items()}
        signature = self._topology_signature(
            root_shape,
            fine_shape,
            root_spacing,
            fine_spacing,
            root_origin,
            fine_origin,
            root_position,
            fine_position,
            patch_spec,
            field_kinds,
        )
        if self._topology is not None and signature != self._topology:
            raise ValueError("FMR patch-series topology changed between outputs")
        self._topology = signature
        return root_shape, fine_shape

    def _expected_alias_names(self, output_index):
        return tuple(
            self._output_path(level, patch, output_index, "opmd").name
            for level, patch in self._PATCHES
        )

    def _parse_manifest(self):
        if not self.visit_path.exists():
            return {}
        if not self.visit_path.is_file() or self.visit_path.is_symlink():
            raise ValueError(f"VisIt manifest path is not a regular file: {self.visit_path}")

        lines = self.visit_path.read_text(encoding="utf-8").splitlines()
        if not lines or lines[0] != "!NBLOCKS 2":
            raise ValueError("VisIt manifest must begin with exactly '!NBLOCKS 2'")
        if (len(lines) - 1) % 3 != 0:
            raise ValueError("VisIt manifest contains an incomplete block group")

        groups = {}
        name = re.escape(self.base_path.name)
        root_re = re.compile(
            rf"^{name}_level_00_patch_000_(\d{{8}})\.opmd$"
        )
        fine_re = re.compile(
            rf"^{name}_level_01_patch_000_(\d{{8}})\.opmd$"
        )
        for offset in range(1, len(lines), 3):
            time_line, root_name, fine_name = lines[offset:offset + 3]
            if not time_line.startswith("!TIME "):
                raise ValueError("VisIt manifest group is missing its !TIME line")
            try:
                time = float(time_line[6:])
            except ValueError as exc:
                raise ValueError("VisIt manifest contains an invalid time") from exc
            if not np.isfinite(time):
                raise ValueError("VisIt manifest contains a non-finite time")
            root_match = root_re.fullmatch(root_name)
            fine_match = fine_re.fullmatch(fine_name)
            if root_match is None or fine_match is None:
                raise ValueError(
                    "VisIt manifest block order or patch topology is invalid"
                )
            root_index = int(root_match.group(1))
            fine_index = int(fine_match.group(1))
            if root_index != fine_index:
                raise ValueError("VisIt manifest patch output indices do not match")
            if root_index in groups:
                raise ValueError(
                    f"VisIt manifest duplicates output index {root_index}"
                )
            groups[root_index] = (time, root_name, fine_name)
        indices = sorted(groups)
        if indices != list(range(len(indices))):
            raise ValueError(
                "VisIt manifest output indices must be contiguous and start at zero"
            )
        ordered_times = [groups[index][0] for index in indices]
        if any(later < earlier and not np.isclose(
            later, earlier, rtol=_ALIGNMENT_RTOL, atol=_ALIGNMENT_ATOL
        ) for earlier, later in zip(ordered_times, ordered_times[1:])):
            raise ValueError(
                "VisIt manifest times must be nondecreasing by output index"
            )
        return groups

    def _validate_existing_patch(
        self,
        output_index,
        manifest_time,
        level,
        expected_fields,
        expected_shape,
        expected_spacing,
        expected_origin,
        expected_position,
        patch_spec,
    ):
        path = self._output_path(level, 0, output_index)
        if not path.is_file() or path.is_symlink():
            raise ValueError(
                f"VisIt manifest references a missing patch file: {path}"
            )

        series = io.Series(str(path), io.Access.read_only)
        try:
            if output_index not in series.iterations:
                raise ValueError(
                    f"patch file {path} does not contain iteration {output_index}"
                )
            iteration = series.iterations[output_index]
            if not np.isclose(
                iteration.time,
                manifest_time,
                rtol=_ALIGNMENT_RTOL,
                atol=_ALIGNMENT_ATOL,
            ):
                raise ValueError(
                    f"patch file {path} has a time conflicting with its manifest"
                )
            if not np.isclose(
                iteration.dt,
                self.dt,
                rtol=_ALIGNMENT_RTOL,
                atol=_ALIGNMENT_ATOL,
            ):
                raise ValueError("FMR patch-series solver timestep changed")

            expected_attributes = self._iteration_attributes(
                level, expected_shape, patch_spec, simulation_step=0
            )
            for name, expected in expected_attributes.items():
                if name == "simulationStep":
                    continue
                try:
                    actual = iteration.get_attribute(name)
                except Exception as exc:
                    raise ValueError(
                        f"patch file {path} is missing hierarchy attribute {name}"
                    ) from exc
                if not np.array_equal(np.asarray(actual), np.asarray(expected)):
                    raise ValueError(
                        f"FMR patch-series topology changed for attribute {name}"
                    )

            if set(iteration.meshes) != set(expected_fields):
                raise ValueError("FMR patch-series field set changed on restart")
            for field_name, field in expected_fields.items():
                mesh = iteration.meshes[field_name]
                expected_components = (
                    set(_VECTOR_COMPONENTS)
                    if _field_kind(field) == "vector"
                    else {io.Mesh_Record_Component.SCALAR}
                )
                if set(mesh) != expected_components:
                    raise ValueError(
                        f"field {field_name!r} changes scalar/vector kind on restart"
                    )
                if not np.allclose(
                    mesh.grid_spacing,
                    expected_spacing,
                    rtol=_ALIGNMENT_RTOL,
                    atol=_ALIGNMENT_ATOL,
                ):
                    raise ValueError("FMR patch-series spacing changed on restart")
                if not np.allclose(
                    mesh.grid_global_offset,
                    expected_origin,
                    rtol=_ALIGNMENT_RTOL,
                    atol=_ALIGNMENT_ATOL,
                ):
                    raise ValueError("FMR patch-series origin changed on restart")
                for component_name in mesh:
                    component = mesh[component_name]
                    if tuple(component.shape) != expected_shape:
                        raise ValueError(
                            "FMR patch-series active shape changed on restart"
                        )
                    if not np.allclose(
                        component.position,
                        expected_position,
                        rtol=_ALIGNMENT_RTOL,
                        atol=_ALIGNMENT_ATOL,
                    ):
                        raise ValueError(
                            "FMR patch-series grid position changed on restart"
                        )
            simulation_step = int(
                iteration.get_attribute("simulationStep")
            )
            if simulation_step < 0:
                raise ValueError(
                    f"patch file {path} contains an invalid simulationStep"
                )
            iteration.close()
            return simulation_step
        finally:
            series.close()

    def _validate_existing_outputs(
        self,
        groups,
        root_fields,
        fine_fields,
        root_shape,
        fine_shape,
        root_spacing,
        fine_spacing,
        root_origin,
        fine_origin,
        root_position,
        fine_position,
        patch_spec,
        current_output_index,
        current_simulation_step,
    ):
        for output_index, (time, _, _) in groups.items():
            if jax.process_index() == 0:
                for level, patch in self._PATCHES:
                    alias = self._output_path(
                        level, patch, output_index, "opmd"
                    )
                    target = self._output_path(
                        level, patch, output_index, "h5"
                    )
                    if (not alias.is_symlink()
                            or os.readlink(alias) != target.name):
                        raise ValueError(
                            f"VisIt manifest references an invalid alias: {alias}"
                        )
            root_step = self._validate_existing_patch(
                output_index,
                time,
                0,
                root_fields,
                root_shape,
                root_spacing,
                root_origin,
                root_position,
                patch_spec,
            )
            fine_step = self._validate_existing_patch(
                output_index,
                time,
                1,
                fine_fields,
                fine_shape,
                fine_spacing,
                fine_origin,
                fine_position,
                patch_spec,
            )
            if root_step != fine_step:
                raise ValueError(
                    "root and fine patches have conflicting simulationStep values"
                )
            if (output_index == current_output_index
                    and root_step != current_simulation_step):
                raise ValueError(
                    f"output index {output_index} already has conflicting "
                    f"simulationStep {root_step}"
                )

    def _rewrite_manifest(self, output_index, time):
        groups = self._parse_manifest()
        aliases = self._expected_alias_names(output_index)
        if output_index in groups:
            old_time, old_root, old_fine = groups[output_index]
            if not np.isclose(
                old_time, time, rtol=_ALIGNMENT_RTOL, atol=_ALIGNMENT_ATOL
            ):
                raise ValueError(
                    f"output index {output_index} already has conflicting time "
                    f"{old_time} in the VisIt manifest"
                )
            if (old_root, old_fine) != aliases:
                raise ValueError(
                    f"output index {output_index} has conflicting block paths"
                )
        groups[output_index] = (float(time), *aliases)

        indices = sorted(groups)
        if indices != list(range(len(indices))):
            raise ValueError(
                "saved output indices must be contiguous and start at zero"
            )

        ordered_times = [groups[index][0] for index in indices]
        if any(later < earlier and not np.isclose(
            later, earlier, rtol=_ALIGNMENT_RTOL, atol=_ALIGNMENT_ATOL
        ) for earlier, later in zip(ordered_times, ordered_times[1:])):
            raise ValueError(
                "VisIt manifest times must be nondecreasing by output index"
            )

        lines = ["!NBLOCKS 2"]
        for index in sorted(groups):
            group_time, root_name, fine_name = groups[index]
            lines.extend((f"!TIME {group_time}", root_name, fine_name))
        contents = "\n".join(lines) + "\n"

        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.visit_path.parent,
            prefix=f".{self.visit_path.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(contents)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, self.visit_path)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def _ensure_relative_alias(alias_path, target_path):
        target = target_path.name
        if alias_path.is_symlink():
            if os.readlink(alias_path) == target:
                return
            raise FileExistsError(
                f"alias {alias_path} points to {os.readlink(alias_path)!r}, "
                f"expected {target!r}"
            )
        if os.path.lexists(alias_path):
            raise FileExistsError(
                f"refusing to overwrite conflicting alias path {alias_path}"
            )
        alias_path.symlink_to(target)

    @staticmethod
    def _preflight_relative_alias(alias_path, target_path):
        if alias_path.is_symlink():
            if os.readlink(alias_path) != target_path.name:
                raise FileExistsError(
                    f"alias {alias_path} points to {os.readlink(alias_path)!r}, "
                    f"expected {target_path.name!r}"
                )
        elif os.path.lexists(alias_path):
            raise FileExistsError(
                f"refusing to overwrite conflicting alias path {alias_path}"
            )

    @staticmethod
    def _ensure_helper(helper_path, pattern_name):
        expected = pattern_name + "\n"
        if os.path.lexists(helper_path):
            if (helper_path.is_file() and not helper_path.is_symlink()
                    and helper_path.read_text(encoding="utf-8") == expected):
                return
            raise FileExistsError(
                f"refusing to overwrite conflicting ParaView helper {helper_path}"
            )
        helper_path.write_text(expected, encoding="utf-8")

    def write(
        self,
        root_fields,
        fine_fields,
        *,
        output_index,
        simulation_step,
        time,
        root_spacing,
        fine_spacing,
        root_origin,
        fine_origin,
        root_ghost_cells,
        fine_ghost_cells,
        patch_spec,
        root_grid_position=(0.0, 0.0, 0.0),
        fine_grid_position=(0.0, 0.0, 0.0),
    ):
        """Write one complete two-patch output and update viewer metadata."""

        output_index = int(output_index)
        simulation_step = int(simulation_step)
        time = float(time)
        if output_index < 0 or simulation_step < 0:
            raise ValueError("output_index and simulation_step must be non-negative")
        if not np.isfinite(time):
            raise ValueError("time must be finite")
        if not isinstance(root_fields, Mapping) or not isinstance(
            fine_fields, Mapping
        ):
            raise ValueError(
                "FMR patch-series output requires exactly one field map per level"
            )
        root_spacing = _float_3tuple(root_spacing, "root_spacing")
        fine_spacing = _float_3tuple(fine_spacing, "fine_spacing")
        root_origin = _float_3tuple(root_origin, "root_origin")
        fine_origin = _float_3tuple(fine_origin, "fine_origin")
        root_position = _float_3tuple(
            root_grid_position, "root_grid_position"
        )
        fine_position = _float_3tuple(
            fine_grid_position, "fine_grid_position"
        )
        fine_ghost_tuple = _integer_3tuple(
            fine_ghost_cells, "fine_ghost_cells"
        )
        if fine_ghost_tuple != (int(patch_spec.ghost_width),) * 3:
            raise ValueError(
                "fine_ghost_cells must match FMRPatchSpec.ghost_width"
            )

        root_active = _strip_ghost_cells(root_fields, root_ghost_cells)
        fine_active = _strip_ghost_cells(fine_fields, fine_ghost_tuple)
        root_shape, fine_shape = self._validate_topology(
            root_active,
            fine_active,
            root_spacing,
            fine_spacing,
            root_origin,
            fine_origin,
            root_position,
            fine_position,
            patch_spec,
        )

        # Reject manifest conflicts before creating or replacing patch files.
        existing = self._parse_manifest()
        candidate_indices = sorted(set(existing) | {output_index})
        if candidate_indices != list(range(len(candidate_indices))):
            raise ValueError(
                "saved output indices must be contiguous and start at zero"
            )
        if output_index in existing and not np.isclose(
            existing[output_index][0],
            float(time),
            rtol=_ALIGNMENT_RTOL,
            atol=_ALIGNMENT_ATOL,
        ):
            raise ValueError(
                f"output index {output_index} already has conflicting time "
                f"{existing[output_index][0]} in the VisIt manifest"
            )
        self._validate_existing_outputs(
            existing,
            root_active,
            fine_active,
            root_shape,
            fine_shape,
            root_spacing,
            fine_spacing,
            root_origin,
            fine_origin,
            root_position,
            fine_position,
            patch_spec,
            output_index,
            simulation_step,
        )

        if jax.process_index() == 0:
            for level, patch in self._PATCHES:
                self._preflight_relative_alias(
                    self._output_path(level, patch, output_index, "opmd"),
                    self._output_path(level, patch, output_index, "h5"),
                )

        self._write_patch(
            0,
            root_active,
            root_shape,
            root_spacing,
            root_origin,
            root_position,
            patch_spec,
            output_index,
            simulation_step,
            time,
        )
        self._write_patch(
            1,
            fine_active,
            fine_shape,
            fine_spacing,
            fine_origin,
            fine_position,
            patch_spec,
            output_index,
            simulation_step,
            time,
        )

        if jax.process_index() == 0:
            for level, patch in self._PATCHES:
                h5_path = self._output_path(
                    level, patch, output_index, "h5"
                )
                alias_path = self._output_path(
                    level, patch, output_index, "opmd"
                )
                self._ensure_relative_alias(alias_path, h5_path)
                self._ensure_helper(
                    self._helper_path(level, patch),
                    self._pattern(level, patch, "h5").name,
                )
            self._rewrite_manifest(output_index, float(time))

        return self.visit_path

    def close(self):
        """Retained for context-manager symmetry; writes close immediately."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
