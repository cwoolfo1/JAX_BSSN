"""Read uniform Cartesian openPMD coordinates without importing JAX."""

import numpy as np


def mesh_axis_coordinates(mesh, component, axis, *, edges=False):
    """Return sample positions or bounding cell edges; reject mapped archives."""
    if mesh.geometry_string != "cartesian" or any(
        name in mesh.attributes
        for name in ("cartoonRadialScale", "cartoonAxialScale",
                     "cartoonRadialOuterRadius", "cartoonAxialOuterRadius",
                     "physicalCoordinatesX", "physicalCoordinatesZ")
    ):
        raise ValueError("Unsupported mesh geometry: only uniform Cartesian output is supported")
    size = component.shape[axis]
    offset = float(mesh.grid_global_offset[axis])
    spacing = float(mesh.grid_spacing[axis])
    position = float(component.position[axis])
    return offset + (
        np.arange(size + int(edges)) + position - (0.5 if edges else 0.0)
    ) * spacing
