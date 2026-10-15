"""Uniform openPMD coordinates survive a cropped field round trip."""

import numpy as np
import openpmd_api as io

from JAX_BSSN.diagnostics.openpmd import OpenPMDWriter


def test_cropped_snapshot_retains_physical_coordinates_and_values(tmp_path):
    from JAX_BSSN.diagnostics.mesh_coordinates import mesh_axis_coordinates
    selection = (slice(2, 6), slice(None), slice(2, 6))
    origin = (-0.75, 0.0, -0.75)
    x = np.arange(-1.75, 2, .5)[:, None, None]
    z = np.arange(-1.75, 2, .5)[None, None, :]
    full = 10*x+z
    data = {'W': full[selection], 'shift': tuple(k*full[selection] for k in (1,2,3))}
    path = tmp_path/'cropped.h5'
    with OpenPMDWriter(path, (.5,)*3, origin, .01, grid_position=(0.,)*3) as writer:
        writer.write(data, 0, 0.)
    series = io.Series(str(path), io.Access.read_only)
    try:
        meshes = series.iterations[0].meshes
        for name in ('W', 'shift'):
            mesh = meshes[name]
            for key in mesh:
                component = mesh[key]
                xs = mesh_axis_coordinates(mesh, component, 0)
                zs = mesh_axis_coordinates(mesh, component, 2)
                np.testing.assert_array_equal(xs, [-.75,-.25,.25,.75])
                np.testing.assert_array_equal(zs, [-.75,-.25,.25,.75])
                actual = component.load_chunk()
                series.flush()
                factor = 'xyz'.index(key)+1 if name == 'shift' else 1
                np.testing.assert_array_equal(actual, factor*(10*xs[:,None,None]+zs[None,None,:]))
    finally:
        series.close()
