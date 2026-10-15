"""Single-tile Cartesian layout adapter for PyPIC3D.

BSSN arrays contain physical C nodes, without EM guards. V nodes are half a
cell ABOVE C nodes, as in PyPIC3D. Conducting axes include both C endpoints;
the final V sample on such an axis is exterior data, not an owned node.
"""

from dataclasses import dataclass
import math

import jax.numpy as jnp
from PyPIC3D.utilities.parameters import (
    build_static_parameters,
    DynamicParameters,
    GridParameters,
)

CENTER_LOCATION = ("C", "C", "C")
BOUNDARY_CODES = {"periodic": 0, "conducting": 1, "absorbing": 2, "constant": 3}


@dataclass(frozen=True)
class EMGrid:
    """Static configuration; construct outside JIT with :func:`make_em_grid`."""

    shape: tuple
    static: object

    @property
    def guard_cells(self):
        return self.static.guard_cells

    @property
    def tile_shape(self):
        return self.static.tile_shape

    def to_tile(self, values):
        """Pad scalar or component-first BSSN data and add tile dimensions.

        Metric exteriors use periodic copies or linear extrapolation. Native
        field exteriors are subsequently replaced by PyPIC3D refresh_fields.
        """
        if values.shape[-3:] != self.shape:
            raise ValueError(
                f"Expected physical shape {self.shape}, got {values.shape[-3:]}"
            )
        for direction, (n, width, bc) in enumerate(
            zip(self.shape, self.tile_shape, self.static.boundary_conditions)
        ):
            axis = values.ndim - 3 + direction
            index = jnp.arange(width + 2 * self.guard_cells) - self.guard_cells
            if bc == 0:
                values = jnp.take(values, index % n, axis=axis)
            else:
                clipped = jnp.clip(index, 0, n - 1)
                padded = jnp.take(values, clipped, axis=axis)
                if n > 1:
                    left = jnp.take(values, jnp.array([1]), axis=axis) - jnp.take(
                        values, jnp.array([0]), axis=axis
                    )
                    right = jnp.take(values, jnp.array([n - 1]), axis=axis) - jnp.take(
                        values, jnp.array([n - 2]), axis=axis
                    )
                    dimensions = [1] * values.ndim
                    dimensions[axis] = index.size
                    padded += jnp.minimum(index, 0).reshape(dimensions) * left
                    padded += (
                        jnp.maximum(index - (n - 1), 0).reshape(dimensions) * right
                    )
                values = padded
        return values.reshape(values.shape[:-3] + (1, 1, 1) + values.shape[-3:])

    def from_tile(self, values):
        """Extract BSSN C-node allocation (including conducting endpoints)."""
        g = self.guard_cells
        return values[(Ellipsis, 0, 0, 0) + tuple(slice(g, g + n) for n in self.shape)]

    def coordinates(self, params, location=CENTER_LOCATION, *, tiled=False):
        """Coordinate arrays for native initial data; V has positive offset."""
        result = []
        for axis, (n, width, origin, loc) in enumerate(
            zip(
                self.shape,
                self.tile_shape,
                (params.x_min, params.y_min, params.z_min),
                location,
            )
        ):
            count = width + 2 * self.guard_cells if tiled else n
            offset = -self.guard_cells if tiled else 0
            line = origin + params.dx * (
                jnp.arange(count) + offset + (0.5 if loc == "V" else 0)
            )
            dims = [1] * (6 if tiled else 3)
            dims[axis + (3 if tiled else 0)] = count
            result.append(line.reshape(dims))
        return tuple(result)

    def dynamic(self, params, dt):
        """PyPIC3D dynamic parameters; units c=eps=mu=1, no particles."""
        origins = (params.x_min, params.y_min, params.z_min)
        center = tuple(
            o + params.dx * jnp.arange(-1, n + 1)
            for o, n in zip(origins, self.tile_shape)
        )
        vertex = tuple(a + 0.5 * params.dx for a in center)
        # PyPIC3D coordinate lines have three tile dimensions plus one axis.
        tc = tuple(
            (
                o + params.dx * jnp.arange(-self.guard_cells, n + self.guard_cells)
            ).reshape(1, 1, 1, -1)
            for o, n in zip(origins, self.tile_shape)
        )
        tv = tuple(a + 0.5 * params.dx for a in tc)
        one = jnp.asarray(1.0)
        return DynamicParameters(
            jnp.asarray(dt),
            *([jnp.asarray(params.dx)] * 3),
            *(jnp.asarray(n) for n in self.tile_shape),
            *(params.dx * n for n in self.tile_shape),
            one,
            one,
            one,
            one,
            one,
            GridParameters(vertex, center, tv, tc),
        )


def make_em_grid(shape, params, *, boundary_conditions=("periodic",) * 3):
    """Build one PyPIC3D tile, with boundary choices independent of BSSN.

    Use names rather than BSSN's integer boundary codes: BSSN code 1 means
    Sommerfeld, whereas PyPIC3D code 1 means conducting.
    """
    shape = tuple(shape)
    if len(shape) != 3 or any(
        not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in shape
    ):
        raise ValueError("EM shape must contain three positive integer sizes")
    if not math.isfinite(float(params.dx)) or params.dx <= 0:
        raise ValueError("EM requires finite dx > 0")
    if not math.isfinite(float(params.dt)) or params.dt <= 0:
        raise ValueError("EM requires finite dt > 0")
    if len(boundary_conditions) != 3 or any(
        b not in BOUNDARY_CODES for b in boundary_conditions
    ):
        raise ValueError(
            "Use PyPIC3D boundary names: periodic, conducting, absorbing, constant; EM Sommerfeld is unsupported"
        )
    codes = tuple(BOUNDARY_CODES[b] for b in boundary_conditions)
    widths = tuple(n - (bc == 1) for n, bc in zip(shape, codes))
    if any((n != 1 and n <= 3) or (bc != 0 and n <= 3) for n, bc in zip(widths, codes)):
        raise ValueError(
            "Active EM axes need more than three cells; singleton axes must be periodic"
        )
    static = build_static_parameters(
        dict(
            solver="static_metric",
            tile_shape=widths,
            Nx=widths[0],
            Ny=widths[1],
            Nz=widths[2],
            boundary_conditions=codes,
            guard_cells=3,
            shape_factor=1,
            particle_pusher="hybrid_boris_geodesic",
            current_deposition="GR_direct",
            metric="flat_cartesian",
            pml_active=False,
            supergaussian_active=False,
        )
    )
    return EMGrid(shape, static)
