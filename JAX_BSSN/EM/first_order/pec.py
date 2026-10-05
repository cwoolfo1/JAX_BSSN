"""Metric-aware PEC walls for the Cartesian, densitized Yee solver.

Ported from PyPIC3D/boundary_conditions/pec.py (Christopher Woolford, 2025,
MIT license). The face projectors, coupled edge solves, and ordered corner
reflections are retained; tile exchange is unnecessary on this uniform grid.

Walls are stationary in the FIDO frame: D is metric-normal and B is tangent.
These constraints do not set the coordinate electric covector to zero when
the shift is nonzero. Geometry is sampled anew from the supplied BSSN state.

On each conducting axis, the input includes explicit exterior samples.
The two wall C nodes are g and size-g-1. JAX BSSN's V nodes are *below* C,
so the owned V nodes are g+1 through size-g-1, unlike PyPIC3D's upper V
convention. Other axes retain their ordinary periodic/Sommerfeld handling.
"""

# MIT License
#
# Copyright (c) 2025 Christopher Woolford
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from dataclasses import dataclass
from functools import partial
from itertools import combinations, product

import jax
import jax.numpy as jnp

from JAX_BSSN.EM.first_order.geometry import _metric_fields_at_location
from JAX_BSSN.EM.first_order.staggering import (
    DISPLACEMENT_FIELD_LOCATIONS as D_LOCATIONS,
    MAGNETIC_FIELD_LOCATIONS as B_LOCATIONS,
    interpolate_between_locations,
)


@dataclass(frozen=True)
class PECBoundary:
    """Paired coordinate-plane EM walls, independent of gravity boundaries.

    ``axes`` selects x/y/z as 0/1/2. Allocate ``cells + 1 + 2*guard_cells``
    samples on those axes. Physical wall coordinates follow the existing C
    grid, including its origin. At least two exterior layers are needed for
    the constitutive/curl stencil; the default also accommodates BSSN's KO
    stencil. Each conducting direction needs more cells than guard layers.

    Pass this object as ``pec_boundary=...`` to the Cartesian initializer
    and stepper. A separate ``stage_boundary`` callback can supply spacetime
    boundary data; this object never modifies BSSN variables.
    """

    axes: tuple[int, ...] = (0, 1, 2)
    guard_cells: int = 3

    def __post_init__(self):
        axes = tuple(self.axes)
        if len(set(axes)) != len(axes) or any(not isinstance(a, int) or a not in (0, 1, 2) for a in axes):
            raise ValueError("PEC axes must be distinct members of (0, 1, 2)")
        if not isinstance(self.guard_cells, int) or self.guard_cells < 2:
            raise ValueError("PEC requires at least two guard cells")
        object.__setattr__(self, "axes", tuple(sorted(axes)))

    def validate(self, shape):
        if len(shape) != 3:
            raise ValueError("PEC requires a Cartesian three-dimensional grid")
        for axis in self.axes:
            cells = shape[axis] - 2 * self.guard_cells - 1
            if cells <= self.guard_cells:
                raise ValueError("Each PEC axis needs more physical cells than guard cells")

    def walls(self, shape, axis):
        return self.guard_cells, shape[axis] - self.guard_cells - 1


def _plane(axis, node):
    return (slice(None),) * axis + (node,) + (slice(None),) * (2 - axis)


def _face_mask(shape, axis, boundary):
    low, high = boundary.walls(shape, axis)
    nodes = jnp.arange(shape[axis])
    return ((nodes == low) | (nodes == high)).reshape(
        (1,) * axis + (shape[axis],) + (1,) * (2 - axis)
    )


def _metric_samples(bssn, params):
    samples = {}
    for location in product(("C", "V"), repeat=3):
        W, _, _, _, inverse = _metric_fields_at_location(bssn, location, params)
        # Only inverse-metric ratios enter the projectors; W^2 cancels.
        samples[location] = (inverse, W**-3)
    return samples


def _reconstruct(vector, locations, metric, target, params):
    volume = metric[target][1]
    return tuple(
        value if source == target else interpolate_between_locations(
            metric[source][1] * value, source, target, params
        ) / volume
        for value, source in zip(vector, locations)
    )


def _normal(inverse, vector, axis, component):
    return inverse[component, axis] / inverse[axis, axis] * vector[axis]


def _images(shape, location, axis, boundary):
    low, high = boundary.walls(shape, axis)
    vertex = int(location[axis] == "V")
    for wall, outside in ((low, range(low + vertex)),
                          (high, range(high + 1, shape[axis]))):
        for node in outside:
            yield wall, node, 2 * wall - node + vertex


def _prepare_normal_D(snapshot, boundary):
    prepared = list(snapshot)
    for axis in boundary.axes:
        for _, node, owner in _images(snapshot[0].shape, D_LOCATIONS[axis], axis, boundary):
            prepared[axis] = prepared[axis].at[_plane(axis, node)].set(
                snapshot[axis][_plane(axis, owner)]
            )
    return tuple(prepared)


def _project_native_nodes(snapshot, locations, kind, metric, params, boundary):
    shape = snapshot[0].shape
    projected = []
    for component, location in enumerate(locations):
        value = snapshot[component]
        vector = _reconstruct(snapshot, locations, metric, location, params)
        inverse = metric[location][0]
        incident = tuple(axis for axis in boundary.axes if location[axis] == "C")
        for count in range(1, len(incident) + 1):
            for faces in combinations(incident, count):
                mask = True
                for axis in faces:
                    mask = mask & _face_mask(shape, axis, boundary)
                if count == 1:
                    normal = _normal(inverse, vector, faces[0], component)
                    constrained = normal if kind == "D" else vector[component] - normal
                else:
                    constrained = 0.0
                value = jnp.where(mask, constrained, value)
        projected.append(value)
    return tuple(projected)


def _edge_nodes(shape, boundary):
    for p, q in combinations(boundary.axes, 2):
        for low_p, low_q in product((True, False), repeat=2):
            def at(p_node, q_node, p=p, q=q):
                index = [slice(None)] * 3
                index[p], index[q] = p_node, q_node
                return tuple(index)

            def nodes(axis, low):
                lower, upper = boundary.walls(shape, axis)
                # C wall, owned lower-V sample, exterior lower-V sample.
                return (lower, lower + 1, lower) if low else (upper, upper, upper + 1)

            yield p, q, at, nodes(p, low_p), nodes(q, low_q)


def _solve_D_edges(projected, snapshot, metric, boundary):
    """Solve both mutually coupled native D rows at each edge exactly."""
    shape = snapshot[0].shape
    result = list(projected)
    for p, q, at, (pw, ph, pg), (qw, qh, qg) in _edge_nodes(shape, boundary):
        a, am = at(pw, qh), at(pw, qg)
        b, bm = at(ph, qw), at(pg, qw)
        inv_a, vol_a = metric[D_LOCATIONS[q]]
        inv_b, vol_b = metric[D_LOCATIONS[p]]
        alpha = inv_a[q, p][a] / inv_a[p, p][a] * (vol_b[b] + vol_b[bm]) / (4 * vol_a[a])
        beta = inv_b[p, q][b] / inv_b[q, q][b] * (vol_a[a] + vol_a[am]) / (4 * vol_b[b])
        a0, b0, a1, b1 = snapshot[q][a], snapshot[p][b], projected[q][a], projected[p][b]
        solved_a = (a1 + alpha * (b1 - b0) - alpha * beta * a0) / (1 - alpha * beta)
        solved_b = b1 + beta * (solved_a - a0)
        third_wall = jnp.zeros(shape, dtype=bool)
        for r in boundary.axes:
            if r not in (p, q):
                third_wall = third_wall | _face_mask(shape, r, boundary)
        result[q] = result[q].at[a].set(jnp.where(third_wall[a], a1, solved_a))
        result[p] = result[p].at[b].set(jnp.where(third_wall[b], b1, solved_b))
    return tuple(result)


def _reflect_vector(vector, metric, location, axis, boundary, kind):
    surface = list(location)
    surface[axis] = "C"
    inverse = metric[tuple(surface)][0]
    result = list(vector)
    for wall, node, owner in _images(vector[0].shape, location, axis, boundary):
        values = tuple(v[_plane(axis, owner)] for v in vector)
        wall_inverse = inverse[(slice(None), slice(None)) + _plane(axis, wall)]
        for component in range(3):
            normal = _normal(wall_inverse, values, axis, component)
            reflected = 2 * normal - values[component] if kind == "D" else values[component] - 2 * normal
            result[component] = result[component].at[_plane(axis, node)].set(reflected)
    return tuple(result)


def _reflect_axis(snapshot, locations, kind, metric, params, boundary, axis, previous):
    result = []
    for component, location in enumerate(locations):
        vector = _reconstruct(snapshot, locations, metric, location, params)
        # Reconstruct at the owned image before composing corner reflections,
        # so no component transfer crosses the end of the allocated halo.
        for earlier in previous:
            vector = _reflect_vector(vector, metric, location, earlier, boundary, kind)
        vector = _reflect_vector(vector, metric, location, axis, boundary, kind)
        result.append(vector[component])
    return tuple(result)


def _solve_B_edges(first, second, metric, boundary):
    """Close the mutually coupled magnetic ghost pair after two sweeps."""
    result = list(second)
    for p, q, at, (pw, ph, pg), (qw, qh, qg) in _edge_nodes(first[0].shape, boundary):
        c, co, d, do = at(pw, qg), at(pw, qh), at(pg, qw), at(ph, qw)
        surface = ["V"] * 3
        surface[p] = surface[q] = "C"
        inverse = metric[tuple(surface)][0][(slice(None), slice(None)) + at(pw, qw)]
        vol_p, vol_q = metric[B_LOCATIONS[p]][1], metric[B_LOCATIONS[q]][1]
        alpha = -2 * inverse[p, q] / inverse[q, q] * vol_q[d] / (4 * vol_p[co])
        beta = -2 * inverse[q, p] / inverse[p, p] * vol_p[c] / (4 * vol_q[do])
        c1, c2, d2 = first[p][c], second[p][c], second[q][d]
        delta = (c2 - c1) / (1 - alpha * beta)
        result[p] = result[p].at[c].set(c2 + alpha * beta * delta)
        result[q] = result[q].at[d].set(d2 + beta * delta)
    return tuple(result)


def _enforce_physical(vector, locations, kind, metric, params, boundary):
    def sweep(fields):
        for position, axis in enumerate(boundary.axes):
            fields = _reflect_axis(fields, locations, kind, metric, params, boundary,
                                   axis, boundary.axes[:position])
        return fields

    snapshot = _prepare_normal_D(vector, boundary) if kind == "D" else vector
    projected = _project_native_nodes(snapshot, locations, kind, metric, params, boundary)
    if kind == "D":
        projected = _solve_D_edges(projected, snapshot, metric, boundary)
    result = sweep(projected)
    if kind == "B" and len(boundary.axes) > 1:
        result = _solve_B_edges(result, sweep(result), metric, boundary)
        if len(boundary.axes) > 2:
            result = sweep(result)
    return result


def _enforce_density(vector, locations, kind, metric, params, boundary):
    physical = tuple(vector[i] / metric[loc][1] for i, loc in enumerate(locations))
    projected = _enforce_physical(physical, locations, kind, metric, params, boundary)
    density = jnp.stack(tuple(value * metric[loc][1] for value, loc in zip(projected, locations)))
    # Retain unconstrained owners bit-for-bit, including density conversion.
    touched = []
    shape = vector.shape[-3:]
    for location in locations:
        mask = jnp.zeros(shape, dtype=bool)
        for axis in boundary.axes:
            low, high = boundary.walls(shape, axis)
            nodes = jnp.arange(shape[axis]).reshape((1,) * axis + (shape[axis],) + (1,) * (2-axis))
            mask = mask | (nodes <= low) | (nodes > high)
            if location[axis] == "C":
                mask = mask | (nodes == high)
        touched.append(mask)
    return jnp.where(jnp.stack(touched), density, vector)


@partial(jax.jit, static_argnames=("boundary",))
def enforce_pec_D(displacement, bssn, params, *, boundary=PECBoundary()):
    """Project densitized D and refresh its exterior using this stage's metric."""
    boundary.validate(displacement.shape[-3:])
    if not boundary.axes:
        return displacement
    return _enforce_density(displacement, D_LOCATIONS, "D", _metric_samples(bssn, params), params, boundary)


@partial(jax.jit, static_argnames=("boundary",))
def enforce_pec_B(magnetic, bssn, params, *, boundary=PECBoundary()):
    """Project densitized B and refresh its exterior using this stage's metric."""
    boundary.validate(magnetic.shape[-3:])
    if not boundary.axes:
        return magnetic
    return _enforce_density(magnetic, B_LOCATIONS, "B", _metric_samples(bssn, params), params, boundary)


@partial(jax.jit, static_argnames=("boundary",))
def apply_pec_boundaries(displacement, magnetic, bssn, params, *, boundary=PECBoundary()):
    """Condition a common-time density pair without changing the spacetime."""
    boundary.validate(displacement.shape[-3:])
    if magnetic.shape != displacement.shape or displacement.shape != (3,) + bssn.conformal_factor.shape:
        raise ValueError("PEC fields and BSSN must share the same Cartesian grid")
    if not boundary.axes:
        return displacement, magnetic
    metric = _metric_samples(bssn, params)
    return (_enforce_density(displacement, D_LOCATIONS, "D", metric, params, boundary),
            _enforce_density(magnetic, B_LOCATIONS, "B", metric, params, boundary))


__all__ = ["PECBoundary", "apply_pec_boundaries", "enforce_pec_D", "enforce_pec_B"]
