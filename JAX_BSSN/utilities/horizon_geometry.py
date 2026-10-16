"""Symmetry-aware sampling of physical gamma_ij, d_k gamma_ij and K_ij.

The apparent horizon solver only ever asks for the spatial geometry at a batch
of Cartesian points.  This module answers that question for the three ways the
code stores a slice: a full 3D grid, the signed y=0 plane of an axisymmetric
Cartoon run, and the radial line of a spherically symmetric Cartoon run.  The
reduced providers rebuild the Cartesian tensors by rotating the stored ones, so
the solver itself never learns which symmetry it is working in.

Derivatives are taken with forward-mode AD of the interpolant, which makes
d_k gamma_ij exactly the derivative of the gamma_ij the expansion uses and
supplies the angular (1/rho, 1/r) contributions of the reduced providers for
free.  Interpolation is the degree-five Lagrange rule already used by the
Cartoon reconstructions.

Queries outside the resolved region return the clipped-stencil value rather
than NaN, so that a Newton line search stays finite, and report themselves
through the companion validity flag, which the caller must honour: an invalid
sample is never a usable one.
"""

import numpy as np
import jax.numpy as jnp
from jax import jacfwd, vmap

from JAX_BSSN.cartoon.interpolation import lagrange6_stencil

# Products of (k - m) over m != k for six consecutive degree-five Lagrange nodes.
_LAGRANGE_DENOMINATORS = np.array([-120.0, 24.0, -12.0, 12.0, -24.0, 120.0])
_LAGRANGE_MASK = np.eye(6)

# Independent components of a symmetric 3x3 tensor, and the map back.
SYMMETRIC_COMPONENTS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
_EXPAND = np.array([[0, 1, 2], [1, 3, 4], [2, 4, 5]])

INNER_EXCLUSION_CELLS = 4.0  # Smallest trial surface the reduced providers will seed.
AXIS_FLOOR_CELLS = 1.e-6  # Below this the rotation onto a query point is undefined.


def _uniform_axis(coordinate, name, minimum_size=10):
    """Validate one increasing uniform coordinate axis and return (origin, spacing)."""

    coordinate = np.asarray(coordinate, dtype=float)
    if coordinate.ndim != 1 or coordinate.size < minimum_size:
        raise ValueError(f"{name} must be a 1-D axis with at least {minimum_size} samples")
    spacing = np.diff(coordinate)
    if np.any(spacing <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    if not np.allclose(spacing, spacing[0], rtol=1.e-10, atol=0.0):
        raise ValueError(f"{name} must be uniformly spaced; horizon interpolation assumes it")
    return float(coordinate[0]), float(spacing[0])


def _symmetric_table(*tensors):
    """Stack the upper-triangular components of symmetric tensors into one table."""

    rows = []
    for tensor in tensors:
        tensor = np.asarray(tensor, dtype=float)
        for i, j in SYMMETRIC_COMPONENTS:
            if not np.allclose(tensor[i, j], tensor[j, i], rtol=1.e-12, atol=1.e-14):
                raise ValueError("gamma and Kij must be symmetric")
            rows.append(tensor[i, j])
    return jnp.asarray(np.stack(rows))


def _expand_symmetric(values):
    """Rebuild a (3,3) symmetric tensor from its six independent components."""

    return values[_EXPAND]


def lagrange_weights(index, nodes):
    """Degree-five Lagrange weights for six consecutive nodes.

    These are the numbers ``lagrange6_weights`` produces, written as one masked
    product instead of thirty scatter updates.  The difference is invisible at
    runtime but dominates the compile time of a horizon solve, which traces
    this once per surface node of every trial surface.
    """

    offsets = jnp.where(_LAGRANGE_MASK, 1.0, (index - nodes)[None, :])
    return jnp.prod(offsets, axis=1)/_LAGRANGE_DENOMINATORS


def interpolate_point(table, indices):
    """Degree-five Lagrange interpolation of ``table`` at one fractional index.

    ``table`` has shape ``(fields,) + grid_shape`` and ``indices`` has one
    entry per grid axis.  The stencil is clipped to the array, so the result is
    always finite; callers decide separately whether the point was resolved.
    """

    nodes, weights = [], []
    for axis, index in enumerate(indices):
        node = lagrange6_stencil(index, table.shape[axis + 1])
        nodes.append(node)
        weights.append(lagrange_weights(index, node))

    if len(indices) == 1:
        block = table[:, nodes[0]]
        return jnp.einsum("fa,a->f", block, weights[0])
    if len(indices) == 2:
        block = table[:, nodes[0][:, None], nodes[1][None, :]]
        return jnp.einsum("fab,a,b->f", block, weights[0], weights[1])
    block = table[:, nodes[0][:, None, None], nodes[1][None, :, None], nodes[2][None, None, :]]
    return jnp.einsum("fabc,a,b,c->f", block, weights[0], weights[1], weights[2])


class _Geometry:
    """Common sampling and bookkeeping for the three storage layouts."""

    symmetry = None

    def __init__(self, axes, tables, margin=4):
        self.origin, self.spacing = zip(*(_uniform_axis(a, n) for a, n in axes))
        self.bounds = tuple((float(a[0]), float(a[-1])) for a, _ in axes)
        self.shape = tuple(len(a) for a, _ in axes)
        self.tables = tables
        self.margin = int(margin)
        if min(self.shape) < 2*self.margin + 6:
            raise ValueError("Insufficient resolved domain: too few samples per axis")

    def _index(self, coordinates):
        """Fractional grid indices of physical coordinates, one per grid axis."""

        return [(c - o)/s for c, o, s in zip(coordinates, self.origin, self.spacing)]

    def _resolved(self, indices):
        """True where every fractional index sits inside the interior margin."""

        inside = jnp.array(True)
        for index, size in zip(indices, self.shape):
            inside &= (index >= self.margin) & (index <= size - 1 - self.margin)
        return inside

    def _interior(self, axis):
        """Physical extent of one axis after dropping the margin cells."""

        low, high = self.bounds[axis]
        offset = self.margin*self.spacing[axis]
        return low + offset, high - offset

    def tensors(self, tables, point):
        """Physical gamma_ij and K_ij at one Cartesian point."""

        raise NotImplementedError

    def resolved(self, point):
        """True when one Cartesian point is inside the resolved region."""

        raise NotImplementedError

    def radial_bounds(self, center):
        """Trial-sphere radii about ``center`` that stay inside the resolved region."""

        raise NotImplementedError

    def sample(self, tables, points):
        """Sample a batch of Cartesian points of shape (n, 3).

        Returns gamma (n,3,3), d_k gamma (n,3,3,3) indexed [point, k, i, j],
        K_ij (n,3,3) and a per-point resolved flag.
        """

        gamma, curvature = vmap(self.tensors, in_axes=(None, 0))(tables, points)
        gradient = vmap(jacfwd(lambda p: self.tensors(tables, p)[0]))(points)
        return gamma, jnp.moveaxis(gradient, -1, 1), curvature, vmap(self.resolved)(points)


class CartesianGeometry(_Geometry):
    """Full 3D storage: gamma and Kij of shape (3,3,nx,ny,nz)."""

    symmetry = "3d"

    def __init__(self, x, y, z, gamma, Kij, margin=4):
        axes = ((x, "x"), (y, "y"), (z, "z"))
        shape = (3, 3, len(x), len(y), len(z))
        if np.shape(gamma) != shape or np.shape(Kij) != shape:
            raise ValueError(f"gamma and Kij must have shape {shape}")
        super().__init__(axes, _symmetric_table(gamma, Kij), margin)
        self.inner_radius = 2*max(self.spacing)

    def tensors(self, tables, point):
        values = interpolate_point(tables, self._index(point))
        return _expand_symmetric(values[:6]), _expand_symmetric(values[6:])

    def resolved(self, point):
        return self._resolved(self._index(point))

    def radial_bounds(self, center):
        center = np.asarray(center, dtype=float)
        reach = min(min(center[k] - low, high - center[k])
                    for k, (low, high) in enumerate(map(self._interior, range(3))))
        return self.inner_radius, float(reach)


class AxisymmetricGeometry(_Geometry):
    """Signed y=0 plane of an axisymmetric slice: gamma and Kij of shape (3,3,nx,nz).

    The stored tensors are rotated about the z axis onto the query point, so
    off-plane and off-axis samples are exact rather than extrapolated.  A
    neighbourhood of the rotation axis is excluded because the azimuthal
    derivative carries a 1/rho that no plane data can resolve there.
    """

    symmetry = "axisymmetric"

    def __init__(self, x, z, gamma, Kij, margin=4):
        axes = ((x, "x"), (z, "z"))
        shape = (3, 3, len(x), len(z))
        if np.shape(gamma) != shape or np.shape(Kij) != shape:
            raise ValueError(f"gamma and Kij must have shape {shape}")
        x = np.asarray(x, dtype=float)
        if not x[margin - 1] < 0 < x[-margin - 1]:
            raise ValueError("Axisymmetric horizon interpolation needs a signed x-z plane")
        super().__init__(axes, _symmetric_table(gamma, Kij), margin)
        self.inner_radius = INNER_EXCLUSION_CELLS*self.spacing[0]
        self.axis_floor = AXIS_FLOOR_CELLS*self.spacing[0]

    def _cylindrical(self, point):
        radius = jnp.sqrt(point[0]**2 + point[1]**2)
        # `resolved` rejects the axis itself; this only keeps trial values finite.
        safe = jnp.where(radius > self.axis_floor, radius, 1.0)
        return radius, point[0]/safe, point[1]/safe

    def tensors(self, tables, point):
        radius, cosine, sine = self._cylindrical(point)
        values = interpolate_point(tables, self._index((radius, point[2])))
        rotation = jnp.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
        plane = (_expand_symmetric(values[:6]), _expand_symmetric(values[6:]))
        return tuple(rotation @ tensor @ rotation.T for tensor in plane)

    def resolved(self, point):
        # Only the axis itself is unusable: a surface crossing it does so at the
        # poles, where the azimuthal terms it needs vanish faster than 1/rho.
        radius = jnp.sqrt(point[0]**2 + point[1]**2)
        return self._resolved(self._index((radius, point[2]))) & (radius > self.axis_floor)

    def radial_bounds(self, center):
        center = np.asarray(center, dtype=float)
        if not np.allclose(center[:2], 0.0, atol=1.e-12):
            raise ValueError("An axisymmetric horizon center must lie on the rotation axis")
        _, outer = self._interior(0)
        low, high = self._interior(1)
        return self.inner_radius, float(min(outer, center[2] - low, high - center[2]))


class SphericalGeometry(_Geometry):
    """Radial line of a spherically symmetric slice: gamma and Kij of shape (3,3,nr).

    Only the radial and tangential eigenvalues carry information, so the
    Cartesian tensors are rebuilt as A(r) n_i n_j + B(r) (delta_ij - n_i n_j).
    """

    symmetry = "spherical"

    def __init__(self, r, gamma, Kij, margin=4):
        shape = (3, 3, len(r))
        if np.shape(gamma) != shape or np.shape(Kij) != shape:
            raise ValueError(f"gamma and Kij must have shape {shape}")
        r = np.asarray(r, dtype=float)
        if r[0] < 0:
            raise ValueError("A spherically symmetric radial axis must be nonnegative")
        rows = []
        for tensor in (np.asarray(gamma, dtype=float), np.asarray(Kij, dtype=float)):
            scale = max(np.max(abs(tensor)), 1.e-300)
            off = max(np.max(abs(tensor[0, 1])), np.max(abs(tensor[0, 2])),
                      np.max(abs(tensor[1, 2])), np.max(abs(tensor[1, 1] - tensor[2, 2])))
            if off > 1.e-9*scale:
                raise ValueError("Spherically symmetric input must be diagonal with "
                                 "equal transverse components on the radial axis")
            rows.append(tensor[0, 0])
            rows.append(0.5*(tensor[1, 1] + tensor[2, 2]))
        super().__init__(((r, "r"),), jnp.asarray(np.stack(rows)), margin)
        self.inner_radius = max(INNER_EXCLUSION_CELLS*self.spacing[0],
                                self.bounds[0][0] + self.margin*self.spacing[0])

    def tensors(self, tables, point):
        radius = jnp.sqrt(jnp.sum(point**2))
        normal = point/jnp.where(radius > 0.0, radius, 1.0)
        projector = jnp.eye(3) - normal[:, None]*normal[None, :]
        radial = normal[:, None]*normal[None, :]
        values = interpolate_point(tables, self._index((radius,)))
        return (values[0]*radial + values[1]*projector,
                values[2]*radial + values[3]*projector)

    def resolved(self, point):
        radius = jnp.sqrt(jnp.sum(point**2))
        return self._resolved(self._index((radius,))) & (radius > 0.0)

    def radial_bounds(self, center):
        center = np.asarray(center, dtype=float)
        if not np.allclose(center, 0.0, atol=1.e-12):
            raise ValueError("A spherically symmetric horizon center must be the origin")
        return self.inner_radius, float(self._interior(0)[1])


def build_geometry(coordinates, gamma, Kij, symmetry=None, margin=4):
    """Choose the provider matching the number of coordinate axes supplied."""

    if np.ndim(coordinates) == 1 and not isinstance(coordinates, tuple):
        coordinates = (coordinates,)  # A lone radial axis needs no wrapping.
    builders = {1: SphericalGeometry, 2: AxisymmetricGeometry, 3: CartesianGeometry}
    if len(coordinates) not in builders:
        raise ValueError("Pass (r,), (x, z) or (x, y, z) coordinate axes")
    builder = builders[len(coordinates)]
    if symmetry is not None and symmetry != builder.symmetry:
        raise ValueError(f"{len(coordinates)} coordinate axes describe "
                         f"{builder.symmetry!r} data, not {symmetry!r}")
    return builder(*coordinates, gamma, Kij, margin=margin)
