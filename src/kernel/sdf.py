"""Signed distance fields - implicit modelling for OpenCad.

A mesh describes a solid by its skin.  A field describes it by a function: for
any point in space, how far are you from the surface, and are you inside it?
That single change makes a set of operations trivial that meshes make hard or
impossible:

===================  ==================================  =========================
Operation            Meshes                              Fields
===================  ==================================  =========================
Boolean union        fragile; needs clean, closed input  ``min(a, b)``
Offset / shell       self-intersects on concave regions  ``f - d``
Fillet between parts not expressible without new topology a polynomial smooth-min
Infinite array       one copy per instance               a coordinate ``mod``
Graded lattice       remodel every cell                  a function of position
===================  ==================================  =========================

Conventions
-----------
- A field maps ``(N, 3)`` points to ``(N,)`` values.
- **Negative inside**, zero on the surface, positive outside.
- Distances are in millimetres, matching :mod:`src.kernel.units`.
- A true distance field also satisfies ``|grad f| == 1``.  Not all of these are
  true distance fields, and it matters: ``shell`` and ``offset`` measure
  thickness in field units, so on a pseudo-distance field the wall comes out
  the wrong size.  Every primitive documents which it is, and the TPMS fields
  are explicitly rescaled so their thickness *is* metric.

Fields are resolution-independent.  Nothing is discretised until
:func:`src.kernel.meshing.surface_nets` samples the field, so the same design
can be meshed coarsely for a preview and finely for export.
"""

from __future__ import annotations

import numpy as np

from src.kernel.mesh import axis_index, rotation_matrix

__all__ = [
    "SDF",
    "TPMS_GRADIENT_SCALE",
    "TPMS_KINDS",
    "box",
    "capsule",
    "cone",
    "cylinder",
    "ellipsoid",
    "extrude",
    "hex_prism",
    "plane",
    "revolve",
    "rounded_box",
    "sample_grid",
    "sphere",
    "torus",
    "tpms",
    "tpms_sheet",
    "tpms_solid",
]


def _as_points(points):
    """Normalise input to ``(N, 3)``, remembering whether it was a single point."""
    array = np.asarray(points, dtype=float)
    if array.ndim == 1:
        return array.reshape(1, 3), True
    return array.reshape(-1, 3), False


def _union_bounds(a, b):
    if a is None or b is None:
        return None
    return np.minimum(a[0], b[0]), np.maximum(a[1], b[1])


def _intersect_bounds(a, b):
    # Intersecting with an unbounded field leaves the other field's bounds.
    if a is None:
        return b
    if b is None:
        return a
    low = np.maximum(a[0], b[0])
    high = np.minimum(a[1], b[1])
    return (low, high) if np.all(low <= high) else None


def _expand_bounds(bounds, amount):
    if bounds is None:
        return None
    amount = max(float(amount), 0.0)
    return bounds[0] - amount, bounds[1] + amount


def _transform_bounds(bounds, matrix):
    """Conservative AABB of a transformed box, via its eight corners."""
    if bounds is None:
        return None
    low, high = bounds
    corners = np.array(
        [
            [x, y, z]
            for x in (low[0], high[0])
            for y in (low[1], high[1])
            for z in (low[2], high[2])
        ]
    )
    homogeneous = np.hstack([corners, np.ones((8, 1))])
    moved = (homogeneous @ np.asarray(matrix, dtype=float).T)[:, :3]
    return moved.min(axis=0), moved.max(axis=0)


class SDF:
    """A composable signed distance field.

    Wraps a callable and gives it CSG operators, transforms, and warps.  Every
    method returns a new field; nothing mutates in place, so a field can be
    shared freely between a preview and an export at different resolutions.
    """

    __slots__ = ("_function", "bounds", "exact", "name")

    def __init__(self, function, bounds=None, name="field", exact=True):
        if not callable(function):
            raise TypeError("An SDF needs a callable of shape (N, 3) -> (N,).")
        self._function = function
        self.bounds = bounds
        self.name = str(name)
        #: Whether this is a true distance field (``|grad| == 1``) rather than
        #: a bound.  Shell and offset thickness is only exact when this is True.
        self.exact = bool(exact)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def __call__(self, points):
        query, was_single = _as_points(points)
        if len(query) == 0:
            return np.zeros(0)
        values = np.asarray(self._function(query), dtype=float).ravel()
        if values.shape[0] != query.shape[0]:
            raise ValueError(
                f"Field {self.name!r} returned {values.shape[0]} values for "
                f"{query.shape[0]} points."
            )
        return float(values[0]) if was_single else values

    def gradient(self, points, eps=1e-5):
        """Central-difference gradient - the outward normal for an exact field."""
        query, was_single = _as_points(points)
        grad = np.zeros_like(query)
        for axis in range(3):
            step = np.zeros(3)
            step[axis] = eps
            grad[:, axis] = (
                self._function(query + step) - self._function(query - step)
            ) / (2.0 * eps)
        return grad[0] if was_single else grad

    def normal(self, points, eps=1e-5):
        """Unit surface normal, pointing out of the solid."""
        grad = np.atleast_2d(self.gradient(points, eps))
        lengths = np.linalg.norm(grad, axis=1)
        safe = np.where(lengths > 1e-12, lengths, 1.0)
        normals = grad / safe[:, None]
        return normals.reshape(np.shape(points))

    def project(self, points, iterations=6, eps=1e-5):
        """Newton-project points onto the zero level set.

        Used to sharpen an extracted mesh: a vertex placed by interpolation sits
        near the surface, and a couple of iterations put it on the surface.
        """
        query, was_single = _as_points(points)
        current = query.copy()
        for _ in range(max(int(iterations), 0)):
            values = np.asarray(self._function(current), dtype=float).ravel()
            grad = np.atleast_2d(self.gradient(current, eps))
            norm_sq = np.einsum("ij,ij->i", grad, grad)
            safe = np.where(norm_sq > 1e-18, norm_sq, 1.0)
            current = current - grad * (values / safe)[:, None]
        return current[0] if was_single else current

    def contains(self, points):
        """Boolean mask of points strictly inside the solid."""
        return np.atleast_1d(self(points)) < 0.0

    # ------------------------------------------------------------------
    # Boolean composition
    # ------------------------------------------------------------------
    def union(self, *others):
        fields = [self, *(_coerce(other) for other in others)]
        bounds = fields[0].bounds
        for field in fields[1:]:
            bounds = _union_bounds(bounds, field.bounds)
        functions = [field._function for field in fields]

        def evaluate(points):
            value = functions[0](points)
            for function in functions[1:]:
                value = np.minimum(value, function(points))
            return value

        return SDF(evaluate, bounds, "union", all(field.exact for field in fields))

    def intersection(self, *others):
        fields = [self, *(_coerce(other) for other in others)]
        bounds = fields[0].bounds
        for field in fields[1:]:
            bounds = _intersect_bounds(bounds, field.bounds)
        functions = [field._function for field in fields]

        def evaluate(points):
            value = functions[0](points)
            for function in functions[1:]:
                value = np.maximum(value, function(points))
            return value

        # max() of two exact fields is only a bound near concave corners.
        return SDF(evaluate, bounds, "intersection", False)

    def difference(self, *others):
        fields = [_coerce(other) for other in others]
        functions = [field._function for field in fields]
        mine = self._function

        def evaluate(points):
            value = mine(points)
            for function in functions:
                value = np.maximum(value, -function(points))
            return value

        # Subtracting can only shrink the solid, so our bounds still hold.
        return SDF(evaluate, self.bounds, "difference", False)

    def __or__(self, other):
        return self.union(other)

    def __and__(self, other):
        return self.intersection(other)

    def __sub__(self, other):
        return self.difference(other)

    def __invert__(self):
        """Swap inside and outside; the result is unbounded."""
        function = self._function
        return SDF(lambda points: -function(points), None, "complement", self.exact)

    # ------------------------------------------------------------------
    # Smooth composition - fillets without topology
    # ------------------------------------------------------------------
    def smooth_union(self, other, k=1.0):
        """Union with a fillet of roughly radius ``k`` along the seam.

        ``k <= 0`` degrades exactly to a sharp union.  The result is never
        greater than the sharp union, which is what makes the blend additive.
        """
        other = _coerce(other)
        if k <= 0:
            return self.union(other)
        a, b = self._function, other._function
        radius = float(k)

        def evaluate(points):
            da, db = a(points), b(points)
            h = np.clip(0.5 + 0.5 * (db - da) / radius, 0.0, 1.0)
            return db + (da - db) * h - radius * h * (1.0 - h)

        bounds = _union_bounds(self.bounds, other.bounds)
        return SDF(evaluate, bounds, "smooth_union", False)

    def smooth_intersection(self, other, k=1.0):
        other = _coerce(other)
        if k <= 0:
            return self.intersection(other)
        a, b = self._function, other._function
        radius = float(k)

        def evaluate(points):
            da, db = a(points), b(points)
            h = np.clip(0.5 - 0.5 * (db - da) / radius, 0.0, 1.0)
            return db + (da - db) * h + radius * h * (1.0 - h)

        bounds = _intersect_bounds(self.bounds, other.bounds)
        return SDF(evaluate, bounds, "smooth_intersection", False)

    def smooth_difference(self, other, k=1.0):
        """Subtract ``other`` leaving a fillet - the usual way to soften a cut."""
        other = _coerce(other)
        if k <= 0:
            return self.difference(other)
        return self.smooth_intersection(~other, k)

    def blend(self, other, t=0.5):
        """Linearly morph between two fields; ``t=0`` is self, ``t=1`` is other."""
        other = _coerce(other)
        a, b = self._function, other._function
        amount = float(t)

        def evaluate(points):
            return a(points) * (1.0 - amount) + b(points) * amount

        return SDF(evaluate, _union_bounds(self.bounds, other.bounds), "blend", False)

    # ------------------------------------------------------------------
    # Thickness
    # ------------------------------------------------------------------
    def offset(self, distance):
        """Grow (positive) or shrink (negative) the solid by a real distance.

        This is the operation meshes cannot do reliably: on a field it is a
        subtraction, and it never self-intersects.
        """
        function = self._function
        amount = float(distance)
        return SDF(
            lambda points: function(points) - amount,
            _expand_bounds(self.bounds, amount),
            "offset",
            self.exact,
        )

    def shell(self, thickness):
        """Hollow the solid, leaving a wall of ``thickness`` centred on the surface."""
        function = self._function
        half = float(thickness) / 2.0
        return SDF(
            lambda points: np.abs(function(points)) - half,
            _expand_bounds(self.bounds, half),
            "shell",
            self.exact,
        )

    # Note: there is deliberately no generic ``rounded()``.  On a distance field
    # ``offset(-r).offset(r)`` is the identity, so edge rounding cannot be added
    # after the fact - it has to be built into the primitive (see
    # :func:`rounded_box`) or produced by :meth:`smooth_union`.

    # ------------------------------------------------------------------
    # Rigid transforms - implemented by moving the query point, not the field
    # ------------------------------------------------------------------
    def translate(self, offset):
        function = self._function
        shift = np.asarray(offset, dtype=float).reshape(3)
        bounds = None if self.bounds is None else (self.bounds[0] + shift, self.bounds[1] + shift)
        return SDF(lambda points: function(points - shift), bounds, "translate", self.exact)

    def scale(self, factor):
        """Uniform scale. Non-uniform scaling would break the distance metric."""
        function = self._function
        amount = float(factor)
        if amount == 0.0:
            raise ValueError("Scale factor must be non-zero.")
        bounds = None if self.bounds is None else (self.bounds[0] * amount, self.bounds[1] * amount)
        if amount < 0:
            bounds = None if bounds is None else (bounds[1], bounds[0])
        return SDF(
            lambda points: function(points / amount) * abs(amount),
            bounds,
            "scale",
            self.exact,
        )

    def rotate(self, angle_degrees, axis="z"):
        function = self._function
        matrix = rotation_matrix(angle_degrees, axis)
        inverse = matrix.T  # rotations are orthonormal
        full = np.eye(4)
        full[:3, :3] = matrix
        return SDF(
            lambda points: function(points @ inverse.T),
            _transform_bounds(self.bounds, full),
            "rotate",
            self.exact,
        )

    def transform(self, matrix):
        """Apply a 4x4 transform.

        Exact only for rigid motions and uniform scale; a shear or non-uniform
        scale leaves a field that still has the right zero level set but no
        longer measures true distance, so ``exact`` is cleared.
        """
        matrix = np.asarray(matrix, dtype=float).reshape(4, 4)
        inverse = np.linalg.inv(matrix)
        function = self._function

        linear = matrix[:3, :3]
        singular = np.linalg.svd(linear, compute_uv=False)
        is_similarity = bool(np.allclose(singular, singular[0]))
        # A similarity scales distance uniformly; anything else does not.
        factor = float(singular[0]) if is_similarity else 1.0

        def evaluate(points):
            homogeneous = np.hstack([points, np.ones((len(points), 1))])
            local = (homogeneous @ inverse.T)[:, :3]
            return function(local) * factor

        return SDF(
            evaluate,
            _transform_bounds(self.bounds, matrix),
            "transform",
            self.exact and is_similarity,
        )

    # ------------------------------------------------------------------
    # Warps - cheap shape variation that meshes would need remodelling for
    # ------------------------------------------------------------------
    def twist(self, turns_per_unit, axis="z"):
        """Twist about an axis; ``turns_per_unit`` full turns per millimetre."""
        function = self._function
        index = axis_index(axis)
        rate = float(turns_per_unit) * 2.0 * np.pi
        others = [i for i in range(3) if i != index]

        def evaluate(points):
            angle = -rate * points[:, index]
            cos, sin = np.cos(angle), np.sin(angle)
            local = points.copy()
            u, v = points[:, others[0]], points[:, others[1]]
            local[:, others[0]] = cos * u - sin * v
            local[:, others[1]] = sin * u + cos * v
            return function(local)

        # A twist keeps the axial extent but sweeps the cross-section around,
        # so the safe bound is the circumscribing cylinder.
        bounds = None
        if self.bounds is not None:
            low, high = self.bounds
            radius = float(np.max(np.abs(np.stack([low[others], high[others]]))))
            bounds = (low.copy(), high.copy())
            bounds[0][others] = -radius
            bounds[1][others] = radius
        return SDF(evaluate, bounds, "twist", False)

    def bend(self, curvature, axis="z"):
        """Bend the solid, curving the given axis by ``curvature`` per millimetre."""
        function = self._function
        index = axis_index(axis)
        rate = float(curvature)
        others = [i for i in range(3) if i != index]

        def evaluate(points):
            angle = rate * points[:, others[0]]
            cos, sin = np.cos(angle), np.sin(angle)
            local = points.copy()
            u = points[:, others[0]]
            w = points[:, index]
            local[:, others[0]] = cos * u - sin * w
            local[:, index] = sin * u + cos * w
            return function(local)

        return SDF(evaluate, None, "bend", False)

    def taper(self, factor, axis="z"):
        """Scale the cross-section along an axis - 1.0 keeps it, 0.0 is a point."""
        function = self._function
        index = axis_index(axis)
        amount = float(factor)
        others = [i for i in range(3) if i != index]

        def evaluate(points):
            span = points[:, index]
            scale = 1.0 + (amount - 1.0) * span
            scale = np.where(np.abs(scale) > 1e-9, scale, 1e-9)
            local = points.copy()
            local[:, others] = points[:, others] / scale[:, None]
            return function(local) * np.minimum(np.abs(scale), 1.0)

        return SDF(evaluate, None, "taper", False)

    def elongate(self, extents):
        """Stretch the solid by sliding its surface apart - rounded ends stay round."""
        function = self._function
        half = np.abs(np.asarray(extents, dtype=float).reshape(3))

        def evaluate(points):
            return function(points - np.clip(points, -half, half))

        return SDF(evaluate, _expand_bounds(self.bounds, float(half.max())), "elongate", self.exact)

    def repeat(self, spacing, count=None):
        """Tile the field on a grid, infinitely or a bounded number of times.

        An infinite array costs nothing extra to evaluate, which is the point:
        a thousand instances is the same field as one.
        """
        function = self._function
        period = np.asarray(spacing, dtype=float).reshape(3)
        if np.any(period <= 0):
            raise ValueError("Repeat spacing must be positive in every axis.")

        if count is None:
            def evaluate(points):
                return function(points - period * np.round(points / period))

            return SDF(evaluate, None, "repeat", self.exact)

        limit = np.abs(np.asarray(count, dtype=float).reshape(3))

        def evaluate(points):
            index = np.clip(np.round(points / period), -limit, limit)
            return function(points - period * index)

        bounds = None
        if self.bounds is not None:
            extent = period * limit
            bounds = (self.bounds[0] - extent, self.bounds[1] + extent)
        return SDF(evaluate, bounds, "repeat", self.exact)

    def bounded(self, low, high):
        """Attach explicit bounds, for fields whose extent cannot be inferred."""
        return SDF(
            self._function,
            (np.asarray(low, dtype=float).reshape(3), np.asarray(high, dtype=float).reshape(3)),
            self.name,
            self.exact,
        )

    def sample_bounds(self, margin=0.05):
        """The box to mesh over: the field's own bounds plus a small margin."""
        if self.bounds is None:
            raise ValueError(
                f"Field {self.name!r} has no known extent - pass explicit bounds, "
                "or call .bounded(low, high) to declare them."
            )
        low, high = self.bounds
        pad = np.maximum((high - low) * float(margin), 1e-6)
        return low - pad, high + pad

    def __repr__(self):
        extent = "unbounded" if self.bounds is None else "bounded"
        kind = "exact" if self.exact else "approximate"
        return f"SDF({self.name!r}, {extent}, {kind})"


def _coerce(value):
    """Accept a field, a bare callable, or a constant everywhere."""
    if isinstance(value, SDF):
        return value
    if callable(value):
        return SDF(value)
    constant = float(value)
    return SDF(lambda points: np.full(len(points), constant), None, "constant")


# ----------------------------------------------------------------------
# Primitives
# ----------------------------------------------------------------------
def sphere(radius=1.0, center=(0.0, 0.0, 0.0)):
    """Exact distance field for a sphere."""
    radius = float(radius)
    if radius <= 0:
        raise ValueError("Sphere radius must be positive.")
    origin = np.asarray(center, dtype=float).reshape(3)

    def evaluate(points):
        return np.linalg.norm(points - origin, axis=1) - radius

    return SDF(evaluate, (origin - radius, origin + radius), "sphere")


def box(size=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0)):
    """Exact distance field for an axis-aligned box, inside and out."""
    half = np.abs(np.broadcast_to(np.asarray(size, dtype=float), (3,))) / 2.0
    if np.any(half <= 0):
        raise ValueError("Box size must be positive in every axis.")
    origin = np.asarray(center, dtype=float).reshape(3)

    def evaluate(points):
        q = np.abs(points - origin) - half
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(q.max(axis=1), 0.0)
        return outside + inside

    return SDF(evaluate, (origin - half, origin + half), "box")


def rounded_box(size=(1.0, 1.0, 1.0), radius=0.1, center=(0.0, 0.0, 0.0)):
    """A box with rounded edges; ``size`` is the overall size including the radius."""
    radius = float(radius)
    full = np.abs(np.broadcast_to(np.asarray(size, dtype=float), (3,)))
    if radius < 0:
        raise ValueError("Round radius cannot be negative.")
    if np.any(full <= 2 * radius):
        raise ValueError("Round radius is too large for the box size.")
    inner = box(full - 2 * radius, center)
    return SDF(
        inner.offset(radius)._function,
        (np.asarray(center, dtype=float) - full / 2, np.asarray(center, dtype=float) + full / 2),
        "rounded_box",
    )


def cylinder(radius=0.5, height=1.0, center=(0.0, 0.0, 0.0), axis="z"):
    """Exact distance field for a capped cylinder."""
    radius, height = float(radius), float(height)
    if radius <= 0 or height <= 0:
        raise ValueError("Cylinder radius and height must be positive.")
    origin = np.asarray(center, dtype=float).reshape(3)
    index = axis_index(axis)
    others = [i for i in range(3) if i != index]
    half = height / 2.0

    def evaluate(points):
        local = points - origin
        radial = np.linalg.norm(local[:, others], axis=1) - radius
        axial = np.abs(local[:, index]) - half
        outside = np.linalg.norm(
            np.stack([np.maximum(radial, 0.0), np.maximum(axial, 0.0)], axis=1), axis=1
        )
        inside = np.minimum(np.maximum(radial, axial), 0.0)
        return outside + inside

    extent = np.empty(3)
    extent[others] = radius
    extent[index] = half
    return SDF(evaluate, (origin - extent, origin + extent), "cylinder")


def capsule(start=(0.0, 0.0, -0.5), end=(0.0, 0.0, 0.5), radius=0.5):
    """Exact distance field for a line segment swept by a sphere."""
    radius = float(radius)
    if radius <= 0:
        raise ValueError("Capsule radius must be positive.")
    a = np.asarray(start, dtype=float).reshape(3)
    b = np.asarray(end, dtype=float).reshape(3)
    axis = b - a
    length_sq = float(axis @ axis)

    def evaluate(points):
        offset = points - a
        if length_sq <= 1e-18:
            return np.linalg.norm(offset, axis=1) - radius
        t = np.clip((offset @ axis) / length_sq, 0.0, 1.0)
        closest = a + t[:, None] * axis
        return np.linalg.norm(points - closest, axis=1) - radius

    low = np.minimum(a, b) - radius
    high = np.maximum(a, b) + radius
    return SDF(evaluate, (low, high), "capsule")


def cone(radius=0.5, height=1.0, center=(0.0, 0.0, 0.0)):
    """A cone standing on Z, base radius ``radius``, apex ``height`` above it."""
    radius, height = float(radius), float(height)
    if radius <= 0 or height <= 0:
        raise ValueError("Cone radius and height must be positive.")
    half = height / 2.0
    # Profile in the (radial, axial) half-plane: base corner, rim, apex.
    profile = np.array([[0.0, -half], [radius, -half], [0.0, half]])
    field = revolve(profile).translate(center)
    field.name = "cone"
    origin = np.asarray(center, dtype=float).reshape(3)
    field.bounds = (
        origin - np.array([radius, radius, half]),
        origin + np.array([radius, radius, half]),
    )
    return field


def torus(major_radius=1.0, minor_radius=0.25, center=(0.0, 0.0, 0.0), axis="z"):
    """Exact distance field for a torus."""
    major, minor = float(major_radius), float(minor_radius)
    if major <= 0 or minor <= 0:
        raise ValueError("Torus radii must be positive.")
    origin = np.asarray(center, dtype=float).reshape(3)
    index = axis_index(axis)
    others = [i for i in range(3) if i != index]

    def evaluate(points):
        local = points - origin
        radial = np.linalg.norm(local[:, others], axis=1) - major
        return np.sqrt(radial**2 + local[:, index] ** 2) - minor

    extent = np.empty(3)
    extent[others] = major + minor
    extent[index] = minor
    return SDF(evaluate, (origin - extent, origin + extent), "torus")


def plane(normal=(0.0, 0.0, 1.0), offset=0.0):
    """A half-space. Solid on the side the normal points away from."""
    direction = np.asarray(normal, dtype=float).reshape(3)
    length = np.linalg.norm(direction)
    if length <= 1e-12:
        raise ValueError("Plane normal must be non-zero.")
    direction = direction / length
    distance = float(offset)

    def evaluate(points):
        return points @ direction - distance

    return SDF(evaluate, None, "plane")


def ellipsoid(radii=(1.0, 0.75, 0.5), center=(0.0, 0.0, 0.0)):
    """An ellipsoid.

    No closed-form distance exists, so this is the standard scaled bound: the
    zero level set is exact, but values away from the surface underestimate the
    true distance.  Offsetting an ellipsoid therefore does not give an exact
    offset surface.
    """
    extent = np.abs(np.asarray(radii, dtype=float).reshape(3))
    if np.any(extent <= 0):
        raise ValueError("Ellipsoid radii must be positive.")
    origin = np.asarray(center, dtype=float).reshape(3)

    def evaluate(points):
        local = (points - origin) / extent
        k0 = np.linalg.norm(local, axis=1)
        k1 = np.linalg.norm(local / extent, axis=1)
        safe = np.where(k1 > 1e-12, k1, 1.0)
        return np.where(k0 > 1e-12, k0 * (k0 - 1.0) / safe, -float(extent.min()))

    return SDF(evaluate, (origin - extent, origin + extent), "ellipsoid", exact=False)


def _polygon_distance_2d(query, polygon, distance_edges=None):
    """Signed distance from 2D points to a closed polygon; negative inside.

    ``distance_edges`` optionally masks which edges contribute to the *distance*.
    The full loop is always used for the inside test.  Revolved profiles need
    this: the segment that closes a profile along the axis of revolution is
    needed to make the polygon closed, but it sweeps out no surface, so
    measuring distance to it would report zero at the centre of a solid.
    """
    polygon = np.asarray(polygon, dtype=float).reshape(-1, 2)
    following = np.roll(polygon, -1, axis=0)

    edge = following - polygon                       # (C, 2)
    to_point = query[:, None, :] - polygon[None, :, :]  # (N, C, 2)
    length_sq = np.einsum("cj,cj->c", edge, edge)
    length_sq = np.where(length_sq > 1e-18, length_sq, 1.0)
    t = np.clip(np.einsum("ncj,cj->nc", to_point, edge) / length_sq, 0.0, 1.0)
    closest = to_point - t[:, :, None] * edge[None, :, :]
    per_edge = np.sqrt(np.einsum("ncj,ncj->nc", closest, closest))
    if distance_edges is not None:
        mask = np.asarray(distance_edges, dtype=bool)
        if not mask.any():
            raise ValueError("A profile needs at least one edge off the axis.")
        per_edge = np.where(mask[None, :], per_edge, np.inf)
    distance = per_edge.min(axis=1)

    # Even-odd crossing test for the sign.
    y0 = polygon[None, :, 1]
    y1 = following[None, :, 1]
    py = query[:, None, 1]
    px = query[:, None, 0]
    straddles = (y0 > py) != (y1 > py)
    denominator = np.where(np.abs(y1 - y0) > 1e-18, y1 - y0, 1.0)
    crossing_x = polygon[None, :, 0] + (py - y0) * (following[None, :, 0] - polygon[None, :, 0]) / denominator
    inside = np.sum(straddles & (px < crossing_x), axis=1) % 2 == 1
    return np.where(inside, -distance, distance)


def extrude(polygon, height=1.0, center=(0.0, 0.0, 0.0)):
    """Extrude a closed 2D polygon along Z into a solid.

    This is the bridge from a sketch to an implicit solid: the profile keeps its
    exact edges, and the result composes with every other field operation.
    """
    profile = np.asarray(polygon, dtype=float).reshape(-1, 2)
    if len(profile) < 3:
        raise ValueError("An extrusion profile needs at least three points.")
    half = float(height) / 2.0
    if half <= 0:
        raise ValueError("Extrusion height must be positive.")
    origin = np.asarray(center, dtype=float).reshape(3)

    def evaluate(points):
        local = points - origin
        planar = _polygon_distance_2d(local[:, :2], profile)
        axial = np.abs(local[:, 2]) - half
        outside = np.linalg.norm(
            np.stack([np.maximum(planar, 0.0), np.maximum(axial, 0.0)], axis=1), axis=1
        )
        inside = np.minimum(np.maximum(planar, axial), 0.0)
        return outside + inside

    low = np.array([profile[:, 0].min(), profile[:, 1].min(), -half]) + origin
    high = np.array([profile[:, 0].max(), profile[:, 1].max(), half]) + origin
    return SDF(evaluate, (low, high), "extrude")


def revolve(profile, axis="z"):
    """Revolve a 2D profile around an axis - a lathe operation.

    ``profile`` is given in the ``(radial, axial)`` half-plane with radial >= 0.
    """
    points_2d = np.asarray(profile, dtype=float).reshape(-1, 2)
    if len(points_2d) < 3:
        raise ValueError("A revolve profile needs at least three points.")
    index = axis_index(axis)
    others = [i for i in range(3) if i != index]

    # An edge with both ends on the axis closes the profile but sweeps out no
    # surface, so it must not contribute to the distance.
    following = np.roll(points_2d, -1, axis=0)
    on_axis = (np.abs(points_2d[:, 0]) < 1e-12) & (np.abs(following[:, 0]) < 1e-12)
    contributing = ~on_axis
    if not contributing.any():
        raise ValueError("A revolve profile must have at least one edge off the axis.")

    def evaluate(points):
        radial = np.linalg.norm(points[:, others], axis=1)
        axial = points[:, index]
        return _polygon_distance_2d(
            np.stack([radial, axial], axis=1), points_2d, contributing
        )

    radius = float(np.abs(points_2d[:, 0]).max())
    low = np.empty(3)
    high = np.empty(3)
    low[others] = -radius
    high[others] = radius
    low[index] = points_2d[:, 1].min()
    high[index] = points_2d[:, 1].max()
    return SDF(evaluate, (low, high), "revolve")


def hex_prism(radius=0.5, height=1.0, center=(0.0, 0.0, 0.0)):
    """A regular hexagonal prism - the usual infill and honeycomb cell."""
    radius = float(radius)
    if radius <= 0:
        raise ValueError("Hex prism radius must be positive.")
    angles = np.deg2rad(np.arange(6) * 60.0)
    profile = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    field = extrude(profile, height, center)
    field.name = "hex_prism"
    return field


# ----------------------------------------------------------------------
# Triply periodic minimal surfaces
# ----------------------------------------------------------------------
TPMS_KINDS = {
    "gyroid": "Gyroid",
    "schwarz_p": "Schwarz P",
    "diamond": "Diamond (Schwarz D)",
    "neovius": "Neovius",
    "iwp": "Schoen I-WP",
    "lidinoid": "Lidinoid",
    "split_p": "Split P",
}

#: Mean ``|grad F|`` on each field's zero level set, measured numerically over a
#: full period.  Dividing the raw field by this turns it into an approximate
#: distance, which is what lets lattice wall thickness be given in millimetres
#: instead of meaningless field units.
#:
#: The conversion is approximate because ``|grad F|`` is not constant over the
#: surface.  Measured against a requested 1 mm wall, the median error is about
#: 2-3% for gyroid, diamond and I-WP, 6% for Schwarz P and Split P, and around
#: 17% for Neovius and Lidinoid, whose gradients vary most.  Check a thin-walled
#: lattice against its intended wall before committing it to a print.
TPMS_GRADIENT_SCALE = {
    "gyroid": 1.5318,
    "schwarz_p": 1.3116,
    "diamond": 1.4974,
    "neovius": 2.4640,
    "iwp": 4.3002,
    "lidinoid": 1.2721,
    "split_p": 2.2576,
}


def normalize_tpms_kind(kind):
    """Accept 'Schwarz P', 'schwarz-p', 'primitive', and friends."""
    normalized = str(kind).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "schwarz": "schwarz_p",
        "schwarzp": "schwarz_p",
        "primitive": "schwarz_p",
        "p": "schwarz_p",
        "schwarz_d": "diamond",
        "d": "diamond",
        "g": "gyroid",
        "i_wp": "iwp",
        "iwp_schoen": "iwp",
        "schoen_iwp": "iwp",
        "split": "split_p",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in TPMS_KINDS:
        supported = ", ".join(TPMS_KINDS.values())
        raise ValueError(f"Unsupported TPMS type {kind!r}. Supported: {supported}.")
    return normalized


def _tpms_field(kind, u, v, w):
    if kind == "gyroid":
        return np.sin(u) * np.cos(v) + np.sin(v) * np.cos(w) + np.sin(w) * np.cos(u)
    if kind == "schwarz_p":
        return np.cos(u) + np.cos(v) + np.cos(w)
    if kind == "diamond":
        return (
            np.sin(u) * np.sin(v) * np.sin(w)
            + np.sin(u) * np.cos(v) * np.cos(w)
            + np.cos(u) * np.sin(v) * np.cos(w)
            + np.cos(u) * np.cos(v) * np.sin(w)
        )
    if kind == "neovius":
        return 3 * (np.cos(u) + np.cos(v) + np.cos(w)) + 4 * np.cos(u) * np.cos(v) * np.cos(w)
    if kind == "iwp":
        return 2 * (
            np.cos(u) * np.cos(v) + np.cos(v) * np.cos(w) + np.cos(w) * np.cos(u)
        ) - (np.cos(2 * u) + np.cos(2 * v) + np.cos(2 * w))
    if kind == "lidinoid":
        return (
            0.5
            * (
                np.sin(2 * u) * np.cos(v) * np.sin(w)
                + np.sin(2 * v) * np.cos(w) * np.sin(u)
                + np.sin(2 * w) * np.cos(u) * np.sin(v)
            )
            - 0.5
            * (
                np.cos(2 * u) * np.cos(2 * v)
                + np.cos(2 * v) * np.cos(2 * w)
                + np.cos(2 * w) * np.cos(2 * u)
            )
            + 0.15
        )
    if kind == "split_p":
        return (
            1.1
            * (
                np.sin(2 * u) * np.sin(w) * np.cos(v)
                + np.sin(2 * v) * np.sin(u) * np.cos(w)
                + np.sin(2 * w) * np.sin(v) * np.cos(u)
            )
            - 0.2
            * (
                np.cos(2 * u) * np.cos(2 * v)
                + np.cos(2 * v) * np.cos(2 * w)
                + np.cos(2 * w) * np.cos(2 * u)
            )
            - 0.4 * (np.cos(2 * u) + np.cos(2 * v) + np.cos(2 * w))
        )
    raise ValueError(f"Unsupported TPMS type {kind!r}.")


def _as_spatial_function(value, name):
    """Accept a constant or a callable of position, and always return a callable.

    This is what makes lattices *graded*: pass a number for a uniform lattice,
    or a function of position to vary cell size or wall thickness through the
    part - denser where it is loaded, lighter where it is not.
    """
    if callable(value):
        return value
    constant = float(value)
    if constant <= 0:
        raise ValueError(f"{name} must be positive.")
    return lambda points: np.full(len(points), constant)


def tpms(kind="gyroid", period=10.0, phase=0.0):
    """The raw periodic field for a triply periodic minimal surface.

    The value is *not* a distance.  Use :func:`tpms_sheet` or :func:`tpms_solid`
    unless you specifically want the underlying implicit function.
    """
    kind = normalize_tpms_kind(kind)
    period_of = _as_spatial_function(period, "TPMS period")
    offset = float(phase)

    def evaluate(points):
        scale = 2.0 * np.pi / np.asarray(period_of(points), dtype=float).ravel()
        u = points[:, 0] * scale
        v = points[:, 1] * scale
        w = points[:, 2] * scale
        return _tpms_field(kind, u, v, w) - offset

    return SDF(evaluate, None, f"tpms:{kind}", exact=False)


def _tpms_pseudo_distance(kind, period_of, phase):
    """Rescale the periodic field into an approximate millimetre distance."""
    gradient_scale = TPMS_GRADIENT_SCALE[kind]
    offset = float(phase)

    def evaluate(points):
        period = np.asarray(period_of(points), dtype=float).ravel()
        scale = 2.0 * np.pi / period
        raw = _tpms_field(kind, points[:, 0] * scale, points[:, 1] * scale, points[:, 2] * scale)
        # |grad| in world units is (2 pi / period) * gradient_scale, so dividing
        # the field by it converts a field value into a length.
        return (raw - offset) * period / (2.0 * np.pi * gradient_scale)

    return evaluate


def tpms_sheet(kind="gyroid", period=10.0, thickness=1.0, phase=0.0):
    """A TPMS *sheet* lattice - a wall of real thickness following the surface.

    ``period`` and ``thickness`` each accept a number for a uniform lattice, or a
    callable ``f(points) -> (N,)`` for a functionally graded one.  Grading is the
    interesting case: stiffness can follow load, and density can follow a
    stress field, without modelling a single cell by hand.

    The result is unbounded by construction; intersect it with a solid to get a
    finite part, which is the normal way to fill a shape with lattice::

        part = sdf.sphere(20) & sdf.tpms_sheet("gyroid", period=6, thickness=0.8)

    Thickness is accurate to roughly 10-15% - see :data:`TPMS_GRADIENT_SCALE`.
    """
    kind = normalize_tpms_kind(kind)
    period_of = _as_spatial_function(period, "TPMS period")
    thickness_of = _as_spatial_function(thickness, "TPMS thickness")
    distance = _tpms_pseudo_distance(kind, period_of, phase)

    def evaluate(points):
        wall = np.asarray(thickness_of(points), dtype=float).ravel()
        return np.abs(distance(points)) - wall / 2.0

    return SDF(evaluate, None, f"tpms_sheet:{kind}", exact=False)


def tpms_solid(kind="gyroid", period=10.0, level=0.0):
    """A TPMS *network* lattice - the solid on one side of the surface.

    ``level`` shifts the surface, trading strut thickness against porosity: 0 is
    the balanced minimal surface, positive values thicken the solid phase.
    """
    kind = normalize_tpms_kind(kind)
    period_of = _as_spatial_function(period, "TPMS period")
    level_of = level if callable(level) else (lambda points, value=float(level): np.full(len(points), value))
    distance = _tpms_pseudo_distance(kind, period_of, 0.0)

    def evaluate(points):
        return distance(points) - np.asarray(level_of(points), dtype=float).ravel()

    return SDF(evaluate, None, f"tpms_solid:{kind}", exact=False)


# ----------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------
def sample_grid(field, bounds=None, resolution=64, chunk=1 << 20):
    """Evaluate a field on a regular grid.

    Returns ``(values, origin, spacing)`` where ``values`` is indexed
    ``[i, j, k]`` over x, y, z.  Evaluation is chunked so a 256^3 grid does not
    build a 16 million by 3 point array in one allocation.
    """
    field = _coerce(field)
    if bounds is None:
        low, high = field.sample_bounds()
    else:
        low = np.asarray(bounds[0], dtype=float).reshape(3)
        high = np.asarray(bounds[1], dtype=float).reshape(3)
    if np.any(high <= low):
        raise ValueError("Sampling bounds must have positive extent in every axis.")

    counts = np.broadcast_to(np.asarray(resolution, dtype=int), (3,)).astype(int)
    if np.any(counts < 2):
        raise ValueError("Sampling resolution must be at least 2 in every axis.")

    axes = [np.linspace(low[i], high[i], counts[i]) for i in range(3)]
    spacing = np.array([(high[i] - low[i]) / (counts[i] - 1) for i in range(3)])

    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    flat = grid.reshape(-1, 3)
    values = np.empty(len(flat))

    step = max(int(chunk), 1024)
    for start in range(0, len(flat), step):
        stop = min(start + step, len(flat))
        values[start:stop] = np.asarray(field(flat[start:stop]), dtype=float).ravel()

    return values.reshape(tuple(counts)), low, spacing
