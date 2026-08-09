"""Bounding volume hierarchy over triangles, and the spatial queries built on it.

Almost everything non-trivial the kernel wants to do with an arbitrary mesh
reduces to one of four questions: where does this ray hit the surface, what is
the nearest point on the surface, is this point inside the solid, and do these
two shells overlap.  Answering any of them by scanning every triangle is O(M)
per query, which is already painful on a 20k-triangle import and hopeless once
booleans, wall-thickness sampling, and vertex snapping start issuing millions of
them.  This module builds the acceleration structure once and answers all four.

Building
--------
The tree is built top-down over triangle centroids.  At each node the triangles
are sorted along the axis with the largest centroid spread and an exact surface
area heuristic (SAH) sweep picks the split, so build cost is O(M log^2 M) rather
than the O(M log M) of a binned approximation.  The sort is the price of not
having to tune a bin count, and it is paid once per mesh; ``split_method
="median"`` drops back to the cheaper balanced split.

Traversal
---------
Traversal deliberately avoids a Python loop per query.  Every query keeps a
*wavefront*: a flat array of ``(query, node)`` pairs that are all tested with a
single numpy expression, then split into leaves to evaluate and internal nodes
to expand.  Each traversal step is therefore O(front size) in numpy instead of
O(1) in Python, at the cost of visiting some nodes a strictly best-first
traversal would have pruned, and of holding the front in memory.  For batched
queries - which is how the kernel uses this - that trade is strongly favourable;
for a single ray it is merely adequate.

Node data lives in flat arrays (``node_min``, ``node_max``, ``node_left``, ...)
rather than one Python object per node, so a wavefront can index them directly.

Where speed is traded for simplicity: candidate ``(query, triangle)`` pairs are
materialised before the exact test runs, so a query that touches a very large
number of triangles costs memory proportional to that count;
:meth:`BVH.winding_number` ignores the hierarchy entirely and is O(P * M), which
is why it is only ever used as a fallback for a handful of points.
"""

from __future__ import annotations

import numpy as np

from src.kernel.mesh import EPS, Mesh

__all__ = [
    "BVH",
    "MeshField",
    "RayHits",
    "closest_point_on_triangles",
    "mesh_sdf",
    "moller_trumbore",
    "triangle_triangle_intersect",
]

# A deliberately "generic" unit direction: no zero components, no two components
# equal, and not parallel to any face or edge of an axis-aligned box.  Parity ray
# casting from this direction almost never lands on a shared edge or vertex.
_PROBE_DIRECTION = np.array([0.3145678, 0.5231419, 0.7918273])

# Seed is fixed so a re-cast produces the same answer on every run; a containment
# test that flickers between calls would be worse than one that is merely slow.
_JITTER_SEED = 0xC0FFEE


def _dot(a, b):
    """Row-wise dot product of two ``(K, 3)`` arrays."""
    return np.einsum("ij,ij->i", a, b)


def _half_area(extents):
    """Half the surface area of boxes with the given ``(..., 3)`` extents."""
    e = np.maximum(extents, 0.0)
    return e[..., 0] * e[..., 1] + e[..., 1] * e[..., 2] + e[..., 2] * e[..., 0]


def _as_points(points):
    """Coerce a point or point cloud to a contiguous ``(P, 3)`` float array."""
    array = np.asarray(points, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("Points must have shape (P, 3).")
    return np.ascontiguousarray(array)


def _as_rays(origins, directions):
    """Broadcast origins/directions to matching ``(R, 3)`` arrays, unit length.

    Directions are normalised so every distance this module reports is a true
    Euclidean distance rather than a parameter in units of the input vector.
    """
    o = np.asarray(origins, dtype=float)
    d = np.asarray(directions, dtype=float)
    if o.ndim == 1:
        o = o.reshape(1, -1)
    if d.ndim == 1:
        d = d.reshape(1, -1)
    if o.ndim != 2 or o.shape[1] != 3 or d.ndim != 2 or d.shape[1] != 3:
        raise ValueError("Ray origins and directions must have shape (R, 3).")
    o, d = np.broadcast_arrays(o, d)
    o = np.ascontiguousarray(o, dtype=float)
    d = np.ascontiguousarray(d, dtype=float)
    lengths = np.linalg.norm(d, axis=1)
    if len(lengths) and np.any(lengths <= EPS):
        raise ValueError("Ray directions must be non-zero.")
    if len(lengths):
        d = d / lengths[:, None]
    return o, d


# ----------------------------------------------------------------------
# Point / triangle and ray / triangle primitives
# ----------------------------------------------------------------------
def _closest_point_on_segments(points, p0, p1):
    """Closest point on each segment ``p0 -> p1`` to the matching query point."""
    d = p1 - p0
    dd = _dot(d, d)
    safe = np.where(dd > 0.0, dd, 1.0)
    t = np.clip(_dot(points - p0, d) / safe, 0.0, 1.0)
    t = np.where(dd > 0.0, t, 0.0)
    return p0 + t[:, None] * d


def closest_point_on_triangles(points, triangles):
    """Closest point on each triangle to the matching query point.

    ``points`` is ``(K, 3)`` and ``triangles`` is ``(K, 3, 3)``; the two are
    paired row by row.  This is Ericson's seven-region Voronoi test - three
    vertex regions, three edge regions, and the face interior - not a projection
    onto the supporting plane, so a query beyond an edge or corner returns the
    point on that edge or corner rather than a point outside the triangle.

    Triangles whose area has collapsed relative to their longest edge fall back
    to the closest point on their three edges, which stays correct when a
    triangle has degenerated into a segment or a point.
    """
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=float)
    if len(points) != len(triangles):
        raise ValueError("points and triangles must have the same leading length.")
    if len(points) == 0:
        return np.zeros((0, 3))

    a = triangles[:, 0]
    b = triangles[:, 1]
    c = triangles[:, 2]
    ab = b - a
    ac = c - a
    bc = c - b

    d1 = _dot(ab, points - a)
    d2 = _dot(ac, points - a)
    d3 = _dot(ab, points - b)
    d4 = _dot(ac, points - b)
    d5 = _dot(ab, points - c)
    d6 = _dot(ac, points - c)

    # Barycentric "outside" indicators for the three edges.
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    def _ratio(num, den):
        safe = np.where(den != 0.0, den, 1.0)
        return np.clip(np.where(den != 0.0, num / safe, 0.0), 0.0, 1.0)

    t_ab = _ratio(d1, d1 - d3)
    t_ac = _ratio(d2, d2 - d6)
    t_bc = _ratio(d4 - d3, (d4 - d3) + (d5 - d6))

    denom = va + vb + vc
    safe_denom = np.where(np.abs(denom) > 0.0, denom, 1.0)
    v_in = np.where(np.abs(denom) > 0.0, vb / safe_denom, 0.0)
    w_in = np.where(np.abs(denom) > 0.0, vc / safe_denom, 0.0)
    interior = a + ab * v_in[:, None] + ac * w_in[:, None]

    # Conditions are evaluated in Ericson's order; np.select takes the first
    # true entry, which reproduces the if/elif chain exactly.
    conditions = [
        (d1 <= 0.0) & (d2 <= 0.0),
        (d3 >= 0.0) & (d4 <= d3),
        (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0),
        (d6 >= 0.0) & (d5 <= d6),
        (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0),
        (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0),
    ]
    choices = [
        a,
        b,
        a + ab * t_ab[:, None],
        c,
        a + ac * t_ac[:, None],
        b + bc * t_bc[:, None],
    ]
    result = np.select([cond[:, None] for cond in conditions], choices, default=interior)

    # Collapsed triangles: Ericson's interior branch divides by an area that is
    # numerically zero, so answer with the edge closest point instead.
    twice_area = np.linalg.norm(np.cross(ab, ac), axis=1)
    longest = np.maximum(
        np.maximum(_dot(ab, ab), _dot(ac, ac)), _dot(bc, bc)
    )
    degenerate = twice_area <= 1e-12 * np.maximum(longest, EPS)
    if np.any(degenerate):
        idx = np.flatnonzero(degenerate)
        q = points[idx]
        candidates = np.stack(
            [
                _closest_point_on_segments(q, a[idx], b[idx]),
                _closest_point_on_segments(q, b[idx], c[idx]),
                _closest_point_on_segments(q, c[idx], a[idx]),
            ],
            axis=1,
        )
        gaps = np.linalg.norm(candidates - q[:, None, :], axis=2)
        result[idx] = candidates[np.arange(len(idx)), np.argmin(gaps, axis=1)]
    return result


def moller_trumbore(origins, directions, triangles, epsilon=1e-12):
    """Moller-Trumbore ray/triangle intersection, vectorised row by row.

    ``origins``/``directions`` are ``(K, 3)`` and ``triangles`` is ``(K, 3, 3)``.
    Returns ``(t, u, v, det)``: the distance along the ray (Euclidean when the
    directions are unit length), the barycentric coordinates of the hit relative
    to the triangle's first vertex, and the determinant whose magnitude measures
    how far the ray is from being parallel to the triangle.

    Triangles are two-sided; a hit is ``u >= 0``, ``v >= 0``, ``u + v <= 1`` and
    ``t`` in the caller's range.  ``epsilon`` is a *relative* parallelism
    tolerance: a row is rejected when ``|det| <= epsilon * |e1| * |e2|``, which
    makes the test independent of model units.  Rejected rows return NaN for
    ``t``, ``u`` and ``v``, so ordinary comparisons filter them out.
    """
    origins = np.asarray(origins, dtype=float)
    directions = np.asarray(directions, dtype=float)
    triangles = np.asarray(triangles, dtype=float)
    if len(origins) == 0:
        empty = np.zeros(0)
        return empty, empty.copy(), empty.copy(), empty.copy()

    v0 = triangles[:, 0]
    e1 = triangles[:, 1] - v0
    e2 = triangles[:, 2] - v0

    pvec = np.cross(directions, e2)
    det = _dot(e1, pvec)
    scale = np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1)
    usable = np.abs(det) > epsilon * np.maximum(scale, EPS)
    safe_det = np.where(usable, det, 1.0)

    tvec = origins - v0
    qvec = np.cross(tvec, e1)
    u = _dot(tvec, pvec) / safe_det
    v = _dot(directions, qvec) / safe_det
    t = _dot(e2, qvec) / safe_det

    nan = np.full(len(origins), np.nan)
    u = np.where(usable, u, nan)
    v = np.where(usable, v, nan)
    t = np.where(usable, t, nan)
    return t, u, v, det


def triangle_triangle_intersect(tri_a, tri_b, tolerance=1e-9):
    """Moller's interval-overlap triangle/triangle test, vectorised row by row.

    ``tri_a`` and ``tri_b`` are ``(K, 3, 3)`` arrays paired row by row; returns
    a ``(K,)`` boolean array.

    Each triangle is tested against the other's supporting plane first, which
    rejects the overwhelming majority of pairs.  Survivors are reduced to two
    intervals on the line where the planes meet, and the triangles intersect iff
    the intervals overlap.  Near-parallel planes take a coplanar branch that
    runs a 2D separating-axis test in the dominant projection plane.

    Triangles are closed sets, so touching counts as intersecting - two shells
    that share a face exactly will report a hit.  ``tolerance`` is relative to
    the longest edge of the pair; a vertex within that distance of the other
    plane is snapped onto it.  Zero-area triangles never intersect anything.
    """
    tri_a = np.asarray(tri_a, dtype=float)
    tri_b = np.asarray(tri_b, dtype=float)
    if len(tri_a) != len(tri_b):
        raise ValueError("tri_a and tri_b must have the same leading length.")
    count = len(tri_a)
    if count == 0:
        return np.zeros(0, dtype=bool)

    normal_a = np.cross(tri_a[:, 1] - tri_a[:, 0], tri_a[:, 2] - tri_a[:, 0])
    normal_b = np.cross(tri_b[:, 1] - tri_b[:, 0], tri_b[:, 2] - tri_b[:, 0])
    len_a = np.linalg.norm(normal_a, axis=1)
    len_b = np.linalg.norm(normal_b, axis=1)

    edges = np.concatenate(
        [
            np.linalg.norm(tri_a - tri_a[:, [1, 2, 0]], axis=2),
            np.linalg.norm(tri_b - tri_b[:, [1, 2, 0]], axis=2),
        ],
        axis=1,
    )
    scale = edges.max(axis=1)
    proper = (len_a > 1e-12 * np.maximum(scale, EPS) ** 2) & (
        len_b > 1e-12 * np.maximum(scale, EPS) ** 2
    )
    if not np.any(proper):
        return np.zeros(count, dtype=bool)

    unit_a = normal_a / np.where(len_a > 0.0, len_a, 1.0)[:, None]
    unit_b = normal_b / np.where(len_b > 0.0, len_b, 1.0)[:, None]
    snap = tolerance * np.maximum(scale, EPS)

    # Signed distance of every vertex of one triangle to the other's plane.
    dist_b = np.einsum("kij,kj->ki", tri_b, unit_a) - _dot(unit_a, tri_a[:, 0])[:, None]
    dist_a = np.einsum("kij,kj->ki", tri_a, unit_b) - _dot(unit_b, tri_b[:, 0])[:, None]
    dist_b = np.where(np.abs(dist_b) <= snap[:, None], 0.0, dist_b)
    dist_a = np.where(np.abs(dist_a) <= snap[:, None], 0.0, dist_a)

    separated = (
        np.all(dist_b > 0.0, axis=1)
        | np.all(dist_b < 0.0, axis=1)
        | np.all(dist_a > 0.0, axis=1)
        | np.all(dist_a < 0.0, axis=1)
    )

    line = np.cross(unit_a, unit_b)
    parallel = np.linalg.norm(line, axis=1) <= 1e-8
    coplanar = parallel & ~separated
    crossing = ~parallel & ~separated

    result = np.zeros(count, dtype=bool)

    if np.any(crossing):
        idx = np.flatnonzero(crossing)
        axis = np.argmax(np.abs(line[idx]), axis=1)
        proj_a = np.take_along_axis(tri_a[idx], axis[:, None, None], axis=2)[:, :, 0]
        proj_b = np.take_along_axis(tri_b[idx], axis[:, None, None], axis=2)[:, :, 0]
        lo_a, hi_a = _plane_interval(proj_a, dist_a[idx])
        lo_b, hi_b = _plane_interval(proj_b, dist_b[idx])
        result[idx] = (lo_a <= hi_b) & (lo_b <= hi_a)

    if np.any(coplanar):
        idx = np.flatnonzero(coplanar)
        result[idx] = _coplanar_overlap(tri_a[idx], tri_b[idx], unit_a[idx])

    return result & proper


def _plane_interval(projections, distances):
    """Interval a triangle carves out of the intersection line.

    ``projections`` is each vertex's coordinate along the chosen axis and
    ``distances`` its signed distance to the other triangle's plane, already
    snapped so an on-plane vertex is exactly zero.  Every on-plane vertex and
    every plane-crossing edge contributes one value; the interval is their
    range.  A triangle that only grazes the plane at one vertex therefore gets a
    zero-length interval rather than being dropped.
    """
    count = len(projections)
    values = np.empty((count, 6))
    valid = np.zeros((count, 6), dtype=bool)

    values[:, 0:3] = projections
    valid[:, 0:3] = distances == 0.0

    for slot, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
        di = distances[:, i]
        dj = distances[:, j]
        denom = di - dj
        safe = np.where(denom != 0.0, denom, 1.0)
        ratio = np.where(denom != 0.0, di / safe, 0.0)
        values[:, 3 + slot] = projections[:, i] + (projections[:, j] - projections[:, i]) * ratio
        valid[:, 3 + slot] = di * dj < 0.0

    lo = np.where(valid, values, np.inf).min(axis=1)
    hi = np.where(valid, values, -np.inf).max(axis=1)
    return lo, hi


def _coplanar_overlap(tri_a, tri_b, normal):
    """2D separating-axis overlap for coplanar triangle pairs.

    Projects both triangles onto the two coordinates the shared normal is least
    aligned with, then tests the six edge normals.  Touching counts as
    overlapping, matching the non-coplanar branch.
    """
    drop = np.argmax(np.abs(normal), axis=1)
    keep = np.stack([(drop + 1) % 3, (drop + 2) % 3], axis=1)  # (K, 2)
    a2 = np.take_along_axis(tri_a, keep[:, None, :], axis=2)  # (K, 3, 2)
    b2 = np.take_along_axis(tri_b, keep[:, None, :], axis=2)

    edges = np.concatenate([a2[:, [1, 2, 0]] - a2, b2[:, [1, 2, 0]] - b2], axis=1)
    axes = np.stack([-edges[:, :, 1], edges[:, :, 0]], axis=2)  # (K, 6, 2)

    pa = np.einsum("kna,kma->knm", axes, a2)
    pb = np.einsum("kna,kma->knm", axes, b2)
    gap = (pa.min(axis=2) > pb.max(axis=2)) | (pb.min(axis=2) > pa.max(axis=2))
    return ~np.any(gap, axis=1)


# ----------------------------------------------------------------------
# Ray hit records
# ----------------------------------------------------------------------
class RayHits:
    """All intersections of a ray batch, in CSR layout.

    Hits for ray ``i`` occupy ``[offsets[i]:offsets[i + 1]]`` of the flat
    ``distances``, ``faces``, and ``points`` arrays, sorted by increasing
    distance.  Indexing the object (``hits[i]``) yields the ``(distances,
    faces)`` pair for one ray; :meth:`as_lists` materialises all of them.  The
    flat layout is what downstream code actually wants - it keeps a per-ray
    Python loop out of the hot path.
    """

    __slots__ = ("n_rays", "offsets", "ray_index", "distances", "faces", "points")

    def __init__(self, n_rays, offsets, ray_index, distances, faces, points):
        self.n_rays = int(n_rays)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.ray_index = np.asarray(ray_index, dtype=np.int64)
        self.distances = np.asarray(distances, dtype=float)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.points = np.asarray(points, dtype=float)

    @property
    def counts(self):
        """Number of hits per ray."""
        return np.diff(self.offsets)

    @property
    def n_hits(self):
        return int(len(self.distances))

    def __len__(self):
        return self.n_rays

    def __getitem__(self, index):
        index = int(index)
        if index < 0:
            index += self.n_rays
        if not 0 <= index < self.n_rays:
            raise IndexError("Ray index out of range.")
        span = slice(int(self.offsets[index]), int(self.offsets[index + 1]))
        return self.distances[span], self.faces[span]

    def as_lists(self):
        """``[(distances, faces), ...]`` - one entry per ray, in ray order."""
        return [self[i] for i in range(self.n_rays)]

    def first(self):
        """``(distances, faces)`` of the nearest hit, ``inf``/``-1`` for misses."""
        distances = np.full(self.n_rays, np.inf)
        faces = np.full(self.n_rays, -1, dtype=np.int64)
        counts = self.counts
        hit = np.flatnonzero(counts > 0)
        if len(hit):
            head = self.offsets[hit]
            distances[hit] = self.distances[head]
            faces[hit] = self.faces[head]
        return distances, faces

    def __repr__(self):
        return f"RayHits(rays={self.n_rays}, hits={self.n_hits})"


class MeshField:
    """Callable signed-distance field backed by a triangle mesh.

    Calling the object with an ``(P, 3)`` array returns ``(P,)`` signed
    distances - negative inside the solid - which is the same call signature the
    analytic fields in :mod:`src.kernel.sdf` expose.  That is the whole point:
    it lets an arbitrary imported STL be used wherever an implicit field is
    expected, which is what makes SDF booleans, offsetting, and wall-thickness
    analysis available on meshes that no analytic primitive describes.

    This is a plain callable, not a subclass of anything in the ``sdf`` module,
    so importing this module never drags that one in.
    """

    __slots__ = ("bvh",)

    def __init__(self, bvh):
        self.bvh = bvh

    def __call__(self, points):
        return self.bvh.signed_distance(points)

    @property
    def mesh(self):
        return self.bvh.mesh

    @property
    def bounds(self):
        """``(min_xyz, max_xyz)`` of the source mesh."""
        return self.bvh.bounds

    def __repr__(self):
        return f"MeshField(faces={self.bvh.n_faces})"


def mesh_sdf(mesh, **kwargs):
    """Wrap a mesh as a callable signed-distance field.

    Extra keyword arguments are forwarded to :class:`BVH`.  Accuracy is exactly
    the accuracy of the tessellation: the field is the distance to the triangle
    soup, so a coarse mesh gives a faceted field.  The sign comes from
    :meth:`BVH.contains`, so the mesh must be closed and consistently wound.
    """
    return MeshField(mesh if isinstance(mesh, BVH) else BVH(mesh, **kwargs))


# ----------------------------------------------------------------------
# The hierarchy
# ----------------------------------------------------------------------
class BVH:
    """Bounding volume hierarchy over the triangles of a :class:`Mesh`.

    The tree is immutable once built; every query is read-only, so a single BVH
    can be shared by the feature tree, the analysis panel, and the boolean
    evaluator without defensive copying.
    """

    __slots__ = (
        "mesh",
        "leaf_size",
        "split_method",
        "order",
        "node_min",
        "node_max",
        "node_left",
        "node_right",
        "node_start",
        "node_count",
        "face_min",
        "face_max",
        "_triangles",
        "_scale",
    )

    def __init__(self, mesh, leaf_size=8, split_method="sah"):
        if not isinstance(mesh, Mesh):
            raise TypeError("BVH requires a kernel Mesh.")
        method = str(split_method).lower()
        if method not in ("sah", "median"):
            raise ValueError("split_method must be 'sah' or 'median'.")

        self.mesh = mesh
        self.leaf_size = max(int(leaf_size), 1)
        self.split_method = method
        self._triangles = mesh.triangles()
        if len(self._triangles):
            self.face_min = self._triangles.min(axis=1)
            self.face_max = self._triangles.max(axis=1)
        else:
            self.face_min = np.zeros((0, 3))
            self.face_max = np.zeros((0, 3))
        self._scale = max(float(mesh.diagonal), EPS)
        self._build()

    @classmethod
    def from_mesh(cls, mesh, **kwargs):
        """Alias for the constructor, for call sites that read better this way."""
        return cls(mesh, **kwargs)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build(self):
        count = len(self._triangles)
        if count == 0:
            # One sentinel leaf with an inverted box: every overlap test fails,
            # so queries degrade to "no hits" without a special case per query.
            self.order = np.zeros(0, dtype=np.int64)
            self.node_min = np.full((1, 3), np.inf)
            self.node_max = np.full((1, 3), -np.inf)
            self.node_left = np.full(1, -1, dtype=np.int64)
            self.node_right = np.full(1, -1, dtype=np.int64)
            self.node_start = np.zeros(1, dtype=np.int64)
            self.node_count = np.zeros(1, dtype=np.int64)
            return

        centroids = self._triangles.mean(axis=1)
        order = np.arange(count, dtype=np.int64)

        node_min = [None]
        node_max = [None]
        node_left = [-1]
        node_right = [-1]
        node_start = [0]
        node_count = [count]

        stack = [(0, count, 0)]
        while stack:
            lo, hi, node = stack.pop()
            span = order[lo:hi]
            node_min[node] = self.face_min[span].min(axis=0)
            node_max[node] = self.face_max[span].max(axis=0)
            node_start[node] = lo
            node_count[node] = hi - lo

            size = hi - lo
            if size <= self.leaf_size:
                continue

            local = centroids[span]
            spread = local.max(axis=0) - local.min(axis=0)
            axis = int(np.argmax(spread))
            if spread[axis] <= EPS * max(self._scale, 1.0):
                # Every centroid coincides: no split plane is meaningful, but a
                # single huge leaf would poison traversal, so halve by index.
                split = size // 2
            else:
                permutation = np.argsort(local[:, axis], kind="stable")
                span = span[permutation]
                order[lo:hi] = span
                split = self._choose_split(span)

            if split <= 0 or split >= size:
                continue

            left = len(node_min)
            right = left + 1
            for container, value in (
                (node_min, None),
                (node_max, None),
                (node_left, -1),
                (node_right, -1),
                (node_start, 0),
                (node_count, 0),
            ):
                container.append(value)
                container.append(value)
            node_left[node] = left
            node_right[node] = right
            node_count[node] = 0
            stack.append((lo, lo + split, left))
            stack.append((lo + split, hi, right))

        self.order = order
        self.node_min = np.asarray(node_min, dtype=float)
        self.node_max = np.asarray(node_max, dtype=float)
        self.node_left = np.asarray(node_left, dtype=np.int64)
        self.node_right = np.asarray(node_right, dtype=np.int64)
        self.node_start = np.asarray(node_start, dtype=np.int64)
        self.node_count = np.asarray(node_count, dtype=np.int64)

    def _choose_split(self, span):
        """Number of primitives that go left, given ``span`` sorted along an axis."""
        size = len(span)
        if self.split_method == "median":
            return size // 2

        low = np.minimum.accumulate(self.face_min[span], axis=0)
        high = np.maximum.accumulate(self.face_max[span], axis=0)
        low_r = np.minimum.accumulate(self.face_min[span][::-1], axis=0)[::-1]
        high_r = np.maximum.accumulate(self.face_max[span][::-1], axis=0)[::-1]

        counts = np.arange(1, size)
        cost = _half_area(high[: size - 1] - low[: size - 1]) * counts + _half_area(
            high_r[1:] - low_r[1:]
        ) * (size - counts)
        return int(np.argmin(cost)) + 1

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------
    @property
    def n_faces(self):
        return int(len(self._triangles))

    @property
    def n_nodes(self):
        return int(len(self.node_left))

    @property
    def is_empty(self):
        return self.n_faces == 0

    @property
    def bounds(self):
        """``(min_xyz, max_xyz)`` of the root node."""
        if self.is_empty:
            return np.zeros(3), np.zeros(3)
        return self.node_min[0].copy(), self.node_max[0].copy()

    @property
    def leaf_nodes(self):
        return np.flatnonzero(self.node_left < 0)

    def depth(self):
        """Maximum root-to-leaf depth, for diagnostics and tuning."""
        if self.is_empty:
            return 0
        depths = np.zeros(self.n_nodes, dtype=np.int64)
        internal = np.flatnonzero(self.node_left >= 0)
        # Children always have a larger index than their parent, so one forward
        # sweep is enough - no traversal required.
        for node in internal:
            depths[self.node_left[node]] = depths[node] + 1
            depths[self.node_right[node]] = depths[node] + 1
        return int(depths.max()) + 1

    def __repr__(self):
        return f"BVH(faces={self.n_faces}, nodes={self.n_nodes}, leaf_size={self.leaf_size})"

    # ------------------------------------------------------------------
    # Traversal helpers
    # ------------------------------------------------------------------
    def _expand_leaves(self, rows, nodes):
        """Explode ``(row, leaf)`` pairs into ``(row, face)`` pairs."""
        if len(nodes) == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        counts = self.node_count[nodes]
        total = int(counts.sum())
        if total == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        before = np.concatenate([[0], np.cumsum(counts)[:-1]])
        base = np.repeat(self.node_start[nodes] - before, counts)
        faces = self.order[base + np.arange(total)]
        return np.repeat(rows, counts), faces

    def _box_distance2(self, nodes, points):
        """Squared distance from each point to its paired node box (0 inside)."""
        low = self.node_min[nodes]
        high = self.node_max[nodes]
        delta = np.maximum(low - points, 0.0) + np.maximum(points - high, 0.0)
        return _dot(delta, delta)

    def _node_ray_range(self, nodes, origins, inv_dir):
        """Slab test: entry and exit parameters of each ray on its node box.

        NaN from ``0 * inf`` (a ray parallel to a slab with its origin exactly on
        the boundary) is replaced by an unbounded interval, which is the correct
        reading: that axis places no constraint on the ray.
        """
        low = self.node_min[nodes]
        high = self.node_max[nodes]
        with np.errstate(invalid="ignore"):
            t1 = (low - origins) * inv_dir
            t2 = (high - origins) * inv_dir
        t1 = np.where(np.isnan(t1), -np.inf, t1)
        t2 = np.where(np.isnan(t2), np.inf, t2)
        return np.minimum(t1, t2).max(axis=1), np.maximum(t1, t2).min(axis=1)

    def _ray_candidates(self, origins, directions, tmin, tmax):
        """``(ray, face)`` pairs whose leaf box the ray enters within range."""
        if self.is_empty or len(origins) == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        with np.errstate(divide="ignore"):
            inv_dir = 1.0 / directions

        rows = np.arange(len(origins), dtype=np.int64)
        nodes = np.zeros(len(origins), dtype=np.int64)
        out_rows = []
        out_faces = []
        while len(rows):
            enter, exit_ = self._node_ray_range(nodes, origins[rows], inv_dir[rows])
            enter = np.maximum(enter, tmin)
            keep = (exit_ >= enter) & (enter <= tmax)
            rows = rows[keep]
            nodes = nodes[keep]
            if not len(rows):
                break
            leaf = self.node_left[nodes] < 0
            if np.any(leaf):
                r, f = self._expand_leaves(rows[leaf], nodes[leaf])
                out_rows.append(r)
                out_faces.append(f)
            internal = np.flatnonzero(~leaf)
            parent = nodes[internal]
            rows = np.concatenate([rows[internal], rows[internal]])
            nodes = np.concatenate([self.node_left[parent], self.node_right[parent]])

        if not out_rows:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        return np.concatenate(out_rows), np.concatenate(out_faces)

    # ------------------------------------------------------------------
    # Ray queries
    # ------------------------------------------------------------------
    def ray_intersections(self, origins, directions, tmin=0.0, tmax=np.inf):
        """Every intersection of every ray, sorted by distance.

        Returns a :class:`RayHits` record (CSR layout - see that class).
        Directions are normalised internally, so distances are Euclidean.
        ``tmin``/``tmax`` clip the ray to a segment, which is how the boolean
        evaluator asks "does this edge cross that shell".
        """
        origins, directions = _as_rays(origins, directions)
        n_rays = len(origins)
        rows, faces = self._ray_candidates(origins, directions, tmin, tmax)

        if len(rows) == 0:
            return RayHits(
                n_rays,
                np.zeros(n_rays + 1, dtype=np.int64),
                np.zeros(0, dtype=np.int64),
                np.zeros(0),
                np.zeros(0, dtype=np.int64),
                np.zeros((0, 3)),
            )

        t, u, v, _ = moller_trumbore(origins[rows], directions[rows], self._triangles[faces])
        hit = (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t >= tmin) & (t <= tmax)
        rows = rows[hit]
        faces = faces[hit]
        t = t[hit]

        order = np.lexsort((t, rows))
        rows = rows[order]
        faces = faces[order]
        t = t[order]
        counts = np.bincount(rows, minlength=n_rays)
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        points = origins[rows] + directions[rows] * t[:, None]
        return RayHits(n_rays, offsets, rows, t, faces, points)

    def first_hit(self, origins, directions, tmin=0.0, tmax=np.inf):
        """Nearest intersection per ray as ``(distances, face_indices)``.

        Misses report ``inf`` and ``-1``.  Unlike :meth:`ray_intersections` this
        prunes the traversal with the best distance found so far, so it does not
        pay for the far side of the model.
        """
        origins, directions = _as_rays(origins, directions)
        n_rays = len(origins)
        best_t = np.full(n_rays, np.inf)
        best_face = np.full(n_rays, -1, dtype=np.int64)
        if self.is_empty or n_rays == 0:
            return best_t, best_face

        with np.errstate(divide="ignore"):
            inv_dir = 1.0 / directions

        rows = np.arange(n_rays, dtype=np.int64)
        nodes = np.zeros(n_rays, dtype=np.int64)
        while len(rows):
            enter, exit_ = self._node_ray_range(nodes, origins[rows], inv_dir[rows])
            enter = np.maximum(enter, tmin)
            limit = np.minimum(best_t[rows], tmax)
            keep = (exit_ >= enter) & (enter <= limit)
            rows = rows[keep]
            nodes = nodes[keep]
            if not len(rows):
                break

            leaf = self.node_left[nodes] < 0
            if np.any(leaf):
                r, f = self._expand_leaves(rows[leaf], nodes[leaf])
                if len(r):
                    t, u, v, _ = moller_trumbore(
                        origins[r], directions[r], self._triangles[f]
                    )
                    hit = (
                        (u >= 0.0)
                        & (v >= 0.0)
                        & (u + v <= 1.0)
                        & (t >= tmin)
                        & (t <= tmax)
                    )
                    r = r[hit]
                    f = f[hit]
                    t = t[hit]
                    if len(r):
                        np.minimum.at(best_t, r, t)
                        winner = t <= best_t[r]
                        best_face[r[winner]] = f[winner]

            internal = np.flatnonzero(~leaf)
            parent = nodes[internal]
            rows = np.concatenate([rows[internal], rows[internal]])
            nodes = np.concatenate([self.node_left[parent], self.node_right[parent]])

        best_t = np.where(best_face >= 0, best_t, np.inf)
        return best_t, best_face

    # ------------------------------------------------------------------
    # Proximity queries
    # ------------------------------------------------------------------
    def closest_point(self, points, max_distance=np.inf):
        """Nearest surface point per query as ``(closest_xyz, distances, faces)``.

        Distances are exact point-to-triangle distances (the seven-region test,
        see :func:`closest_point_on_triangles`), not point-to-plane.  Queries
        farther than ``max_distance`` from the surface report ``inf``, face
        ``-1``, and echo the query point back as the closest point.

        A descent to one leaf per query seeds the pruning bound before the
        breadth-first sweep starts, which is what keeps the wavefront small.
        """
        points = _as_points(points)
        n_points = len(points)
        limit = float(max_distance)
        best = np.full(n_points, limit**2 if np.isfinite(limit) else np.inf)
        best_face = np.full(n_points, -1, dtype=np.int64)
        best_xyz = points.copy()
        if self.is_empty or n_points == 0:
            return best_xyz, np.full(n_points, np.inf), best_face

        # Phase 1: greedy descent, always into the nearer child, for a bound.
        nodes = np.zeros(n_points, dtype=np.int64)
        active = np.arange(n_points, dtype=np.int64)
        while len(active):
            internal = np.flatnonzero(self.node_left[nodes[active]] >= 0)
            if not len(internal):
                break
            active = active[internal]
            parent = nodes[active]
            left = self.node_left[parent]
            right = self.node_right[parent]
            to_left = self._box_distance2(left, points[active])
            to_right = self._box_distance2(right, points[active])
            nodes[active] = np.where(to_left <= to_right, left, right)
        self._update_closest(np.arange(n_points, dtype=np.int64), nodes, points, best, best_face, best_xyz)

        # Phase 2: breadth-first sweep, pruned by the bound from phase 1.
        rows = np.arange(n_points, dtype=np.int64)
        nodes = np.zeros(n_points, dtype=np.int64)
        while len(rows):
            keep = self._box_distance2(nodes, points[rows]) <= best[rows]
            rows = rows[keep]
            nodes = nodes[keep]
            if not len(rows):
                break
            leaf = self.node_left[nodes] < 0
            if np.any(leaf):
                self._update_closest(
                    rows[leaf], nodes[leaf], points, best, best_face, best_xyz
                )
            internal = np.flatnonzero(~leaf)
            parent = nodes[internal]
            rows = np.concatenate([rows[internal], rows[internal]])
            nodes = np.concatenate([self.node_left[parent], self.node_right[parent]])

        distances = np.where(best_face >= 0, np.sqrt(np.maximum(best, 0.0)), np.inf)
        return best_xyz, distances, best_face

    def _update_closest(self, rows, nodes, points, best, best_face, best_xyz):
        """Evaluate the triangles of ``nodes`` and keep the improvements."""
        r, f = self._expand_leaves(rows, nodes)
        if not len(r):
            return
        candidate = closest_point_on_triangles(points[r], self._triangles[f])
        delta = candidate - points[r]
        d2 = _dot(delta, delta)
        np.minimum.at(best, r, d2)
        winner = d2 <= best[r]
        best_face[r[winner]] = f[winner]
        best_xyz[r[winner]] = candidate[winner]

    def distance(self, points):
        """Unsigned distance to the surface - shorthand for ``closest_point``."""
        return self.closest_point(points)[1]

    # ------------------------------------------------------------------
    # Inside / outside
    # ------------------------------------------------------------------
    def winding_number(self, points, chunk=None):
        """Generalised winding number of the mesh around each point.

        Exactly 1 inside a closed, outward-wound solid and 0 outside, with the
        van Oosterom-Strackee solid-angle formula summed over every triangle.
        Fractional values mean the mesh is open or inconsistently wound, and a
        point exactly on the surface gives 0.5.

        Cost is O(P * M) - it ignores the hierarchy completely - which is why
        :meth:`contains` only calls it for the handful of points parity casting
        could not settle.  ``chunk`` bounds peak memory; the default keeps the
        working set near a few million floats.
        """
        points = _as_points(points)
        if self.is_empty or len(points) == 0:
            return np.zeros(len(points))
        faces = self._triangles
        if chunk is None:
            chunk = max(1, int(2_000_000 // max(len(faces), 1)))
        result = np.zeros(len(points))
        for start in range(0, len(points), chunk):
            block = points[start : start + chunk][:, None, :]
            a = faces[None, :, 0, :] - block
            b = faces[None, :, 1, :] - block
            c = faces[None, :, 2, :] - block
            la = np.linalg.norm(a, axis=2)
            lb = np.linalg.norm(b, axis=2)
            lc = np.linalg.norm(c, axis=2)
            numerator = np.einsum("pfi,pfi->pf", a, np.cross(b, c))
            denominator = (
                la * lb * lc
                + np.einsum("pfi,pfi->pf", a, b) * lc
                + np.einsum("pfi,pfi->pf", a, c) * lb
                + np.einsum("pfi,pfi->pf", b, c) * la
            )
            result[start : start + chunk] = 2.0 * np.arctan2(numerator, denominator).sum(axis=1)
        return result / (4.0 * np.pi)

    def contains(self, points, max_attempts=8):
        """Point-in-solid test, ``True`` for points strictly inside.

        The primary method is parity ray casting: one BVH ray query per point,
        odd crossing count means inside.  It is chosen over evaluating the
        generalised winding number everywhere because parity costs O(log M) per
        point against the winding number's O(M).

        The classic failure of parity casting is a ray that grazes an edge or a
        vertex, where one crossing gets counted twice or not at all - exactly
        what happens when you fire an axis-aligned ray at an axis-aligned
        tessellation, or through the hole of a torus whose vertex rings line up
        with the ray.  So the default probe direction is deliberately generic,
        *and* every candidate hit is checked for fragility: a near-parallel
        triangle, or a barycentric coordinate within tolerance of an edge.  Any
        point with a fragile candidate is re-cast along a fresh pseudo-random
        direction (seeded, so results are reproducible).  Points still fragile
        after ``max_attempts`` fall back to :meth:`winding_number`, which is
        exact but expensive - by then there are almost never more than a few.

        The mesh must be closed and consistently wound; on an open surface the
        answer is meaningless.  Points within roughly ``1e-12`` of the surface
        relative to the model size are genuinely ambiguous and may go either way.
        """
        points = _as_points(points)
        inside = np.zeros(len(points), dtype=bool)
        if self.is_empty or len(points) == 0:
            return inside

        low, high = self.bounds
        pad = 1e-9 * self._scale
        maybe = np.all(points >= low - pad, axis=1) & np.all(points <= high + pad, axis=1)
        todo = np.flatnonzero(maybe)
        if not len(todo):
            return inside

        rng = np.random.default_rng(_JITTER_SEED)
        direction = _PROBE_DIRECTION / np.linalg.norm(_PROBE_DIRECTION)
        for attempt in range(max(int(max_attempts), 1)):
            if not len(todo):
                break
            counts, fragile = self._parity_counts(points[todo], direction)
            settled = ~fragile
            inside[todo[settled]] = (counts[settled] % 2) == 1
            todo = todo[fragile]
            if len(todo):
                direction = rng.normal(size=3)
                norm = np.linalg.norm(direction)
                while norm <= EPS:  # pragma: no cover - astronomically unlikely
                    direction = rng.normal(size=3)
                    norm = np.linalg.norm(direction)
                direction = direction / norm
        if len(todo):
            inside[todo] = self.winding_number(points[todo]) >= 0.5
        return inside

    def _parity_counts(self, origins, direction):
        """Crossing counts along one shared direction, plus a fragility flag."""
        directions = np.broadcast_to(np.asarray(direction, dtype=float), origins.shape)
        directions = np.ascontiguousarray(directions)
        count = len(origins)
        rows, faces = self._ray_candidates(origins, directions, 0.0, np.inf)
        if not len(rows):
            return np.zeros(count, dtype=np.int64), np.zeros(count, dtype=bool)

        t, u, v, det = moller_trumbore(origins[rows], directions[rows], self._triangles[faces])
        t_eps = 1e-12 * self._scale
        bary_tol = 1e-9

        hit = (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > t_eps)
        near = (u >= -bary_tol) & (v >= -bary_tol) & (u + v <= 1.0 + bary_tol) & (t > t_eps)
        on_edge = (
            (np.abs(u) <= bary_tol)
            | (np.abs(v) <= bary_tol)
            | (np.abs(1.0 - u - v) <= bary_tol)
        )
        # NaN u/v means the triangle was rejected as parallel to the ray; that is
        # exactly the coplanar-grazing case parity casting cannot resolve.
        parallel = np.isnan(t)
        fragile_pair = (near & on_edge) | parallel

        counts = np.bincount(rows[hit], minlength=count)
        fragile = np.zeros(count, dtype=bool)
        flagged = rows[fragile_pair]
        if len(flagged):
            fragile[flagged] = True
        return counts, fragile

    def signed_distance(self, points):
        """Distance to the surface, negative inside the solid.

        This is the bridge from "a pile of triangles someone imported" to "an
        implicit field": with a sign, a mesh can be resampled, offset, blended,
        or booleaned by the same machinery that handles analytic primitives, and
        wall thickness becomes a field evaluation instead of a topology problem.

        Magnitude is exact for the tessellation; the sign inherits every caveat
        of :meth:`contains`, so the mesh must be closed and consistently wound.
        """
        points = _as_points(points)
        distances = self.closest_point(points)[1]
        if self.is_empty or len(points) == 0:
            return distances
        return np.where(self.contains(points), -distances, distances)

    def as_field(self):
        """This hierarchy as a callable :class:`MeshField`."""
        return MeshField(self)

    # ------------------------------------------------------------------
    # Box queries
    # ------------------------------------------------------------------
    def box_query(self, min_xyz, max_xyz):
        """Indices of faces whose bounding box overlaps the given box.

        This is a *bounds* overlap, so the result is a conservative superset of
        the faces that truly touch the box - which is what callers doing
        selection rubber-banding or spatial binning want.
        """
        low = np.asarray(min_xyz, dtype=float).reshape(3)
        high = np.asarray(max_xyz, dtype=float).reshape(3)
        low, high = np.minimum(low, high), np.maximum(low, high)
        if self.is_empty:
            return np.zeros(0, dtype=np.int64)

        nodes = np.zeros(1, dtype=np.int64)
        found = []
        while len(nodes):
            overlap = np.all(self.node_min[nodes] <= high, axis=1) & np.all(
                self.node_max[nodes] >= low, axis=1
            )
            nodes = nodes[overlap]
            if not len(nodes):
                break
            leaf = self.node_left[nodes] < 0
            if np.any(leaf):
                selected = nodes[leaf]
                _, f = self._expand_leaves(np.zeros(len(selected), dtype=np.int64), selected)
                found.append(f)
            parent = nodes[~leaf]
            nodes = np.concatenate([self.node_left[parent], self.node_right[parent]])

        if not found:
            return np.zeros(0, dtype=np.int64)
        faces = np.unique(np.concatenate(found))
        keep = np.all(self.face_min[faces] <= high, axis=1) & np.all(
            self.face_max[faces] >= low, axis=1
        )
        return faces[keep]

    # ------------------------------------------------------------------
    # Pairwise overlap
    # ------------------------------------------------------------------
    def _iter_candidate_pairs(self, other, same_tree):
        """Yield batches of ``(face_self, face_other)`` pairs from overlapping leaves.

        Batches arrive one wavefront level at a time so callers that only need a
        yes/no answer can stop early.  For ``same_tree`` the traversal expands a
        node against itself as ``(l, l)``, ``(l, r)``, ``(r, r)`` so each
        unordered pair is produced exactly once.
        """
        if self.is_empty or other.is_empty:
            return
        front = np.zeros((1, 2), dtype=np.int64)
        while len(front):
            node_a = front[:, 0]
            node_b = front[:, 1]
            same = node_a == node_b if same_tree else np.zeros(len(node_a), dtype=bool)
            overlap = np.all(self.node_min[node_a] <= other.node_max[node_b], axis=1) & np.all(
                other.node_min[node_b] <= self.node_max[node_a], axis=1
            )
            keep = overlap | same
            node_a = node_a[keep]
            node_b = node_b[keep]
            same = same[keep]
            if not len(node_a):
                break

            leaf_a = self.node_left[node_a] < 0
            leaf_b = other.node_left[node_b] < 0
            both = leaf_a & leaf_b
            if np.any(both):
                pairs = self._leaf_pair_faces(
                    other, node_a[both], node_b[both], same[both], same_tree
                )
                if len(pairs):
                    yield pairs

            rest = ~both
            batches = []

            mirrored = np.flatnonzero(rest & same)
            if len(mirrored):
                node = node_a[mirrored]
                left = self.node_left[node]
                right = self.node_right[node]
                batches.append(np.stack([left, left], axis=1))
                batches.append(np.stack([left, right], axis=1))
                batches.append(np.stack([right, right], axis=1))

            crossed = np.flatnonzero(rest & ~same)
            if len(crossed):
                a = node_a[crossed]
                b = node_b[crossed]
                a_leaf = self.node_left[a] < 0
                b_leaf = other.node_left[b] < 0
                size_a = _half_area(self.node_max[a] - self.node_min[a])
                size_b = _half_area(other.node_max[b] - other.node_min[b])
                split_a = np.flatnonzero(~a_leaf & (b_leaf | (size_a >= size_b)))
                split_b = np.flatnonzero(~(~a_leaf & (b_leaf | (size_a >= size_b))))
                if len(split_a):
                    batches.append(np.stack([self.node_left[a[split_a]], b[split_a]], axis=1))
                    batches.append(np.stack([self.node_right[a[split_a]], b[split_a]], axis=1))
                if len(split_b):
                    batches.append(np.stack([a[split_b], other.node_left[b[split_b]]], axis=1))
                    batches.append(np.stack([a[split_b], other.node_right[b[split_b]]], axis=1))

            front = np.concatenate(batches) if batches else np.zeros((0, 2), dtype=np.int64)

    def _leaf_pair_faces(self, other, node_a, node_b, same, same_tree):
        """Cartesian product of the faces in each overlapping pair of leaves."""
        count_a = self.node_count[node_a]
        count_b = other.node_count[node_b]
        sizes = count_a * count_b
        total = int(sizes.sum())
        if total == 0:
            return np.zeros((0, 2), dtype=np.int64)
        before = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        local = np.arange(total) - np.repeat(before, sizes)
        wide = np.repeat(count_b, sizes)
        face_a = self.order[np.repeat(self.node_start[node_a], sizes) + local // wide]
        face_b = other.order[np.repeat(other.node_start[node_b], sizes) + local % wide]

        pairs = np.stack([face_a, face_b], axis=1)
        if same_tree:
            mirrored = np.repeat(same, sizes)
            pairs = pairs[~mirrored | (face_a < face_b)]
            pairs = np.sort(pairs, axis=1)
        return pairs

    def intersection_pairs(self, other, exact=True, tolerance=1e-9):
        """Triangle index pairs ``(face_self, face_other)`` that overlap.

        With ``exact=True`` every pair has passed
        :func:`triangle_triangle_intersect`; with ``exact=False`` the raw BVH
        candidates come back, which is what a boolean evaluator wants when it is
        going to do its own clipping anyway.  This is a *surface* test: one solid
        entirely inside another with no shared surface reports nothing.
        """
        batches = []
        for candidates in self._iter_candidate_pairs(other, same_tree=False):
            if exact:
                keep = triangle_triangle_intersect(
                    self._triangles[candidates[:, 0]],
                    other._triangles[candidates[:, 1]],
                    tolerance=tolerance,
                )
                candidates = candidates[keep]
            if len(candidates):
                batches.append(candidates)
        if not batches:
            return np.zeros((0, 2), dtype=np.int64)
        pairs = np.unique(np.concatenate(batches), axis=0)
        return pairs.astype(np.int64)

    def intersects(self, other, tolerance=1e-9):
        """``True`` if any triangle of this mesh meets any triangle of ``other``.

        Stops at the first wavefront level that produces a real hit rather than
        enumerating every pair.
        """
        for candidates in self._iter_candidate_pairs(other, same_tree=False):
            hit = triangle_triangle_intersect(
                self._triangles[candidates[:, 0]],
                other._triangles[candidates[:, 1]],
                tolerance=tolerance,
            )
            if np.any(hit):
                return True
        return False

    def self_intersections(self, exact=True, tolerance=1e-9):
        """Triangle index pairs ``(i, j)``, ``i < j``, where the mesh crosses itself.

        Triangles that share a vertex index are skipped: neighbours always touch
        along their shared edge or corner, and reporting that would drown the
        real defects.  A consequence is that a fold where two adjacent triangles
        genuinely overlap is not reported here - that is a normals/orientation
        problem, better caught by :meth:`Mesh.is_oriented`.
        """
        faces = self.mesh.faces
        batches = []
        for candidates in self._iter_candidate_pairs(self, same_tree=True):
            fa = faces[candidates[:, 0]]
            fb = faces[candidates[:, 1]]
            shared = np.any(fa[:, :, None] == fb[:, None, :], axis=(1, 2))
            candidates = candidates[~shared]
            if exact and len(candidates):
                keep = triangle_triangle_intersect(
                    self._triangles[candidates[:, 0]],
                    self._triangles[candidates[:, 1]],
                    tolerance=tolerance,
                )
                candidates = candidates[keep]
            if len(candidates):
                batches.append(candidates)
        if not batches:
            return np.zeros((0, 2), dtype=np.int64)
        return np.unique(np.concatenate(batches), axis=0).astype(np.int64)
