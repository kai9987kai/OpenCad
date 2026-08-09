"""Pure-numpy triangle mesh type for the OpenCad geometry kernel.

The kernel deliberately depends on nothing beyond numpy so geometry can be
built, analysed, and tested headlessly - without VTK, OpenGL, or Qt.  The UI
layer converts at the boundary via :meth:`Mesh.to_polydata` and
:meth:`Mesh.from_polydata`.

Conventions
-----------
- ``vertices`` is an ``(N, 3)`` float64 array.
- ``faces`` is an ``(M, 3)`` int64 array of vertex indices.
- Triangles are counter-clockwise when viewed from outside, so
  :meth:`Mesh.volume` of a closed solid is positive.
"""

from __future__ import annotations

import contextlib

import numpy as np

__all__ = [
    "EPS",
    "MassProperties",
    "Mesh",
    "axis_index",
    "rotation_matrix",
    "transform_matrix",
]

EPS = 1e-12


class MassProperties:
    """Rigid-body mass properties of a closed mesh, assuming unit density."""

    __slots__ = ("area", "center_of_mass", "inertia_tensor", "volume")

    def __init__(self, volume, center_of_mass, inertia_tensor, area):
        self.volume = float(volume)
        self.center_of_mass = np.asarray(center_of_mass, dtype=float)
        self.inertia_tensor = np.asarray(inertia_tensor, dtype=float)
        self.area = float(area)

    @property
    def principal_moments(self):
        """Eigenvalues of the inertia tensor, ascending."""
        return np.linalg.eigvalsh(self.inertia_tensor)

    @property
    def principal_axes(self):
        """Eigenvectors of the inertia tensor as columns, matching moments."""
        return np.linalg.eigh(self.inertia_tensor)[1]

    def as_dict(self):
        return {
            "volume": self.volume,
            "area": self.area,
            "center_of_mass": self.center_of_mass.tolist(),
            "inertia_tensor": self.inertia_tensor.tolist(),
            "principal_moments": self.principal_moments.tolist(),
        }

    def __repr__(self):
        com = np.round(self.center_of_mass, 6).tolist()
        return f"MassProperties(volume={self.volume:.6g}, area={self.area:.6g}, com={com})"


class Mesh:
    """An indexed triangle mesh.

    Every operation returns a new :class:`Mesh` unless documented otherwise;
    the type is treated as immutable-by-convention so undo/redo and the feature
    tree can share references without defensive copying.
    """

    __slots__ = ("attributes", "faces", "vertices")

    def __init__(self, vertices=None, faces=None, attributes=None):
        self.vertices = _as_vertices(vertices)
        self.faces = _as_faces(faces)
        self.attributes = dict(attributes) if attributes else {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def empty(cls):
        return cls()

    @classmethod
    def from_arrays(cls, vertices, faces, attributes=None):
        return cls(vertices, faces, attributes)

    @classmethod
    def from_polygons(cls, vertices, polygons, attributes=None):
        """Build from mixed-size polygons by fan-triangulating each one."""
        faces = []
        for polygon in polygons:
            indices = list(polygon)
            for i in range(1, len(indices) - 1):
                faces.append((indices[0], indices[i], indices[i + 1]))
        return cls(vertices, faces, attributes)

    @classmethod
    def from_polydata(cls, polydata):
        """Convert a :class:`pyvista.PolyData` into a kernel mesh."""
        surface = polydata
        extract = getattr(surface, "extract_surface", None)
        if extract is not None and getattr(surface, "faces", None) is None:
            surface = extract()
        if getattr(surface, "is_all_triangles", True) is False:
            surface = surface.triangulate()

        vertices = np.asarray(surface.points, dtype=float)
        raw = np.asarray(surface.faces, dtype=np.int64).ravel()
        faces = _decode_vtk_faces(raw)
        attributes = {}
        for key in getattr(surface, "field_data", {}):
            value = surface.field_data[key]
            attributes[key] = value[0] if len(value) == 1 else list(value)
        return cls(vertices, faces, attributes)

    def to_polydata(self):
        """Convert to :class:`pyvista.PolyData` (imported lazily)."""
        import pyvista as pv

        if self.is_empty:
            return pv.PolyData()
        padded = np.hstack(
            [np.full((len(self.faces), 1), 3, dtype=np.int64), self.faces]
        ).ravel()
        polydata = pv.PolyData(self.vertices.copy(), padded)
        for key, value in self.attributes.items():
            # VTK rejects values it cannot store in a field array; an attribute
            # that will not travel is not worth failing an export over.
            with contextlib.suppress(Exception):
                polydata.field_data[key] = [value]
        return polydata

    def copy(self):
        return Mesh(self.vertices.copy(), self.faces.copy(), dict(self.attributes))

    def with_attributes(self, **attributes):
        merged = dict(self.attributes)
        merged.update(attributes)
        return Mesh(self.vertices, self.faces, merged)

    # ------------------------------------------------------------------
    # Basic queries
    # ------------------------------------------------------------------
    @property
    def n_vertices(self):
        return len(self.vertices)

    @property
    def n_faces(self):
        return len(self.faces)

    @property
    def is_empty(self):
        return self.n_vertices == 0 or self.n_faces == 0

    @property
    def bounds(self):
        """``(xmin, xmax, ymin, ymax, zmin, zmax)`` matching PyVista's order."""
        if self.n_vertices == 0:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        low = self.vertices.min(axis=0)
        high = self.vertices.max(axis=0)
        return (
            float(low[0]),
            float(high[0]),
            float(low[1]),
            float(high[1]),
            float(low[2]),
            float(high[2]),
        )

    @property
    def bounding_box(self):
        """``(min_xyz, max_xyz)`` as two ``(3,)`` arrays."""
        if self.n_vertices == 0:
            return np.zeros(3), np.zeros(3)
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    @property
    def extents(self):
        low, high = self.bounding_box
        return high - low

    @property
    def center(self):
        """Centre of the axis-aligned bounding box."""
        low, high = self.bounding_box
        return (low + high) / 2.0

    @property
    def diagonal(self):
        return float(np.linalg.norm(self.extents))

    # ------------------------------------------------------------------
    # Differential quantities
    # ------------------------------------------------------------------
    def triangles(self):
        """``(M, 3, 3)`` array of triangle corner coordinates."""
        if self.is_empty:
            return np.zeros((0, 3, 3))
        return self.vertices[self.faces]

    def face_normals(self, normalize=True):
        tri = self.triangles()
        if len(tri) == 0:
            return np.zeros((0, 3))
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        if not normalize:
            return normals
        lengths = np.linalg.norm(normals, axis=1)
        safe = np.where(lengths > EPS, lengths, 1.0)
        return normals / safe[:, None]

    def face_areas(self):
        normals = self.face_normals(normalize=False)
        if len(normals) == 0:
            return np.zeros(0)
        return 0.5 * np.linalg.norm(normals, axis=1)

    def face_centroids(self):
        tri = self.triangles()
        if len(tri) == 0:
            return np.zeros((0, 3))
        return tri.mean(axis=1)

    def vertex_normals(self):
        """Area-weighted vertex normals."""
        if self.is_empty:
            return np.zeros((self.n_vertices, 3))
        weighted = self.face_normals(normalize=False)  # magnitude == 2 * area
        normals = np.zeros((self.n_vertices, 3))
        for corner in range(3):
            np.add.at(normals, self.faces[:, corner], weighted)
        lengths = np.linalg.norm(normals, axis=1)
        safe = np.where(lengths > EPS, lengths, 1.0)
        return normals / safe[:, None]

    @property
    def area(self):
        return float(self.face_areas().sum())

    @property
    def volume(self):
        """Signed volume via the divergence theorem; positive for CCW solids."""
        tri = self.triangles()
        if len(tri) == 0:
            return 0.0
        cross = np.cross(tri[:, 1], tri[:, 2])
        return float(np.einsum("ij,ij->i", tri[:, 0], cross).sum() / 6.0)

    @property
    def centroid(self):
        """Centre of mass for a closed solid, area centroid for an open surface.

        The distinction is made on watertightness rather than on whether the
        computed volume happens to be non-zero: an open hemisphere has a
        perfectly plausible-looking signed volume, and reporting its solid
        centroid would be quietly wrong.
        """
        if self.is_watertight and abs(self.volume) > EPS:
            return self.mass_properties().center_of_mass
        areas = self.face_areas()
        total = areas.sum()
        if total <= EPS:
            return self.center
        return (self.face_centroids() * areas[:, None]).sum(axis=0) / total

    def mass_properties(self):
        """Volume, centroid, and inertia tensor about the centre of mass.

        Uses the tetrahedron covariance decomposition: each triangle forms a
        tetrahedron with the origin whose second-moment contribution has a
        closed form, and signed accumulation cancels the exterior parts.
        """
        tri = self.triangles()
        if len(tri) == 0:
            return MassProperties(0.0, np.zeros(3), np.zeros((3, 3)), 0.0)

        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        # Signed volume of each tetrahedron (origin, a, b, c).
        det = np.einsum("ij,ij->i", a, np.cross(b, c))
        volume = float(det.sum() / 6.0)

        if abs(volume) <= EPS:
            return MassProperties(volume, self.center, np.zeros((3, 3)), self.area)

        centroids = (a + b + c) / 4.0
        com = (centroids * (det / 6.0)[:, None]).sum(axis=0) / volume

        # Canonical covariance of the unit tetrahedron with corners at the
        # origin and the three basis vectors.
        canonical = np.array(
            [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]
        ) / 120.0
        # Transform matrices with tetra edges as columns: (M, 3, 3).
        transform = np.stack([a, b, c], axis=2)
        covariance = np.einsum(
            "m,mij,jk,mlk->il", det, transform, canonical, transform
        )

        # Shift from the origin to the centre of mass.
        covariance = covariance - volume * np.outer(com, com)
        inertia = np.trace(covariance) * np.eye(3) - covariance
        inertia = (inertia + inertia.T) / 2.0
        return MassProperties(volume, com, inertia, self.area)

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------
    def transform(self, matrix):
        """Apply a 4x4 (or 3x3) matrix, flipping winding on mirror transforms."""
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape == (3, 3):
            full = np.eye(4)
            full[:3, :3] = matrix
            matrix = full
        if matrix.shape != (4, 4):
            raise ValueError("Transform matrix must be 3x3 or 4x4.")
        if self.n_vertices == 0:
            return self.copy()

        homogeneous = np.hstack([self.vertices, np.ones((self.n_vertices, 1))])
        transformed = homogeneous @ matrix.T
        w = transformed[:, 3:4]
        w = np.where(np.abs(w) > EPS, w, 1.0)
        vertices = transformed[:, :3] / w

        faces = self.faces
        if np.linalg.det(matrix[:3, :3]) < 0:
            faces = faces[:, ::-1]
        return Mesh(vertices, faces, dict(self.attributes))

    def translated(self, offset):
        offset = np.asarray(offset, dtype=float).reshape(3)
        return Mesh(self.vertices + offset, self.faces, dict(self.attributes))

    def scaled(self, factor, origin=None):
        factor = np.broadcast_to(np.asarray(factor, dtype=float), (3,)).astype(float)
        origin = np.zeros(3) if origin is None else np.asarray(origin, dtype=float)
        vertices = (self.vertices - origin) * factor + origin
        faces = self.faces[:, ::-1] if float(np.prod(factor)) < 0 else self.faces
        return Mesh(vertices, faces, dict(self.attributes))

    def rotated(self, angle_degrees, axis="z", origin=None):
        matrix = rotation_matrix(angle_degrees, axis)
        origin = np.zeros(3) if origin is None else np.asarray(origin, dtype=float)
        vertices = (self.vertices - origin) @ matrix.T + origin
        return Mesh(vertices, self.faces, dict(self.attributes))

    def mirrored(self, axis="x", origin=0.0):
        index = axis_index(axis)
        vertices = self.vertices.copy()
        vertices[:, index] = 2.0 * float(origin) - vertices[:, index]
        return Mesh(vertices, self.faces[:, ::-1], dict(self.attributes))

    def centered(self):
        return self.translated(-self.center)

    def flipped(self):
        """Reverse triangle winding (and therefore the surface orientation)."""
        return Mesh(self.vertices, self.faces[:, ::-1], dict(self.attributes))

    # ------------------------------------------------------------------
    # Topology and cleanup
    # ------------------------------------------------------------------
    def edges(self, unique=True):
        """Vertex-index pairs. Unique edges are sorted low-to-high per row."""
        if self.is_empty:
            return np.zeros((0, 2), dtype=np.int64)
        raw = np.vstack(
            [self.faces[:, [0, 1]], self.faces[:, [1, 2]], self.faces[:, [2, 0]]]
        )
        if not unique:
            return raw
        return np.unique(np.sort(raw, axis=1), axis=0)

    def edge_counts(self):
        """``(edges, counts)`` where counts is how many faces use each edge."""
        if self.is_empty:
            return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.int64)
        raw = np.sort(self.edges(unique=False), axis=1)
        return np.unique(raw, axis=0, return_counts=True)

    def boundary_edges(self):
        """Edges used by exactly one face - the open border of the surface."""
        edges, counts = self.edge_counts()
        return edges[counts == 1]

    def non_manifold_edges(self):
        """Edges shared by three or more faces."""
        edges, counts = self.edge_counts()
        return edges[counts > 2]

    @property
    def is_watertight(self):
        """True when every edge is shared by exactly two faces."""
        if self.is_empty:
            return False
        _, counts = self.edge_counts()
        return bool(np.all(counts == 2))

    @property
    def is_edge_manifold(self):
        if self.is_empty:
            return False
        _, counts = self.edge_counts()
        return bool(np.all(counts <= 2))

    @property
    def is_oriented(self):
        """True when neighbouring faces agree on winding.

        A consistently oriented surface traverses each directed edge at most
        once; a flipped neighbour produces the same directed edge twice.
        """
        if self.is_empty:
            return False
        directed = self.edges(unique=False)
        _, counts = np.unique(directed, axis=0, return_counts=True)
        return bool(np.all(counts == 1))

    @property
    def euler_characteristic(self):
        return self.n_vertices - len(self.edges()) + self.n_faces

    @property
    def genus(self):
        """Topological genus, valid for closed orientable surfaces."""
        if not self.is_watertight:
            return None
        shells = max(1, self.n_components())
        return int((2 * shells - self.euler_characteristic) // 2)

    def weld(self, tolerance=1e-8):
        """Merge coincident vertices and drop the resulting degenerate faces.

        Welding quantises coordinates onto a grid of ``tolerance``, so two
        vertices closer than the tolerance still separate if they straddle a
        cell boundary.  That is the usual trade for an O(n log n) weld; raise
        the tolerance rather than expecting it to be exact, and use
        :meth:`decimate_vertex_cluster` when you actually want clustering.
        """
        if self.is_empty:
            return self.copy()
        tolerance = max(float(tolerance), 0.0)
        if tolerance <= 0.0:
            keys = self.vertices
        else:
            keys = np.round(self.vertices / tolerance) * tolerance
            keys = keys + 0.0  # normalise -0.0 to 0.0
        _, first, inverse = np.unique(
            keys, axis=0, return_index=True, return_inverse=True
        )
        inverse = inverse.ravel()
        vertices = self.vertices[np.sort(first)]
        # np.unique returns lexicographic order; remap to the sorted-first order.
        order = np.argsort(np.argsort(first))
        faces = order[inverse][self.faces]
        return Mesh(vertices, faces, dict(self.attributes)).remove_degenerate_faces()

    def remove_degenerate_faces(self, area_tolerance=0.0):
        """Drop faces with repeated indices or (optionally) near-zero area."""
        if self.is_empty:
            return self.copy()
        faces = self.faces
        distinct = (
            (faces[:, 0] != faces[:, 1])
            & (faces[:, 1] != faces[:, 2])
            & (faces[:, 0] != faces[:, 2])
        )
        keep = distinct
        if area_tolerance > 0.0:
            areas = self.face_areas()
            keep = keep & (areas > float(area_tolerance))
        return Mesh(self.vertices, faces[keep], dict(self.attributes))

    def remove_duplicate_faces(self):
        """Drop faces that repeat the same vertex triple, ignoring rotation."""
        if self.is_empty:
            return self.copy()
        keys = np.sort(self.faces, axis=1)
        _, index = np.unique(keys, axis=0, return_index=True)
        return Mesh(self.vertices, self.faces[np.sort(index)], dict(self.attributes))

    def remove_unreferenced_vertices(self):
        if self.is_empty:
            return Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
        used = np.unique(self.faces)
        remap = np.full(self.n_vertices, -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        return Mesh(self.vertices[used], remap[self.faces], dict(self.attributes))

    def cleaned(self, tolerance=1e-8):
        """Weld, drop degenerate/duplicate faces, and compact the vertex list."""
        mesh = self.weld(tolerance)
        mesh = mesh.remove_duplicate_faces()
        mesh = mesh.remove_unreferenced_vertices()
        return mesh

    # ------------------------------------------------------------------
    # Connectivity
    # ------------------------------------------------------------------
    def vertex_components(self):
        """Component label per vertex, using union-find over face corners."""
        parent = np.arange(max(self.n_vertices, 1), dtype=np.int64)

        def find(node):
            root = node
            while parent[root] != root:
                root = parent[root]
            while parent[node] != root:
                parent[node], node = root, parent[node]
            return root

        for face in self.faces:
            root = find(face[0])
            for index in face[1:]:
                other = find(index)
                if other != root:
                    parent[other] = root

        labels = np.array([find(i) for i in range(self.n_vertices)], dtype=np.int64)
        _, labels = np.unique(labels, return_inverse=True)
        return labels.ravel()

    def n_components(self):
        if self.is_empty:
            return 0
        labels = self.vertex_components()
        return len(np.unique(labels[np.unique(self.faces)]))

    def split(self):
        """Split into connected components, largest first."""
        if self.is_empty:
            return []
        labels = self.vertex_components()
        face_labels = labels[self.faces[:, 0]]
        pieces = []
        for label in np.unique(face_labels):
            subset = Mesh(
                self.vertices, self.faces[face_labels == label], dict(self.attributes)
            )
            pieces.append(subset.remove_unreferenced_vertices())
        pieces.sort(key=lambda piece: piece.n_faces, reverse=True)
        return pieces

    def largest_component(self):
        pieces = self.split()
        return pieces[0] if pieces else self.copy()

    # ------------------------------------------------------------------
    # Combination
    # ------------------------------------------------------------------
    @classmethod
    def concatenate(cls, meshes):
        """Merge meshes into one without any boolean evaluation."""
        meshes = [mesh for mesh in meshes if mesh is not None and not mesh.is_empty]
        if not meshes:
            return cls.empty()
        vertices, faces, offset = [], [], 0
        for mesh in meshes:
            vertices.append(mesh.vertices)
            faces.append(mesh.faces + offset)
            offset += mesh.n_vertices
        return cls(np.vstack(vertices), np.vstack(faces))

    def merged(self, other):
        return Mesh.concatenate([self, other])

    def __add__(self, other):
        return self.merged(other)

    # ------------------------------------------------------------------
    # Refinement and smoothing
    # ------------------------------------------------------------------
    def subdivide(self, levels=1):
        """Midpoint (1-to-4) subdivision, repeated ``levels`` times."""
        mesh = self
        for _ in range(max(int(levels), 0)):
            mesh = mesh._subdivide_once()
        return mesh

    def _subdivide_once(self):
        if self.is_empty:
            return self.copy()
        edges = np.sort(self.edges(unique=False), axis=1)
        unique_edges, inverse = np.unique(edges, axis=0, return_inverse=True)
        inverse = inverse.ravel()
        midpoints = self.vertices[unique_edges].mean(axis=1)
        vertices = np.vstack([self.vertices, midpoints])

        count = self.n_faces
        offset = self.n_vertices
        m01 = inverse[0:count] + offset
        m12 = inverse[count : 2 * count] + offset
        m20 = inverse[2 * count : 3 * count] + offset
        v0, v1, v2 = self.faces[:, 0], self.faces[:, 1], self.faces[:, 2]
        faces = np.vstack(
            [
                np.stack([v0, m01, m20], axis=1),
                np.stack([m01, v1, m12], axis=1),
                np.stack([m20, m12, v2], axis=1),
                np.stack([m01, m12, m20], axis=1),
            ]
        )
        return Mesh(vertices, faces, dict(self.attributes))

    def vertex_adjacency(self):
        """``(indices, offsets)`` CSR-style neighbour lists per vertex."""
        edges = self.edges()
        if len(edges) == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(self.n_vertices + 1, dtype=np.int64)
        both = np.vstack([edges, edges[:, ::-1]])
        order = np.lexsort((both[:, 1], both[:, 0]))
        both = both[order]
        counts = np.bincount(both[:, 0], minlength=self.n_vertices)
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        return both[:, 1].astype(np.int64), offsets

    def smooth_laplacian(self, iterations=10, factor=0.5, preserve_boundary=True):
        """Uniform-weight Laplacian smoothing (shrinks volume)."""
        return self._smooth(iterations, [float(factor)], preserve_boundary)

    def smooth_taubin(self, iterations=20, lamb=0.5, mu=-0.53, preserve_boundary=True):
        """Taubin lambda/mu smoothing - low-pass filtering with little shrinkage."""
        return self._smooth(iterations, [float(lamb), float(mu)], preserve_boundary)

    def _smooth(self, iterations, factors, preserve_boundary):
        if self.is_empty or iterations <= 0:
            return self.copy()
        neighbours, offsets = self.vertex_adjacency()
        degree = np.diff(offsets)
        movable = degree > 0
        if preserve_boundary:
            boundary = np.unique(self.boundary_edges())
            if len(boundary):
                movable = movable.copy()
                movable[boundary] = False

        source = np.repeat(np.arange(self.n_vertices), degree)
        vertices = self.vertices.copy()
        safe_degree = np.where(degree > 0, degree, 1)[:, None]
        for step in range(int(iterations)):
            factor = factors[step % len(factors)]
            accumulated = np.zeros_like(vertices)
            np.add.at(accumulated, source, vertices[neighbours])
            average = accumulated / safe_degree
            delta = (average - vertices) * factor
            delta[~movable] = 0.0
            vertices = vertices + delta
        return Mesh(vertices, self.faces, dict(self.attributes))

    def decimate_vertex_cluster(self, cell_size=None, divisions=48):
        """Fast decimation by snapping vertices to a grid and collapsing cells.

        Topology-insensitive but robust: it never fails on non-manifold input,
        which makes it a safe fallback for imported scan meshes.
        """
        if self.is_empty:
            return self.copy()
        if cell_size is None:
            cell_size = max(self.diagonal / max(int(divisions), 1), EPS)
        cell_size = max(float(cell_size), EPS)

        low, _ = self.bounding_box
        cells = np.floor((self.vertices - low) / cell_size).astype(np.int64)
        _, inverse = np.unique(cells, axis=0, return_inverse=True)
        inverse = inverse.ravel()
        count = int(inverse.max()) + 1

        accumulated = np.zeros((count, 3))
        np.add.at(accumulated, inverse, self.vertices)
        totals = np.bincount(inverse, minlength=count)[:, None]
        vertices = accumulated / np.where(totals > 0, totals, 1)

        mesh = Mesh(vertices, inverse[self.faces], dict(self.attributes))
        return mesh.remove_degenerate_faces().remove_duplicate_faces().remove_unreferenced_vertices()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample_surface(self, count=1000, seed=None):
        """Area-weighted uniform random points on the surface."""
        if self.is_empty or count <= 0:
            return np.zeros((0, 3))
        rng = np.random.default_rng(seed)
        areas = self.face_areas()
        total = areas.sum()
        if total <= EPS:
            return np.zeros((0, 3))
        picks = rng.choice(len(areas), size=int(count), p=areas / total)
        tri = self.triangles()[picks]
        u = rng.random((int(count), 1))
        v = rng.random((int(count), 1))
        fold = (u + v) > 1.0
        u[fold] = 1.0 - u[fold]
        v[fold] = 1.0 - v[fold]
        return tri[:, 0] + u * (tri[:, 1] - tri[:, 0]) + v * (tri[:, 2] - tri[:, 0])

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def validate(self):
        """Return a list of human-readable structural problems (empty is good)."""
        problems = []
        if self.n_vertices == 0:
            problems.append("Mesh has no vertices.")
        if self.n_faces == 0:
            problems.append("Mesh has no faces.")
        if self.n_faces and self.faces.max(initial=-1) >= self.n_vertices:
            problems.append("Face indices reference vertices that do not exist.")
        if self.n_faces and self.faces.min(initial=0) < 0:
            problems.append("Face indices contain negative values.")
        if not np.all(np.isfinite(self.vertices)):
            problems.append("Vertex coordinates contain NaN or infinity.")
        degenerate = self.n_faces - self.remove_degenerate_faces().n_faces
        if degenerate:
            problems.append(f"{degenerate} degenerate face(s) with repeated indices.")
        return problems

    def summary(self):
        low, high = self.bounding_box
        return {
            "vertices": self.n_vertices,
            "faces": self.n_faces,
            "components": self.n_components(),
            "watertight": self.is_watertight,
            "edge_manifold": self.is_edge_manifold,
            "area": self.area,
            "volume": self.volume,
            "bounds_min": low.tolist(),
            "bounds_max": high.tolist(),
        }

    def __repr__(self):
        return f"Mesh(vertices={self.n_vertices}, faces={self.n_faces})"

    def __eq__(self, other):
        if not isinstance(other, Mesh):
            return NotImplemented
        return (
            self.vertices.shape == other.vertices.shape
            and self.faces.shape == other.faces.shape
            and np.array_equal(self.faces, other.faces)
            and np.allclose(self.vertices, other.vertices)
        )

    def __hash__(self):
        return id(self)


# ----------------------------------------------------------------------
# Module helpers
# ----------------------------------------------------------------------
def axis_index(axis):
    """Map ``'x'``/``'y'``/``'z'`` (or 0/1/2) onto a column index."""
    if isinstance(axis, (int, np.integer)):
        if axis in (0, 1, 2):
            return int(axis)
        raise ValueError("Axis index must be 0, 1, or 2.")
    normalized = str(axis).strip().lower()
    if normalized in ("x", "y", "z"):
        return "xyz".index(normalized)
    raise ValueError("Axis must be X, Y, or Z.")


def rotation_matrix(angle_degrees, axis="z"):
    """3x3 right-handed rotation matrix about a principal or arbitrary axis."""
    angle = np.deg2rad(float(angle_degrees))
    cos, sin = np.cos(angle), np.sin(angle)

    if isinstance(axis, (str, int, np.integer)):
        index = axis_index(axis)
        direction = np.zeros(3)
        direction[index] = 1.0
    else:
        direction = np.asarray(axis, dtype=float).reshape(3)

    length = np.linalg.norm(direction)
    if length <= EPS:
        raise ValueError("Rotation axis must be non-zero.")
    x, y, z = direction / length

    # Rodrigues' rotation formula.
    outer = np.outer([x, y, z], [x, y, z])
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return cos * np.eye(3) + sin * skew + (1.0 - cos) * outer


def transform_matrix(translate=None, rotate=None, scale=None, order="srt"):
    """Compose a 4x4 transform from scale, XYZ Euler rotation, and translation.

    ``rotate`` is degrees about X, Y, Z applied in that order, matching the
    orientation convention used by VTK actors.
    """
    matrix = np.eye(4)
    scale_matrix = np.eye(4)
    if scale is not None:
        scale_matrix[:3, :3] = np.diag(
            np.broadcast_to(np.asarray(scale, dtype=float), (3,))
        )

    rotate_matrix = np.eye(4)
    if rotate is not None:
        rx, ry, rz = np.asarray(rotate, dtype=float).reshape(3)
        rotation = rotation_matrix(rz, "z") @ rotation_matrix(ry, "y") @ rotation_matrix(rx, "x")
        rotate_matrix[:3, :3] = rotation

    translate_matrix = np.eye(4)
    if translate is not None:
        translate_matrix[:3, 3] = np.asarray(translate, dtype=float).reshape(3)

    # ``order`` lists the operations in application order, so later entries
    # multiply on the left: "srt" builds T @ R @ S.
    steps = {"s": scale_matrix, "r": rotate_matrix, "t": translate_matrix}
    for key in order.lower():
        matrix = steps[key] @ matrix
    return matrix


def _as_vertices(vertices):
    if vertices is None:
        return np.zeros((0, 3), dtype=float)
    array = np.asarray(vertices, dtype=float)
    if array.size == 0:
        return np.zeros((0, 3), dtype=float)
    array = np.atleast_2d(array)
    if array.shape[1] == 2:  # promote 2D sketch points to the Z=0 plane
        array = np.hstack([array, np.zeros((len(array), 1))])
    if array.shape[1] != 3:
        raise ValueError("Vertices must have shape (N, 3).")
    return np.ascontiguousarray(array, dtype=float)


def _as_faces(faces):
    if faces is None:
        return np.zeros((0, 3), dtype=np.int64)
    array = np.asarray(faces, dtype=np.int64)
    if array.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    array = np.atleast_2d(array)
    if array.shape[1] != 3:
        raise ValueError("Faces must be triangles with shape (M, 3).")
    return np.ascontiguousarray(array, dtype=np.int64)


def _decode_vtk_faces(raw):
    """Decode VTK's ``[n, i0, i1, ..., n, ...]`` connectivity into triangles."""
    faces = []
    cursor = 0
    while cursor < len(raw):
        size = int(raw[cursor])
        indices = raw[cursor + 1 : cursor + 1 + size]
        for i in range(1, size - 1):
            faces.append((indices[0], indices[i], indices[i + 1]))
        cursor += size + 1
    if not faces:
        return np.zeros((0, 3), dtype=np.int64)
    return np.asarray(faces, dtype=np.int64)
