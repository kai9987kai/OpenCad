"""Analytic mesh primitives for the OpenCad geometry kernel.

OpenCad used to reach for ``pyvista.Sphere`` and friends whenever it needed a
solid, which meant no geometry could be authored without VTK, a GPU, or a
display.  This module generates the same family of shapes from numpy alone, so
parts can be built in scripts, in headless CI, and in batch jobs, and so the
resulting topology is ours to reason about rather than VTK's.

Conventions
-----------
- Every *solid* primitive is watertight, consistently wound counter-clockwise
  seen from outside, and therefore has a positive :attr:`Mesh.volume`.
- ``plane`` and ``disc`` are deliberately open surfaces: they report boundary
  edges, zero volume, and ``genus is None``.
- Shapes are generated centred on the origin and aligned to +Z, then rotated
  onto ``direction``/``normal`` and translated to ``center``.
- Faceted primitives are *inscribed* in the ideal shape they approximate - every
  vertex lies exactly on the ideal surface - so their area and volume are
  slightly small and converge from below as the resolution rises.  The error of
  a revolved primitive falls as O(1/resolution^2).

The one shared workhorse is :func:`_lathe`, which revolves a ``(rho, z)``
profile about +Z.  Spheres, cylinders, cones, capsules, tori, and tubes are all
that same operation with a different profile, which is why they share their
winding rules and cannot disagree about which way is out.
"""

from __future__ import annotations

import numpy as np

from src.kernel.mesh import EPS, Mesh, rotation_matrix

__all__ = [
    "OPEN_PRIMITIVES",
    "PRIMITIVES",
    "box",
    "capsule",
    "cone",
    "create",
    "cube",
    "cylinder",
    "direction_matrix",
    "disc",
    "icosahedron",
    "icosphere",
    "octahedron",
    "plane",
    "prism",
    "pyramid",
    "sphere",
    "tetrahedron",
    "torus",
    "tube",
    "wedge",
]


# ----------------------------------------------------------------------
# Argument validation
# ----------------------------------------------------------------------
def _positive(value, name):
    """Coerce to a strictly positive finite float or raise ``ValueError``."""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a number (got {value!r}).") from error
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite number (got {value!r}).")
    return number


def _non_negative(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a number (got {value!r}).") from error
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number (got {value!r}).")
    return number


def _count(value, name, minimum=3):
    """Coerce a resolution/segment count to an int of at least ``minimum``."""
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer (got {value!r}).") from error
    if number < minimum:
        raise ValueError(f"{name} must be at least {minimum} (got {value!r}).")
    return number


def _vector(value, size, name):
    """Broadcast a scalar or sequence to a finite float array of ``size``."""
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 1:
        array = np.repeat(array, size)
    if array.size != size:
        raise ValueError(f"{name} must have {size} components (got {value!r}).")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite (got {value!r}).")
    return array


def _positive_vector(value, size, name):
    array = _vector(value, size, name)
    if np.any(array <= 0.0):
        raise ValueError(f"Every component of {name} must be positive (got {value!r}).")
    return array


def _count_vector(value, size, name, minimum=1):
    array = np.asarray(value).reshape(-1)
    if array.size == 1:
        array = np.repeat(array, size)
    if array.size != size:
        raise ValueError(f"{name} must have {size} components (got {value!r}).")
    return [_count(component, name, minimum) for component in array.tolist()]


# ----------------------------------------------------------------------
# Placement
# ----------------------------------------------------------------------
def direction_matrix(direction, reference=(0.0, 0.0, 1.0)):
    """3x3 rotation carrying ``reference`` onto ``direction`` by the shortest arc.

    Antiparallel inputs have no unique shortest arc; rather than dividing by a
    zero-length cross product (which is where naive implementations produce
    NaNs) a deterministic perpendicular axis is chosen and the rotation is a
    half turn.  The result is always a proper rotation - determinant +1 - so
    applying it never flips triangle winding.
    """
    target = _vector(direction, 3, "direction")
    source = _vector(reference, 3, "reference")
    target_length = float(np.linalg.norm(target))
    source_length = float(np.linalg.norm(source))
    if target_length <= EPS:
        raise ValueError("direction must be a non-zero vector.")
    if source_length <= EPS:
        raise ValueError("reference must be a non-zero vector.")
    target = target / target_length
    source = source / source_length

    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    axis = np.cross(source, target)
    sine = float(np.linalg.norm(axis))
    if sine <= 1e-12:
        if cosine > 0.0:
            return np.eye(3)
        # Antiparallel: any axis perpendicular to ``source`` will do.
        axis = np.cross(source, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) <= 1e-9:
            axis = np.cross(source, [0.0, 1.0, 0.0])
        return rotation_matrix(180.0, axis)
    return rotation_matrix(np.degrees(np.arctan2(sine, cosine)), axis)


def _place(mesh, center, direction=None):
    """Rotate a +Z-aligned mesh onto ``direction`` then move it to ``center``."""
    if direction is not None:
        matrix = direction_matrix(direction)
        if not np.allclose(matrix, np.eye(3)):
            mesh = mesh.transform(matrix)
    offset = _vector(center, 3, "center")
    if np.any(offset != 0.0):
        mesh = mesh.translated(offset)
    return mesh


# ----------------------------------------------------------------------
# Surface of revolution
# ----------------------------------------------------------------------
def _lathe(profile, resolution, closed=False, cap_bottom=True, cap_top=True):
    """Revolve a ``(rho, z)`` profile about the +Z axis into a triangle mesh.

    ``profile`` is walked in order and each consecutive pair becomes a band of
    quads split into two triangles.  Outward orientation follows from the
    profile direction: the outward normal is the profile tangent turned -90
    degrees in the ``(rho, z)`` half-plane, so an open profile must run from
    low z to high z and a ``closed`` profile loop must be counter-clockwise in
    ``(rho, z)``.

    An open profile whose first or last station sits on the axis (``rho == 0``)
    becomes a pole/apex: the band collapses to a triangle fan, which is why the
    poles of a sphere carry no zero-area triangles.  Otherwise ``cap_bottom`` /
    ``cap_top`` close the ends with a flat fan around a hub vertex.  ``closed``
    profiles form a torus-like tube and are never capped.
    """
    profile = np.asarray(profile, dtype=float).reshape(-1, 2)
    segments = _count(resolution, "resolution")
    rho = profile[:, 0]
    height = profile[:, 1]
    if np.any(rho < 0.0):
        raise ValueError("Revolved profile radii must be non-negative.")

    apex_bottom = (not closed) and rho[0] <= EPS
    apex_top = (not closed) and rho[-1] <= EPS
    first = 1 if apex_bottom else 0
    last = len(profile) - 1 if apex_top else len(profile)
    ring_rho = rho[first:last]
    ring_z = height[first:last]
    rings = len(ring_rho)
    if rings < 1 or np.any(ring_rho <= EPS):
        raise ValueError("A revolved profile may only touch the axis at its ends.")
    if closed:
        cap_bottom = cap_top = False
        if rings < 3:
            raise ValueError("A closed revolved profile needs at least three stations.")

    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    ring = np.empty((rings, segments, 3))
    ring[:, :, 0] = ring_rho[:, None] * np.cos(theta)[None, :]
    ring[:, :, 1] = ring_rho[:, None] * np.sin(theta)[None, :]
    ring[:, :, 2] = ring_z[:, None]

    vertices = [ring.reshape(-1, 3)]
    faces = []

    step = np.arange(segments)
    nxt = (step + 1) % segments
    lower = np.arange(rings if closed else rings - 1)
    if len(lower):
        upper = (lower + 1) % rings
        low = lower[:, None] * segments + step[None, :]
        low_next = lower[:, None] * segments + nxt[None, :]
        high = upper[:, None] * segments + step[None, :]
        high_next = upper[:, None] * segments + nxt[None, :]
        faces.append(np.stack([low, low_next, high_next], axis=-1).reshape(-1, 3))
        faces.append(np.stack([low, high_next, high], axis=-1).reshape(-1, 3))

    cursor = rings * segments
    if apex_bottom or cap_bottom:
        hub_z = height[0] if apex_bottom else ring_z[0]
        vertices.append(np.array([[0.0, 0.0, hub_z]]))
        hub = np.full(segments, cursor)
        faces.append(np.stack([hub, nxt, step], axis=-1))
        cursor += 1
    if apex_top or cap_top:
        hub_z = height[-1] if apex_top else ring_z[-1]
        vertices.append(np.array([[0.0, 0.0, hub_z]]))
        hub = np.full(segments, cursor)
        top = (rings - 1) * segments
        faces.append(np.stack([top + step, top + nxt, hub], axis=-1))
        cursor += 1

    return Mesh(np.vstack(vertices), np.vstack(faces))


# ----------------------------------------------------------------------
# Boxes
# ----------------------------------------------------------------------
_BOX_CORNERS = np.array(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    dtype=float,
)

_BOX_FACES = np.array(
    [
        [0, 3, 2], [0, 2, 1],  # -Z
        [4, 5, 6], [4, 6, 7],  # +Z
        [0, 1, 5], [0, 5, 4],  # -Y
        [2, 3, 7], [2, 7, 6],  # +Y
        [1, 2, 6], [1, 6, 5],  # +X
        [0, 4, 7], [0, 7, 3],  # -X
    ],
    dtype=np.int64,
)


def box(size=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0)):
    """Axis-aligned rectangular solid with full side lengths ``size``.

    Volume is exactly ``sx * sy * sz`` and area exactly
    ``2 * (sx*sy + sy*sz + sz*sx)`` - no discretisation error.
    """
    extents = _positive_vector(size, 3, "size")
    vertices = _BOX_CORNERS * (extents / 2.0)
    return _place(Mesh(vertices, _BOX_FACES), center)


def cube(size=1.0, center=(0.0, 0.0, 0.0)):
    """Cube of edge length ``size``; volume ``size ** 3`` exactly."""
    edge = _positive(size, "size")
    return box(size=(edge, edge, edge), center=center)


# ----------------------------------------------------------------------
# Spheres
# ----------------------------------------------------------------------
def sphere(radius=1.0, theta_resolution=32, phi_resolution=16, center=(0.0, 0.0, 0.0)):
    """UV (latitude/longitude) sphere.

    ``theta_resolution`` is the number of segments around the equator and
    ``phi_resolution`` the number of latitude bands from pole to pole, giving
    ``2 * theta_resolution * (phi_resolution - 1)`` triangles.  The caps are
    triangle fans rather than collapsed quads, so no face has zero area.

    Every vertex lies exactly on the sphere, so the result is inscribed: area
    and volume are always slightly below ``4*pi*r^2`` and ``4/3*pi*r^3`` and
    converge from below as O(1/resolution^2).  Triangle size varies strongly
    with latitude; prefer :func:`icosphere` for simulation or analysis.
    """
    radius = _positive(radius, "radius")
    theta_resolution = _count(theta_resolution, "theta_resolution")
    phi_resolution = _count(phi_resolution, "phi_resolution")

    phi = np.linspace(np.pi, 0.0, phi_resolution + 1)
    profile = np.column_stack([radius * np.sin(phi), radius * np.cos(phi)])
    # sin(pi) is 1.2e-16, not 0; pin the poles so the fan logic sees them.
    profile[0] = (0.0, -radius)
    profile[-1] = (0.0, radius)
    return _place(_lathe(profile, theta_resolution), center)


_ICOSAHEDRON_FACES = np.array(
    [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ],
    dtype=np.int64,
)


def _icosahedron_vertices():
    """The 12 corners of a regular icosahedron on the unit sphere."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    raw = np.array(
        [
            [-1.0, phi, 0.0], [1.0, phi, 0.0], [-1.0, -phi, 0.0], [1.0, -phi, 0.0],
            [0.0, -1.0, phi], [0.0, 1.0, phi], [0.0, -1.0, -phi], [0.0, 1.0, -phi],
            [phi, 0.0, -1.0], [phi, 0.0, 1.0], [-phi, 0.0, -1.0], [-phi, 0.0, 1.0],
        ]
    )
    return raw / np.linalg.norm(raw[0])


def icosahedron(size=1.0, center=(0.0, 0.0, 0.0)):
    """Regular icosahedron whose vertices sit at radius ``size`` (circumradius).

    Volume is ``(5/12) * (3 + sqrt(5)) * a^3`` and area ``5 * sqrt(3) * a^2``
    for edge length ``a = 4 * size / sqrt(10 + 2*sqrt(5))``.
    """
    radius = _positive(size, "size")
    return _place(Mesh(_icosahedron_vertices() * radius, _ICOSAHEDRON_FACES), center)


def icosphere(radius=1.0, subdivisions=2, center=(0.0, 0.0, 0.0)):
    """Geodesic sphere: an icosahedron subdivided ``subdivisions`` times.

    Face count is ``20 * 4 ** subdivisions``.  Triangles stay near-equilateral
    everywhere - unlike a UV sphere, which crowds slivers at the poles - which
    makes this the primitive to use when triangle quality matters (FEA meshes,
    curvature estimates, ray sampling).

    Like the UV sphere it is inscribed, so area and volume approach the analytic
    values from below, roughly quartering the error per subdivision level.
    """
    radius = _positive(radius, "radius")
    levels = _count(subdivisions, "subdivisions", minimum=0)

    mesh = Mesh(_icosahedron_vertices(), _ICOSAHEDRON_FACES).subdivide(levels)
    lengths = np.linalg.norm(mesh.vertices, axis=1)
    vertices = mesh.vertices * (radius / np.maximum(lengths, EPS))[:, None]
    return _place(Mesh(vertices, mesh.faces), center)


# ----------------------------------------------------------------------
# Revolved solids
# ----------------------------------------------------------------------
def cylinder(
    radius=0.5,
    height=1.0,
    resolution=32,
    center=(0.0, 0.0, 0.0),
    direction=(0.0, 0.0, 1.0),
    capped=True,
):
    """Right circular cylinder, built along +Z then rotated onto ``direction``.

    The cross-section is a regular ``resolution``-gon inscribed in the circle,
    so the volume is exactly ``height * (n/2) * r^2 * sin(2*pi/n)`` - slightly
    under ``pi*r^2*h`` - and converges to it as O(1/n^2).

    With ``capped=False`` the result is an open tube with two boundary loops,
    not a solid; it reports zero volume by design.
    """
    radius = _positive(radius, "radius")
    height = _positive(height, "height")
    resolution = _count(resolution, "resolution")
    half = height / 2.0
    profile = [(radius, -half), (radius, half)]
    mesh = _lathe(profile, resolution, cap_bottom=bool(capped), cap_top=bool(capped))
    return _place(mesh, center, direction)


def cone(
    radius=0.5,
    height=1.0,
    resolution=32,
    center=(0.0, 0.0, 0.0),
    direction=(0.0, 0.0, 1.0),
    capped=True,
):
    """Right circular cone with its apex on the +``direction`` end.

    ``center`` is the midpoint of the axis, so the base sits at ``-height/2``
    and the apex at ``+height/2`` along ``direction``.  Volume is exactly
    ``height/3 * (n/2) * r^2 * sin(2*pi/n)``.

    ``capped=False`` omits the base, leaving an open surface.
    """
    radius = _positive(radius, "radius")
    height = _positive(height, "height")
    resolution = _count(resolution, "resolution")
    half = height / 2.0
    profile = [(radius, -half), (0.0, half)]
    mesh = _lathe(profile, resolution, cap_bottom=bool(capped))
    return _place(mesh, center, direction)


def torus(
    major_radius=1.0,
    minor_radius=0.25,
    major_resolution=48,
    minor_resolution=24,
    center=(0.0, 0.0, 0.0),
):
    """Torus in the XY plane; a closed genus-1 surface with no caps.

    ``minor_radius`` must stay below ``major_radius`` - a self-intersecting
    "spindle" torus would not bound a well-defined volume.  Volume converges to
    ``2*pi^2 * R * r^2`` and area to ``4*pi^2 * R * r`` as O(1/resolution^2).
    """
    major_radius = _positive(major_radius, "major_radius")
    minor_radius = _positive(minor_radius, "minor_radius")
    if minor_radius >= major_radius:
        raise ValueError(
            "minor_radius must be smaller than major_radius; "
            f"got minor_radius={minor_radius!r} and major_radius={major_radius!r}."
        )
    major_resolution = _count(major_resolution, "major_resolution")
    minor_resolution = _count(minor_resolution, "minor_resolution")

    angle = np.linspace(0.0, 2.0 * np.pi, minor_resolution, endpoint=False)
    profile = np.column_stack(
        [major_radius + minor_radius * np.cos(angle), minor_radius * np.sin(angle)]
    )
    return _place(_lathe(profile, major_resolution, closed=True), center)


def tube(
    outer_radius=1.0,
    inner_radius=0.6,
    height=1.0,
    resolution=48,
    center=(0.0, 0.0, 0.0),
):
    """Hollow pipe: two coaxial prisms joined by annular end caps.

    Watertight and genus 1 - the bore is a real through-hole, not a dimple.
    The solid is exactly the outer prism minus the inner prism, so the volume is
    ``height * (n/2) * (ro^2 - ri^2) * sin(2*pi/n)`` with no further error.
    """
    outer_radius = _positive(outer_radius, "outer_radius")
    inner_radius = _positive(inner_radius, "inner_radius")
    if inner_radius >= outer_radius:
        raise ValueError(
            "inner_radius must be smaller than outer_radius; "
            f"got inner_radius={inner_radius!r} and outer_radius={outer_radius!r}."
        )
    height = _positive(height, "height")
    resolution = _count(resolution, "resolution")

    half = height / 2.0
    # Counter-clockwise in (rho, z) so the revolved normals point outward.
    profile = [
        (inner_radius, -half),
        (outer_radius, -half),
        (outer_radius, half),
        (inner_radius, half),
    ]
    return _place(_lathe(profile, resolution, closed=True), center)


def capsule(radius=0.5, height=1.0, resolution=24, center=(0.0, 0.0, 0.0)):
    """Cylinder of length ``height`` closed by two hemispherical caps.

    Total extent along Z is ``height + 2 * radius``.  Volume converges to
    ``pi*r^2*h + 4/3*pi*r^3`` and area to ``2*pi*r*h + 4*pi*r^2`` from below.
    Each cap uses ``max(2, resolution // 4)`` latitude bands, which keeps the
    quads roughly square without a second resolution knob.
    """
    radius = _positive(radius, "radius")
    height = _positive(height, "height")
    resolution = _count(resolution, "resolution")

    bands = max(2, resolution // 4)
    half = height / 2.0
    lower_phi = np.linspace(np.pi, np.pi / 2.0, bands + 1)
    upper_phi = np.linspace(np.pi / 2.0, 0.0, bands + 1)
    lower = np.column_stack(
        [radius * np.sin(lower_phi), radius * np.cos(lower_phi) - half]
    )
    upper = np.column_stack(
        [radius * np.sin(upper_phi), radius * np.cos(upper_phi) + half]
    )
    profile = np.vstack([lower, upper])
    # Pin the exact stations; sin/cos at pi and pi/2 leave 1e-16 residue.
    profile[0] = (0.0, -half - radius)
    profile[bands] = (radius, -half)
    profile[bands + 1] = (radius, half)
    profile[-1] = (0.0, half + radius)
    return _place(_lathe(profile, resolution), center)


def prism(sides=6, radius=1.0, height=1.0, center=(0.0, 0.0, 0.0)):
    """Regular ``sides``-gon extruded along Z, first vertex on the +X axis.

    ``radius`` is the circumradius of the polygon, so the volume is exactly
    ``height * (n/2) * radius^2 * sin(2*pi/n)``.
    """
    sides = _count(sides, "sides")
    radius = _positive(radius, "radius")
    height = _positive(height, "height")
    return cylinder(radius=radius, height=height, resolution=sides, center=center)


# ----------------------------------------------------------------------
# Open surfaces
# ----------------------------------------------------------------------
def plane(
    size=(1.0, 1.0),
    resolution=(1, 1),
    center=(0.0, 0.0, 0.0),
    normal=(0.0, 0.0, 1.0),
):
    """Open rectangular grid patch, built in XY then rotated onto ``normal``.

    ``resolution`` counts quads per axis, so the patch has
    ``2 * nx * ny`` triangles and ``2 * (nx + ny)`` boundary edges.  Area is
    exactly ``size[0] * size[1]``; volume is zero because the surface is open.
    """
    extents = _positive_vector(size, 2, "size")
    nx, ny = _count_vector(resolution, 2, "resolution", minimum=1)

    xs = np.linspace(-extents[0] / 2.0, extents[0] / 2.0, nx + 1)
    ys = np.linspace(-extents[1] / 2.0, extents[1] / 2.0, ny + 1)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    vertices = np.column_stack(
        [grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)]
    )

    corner = np.arange(nx)[:, None] * (ny + 1) + np.arange(ny)[None, :]
    right = corner + (ny + 1)
    faces = np.vstack(
        [
            np.stack([corner, right, right + 1], axis=-1).reshape(-1, 3),
            np.stack([corner, right + 1, corner + 1], axis=-1).reshape(-1, 3),
        ]
    )
    return _place(Mesh(vertices, faces), center, normal)


def disc(
    radius=1.0,
    inner_radius=0.0,
    resolution=48,
    center=(0.0, 0.0, 0.0),
    normal=(0.0, 0.0, 1.0),
):
    """Open flat disc, or an annulus when ``inner_radius`` is non-zero.

    The rim is a regular ``resolution``-gon inscribed in the circle, so the area
    is exactly ``(n/2) * (ro^2 - ri^2) * sin(2*pi/n)``.  A full disc has
    ``resolution`` boundary edges, an annulus ``2 * resolution``.
    """
    radius = _positive(radius, "radius")
    inner_radius = _non_negative(inner_radius, "inner_radius")
    if inner_radius >= radius:
        raise ValueError(
            "inner_radius must be smaller than radius; "
            f"got inner_radius={inner_radius!r} and radius={radius!r}."
        )
    resolution = _count(resolution, "resolution")

    theta = np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=False)
    unit = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(resolution)])
    step = np.arange(resolution)
    nxt = (step + 1) % resolution

    if inner_radius <= EPS:
        vertices = np.vstack([unit * radius, [[0.0, 0.0, 0.0]]])
        hub = np.full(resolution, resolution)
        faces = np.stack([hub, step, nxt], axis=-1)
    else:
        vertices = np.vstack([unit * inner_radius, unit * radius])
        outer = resolution
        faces = np.vstack(
            [
                np.stack([step, outer + step, outer + nxt], axis=-1),
                np.stack([step, outer + nxt, nxt], axis=-1),
            ]
        )
    return _place(Mesh(vertices, faces), center, normal)


# ----------------------------------------------------------------------
# Faceted solids
# ----------------------------------------------------------------------
def pyramid(base=(1.0, 1.0), height=1.0, center=(0.0, 0.0, 0.0)):
    """Rectangular pyramid: base at ``-height/2``, apex at ``+height/2``.

    Volume is exactly ``base_x * base_y * height / 3``.
    """
    extents = _positive_vector(base, 2, "base")
    height = _positive(height, "height")
    half_x, half_y, half_z = extents[0] / 2.0, extents[1] / 2.0, height / 2.0

    vertices = np.array(
        [
            [-half_x, -half_y, -half_z],
            [half_x, -half_y, -half_z],
            [half_x, half_y, -half_z],
            [-half_x, half_y, -half_z],
            [0.0, 0.0, half_z],
        ]
    )
    faces = np.array(
        [
            [0, 3, 2], [0, 2, 1],  # base, facing -Z
            [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4],
        ],
        dtype=np.int64,
    )
    return _place(Mesh(vertices, faces), center)


def wedge(size=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0)):
    """Triangular prism filling half of the ``size`` box - a ramp along +X.

    The sloped face runs from the low-X/high-Z edge down to the high-X/low-Z
    edge, so the volume is exactly ``sx * sy * sz / 2``.
    """
    extents = _positive_vector(size, 3, "size")
    half_x, half_y, half_z = extents / 2.0

    vertices = np.array(
        [
            [-half_x, -half_y, -half_z],
            [half_x, -half_y, -half_z],
            [-half_x, -half_y, half_z],
            [-half_x, half_y, -half_z],
            [half_x, half_y, -half_z],
            [-half_x, half_y, half_z],
        ]
    )
    faces = np.array(
        [
            [0, 1, 2],  # -Y triangle
            [3, 5, 4],  # +Y triangle
            [0, 3, 4], [0, 4, 1],  # -Z base
            [0, 2, 5], [0, 5, 3],  # -X back
            [1, 4, 5], [1, 5, 2],  # sloped face
        ],
        dtype=np.int64,
    )
    return _place(Mesh(vertices, faces), center)


def tetrahedron(size=1.0, center=(0.0, 0.0, 0.0)):
    """Regular tetrahedron whose vertices sit at radius ``size`` (circumradius).

    Volume is ``8 / (9 * sqrt(3)) * size^3`` and area ``(8 / sqrt(3)) * size^2``,
    which follow from the edge length ``a = size * sqrt(8/3)``.
    """
    radius = _positive(size, "size")
    vertices = (
        np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
            ]
        )
        * (radius / np.sqrt(3.0))
    )
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int64)
    return _place(Mesh(vertices, faces), center)


def octahedron(size=1.0, center=(0.0, 0.0, 0.0)):
    """Regular octahedron with vertices at ``+-size`` on each axis.

    Volume is exactly ``4/3 * size^3`` and area ``4 * sqrt(3) * size^2``.
    """
    radius = _positive(size, "size")
    vertices = (
        np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        )
        * radius
    )
    faces = np.array(
        [
            [0, 2, 4], [2, 1, 4], [3, 0, 4], [1, 3, 4],
            [2, 0, 5], [1, 2, 5], [0, 3, 5], [3, 1, 5],
        ],
        dtype=np.int64,
    )
    return _place(Mesh(vertices, faces), center)


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------
PRIMITIVES = {
    "box": box,
    "cube": cube,
    "sphere": sphere,
    "icosphere": icosphere,
    "cylinder": cylinder,
    "cone": cone,
    "torus": torus,
    "tube": tube,
    "capsule": capsule,
    "prism": prism,
    "pyramid": pyramid,
    "wedge": wedge,
    "tetrahedron": tetrahedron,
    "octahedron": octahedron,
    "icosahedron": icosahedron,
    "plane": plane,
    "disc": disc,
}

#: Names that are not solids - they come out open, with boundary edges.
OPEN_PRIMITIVES = frozenset({"plane", "disc"})

_ALIASES = {
    "ball": "sphere",
    "circle": "disc",
    "disk": "disc",
    "geodesic": "icosphere",
    "geodesic_sphere": "icosphere",
    "pipe": "tube",
    "quad": "plane",
    "rect": "plane",
    "rectangle": "plane",
    "tetra": "tetrahedron",
    "uv_sphere": "sphere",
}


def create(name, **params):
    """Build a primitive by name, e.g. ``create("cylinder", radius=2.0)``.

    Names are matched case-insensitively with spaces and hyphens treated as
    underscores, and a handful of aliases ("ball", "disk", "pipe", ...) map onto
    the canonical factories.  Unknown names raise ``ValueError`` listing every
    supported name so a UI can surface the choice directly.
    """
    key = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    key = _ALIASES.get(key, key)
    factory = PRIMITIVES.get(key)
    if factory is None:
        supported = ", ".join(sorted(PRIMITIVES))
        raise ValueError(f"Unknown primitive '{name}'. Supported names: {supported}.")
    return factory(**params)
