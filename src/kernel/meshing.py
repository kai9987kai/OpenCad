"""Isosurface extraction - turning a field into a mesh.

OpenCad previously reached VTK's ``contour`` filter for this, which meant
lattice generation could not run headlessly, could not be unit tested, and could
not be cancelled halfway through.  This module does it in numpy.

Why Surface Nets rather than marching cubes
-------------------------------------------
Marching cubes needs a 256-entry case table, produces triangles of wildly
varying quality, and creates cracks between cases unless the table is exactly
right.  Surface Nets (naive dual contouring) places **one vertex per cell that
straddles the surface** and joins neighbouring cells across each sign-changing
edge.  That gives:

- a manifold, watertight result by construction - every interior edge is shared
  by exactly two triangles, because every quad is built from four cells;
- far more uniform triangles, which matters for anything downstream that does
  numerics on the mesh;
- a fully vectorised implementation with no case table to get wrong.

The trade is that a dual method rounds sharp creases.  For lattices, organic
blends, and offset surfaces - what fields are actually good at - that is the
right trade.  ``sharpen`` recovers some of it by projecting vertices back onto
the true zero level set.

Known limits
------------
One vertex per cell cannot represent two separate sheets of surface passing
through the same cell.  Where a design pinches to less than one cell thick - a
lattice grazed tangentially by its bounding solid, say - the two sheets merge
into a non-manifold edge.  The result stays *closed* (no boundary edges, so it
still exports and slices), but ``is_watertight`` reports False because those
edges are shared by four triangles rather than two.  Raising the resolution
helps only if the feature has real thickness; if the geometry genuinely pinches
to zero, no resolution will fix it, and the design is the thing to change.
"""

from __future__ import annotations

import numpy as np

from src.kernel.mesh import Mesh
from src.kernel.sdf import SDF, sample_grid

__all__ = ["grid_to_mesh", "surface_nets", "voxelize"]


# The twelve edges of a cell, as pairs of corner indices in a (2, 2, 2) corner
# ordering where corner index = 4*i + 2*j + k.
_EDGE_CORNERS = np.array(
    [
        [0, 1], [2, 3], [4, 5], [6, 7],   # along k (z)
        [0, 2], [1, 3], [4, 6], [5, 7],   # along j (y)
        [0, 4], [1, 5], [2, 6], [3, 7],   # along i (x)
    ]
)

_CORNER_OFFSETS = np.array(
    [[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=np.int64
)


def grid_to_mesh(values, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0), level=0.0, sharpen=None):
    """Extract the ``level`` isosurface from a 3-D scalar array.

    ``values`` is indexed ``[i, j, k]`` over x, y, z, matching
    :func:`src.kernel.sdf.sample_grid`.  Negative is treated as inside, so a
    signed distance field meshes with outward-facing normals and positive
    volume.

    ``sharpen`` optionally takes the field the grid was sampled from; vertices
    are then Newton-projected onto its true zero level set, which removes the
    staircase a coarse grid leaves behind.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 3:
        raise ValueError("Isosurface extraction needs a 3-D array of samples.")
    if min(values.shape) < 2:
        raise ValueError("Each grid axis needs at least two samples.")

    origin = np.asarray(origin, dtype=float).reshape(3)
    spacing = np.broadcast_to(np.asarray(spacing, dtype=float), (3,)).astype(float)
    if np.any(spacing <= 0):
        raise ValueError("Grid spacing must be positive in every axis.")

    field = values - float(level)
    inside = field < 0.0

    # ------------------------------------------------------------------
    # 1. Find cells the surface passes through.
    # ------------------------------------------------------------------
    # Gather the eight corner values of every cell: shape (nx-1, ny-1, nz-1, 8).
    nx, ny, nz = field.shape
    corners = np.stack(
        [
            field[a : nx - 1 + a, b : ny - 1 + b, c : nz - 1 + c]
            for a, b, c in _CORNER_OFFSETS
        ],
        axis=-1,
    )
    corner_inside = corners < 0.0
    straddles = corner_inside.any(axis=-1) & ~corner_inside.all(axis=-1)

    cell_index = np.argwhere(straddles)
    if len(cell_index) == 0:
        return Mesh.empty()

    # ------------------------------------------------------------------
    # 2. Place one vertex per straddling cell, at the centroid of the
    #    zero crossings along its twelve edges.
    # ------------------------------------------------------------------
    active = corners[straddles]                      # (C, 8)
    a_values = active[:, _EDGE_CORNERS[:, 0]]        # (C, 12)
    b_values = active[:, _EDGE_CORNERS[:, 1]]
    crossing = (a_values < 0.0) != (b_values < 0.0)

    denominator = b_values - a_values
    denominator = np.where(np.abs(denominator) > 1e-30, denominator, 1.0)
    t = np.clip(-a_values / denominator, 0.0, 1.0)

    corner_a = _CORNER_OFFSETS[_EDGE_CORNERS[:, 0]].astype(float)  # (12, 3)
    corner_b = _CORNER_OFFSETS[_EDGE_CORNERS[:, 1]].astype(float)
    # Local position of each edge crossing within the unit cell.
    local = corner_a[None, :, :] + t[:, :, None] * (corner_b - corner_a)[None, :, :]

    weight = crossing.astype(float)
    total = weight.sum(axis=1)
    total = np.where(total > 0, total, 1.0)
    offset = (local * weight[:, :, None]).sum(axis=1) / total[:, None]

    vertices = origin + (cell_index + offset) * spacing

    # A lookup from cell coordinate to vertex index, for stitching quads.
    lookup = np.full(np.array(values.shape) - 1, -1, dtype=np.int64)
    lookup[tuple(cell_index.T)] = np.arange(len(cell_index))

    # ------------------------------------------------------------------
    # 3. Emit one quad per sign-changing grid edge, joining the four cells
    #    that share it.  This is what guarantees a manifold result.
    # ------------------------------------------------------------------
    faces = []
    shape = np.array(values.shape)
    # Walking the four cells around an edge in a fixed rotational order gives a
    # loop that is counter-clockwise in the plane of the two perpendicular axes.
    # Whether that loop's normal points along +axis or -axis depends on the
    # handedness of that axis pair: y-z and x-y are right-handed with respect to
    # x and z, but x-z is left-handed with respect to y.
    handedness = {0: 1.0, 1: -1.0, 2: 1.0}
    for axis in range(3):
        # Interior edges along `axis` whose endpoints differ in sign.
        slicer_low = [slice(1, shape[i] - 1) for i in range(3)]
        slicer_high = list(slicer_low)
        slicer_low[axis] = slice(0, shape[axis] - 1)
        slicer_high[axis] = slice(1, shape[axis])

        start = inside[tuple(slicer_low)]
        end = inside[tuple(slicer_high)]
        changed = start != end
        if not changed.any():
            continue

        edge_index = np.argwhere(changed)
        # Re-express in full-grid coordinates: the two non-axis dimensions were
        # sliced from 1, so add the offset back.
        base = edge_index.copy()
        for i in range(3):
            if i != axis:
                base[:, i] += 1

        # The four cells around this edge are the cells whose minimum corner is
        # the edge start minus 0 or 1 in each of the two perpendicular axes.
        others = [i for i in range(3) if i != axis]
        quad = np.empty((len(base), 4), dtype=np.int64)
        deltas = [(0, 0), (-1, 0), (-1, -1), (0, -1)]
        valid = np.ones(len(base), dtype=bool)
        for slot, (du, dv) in enumerate(deltas):
            cell = base.copy()
            cell[:, others[0]] += du
            cell[:, others[1]] += dv
            in_range = np.ones(len(base), dtype=bool)
            for i in range(3):
                in_range &= (cell[:, i] >= 0) & (cell[:, i] < lookup.shape[i])
            picked = np.where(in_range, lookup[tuple(np.clip(cell, 0, np.array(lookup.shape) - 1).T)], -1)
            quad[:, slot] = picked
            valid &= picked >= 0

        quad = quad[valid]
        if len(quad) == 0:
            continue

        # Orient the quad so its normal points from inside to outside.  When the
        # low end of the edge is the inside one, "outward" is +axis.
        starts_inside = start[tuple(edge_index.T)][valid]
        reverse = (handedness[axis] > 0) != starts_inside
        ordered = np.where(reverse[:, None], quad[:, ::-1], quad)

        faces.append(np.stack([ordered[:, 0], ordered[:, 1], ordered[:, 2]], axis=1))
        faces.append(np.stack([ordered[:, 0], ordered[:, 2], ordered[:, 3]], axis=1))

    if not faces:
        return Mesh.empty()

    mesh = Mesh(vertices, np.vstack(faces)).remove_degenerate_faces()

    if sharpen is not None and not mesh.is_empty:
        field_object = sharpen if isinstance(sharpen, SDF) else SDF(sharpen)
        projected = field_object.project(mesh.vertices, iterations=3)
        # Reject projections that moved a vertex further than one cell: those
        # are points where Newton diverged, and keeping the original is safer.
        limit = float(np.linalg.norm(spacing))
        moved = np.linalg.norm(projected - mesh.vertices, axis=1)
        safe = moved < limit
        vertices = mesh.vertices.copy()
        vertices[safe] = projected[safe]
        mesh = Mesh(vertices, mesh.faces)

    return mesh


def surface_nets(field, bounds=None, resolution=64, level=0.0, sharpen=True, progress=None):
    """Mesh a signed distance field.

    ``resolution`` is the number of samples along each axis (an int, or a
    3-tuple for an anisotropic grid).  Cost grows with its cube, so 64 is a
    good preview and 192 is a good export.

    ``bounds`` defaults to the field's own extent.  Fields that are unbounded by
    construction - a TPMS lattice, an infinite array, a half-space - have no
    extent to infer, so intersect them with a solid first or pass bounds
    explicitly.

    ``progress`` is an optional ``callable(fraction, message)`` so the UI can
    show something during a long extraction.
    """
    if progress:
        progress(0.05, "Preparing field")

    values, origin, spacing = sample_grid(field, bounds, resolution)

    if progress:
        progress(0.6, "Extracting isosurface")

    mesh = grid_to_mesh(
        values,
        origin,
        spacing,
        level=level,
        sharpen=field if sharpen and isinstance(field, SDF) else None,
    )

    if progress:
        progress(1.0, f"Meshed {mesh.n_faces} triangles")
    return mesh


def voxelize(field, bounds=None, resolution=64, level=0.0):
    """Sample a field into a boolean occupancy grid.

    Returns ``(occupied, origin, spacing)``.  Useful for volume fraction
    estimates and as the input to voxel-based analysis, where the surface is
    not needed at all.
    """
    values, origin, spacing = sample_grid(field, bounds, resolution)
    return values < float(level), origin, spacing
