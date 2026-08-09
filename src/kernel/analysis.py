"""Engineering and printability analysis.

OpenCad could already tell you a part's area and volume.  It could not tell you
whether the part would actually print, which is the question that matters before
committing eight hours of machine time.  This module answers the cheap, useful
version of that question: is the mesh closed, does it overhang, is it thicker
than the nozzle, how much material will it use, and where is its centre of mass.

Everything here is deliberately geometric.  There is no finite element solver
and no slicer simulation, and nothing pretends otherwise - a finding says what
was measured and against which threshold, so it can be argued with.
"""

from __future__ import annotations

import numpy as np

from src.kernel.mesh import Mesh
from src.kernel.units import UnitSystem

__all__ = [
    "Finding",
    "mesh_report",
    "oriented_bounding_box",
    "overhangs",
    "printability",
    "triangle_quality",
    "volume_fraction",
    "wall_thickness",
]

INFO = "info"
WARNING = "warning"
ERROR = "error"

_SEVERITY_ORDER = {ERROR: 0, WARNING: 1, INFO: 2}


class Finding:
    """One analysis result, at a severity the UI can colour and sort by."""

    __slots__ = ("detail", "severity", "title", "value")

    def __init__(self, severity, title, detail="", value=None):
        self.severity = severity
        self.title = str(title)
        self.detail = str(detail)
        self.value = value

    @property
    def is_problem(self):
        return self.severity in (WARNING, ERROR)

    def as_dict(self):
        return {
            "severity": self.severity,
            "title": self.title,
            "detail": self.detail,
            "value": self.value,
        }

    def __repr__(self):
        return f"Finding({self.severity}, {self.title!r})"


def overhangs(mesh, build_direction=(0.0, 0.0, 1.0), max_angle=45.0):
    """Find downward-facing triangles too steep to print without support.

    ``max_angle`` is measured from the build direction: a face whose normal
    points more than this far below horizontal needs support.  45 degrees is the
    usual FDM rule of thumb.

    Returns ``(mask, area, fraction)`` - which faces overhang, their total area,
    and the fraction of the model's surface they represent.
    """
    if mesh.is_empty:
        return np.zeros(0, dtype=bool), 0.0, 0.0

    direction = np.asarray(build_direction, dtype=float).reshape(3)
    length = np.linalg.norm(direction)
    if length <= 1e-12:
        raise ValueError("Build direction must be non-zero.")
    direction = direction / length

    normals = mesh.face_normals()
    # cos of the angle between the face normal and the *downward* direction.
    facing_down = normals @ (-direction)
    threshold = np.cos(np.deg2rad(90.0 - float(max_angle)))

    mask = facing_down > threshold
    areas = mesh.face_areas()
    total = float(areas.sum())
    overhang_area = float(areas[mask].sum())
    fraction = overhang_area / total if total > 0 else 0.0
    return mask, overhang_area, fraction


def oriented_bounding_box(mesh):
    """The smallest box aligned to the model's own principal axes.

    Returns ``(extents, rotation, center)``.  Useful for deciding how a part
    should sit on a build plate: the axis-aligned box of a diagonally modelled
    part badly overstates how much room it needs.

    The axes come from the principal axes of the surface, so this is the
    principal-axis box rather than the true minimum-volume box - close enough to
    choose an orientation, and far cheaper to compute.

    The covariance is integrated *continuously* over each triangle rather than
    sampled at its centroid.  That distinction matters: a box face split into
    two triangles has centroids that sit off-centre, and using them tilts the
    axes away from the box's own, which is exactly the case this function has to
    get right.
    """
    if mesh.is_empty:
        return np.zeros(3), np.eye(3), np.zeros(3)

    areas = mesh.face_areas()
    total = float(areas.sum())
    if total <= 0:
        return mesh.extents, np.eye(3), mesh.center

    tri = mesh.triangles()
    mean = (mesh.face_centroids() * areas[:, None]).sum(axis=0) / total

    # Closed form for the second moment of a triangle over its own area:
    #   int_T x x^T dA = (A/12) ((v0+v1+v2)(v0+v1+v2)^T + sum_i vi vi^T)
    corner_sum = tri.sum(axis=1)
    second_moment = np.einsum("m,mi,mj->ij", areas / 12.0, corner_sum, corner_sum)
    second_moment += np.einsum("m,mci,mcj->ij", areas / 12.0, tri, tri)

    covariance = second_moment / total - np.outer(mean, mean)
    covariance = (covariance + covariance.T) / 2.0
    _, axes = np.linalg.eigh(covariance)

    projected = mesh.vertices @ axes
    low, high = projected.min(axis=0), projected.max(axis=0)
    extents = high - low
    center = axes @ ((low + high) / 2.0)
    return extents, axes, center


def triangle_quality(mesh):
    """Aspect ratio and minimum angle statistics for the mesh's triangles.

    Sliver triangles break slicers and finite element solvers alike, and they
    are usually a symptom of a bad boolean rather than a problem in themselves.
    """
    if mesh.is_empty:
        return {"count": 0, "min_angle_deg": 0.0, "max_aspect_ratio": 0.0, "slivers": 0}

    tri = mesh.triangles()
    edges = np.stack(
        [
            np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
            np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
            np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1),
        ],
        axis=1,
    )
    longest = edges.max(axis=1)
    areas = mesh.face_areas()

    # Aspect ratio as longest edge over the radius of the inscribed circle.
    perimeter = edges.sum(axis=1)
    inradius = np.where(perimeter > 1e-15, 2.0 * areas / perimeter, 0.0)
    aspect = np.where(inradius > 1e-15, longest / (2.0 * np.sqrt(3.0) * inradius), np.inf)

    # Smallest angle sits opposite the shortest edge; use the law of cosines.
    ordered = np.sort(edges, axis=1)
    a, b, c = ordered[:, 0], ordered[:, 1], ordered[:, 2]
    denominator = 2.0 * b * c
    cosine = np.where(denominator > 1e-15, (b**2 + c**2 - a**2) / denominator, 1.0)
    min_angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    finite = aspect[np.isfinite(aspect)]
    return {
        "count": mesh.n_faces,
        "min_angle_deg": float(min_angle.min()),
        "mean_min_angle_deg": float(min_angle.mean()),
        "max_aspect_ratio": float(finite.max()) if len(finite) else float("inf"),
        "slivers": int((min_angle < 1.0).sum()),
        "degenerate": int((areas <= 1e-15).sum()),
    }


def volume_fraction(field, bounds=None, resolution=64):
    """Relative density of a field - the fraction of the box that is solid.

    This is the number that matters for a lattice: a gyroid at 15% relative
    density weighs 15% of the solid it replaces.

    Sampling happens at *cell centres* rather than grid nodes, making this a
    midpoint Riemann sum over the box.  Node sampling would bias the answer by a
    factor of ``((n-1)/n)^3`` for any solid that fills its own bounding box - a
    33% error at resolution 16, which is exactly the case a lattice's bounding
    box hits.
    """
    from src.kernel.sdf import _coerce

    field = _coerce(field)
    if bounds is None:
        low, high = field.sample_bounds()
    else:
        low = np.asarray(bounds[0], dtype=float).reshape(3)
        high = np.asarray(bounds[1], dtype=float).reshape(3)
    if np.any(high <= low):
        raise ValueError("Analysis bounds must have positive extent in every axis.")

    counts = np.broadcast_to(np.asarray(resolution, dtype=int), (3,)).astype(int)
    if np.any(counts < 1):
        raise ValueError("Analysis resolution must be at least 1 in every axis.")

    cell = (high - low) / counts
    axes = [low[i] + (np.arange(counts[i]) + 0.5) * cell[i] for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    occupied = np.zeros(len(grid), dtype=bool)
    step = 1 << 20
    for start in range(0, len(grid), step):
        stop = min(start + step, len(grid))
        occupied[start:stop] = np.asarray(field(grid[start:stop])).ravel() < 0.0

    cell_volume = float(np.prod(cell))
    return {
        "fraction": float(occupied.mean()),
        "solid_volume": float(occupied.sum() * cell_volume),
        "sampled_volume": float(np.prod(high - low)),
        "resolution": int(counts.max()),
    }


def wall_thickness(field, bounds=None, resolution=64):
    """Estimate the thinnest wall in an implicit solid.

    Inside the solid the field *is* the distance to the nearest surface, so
    twice the largest interior distance along a wall gives its thickness.  This
    reports the distribution of ``2 * |f|`` over solid samples, whose small
    percentiles indicate thin regions.

    This is an estimate on a grid: a feature thinner than one cell is invisible
    to it, so treat the resolution as the limit of what it can see.
    """
    from src.kernel.sdf import sample_grid

    values, _, spacing = sample_grid(field, bounds, resolution)
    inside = values[values < 0]
    cell = float(np.min(spacing))
    if inside.size == 0:
        return {"min": 0.0, "p01": 0.0, "p05": 0.0, "median": 0.0, "resolution_limit": cell}

    thickness = 2.0 * np.abs(inside)
    return {
        "min": float(thickness.min()),
        "p01": float(np.percentile(thickness, 1)),
        "p05": float(np.percentile(thickness, 5)),
        "median": float(np.median(thickness)),
        "resolution_limit": cell,
    }


def mesh_report(mesh, units=None):
    """A structured summary of a mesh: geometry, topology, and quality."""
    units = units or UnitSystem.millimetres()
    if not isinstance(mesh, Mesh):
        raise TypeError("mesh_report expects a kernel Mesh.")

    properties = mesh.mass_properties()
    low, high = mesh.bounding_box
    extents, _, _ = oriented_bounding_box(mesh)

    return {
        "geometry": {
            "vertices": mesh.n_vertices,
            "triangles": mesh.n_faces,
            "area": properties.area,
            "volume": properties.volume,
            "area_text": units.format_area(properties.area),
            "volume_text": units.format_volume(properties.volume),
            "bounds_min": low.tolist(),
            "bounds_max": high.tolist(),
            "size": (high - low).tolist(),
            "size_text": units.format_point(high - low, with_unit=True),
            "oriented_size": extents.tolist(),
            "center_of_mass": properties.center_of_mass.tolist(),
            "center_of_mass_text": units.format_point(properties.center_of_mass, with_unit=True),
            "principal_moments": properties.principal_moments.tolist(),
        },
        "topology": {
            "watertight": mesh.is_watertight,
            "edge_manifold": mesh.is_edge_manifold,
            "oriented": mesh.is_oriented,
            "components": mesh.n_components(),
            "boundary_edges": len(mesh.boundary_edges()),
            "non_manifold_edges": len(mesh.non_manifold_edges()),
            "euler_characteristic": mesh.euler_characteristic,
            "genus": mesh.genus,
        },
        "quality": triangle_quality(mesh),
    }


def printability(
    mesh,
    units=None,
    build_direction=(0.0, 0.0, 1.0),
    max_overhang_angle=45.0,
    min_feature_size=0.8,
    build_volume=None,
):
    """Check a mesh against the constraints of a typical FDM printer.

    ``min_feature_size`` is the smallest wall the process can produce - roughly
    two nozzle widths.  ``build_volume`` is an optional ``(x, y, z)`` in
    millimetres to check the part fits.

    Returns a list of :class:`Finding`, most severe first.  An empty list is not
    possible: a clean part still reports its material usage.
    """
    units = units or UnitSystem.millimetres()
    findings = []

    if mesh.is_empty:
        return [Finding(ERROR, "Empty model", "There is no geometry to print.")]

    # --- Topology: the difference between "prints" and "does not slice" -----
    if not mesh.is_watertight:
        boundary = len(mesh.boundary_edges())
        non_manifold = len(mesh.non_manifold_edges())
        if boundary:
            findings.append(
                Finding(
                    ERROR,
                    "Model is not closed",
                    f"{boundary} boundary edge(s) leave holes in the surface. "
                    "A slicer cannot tell inside from outside across a hole.",
                    boundary,
                )
            )
        if non_manifold:
            findings.append(
                Finding(
                    WARNING,
                    "Non-manifold edges",
                    f"{non_manifold} edge(s) are shared by more than two faces, "
                    "usually where the model pinches to zero thickness. "
                    "Most slicers cope, but the result there is undefined.",
                    non_manifold,
                )
            )
    if not mesh.is_oriented:
        findings.append(
            Finding(
                WARNING,
                "Inconsistent face winding",
                "Neighbouring triangles disagree about which side is outside. "
                "Recompute normals before exporting.",
            )
        )

    components = mesh.n_components()
    if components > 1:
        findings.append(
            Finding(
                INFO,
                "Multiple bodies",
                f"The model contains {components} separate shells; they will print "
                "as separate objects unless they are joined.",
                components,
            )
        )

    # --- Size ---------------------------------------------------------------
    size = mesh.extents
    if build_volume is not None:
        volume = np.asarray(build_volume, dtype=float).reshape(3)
        if np.any(size > volume + 1e-9):
            findings.append(
                Finding(
                    ERROR,
                    "Larger than the build volume",
                    f"The part measures {units.format_point(size, with_unit=True)} "
                    f"but the printer accepts {units.format_point(volume, with_unit=True)}.",
                    size.tolist(),
                )
            )

    # --- Overhangs ----------------------------------------------------------
    _, overhang_area, fraction = overhangs(mesh, build_direction, max_overhang_angle)
    if fraction > 0.01:
        severity = WARNING if fraction > 0.20 else INFO
        findings.append(
            Finding(
                severity,
                "Overhanging surfaces",
                f"{fraction * 100:.1f}% of the surface "
                f"({units.format_area(overhang_area)}) is steeper than "
                f"{max_overhang_angle:g}° and will need support.",
                fraction,
            )
        )

    # --- Thin features ------------------------------------------------------
    quality = triangle_quality(mesh)
    if quality["degenerate"]:
        findings.append(
            Finding(
                WARNING,
                "Degenerate triangles",
                f"{quality['degenerate']} triangle(s) have no area. Clean the mesh "
                "before exporting.",
                quality["degenerate"],
            )
        )
    elif quality["slivers"]:
        findings.append(
            Finding(
                INFO,
                "Sliver triangles",
                f"{quality['slivers']} triangle(s) have an angle under 1°, usually "
                "left behind by a boolean operation.",
                quality["slivers"],
            )
        )

    smallest = float(np.min(size))
    if smallest < float(min_feature_size):
        findings.append(
            Finding(
                WARNING,
                "Very thin part",
                f"The smallest overall dimension is {units.format_length(smallest)}, "
                f"below the {units.format_length(min_feature_size)} minimum feature size.",
                smallest,
            )
        )

    # --- Material -----------------------------------------------------------
    properties = mesh.mass_properties()
    if properties.volume > 0:
        findings.append(
            Finding(
                INFO,
                "Material estimate",
                f"Solid volume {units.format_volume(properties.volume)}; "
                f"at 1.24 g/cm³ (PLA) that is about "
                f"{properties.volume * 1.24e-3:.1f} g if printed solid.",
                properties.volume,
            )
        )

    # --- Stability ----------------------------------------------------------
    if properties.volume > 0:
        centre = properties.center_of_mass
        low, high = mesh.bounding_box
        height = high[2] - low[2]
        if height > 1e-9:
            relative = (centre[2] - low[2]) / height
            if relative > 0.6:
                findings.append(
                    Finding(
                        INFO,
                        "Top-heavy",
                        f"The centre of mass sits {relative * 100:.0f}% of the way up. "
                        "Consider a raft, or printing it another way up.",
                        relative,
                    )
                )

    findings.sort(key=lambda finding: _SEVERITY_ORDER.get(finding.severity, 3))
    return findings
