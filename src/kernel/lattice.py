"""Building a finite, closed lattice solid from a lattice specification.

A TPMS field is infinite by construction.  Meshing one directly over a box
slices it open at every wall that reaches the edge, giving a surface full of
holes rather than a part - so this module caps the lattice with the region it is
meant to fill and works out the grid to sample it on.

The specification is a plain dictionary, which keeps this usable from three
places at once: the Implicit Lab dock, the headless CLI, and the test suite.
Nothing here imports Qt.

Keys
----
``kind``          TPMS family; see :data:`src.kernel.sdf.TPMS_KINDS`.
``mode``          ``"sheet"`` for walls, ``"solid"`` for a strut network.
``period``        Unit cell size in millimetres.
``thickness``     Wall thickness in millimetres (sheet mode).
``level``         Surface offset (network mode); positive thickens the struts.
``grade_target``  ``"none"``, ``"thickness"``, or ``"period"``.
``grade_axis``    ``"x"``, ``"y"``, ``"z"``, or ``"radial"``.
``grade_amount``  Total change across the part, in millimetres.
``size``          Cube size, used when no explicit region is given.
"""

from __future__ import annotations

import numpy as np

from src.kernel import sdf

__all__ = ["DEFAULT_SPEC", "build_lattice_field", "cells_per_wall"]

DEFAULT_SPEC = {
    "kind": "gyroid",
    "mode": "sheet",
    "period": 6.0,
    "thickness": 0.8,
    "level": 0.0,
    "grade_target": "none",
    "grade_axis": "z",
    "grade_amount": 0.0,
    "size": 30.0,
    "resolution": 96,
}

#: Below roughly this many grid cells across a wall, Surface Nets cannot
#: represent the wall and the meshed lattice comes out lighter than specified.
MIN_CELLS_PER_WALL = 2.0


def cells_per_wall(spec, resolution, span=None):
    """How many grid cells fall across one lattice wall.

    Under :data:`MIN_CELLS_PER_WALL` the extracted mesh loses material: a 0.8 mm
    wall in a 30 mm cube at resolution 48 measures about 11% light, and the loss
    is silent.  Callers should warn rather than let that reach an export.
    """
    if span is None:
        span = float(spec.get("size", DEFAULT_SPEC["size"]))
    resolution = int(resolution)
    if resolution < 2 or span <= 0:
        return 0.0
    pitch = span * 1.06 / (resolution - 1)  # 1.06 accounts for the sampling pad
    if pitch <= 0:
        return 0.0
    return float(spec.get("thickness", DEFAULT_SPEC["thickness"])) / pitch


def build_lattice_field(spec, bounds=None):
    """Turn a lattice specification into a field and the box to sample it on.

    ``bounds`` is the region to fill as ``(min_xyz, max_xyz)``; when omitted a
    cube of ``spec["size"]`` centred on the origin is used.  Grading is
    expressed relative to that region, so "0.6 mm thicker along Z" means across
    the part rather than across some absolute distance.

    Returns ``(field, sampling_bounds, region_bounds)``:

    - ``field`` is the lattice **intersected with the region**, so it is a
      closed solid rather than an open surface;
    - ``sampling_bounds`` is padded slightly beyond the region, keeping the
      capped surface off the grid boundary where there are too few neighbouring
      cells to close the mesh;
    - ``region_bounds`` is what the caller asked to fill, which is the volume a
      relative-density figure has to be measured against.
    """
    merged = dict(DEFAULT_SPEC)
    merged.update(spec or {})

    kind = merged["kind"]
    period = float(merged["period"])
    thickness = float(merged["thickness"])

    if bounds is None:
        half = float(merged["size"]) / 2.0
        bounds = (np.full(3, -half), np.full(3, half))
    low = np.asarray(bounds[0], dtype=float).reshape(3)
    high = np.asarray(bounds[1], dtype=float).reshape(3)
    if np.any(high <= low):
        raise ValueError("The lattice region must have positive extent in every axis.")

    target = merged.get("grade_target", "none")
    amount = float(merged.get("grade_amount", 0.0))
    axis = str(merged.get("grade_axis", "z")).lower()

    def ramp(points):
        """0 at one end of the region, 1 at the other."""
        if axis == "radial":
            centre = (low + high) / 2.0
            radius = float(np.linalg.norm(high - low)) / 2.0
            if radius <= 0:
                return np.zeros(len(points))
            return np.clip(np.linalg.norm(points - centre, axis=1) / radius, 0.0, 1.0)
        index = "xyz".index(axis) if axis in "xyz" else 2
        span = high[index] - low[index]
        if span <= 0:
            return np.zeros(len(points))
        return np.clip((points[:, index] - low[index]) / span, 0.0, 1.0)

    if target == "thickness" and amount:
        # Clamp the wall positive: a zero or negative thickness turns the sheet
        # field inside out rather than simply vanishing.
        floor = max(0.05, thickness - abs(amount) / 2.0)

        def wall(points):
            return np.maximum(thickness + amount * (ramp(points) - 0.5), floor)
    else:
        wall = thickness

    if target == "period" and amount:
        floor = max(0.5, period - abs(amount) / 2.0)

        def cell(points):
            return np.maximum(period + amount * (ramp(points) - 0.5), floor)
    else:
        cell = period

    if merged.get("mode", "sheet") == "sheet":
        lattice = sdf.tpms_sheet(kind, period=cell, thickness=wall)
    else:
        lattice = sdf.tpms_solid(kind, period=cell, level=float(merged.get("level", 0.0)))

    region = sdf.box(high - low, center=(low + high) / 2.0)
    field = lattice & region

    pad = np.maximum((high - low) * 0.03, 1e-3)
    return field, (low - pad, high + pad), (low, high)
