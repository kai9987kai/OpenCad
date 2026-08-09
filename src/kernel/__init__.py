"""OpenCad geometry kernel.

A dependency-light CAD kernel: everything in this package runs on numpy alone,
with no VTK, OpenGL, or Qt.  That keeps the geometry layer testable headlessly,
scriptable from the command line, and reusable outside the desktop app.

The kernel carries two representations of a solid and converts between them:

- **Meshes** (:mod:`~src.kernel.mesh`) - a boundary representation, good for
  imported scans, file exchange, and anything measured.
- **Fields** (:mod:`~src.kernel.sdf`) - an implicit representation, good for
  exact offsets, smooth blends, infinite arrays, and graded lattices.

:mod:`~src.kernel.meshing` turns a field into a mesh.

A quick tour::

    from src.kernel import primitives, sdf
    from src.kernel.analysis import printability
    from src.kernel.meshing import surface_nets

    bracket = sdf.rounded_box((40, 20, 8), radius=2)
    bracket -= sdf.cylinder(3, 20).translate((12, 0, 0))
    mesh = surface_nets(bracket, resolution=128)

    for finding in printability(mesh):
        print(finding.severity, finding.title)
"""

from __future__ import annotations

from src.kernel.mesh import (
    EPS,
    MassProperties,
    Mesh,
    axis_index,
    rotation_matrix,
    transform_matrix,
)
from src.kernel.units import UnitSystem, convert_length, format_length, parse_length

__all__ = [
    "EPS",
    "MassProperties",
    "Mesh",
    "UnitSystem",
    "axis_index",
    "convert_length",
    "format_length",
    "parse_length",
    "rotation_matrix",
    "transform_matrix",
]

__version__ = "0.3.0"
