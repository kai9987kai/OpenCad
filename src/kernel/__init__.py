"""OpenCad geometry kernel.

A dependency-light CAD kernel: everything in this package runs on numpy alone,
with no VTK, OpenGL, or Qt.  That keeps the geometry layer testable headlessly,
scriptable from the command line, and reusable outside the desktop app.

The public surface is re-exported lazily so importing a single submodule does
not drag in the whole kernel.
"""

from __future__ import annotations

from src.kernel.mesh import EPS, MassProperties, Mesh, rotation_matrix, transform_matrix

__all__ = [
    "EPS",
    "MassProperties",
    "Mesh",
    "rotation_matrix",
    "transform_matrix",
]

__version__ = "0.3.0"
