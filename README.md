# OpenCad

OpenCad is an early-stage Python CAD playground built around PySide6 and
PyVista. It currently focuses on interactive scene editing, mesh operations,
and experimental geometry generation.

## Current Features

- 3D PyVista viewport with grid, axes, picking, and camera reset
- Scene graph with selectable and renameable objects
- Basic primitives: cube, sphere, cylinder, cone, torus, plane, and circle
- Transform tools for selection, move, scale, rotate, and numeric transforms
- Mesh operations: subdivide, extrude, smooth/bevel proxy, clean/repair, and
  decimate
- Boolean union, difference, and intersection for two selected meshes
- Mesh import for STL, PLY, OBJ, VTP, and VTK files
- Generative Lab for implicit TPMS lattice generation
- Object properties with position, scale, rotation, color, point count, cell
  count, area, and volume
- STL/PLY export for the selected object

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## Generative Lab

The Generative Lab creates implicit triply periodic minimal surface (TPMS)
meshes using Gyroid, Schwarz P, and Diamond equations. These structures are
useful for lightweight CAD and additive-manufacturing workflows because a dense
lattice can be controlled through a few parameters instead of manual topology.

Controls:

- `Surface`: implicit field family
- `Size`: generated cube domain size
- `Cells`: number of repeated TPMS cells across the domain
- `Resolution`: sampling density for contour extraction
- `Iso Level`: contour threshold
- `Z Gradient`: graded density bias along the Z axis

See [docs/research_roadmap.md](docs/research_roadmap.md) for the current
research-backed direction.
