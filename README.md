# OpenCad

OpenCad is an experimental Python CAD playground built with PySide6, PyVista,
and VTK. It combines interactive mesh editing with a **signed distance field
kernel**, so it can do things a mesh-only tool cannot: exact offsets, smooth
blends, and functionally graded lattices.

Underneath the desktop app is a geometry kernel that runs on **numpy alone** —
no VTK, no OpenGL, no Qt. That is what makes the geometry unit tested (800
tests), usable headlessly, and reusable from your own scripts.

## Install

Grab the Windows installer from
[Releases](https://github.com/kai9987kai/OpenCad/releases) and run it. It needs
no Python, installs per-user without an administrator prompt, associates `.ocad`
project files, and can optionally put the command line tool on your PATH.

The builds are unsigned, so SmartScreen warns on first run — choose **More info**
then **Run anyway**.

## Run from source

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[gui,accel]"
python main.py
```

To work on the geometry alone, skip the heavy GUI wheels:

```bash
pip install -e ".[accel,dev]"
```

## Two ways to model

OpenCad carries both a boundary representation (triangle meshes) and an implicit
one (signed distance fields), because they are good at different things.

| | Meshes | Fields |
| --- | --- | --- |
| Best for | imported scans, STL, measured parts | analytic solids, lattices, blends |
| Booleans | fragile on dirty input | exact — `min` and `max` |
| Offset / shell | approximate, self-intersects | exact by construction |
| Fillet between parts | needs new topology | a polynomial smooth-min |
| Infinite arrays | one copy per instance | a coordinate `mod` |

The conversion runs both ways, so the two compose: `surface_nets` extracts a mesh
from a field, and `mesh_sdf` exposes an imported mesh *as* a field — which means
you can offset or hollow an STL exactly, something the mesh pipeline cannot do.

```python
from src.kernel import primitives, sdf
from src.kernel.bvh import mesh_sdf
from src.kernel.meshing import surface_nets

# Implicit modelling: a bracket with a filleted boss and a bore.
plate = sdf.rounded_box((60, 30, 8), radius=3)
boss = sdf.cylinder(8, 20).translate((18, 0, 6))
part = plate.smooth_union(boss, k=3) - sdf.cylinder(4, 40).translate((18, 0, 0))
mesh = surface_nets(part, resolution=160)

# Hollow an imported mesh exactly - no self-intersection.
imported = primitives.icosphere(radius=20, subdivisions=4)
hollow = surface_nets(mesh_sdf(imported).shell(1.5), resolution=128)
```

## Main Features

### Implicit Lab — lattices and fields

- Seven TPMS families: **Gyroid, Schwarz P, Diamond, Neovius, Schoen I-WP,
  Lidinoid, Split P**
- **Sheet** (walls) and **network** (struts) solids
- Wall thickness in **millimetres**, not unitless field levels
- **Functionally graded lattices** — wall thickness or cell size varying along
  X, Y, Z, or radially, so density can follow load
- Lattices are **capped to their region**, so the result is a closed, printable
  solid rather than a surface sliced open at the boundary
- **Relative density estimate** — what the lattice weighs versus solid
- Resolution guidance that warns when the grid is too coarse to resolve the wall
- Runs on a **worker thread** with progress; the viewport stays responsive

### Analysis — will it print?

- Watertight, manifold, and winding checks with the specific edge counts
- **Overhang detection** against a configurable angle and build direction
- Build-volume fit, thin-feature, and separate-body warnings
- Mass properties: volume, surface area, **centre of mass**, inertia tensor,
  principal moments
- Oriented bounding box (principal-axis), for choosing a print orientation
- Triangle quality: smallest angle, aspect ratio, slivers, degenerates
- Material estimate in grams
- Findings are severity-ranked and each states what was measured against which
  threshold

### Geometry kernel

- Mesh type with topology (genus, components, manifold and boundary edges),
  cleanup, welding, smoothing, decimation, and exact mass properties
- **BVH** with ray casting, closest point, robust containment (generalized
  winding number, so torus holes are correct), and self-intersection detection
- **Surface Nets** isosurfacing — manifold by construction, no case table
- Analytic primitives: box, sphere, icosphere, cylinder, cone, torus, capsule,
  tube, prism, pyramid, wedge, and the platonic solids
- **Safe expression evaluator** and parameter table with dependency ordering,
  cycle detection, and reference-rewriting rename
- Length and angle **units** (mm, cm, m, in, ft, thou…), stored in millimetres

### Viewport and scene

- 3D PyVista viewport with grid, axes, picking, camera reset, and named views
- Scene graph with selectable, renameable objects; multi-select
- Hide/show, isolate, lock/unlock, with `[H]` and `[L]` markers
- Orthographic and perspective projection

### Creation and editing

- Toolbar primitives and a parameterized Primitive Lab
- Linear and circular arrays, duplicate, mirror across X/Y/Z
- Select, move, scale, rotate tools plus numeric transforms
- Center to origin, drop to floor, freeze transforms, fit selection
- Grid/snap for object positions and mesh vertices
- Undo/redo

### Mesh operations

Subdivide, extrude, smooth/bevel, clean/repair, decimate, clip, offset surface,
keep largest component, recompute and flip normals, and boolean union,
difference, and intersection.

### File workflow

- Import and export **STL, OBJ, PLY, OFF, and 3MF** — read and written in pure
  Python, so they work headlessly too. VTK formats (VTP, VTK) still import.
- Export the selected object or all visible objects as one mesh
- Save/open `.ocad` projects, preserving meshes, names, transforms, colour,
  opacity, edge and render style, visibility, and lock state

## Common Workflows

### Fill a part with a graded lattice

1. Create or import the part and select it.
2. Open the **Implicit Lab** dock (`Ctrl+L`).
3. Choose a surface type and set the cell size and wall thickness in mm.
4. Set **Fill** to `Selected object bounds`.
5. Under **Grading**, vary wall thickness along an axis to put material where
   the load is.
6. Check the cost line resolves at least 2 cells across the wall, then
   `Generate Lattice`.
7. `Estimate Relative Density` tells you what it will weigh.

### Check a part before printing

1. Select the object and press `Ctrl+R`, or use `Analyse > Printability Report`.
2. Set the overhang angle, minimum feature size, and build volume.
3. Errors are things that will not slice; warnings are things to look at.

## Keyboard Shortcuts

- `Delete` — delete selected
- `Ctrl+Z` / `Ctrl+Y` — undo / redo
- `Ctrl+L` — Implicit Lab
- `Ctrl+R` — printability report

## Project Structure

```text
main.py
src/
  kernel/            numpy only - no VTK, OpenGL, or Qt
    mesh.py          the Mesh type: topology, mass properties, cleanup
    primitives.py    analytic solids
    sdf.py           signed distance fields, CSG, warps, TPMS
    meshing.py       Surface Nets isosurfacing
    lattice.py       lattice specification -> closed solid
    bvh.py           ray casts, closest point, containment, mesh-as-field
    analysis.py      printability and engineering analysis
    expressions.py   safe expression evaluation and parameters
    units.py         length and angle units
    io_mesh.py       STL, OBJ, PLY, OFF, 3MF
  core/
    scene.py         object registry and metadata
    bridge.py        the one seam between kernel meshes and VTK actors
    history.py       command-based undo, budgeted by memory
    project_io.py    .ocad save/load
    viewport.py      PyVista Qt viewport
  ui/
    main_window.py   menus, docks, wiring
    implicit_panel.py, analysis_panel.py, primitive_panel.py, ...
    tasks.py         background geometry on a thread pool
packaging/           PyInstaller spec, Inno Setup script, icon generator
docs/
  architecture.md    the layering, and why it is that way
  research_roadmap.md
tests/               825 tests, no display required
```

## Building a Windows release

```powershell
.\packaging\build.ps1
```

Produces `dist\OpenCad\` (a self-contained application folder) and
`dist\installer\OpenCad-<version>-windows-<arch>-setup.exe`. See
[packaging/README.md](packaging/README.md) for prerequisites and the decisions
behind the build.

See [docs/architecture.md](docs/architecture.md) for how the layers fit together
and [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow.

## Testing

```bash
python -m pytest
python -m ruff check .
```

The suite needs no display, no OpenGL, and no VTK, and runs on Linux, macOS, and
Windows across Python 3.10–3.12 in CI.

Tests assert **closed-form ground truth** rather than recorded output — a cube's
volume, a rectangle's second moment of area, the exact ray distance to a plane.
Where an algorithm is approximate, they assert a convergence property instead:
refine the mesh and the error must shrink at the expected rate.

## Notes and Limits

- OpenCad is mesh- and field-based. It is **not a B-rep CAD kernel**, and there
  is no parametric feature history yet — see the roadmap.
- Mesh boolean quality depends on closed, clean input. Field booleans do not.
- TPMS wall thickness is accurate to roughly 2–3% for Gyroid, Diamond and I-WP,
  and around 17% for Neovius and Lidinoid, whose field gradients vary most.
  Check a thin-walled lattice before committing it to a print.
- Surface Nets rounds sharp creases, and where a design pinches to less than one
  grid cell it can leave non-manifold edges. The result stays closed, so it
  still exports and slices.
- High resolutions produce large meshes. Cost grows with the cube of the
  resolution; the Implicit Lab shows the sample count before you commit.

## Research Direction

OpenCad tracks practical CAD research: implicit modelling, functionally graded
TPMS lattices, mesh-to-CAD reconstruction, constraint-based sketching, and
analysis-aware generative design.

See [docs/research_roadmap.md](docs/research_roadmap.md) for the current roadmap
and references.
