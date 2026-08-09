# OpenCad Architecture

OpenCad is layered so that geometry does not depend on the desktop. The split is
the most important design decision in the project, and it is worth understanding
before changing anything.

```text
┌─────────────────────────────────────────────────────────────┐
│  src/ui/          PySide6 widgets, docks, main window       │  needs Qt + display
├─────────────────────────────────────────────────────────────┤
│  src/core/        scene, project I/O, PyVista bridge        │  needs pyvista
├─────────────────────────────────────────────────────────────┤
│  src/kernel/      geometry: numpy only, no UI, no VTK       │  needs numpy
└─────────────────────────────────────────────────────────────┘
```

Dependencies point downward only. `src/kernel` never imports `src/core` or
`src/ui`, and never imports `pyvista`, `vtk`, `PySide6`, or `pyvistaqt` at module
level. CI enforces the rule with a static check.

## Why the kernel is dependency-light

Three concrete things fall out of the constraint, and all three were impossible
before:

1. **The geometry is unit tested.** The suite runs on any machine with numpy —
   no display server, no OpenGL, no VTK wheels — so continuous integration
   covers three operating systems and three Python versions in under a minute.
2. **There is a headless mode.** `python -m src.cli` can build, analyse, convert,
   and export geometry on a machine with no graphics stack at all.
3. **The geometry is reusable.** A kernel mesh is two numpy arrays. It can be
   handed to a worker thread, pickled, or used from someone else's script
   without dragging a renderer along.

The cost is a conversion at the boundary. That conversion lives in exactly one
place, [`src/core/bridge.py`](../src/core/bridge.py), so it cannot drift.

## The layers in detail

### `src/kernel` — geometry

| Module | Responsibility |
| --- | --- |
| `mesh.py` | The `Mesh` type: indexed triangles, topology, mass properties, transforms, cleanup |
| `primitives.py` | Analytic solids built without VTK — box, sphere, icosphere, cylinder, torus, tube, platonics |
| `sdf.py` | Signed distance fields: CSG, smooth blends, warps, and TPMS lattice fields |
| `meshing.py` | Isosurface extraction — turns a field into a `Mesh` |
| `bvh.py` | Bounding volume hierarchy: ray casts, closest point, containment, self-intersection |
| `polygon.py` | 2D polygons: triangulation, offsetting, section properties |
| `sketch.py` / `constraints.py` | Constraint-based 2D sketching and the solver behind it |
| `sweeps.py` | Extrude, revolve, loft — profile to solid |
| `booleans.py` | Mesh booleans |
| `analysis.py` | Printability and engineering analysis |
| `slicer.py` | Plane sections and layer slicing |
| `repair.py` | Hole filling, orientation repair, non-manifold cleanup |
| `expressions.py` | Safe expression evaluation and the parameter table |
| `units.py` | Length and angle units; the kernel stores millimetres |
| `document.py` | The parametric feature tree |

### `src/core` — application state

`scene.py` tracks what is in the document and its per-object metadata.
`bridge.py` converts between kernel meshes and VTK actors, and owns the matrix
conventions. `history.py` is the undo stack — command based, budgeted by memory
rather than by a fixed count, and free of any Qt import so it can be tested.
`project_io.py` reads and writes `.ocad` files.

### `src/ui` — the desktop

Widgets, docks, and the main window. `tasks.py` runs heavy geometry on a
`QThreadPool` so the viewport keeps repainting; workers must not touch actors or
widgets, only return geometry.

## Two geometry representations

OpenCad carries both a boundary representation (triangle meshes) and an implicit
one (signed distance fields), because they are good at different things.

| | Meshes | Fields |
| --- | --- | --- |
| Exact for | imported scans, STL, anything measured | analytic solids, lattices, blends |
| Booleans | fragile on dirty input | trivially exact — `min`, `max` |
| Offset / shell | approximate, self-intersects | exact, by construction |
| Smooth blends | not expressible | a polynomial smooth-min |
| Infinite arrays | impossible | a coordinate `mod` |
| Cost | proportional to triangles | proportional to sampled volume |

The conversion runs both ways: `meshing.py` extracts a mesh from a field, and
`bvh.py` exposes a mesh as a signed distance function. That round trip is what
makes robust booleans possible on meshes that VTK's boolean filter refuses.

Fields are resolution-independent until they are meshed, so the recommended
workflow is to keep a design implicit while it is being shaped and mesh it once
at the end.

## The parametric feature tree

A feature stores *parameters*, not geometry. `document.py` holds an ordered list
of features, each of which rebuilds its own geometry from its inputs, so editing
the diameter of a hole re-runs the downstream features rather than leaving a hole
of the old size baked into a mesh.

This is what the `.ocad` v2 format persists: the recipe, not just the result.
Meshes are cached alongside it so opening a project does not have to rebuild
everything, but the recipe is the source of truth.

## Threading

The GUI thread owns every widget, actor, and the plotter. Geometry work runs on
a `QThreadPool` through `src/ui/tasks.py`. Workers receive a context object, call
`context.raise_if_cancelled()` inside their loops, report progress, and return a
kernel mesh. Actor creation and scene mutation happen in the result callback,
back on the GUI thread.

Kernel meshes are plain numpy arrays and safe to move across threads. VTK actors
are not.

## Testing philosophy

Geometry fails quietly, so the suite asserts against closed-form ground truth —
a cube's volume, a rectangle's second moment of area, the exact ray distance to a
plane — rather than recorded output. Where an algorithm is approximate, tests
assert a convergence property instead: refine the mesh, and the error must
shrink.

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the practical version of this.
