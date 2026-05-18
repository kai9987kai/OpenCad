# OpenCad

OpenCad is an experimental Python CAD playground built with PySide6, PyVista,
and VTK. It is focused on interactive mesh-based CAD workflows, generative
geometry, and research-inspired modeling tools.

The project is still early-stage, but it now includes enough core tooling to
prototype parts, inspect mesh properties, generate lattices, import/export
geometry, and save full OpenCad projects.

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Requirements are listed in `requirements.txt`:

- `PySide6`
- `pyvista`
- `pyvistaqt`
- `numpy`

## Main Features

### Viewport and Scene

- 3D PyVista viewport with grid, axes, picking, camera reset, and named views
- Scene graph with selectable, renameable objects
- Multi-select support for boolean, measurement, visibility, and batch tools
- Hide/show, isolate selected, show all, lock/unlock
- Hidden and locked state markers in the scene graph: `[H]`, `[L]`
- Orthographic and perspective projection controls

### Creation Tools

- Toolbar primitives: cube, sphere, cylinder, cone, torus, plane, circle
- Primitive Lab dock for parameterized boxes, spheres, cylinders, cones, tori,
  and planes
- Generative Lab for TPMS lattices: Gyroid, Schwarz P, Diamond
- Linear array and circular array
- Duplicate and mirror across X/Y/Z origin planes

### Transform and Editing

- Select, move, scale, and rotate tools
- Numeric position, scale, and rotation in the Properties dock
- Center to origin, drop to floor, freeze transforms
- Fit selection
- Grid/Snap dock for snapping object positions or mesh vertices to a chosen
  spacing
- Undo/redo scene snapshots

### Mesh Operations

- Subdivide
- Extrude along Z
- Smooth/bevel proxy
- Clean/repair
- Parameterized decimate
- Parameterized clip
- Experimental offset surface
- Keep largest connected component
- Recompute normals
- Flip normals
- Boolean union, difference, and intersection

### Inspection and Appearance

- Properties dock: position, scale, rotation, color, point count, cell count,
  area, and volume
- Measurements dock: dimensions, center, total area/volume, and two-object
  center distance
- Appearance dock: material presets, custom color, opacity, representation
  style, and edge visibility
- Status bar with object count, visible count, locked count, and selection

### File Workflow

- Import mesh files: STL, PLY, OBJ, VTP, VTK
- Export selected object as STL/PLY
- Export all visible objects as one combined scene mesh
- Save/open `.ocad` project files

`.ocad` projects preserve:

- Meshes
- Object names and metadata
- Transforms
- Color, opacity, edge visibility, and render style
- Hidden/visible state
- Lock state

## Common Workflows

### Create a parameterized part

1. Open the Primitive Lab dock.
2. Choose a primitive type and dimensions.
3. Click `Create Primitive`.
4. Use the Properties dock for exact transforms.
5. Use Appearance for material, opacity, and edge display.

### Generate a lattice

1. Open the Generative Lab dock.
2. Choose `Gyroid`, `Schwarz P`, or `Diamond`.
3. Adjust size, cell count, resolution, iso level, and Z gradient.
4. Click `Generate TPMS`.

### Run a boolean operation

1. Select two objects in the scene graph.
2. Use `Mesh > Boolean Union`, `Boolean Difference`, or `Boolean Intersection`.
3. Clean or recompute normals afterward if the imported topology is rough.

### Save a project

Use `File > Save Project...` to write an `.ocad` file. Use
`File > Open Project...` to restore it later.

## Keyboard Shortcuts

- `Delete`: delete selected objects
- `Ctrl+Z`: undo
- `Ctrl+Y`: redo

## Project Structure

```text
main.py
src/
  core/
    mesh_ops.py       # mesh algorithms and generative geometry
    project_io.py     # .ocad save/load
    scene.py          # scene object registry and metadata
    transform.py      # viewport transform widgets
    viewport.py       # PyVista Qt viewport
  ui/
    main_window.py
    appearance_panel.py
    generative_panel.py
    grid_panel.py
    measurements_panel.py
    primitive_panel.py
    properties_panel.py
    tool_dialogs.py
    styles.py
docs/
  research_roadmap.md
```

## Research Direction

OpenCad is tracking practical CAD research directions such as implicit
modeling, TPMS lattices, mesh-to-CAD reconstruction, constraint-based sketching,
and analysis-aware generative design.

See [docs/research_roadmap.md](docs/research_roadmap.md) for the current
roadmap and references.

## Notes

- OpenCad is mesh-based today. It is not yet a full B-rep CAD kernel.
- Boolean quality depends on closed, clean input meshes.
- High TPMS resolutions can generate large meshes and may take time to render.
