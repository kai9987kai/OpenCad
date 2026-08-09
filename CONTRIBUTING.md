# Contributing to OpenCad

Thanks for taking an interest. OpenCad is an experimental CAD playground, so the
bar is "correct, tested, and honest about its limits" rather than "feature
complete".

## The one architectural rule

OpenCad is split into two layers, and the split is load-bearing:

```text
src/kernel/   numpy only - geometry, no UI, no VTK, no Qt.  Fully unit tested.
src/core/     scene, project I/O, and the PyVista bridge.   Needs pyvista.
src/ui/       PySide6 widgets and the main window.          Needs Qt + a display.
```

**Nothing in `src/kernel/` may import `pyvista`, `vtk`, `PySide6`, or
`pyvistaqt` at module level.** CI enforces this. The reason is practical: the
kernel is what runs in continuous integration, in the headless CLI, and in
anyone's script. The moment it needs a display server, all three stop working.

If a kernel module genuinely needs to hand geometry to the viewport, convert at
the boundary with `Mesh.to_polydata()` / `Mesh.from_polydata()`.

`scipy` is allowed in the kernel, but only behind a guarded optional import with
a working numpy fallback, because it is an optional dependency.

## Getting set up

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[gui,accel,dev]"
```

To work on the kernel alone you can skip the heavy GUI wheels:

```bash
pip install -e ".[accel,dev]"
```

Run the app with `python main.py`, and the headless tools with `python -m src.cli --help`.

## Before you open a pull request

```bash
python -m ruff check .
python -m pytest
```

Both must be clean. `pytest` needs no display and should finish in well under a
minute; if a test you add is genuinely slow, mark it `@pytest.mark.slow`.

## Writing tests

Geometry code fails quietly, so the suite is built around **closed-form ground
truth** rather than recorded output. A test that asserts a cube's volume is
`8.0` stays meaningful when the implementation is rewritten; a test that asserts
the mesh has 4 096 triangles does not.

Prefer, in order:

1. An analytic value (volume, area, second moment, distance) computed by hand.
2. An invariant (volume is unchanged by rotation, a closed solid is watertight,
   a transform composed with its inverse is the identity).
3. A convergence property (a sphere's meshed volume approaches `4/3 pi r^3` from
   below, and the error shrinks as resolution rises).

Only fall back to golden values when none of the above applies, and say why in
a comment.

`tests/conftest.py` provides `cube`, `unit_cube`, `tetrahedron`, `grid_patch`,
and a seeded `rng` fixture. Use the seeded generator so failures reproduce.

## Style

- 4-space indent, double quotes, `from __future__ import annotations`.
- Docstrings explain *why* and state conventions (sign, winding, units, whether
  a result is exact or approximate). They should not restate the signature.
- Vectorize with numpy. A Python loop over vertices is a bug in waiting.
- Approximations are fine; undocumented approximations are not. If an algorithm
  degrades on non-manifold input or assumes a closed surface, say so in the
  docstring, and raise a clear error rather than returning a quietly wrong
  answer.

## Reporting bugs

A geometry bug report is far more useful with the model attached. `.ocad`,
`.stl`, or a short script using the kernel API all work:

```python
from src.kernel import primitives, sdf
from src.kernel.meshing import surface_nets

field = sdf.sphere(1.0) - sdf.box((0.5, 0.5, 4.0))
mesh = surface_nets(field, resolution=96)
print(mesh.summary())
```

See [SECURITY.md](SECURITY.md) for anything security-sensitive, and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community expectations.
