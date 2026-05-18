# OpenCad Research Roadmap

Last updated: 2026-05-17

This roadmap keeps OpenCad pointed at features that are plausible for the
current codebase while still tracking where CAD research is moving.

## Implemented First: Implicit TPMS Lattices

OpenCad now includes a Generative Lab that samples implicit TPMS fields and
extracts meshes with PyVista contouring. This was chosen first because it gives
the app an advanced CAD capability without needing a new solver, model format,
or ML pipeline.

## Essential CAD Tooling Added

The second pass focused on the basic tools users expect before deeper research
features are useful: import/export, duplicate, additional primitives, numerical
rotation, interactive move/scale/rotate widgets, boolean operations, mesh
repair, mesh decimation, and named camera views.

The third pass adds workflow depth: undo/redo snapshots, measurement feedback,
linear and circular arrays, mirroring, alignment tools, transform freezing, fit
selection, and an experimental surface offset operation. These tools make the
app more useful for iterative CAD modeling while keeping the geometry pipeline
mesh-based and inspectable.

The fourth pass improves project-scale use: `.ocad` project save/open, outliner
visibility and lock controls, isolate/show-all workflows, status feedback, and
UI styling for dialogs and status bars. This gives future parametric and
analysis features a persistent workspace instead of only a live viewport state.

The fifth pass adds adjustable modeling tools and appearance workflow:
parameterized decimation, surface offset, clipping, largest-component cleanup,
visible-scene export, orthographic/perspective projection controls, and an
Appearance dock for color presets, opacity, representation style, and edge
visibility. This starts moving fixed commands toward inspectable tool panels.

The sixth pass adds parameterized primitive creation and grid-aware editing:
Primitive Lab creates dimensioned boxes, spheres, cylinders, cones, tori, and
planes, while Grid/Snap supports object-position snapping and vertex snapping.
Normal recompute and flip-normal tools improve imported and boolean-generated
mesh repair.

Relevant research direction:

- Functionally graded TPMS and gyroid lattices are active additive
  manufacturing topics for lightweight, energy-absorbing, and biomedical
  structures.
- GLU3D shows the longer-term direction: generating lattice unit cells through
  diffusion models for inverse design. OpenCad does not include diffusion
  generation yet, but the current implicit lattice interface gives the project
  a compatible geometry target.

References:

- GLU3D, "Generative Lattice Units with 3D Diffusion for Inverse Design",
  Advanced Functional Materials 2024, DOI 10.1002/adfm.202404165:
  https://doi.org/10.1002/adfm.202404165
- PyVista contouring over scalar fields:
  https://docs.pyvista.org/api/core/_autosummary/pyvista.DataSetFilters.contour.html

## Next High-Value Directions

1. Parametric history tree

   Store each primitive and operation as editable parameters rather than only
   mutating actor meshes. This is the foundation for real CAD behavior.

2. Constraint-based sketching

   Add 2D sketch entities, constraints, and sketch-to-extrude workflows. Recent
   sketch inference work points toward recovering editable primitives and
   constraints from drawings and images.

   Reference: PICASSO, arXiv 2407.13394:
   https://arxiv.org/abs/2407.13394

3. Mesh-to-CAD reconstruction

   Build tools to fit primitives and simple sketch-extrude programs from meshes
   or point clouds. This would bridge imported meshes and editable CAD.

   Reference: CADFit, arXiv 2605.01171:
   https://arxiv.org/abs/2605.01171

4. Differentiable and multimodal CAD generation

   Track text/image/point-to-editable-BRep work, but defer implementation until
   the project has a robust parametric representation.

   Reference: DreamCAD, arXiv 2603.05607:
   https://arxiv.org/abs/2603.05607

5. Analysis-aware generative design

   Add lightweight structural heuristics first: bounding-box mass estimates,
   lattice density estimates, and printable feature-size warnings. Later, add
   actual finite element analysis or external solver integration.
