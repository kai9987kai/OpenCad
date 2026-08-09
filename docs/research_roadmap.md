# OpenCad Research Roadmap

Last updated: 2026-08-09

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

The seventh pass rebuilt the foundation. A numpy-only geometry kernel now sits
under the app, which changed what the project can credibly attempt:

- **Implicit modelling proper.** Signed distance fields with exact CSG, smooth
  blends, warps, and exact offsets and shells. The old TPMS code sampled three
  hard-coded fields and contoured them through VTK; the kernel now has seven
  families, sheet and network solids, and metric wall thickness.
- **Functionally graded lattices**, the direction the references below point at.
  Wall thickness and cell size accept a function of position, so density can
  follow load rather than being uniform by construction.
- **Isosurfacing without VTK.** Surface Nets in numpy, manifold by construction.
  Lattice generation is now testable, cancellable, and runs headlessly.
- **Meshes as fields.** A BVH gives any imported mesh a signed distance
  function, so an STL can be offset or hollowed exactly - an operation the mesh
  pipeline could only approximate, and badly, on concave geometry.
- **Analysis worth acting on.** Watertightness, overhangs, wall thickness,
  relative density, mass properties, and inertia - the inputs an
  analysis-aware generative loop needs before it can optimise anything.
- **800 tests against closed-form ground truth**, running with no display, so
  the geometry can be refactored without silently changing its answers.

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
   mutating actor meshes. This is the foundation for real CAD behavior, and it
   is now the single biggest gap: the pieces it needs already exist. The safe
   expression evaluator and `ParameterTable` in `src/kernel/expressions.py`
   give named, dependency-ordered, cycle-checked parameters; the kernel is
   deterministic and side-effect free, so a feature can be defined as
   parameters plus a rebuild function. What is missing is the document model
   that orders features, tracks dependencies between them, and re-runs the
   downstream ones when a parameter changes - plus an `.ocad` v2 format that
   stores the recipe rather than only the resulting meshes.

2. Constraint-based sketching

   Add 2D sketch entities, constraints, and sketch-to-extrude workflows. The
   implicit side of this already works: `sdf.extrude` turns a closed 2D profile
   into a solid and `sdf.revolve` lathes one, both exactly. What is missing is
   the sketch itself - entities, geometric constraints, and a solver to satisfy
   them - and the 2D polygon layer (triangulation with holes, offsetting,
   section properties) that a sketch needs to become a mesh as well as a field.

   Recent sketch inference work points toward recovering editable primitives
   and constraints from drawings and images.

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

   The lightweight heuristics are now in place: `src/kernel/analysis.py` reports
   mass properties and inertia, relative density, wall thickness, overhangs, and
   printable feature-size warnings, and `volume_fraction` measures a lattice
   against the volume it fills.

   The next step is to close the loop - drive lattice grading from an analysis
   result rather than from a hand-set gradient. A cheap, honest version is
   within reach without a finite element solver: grade density by distance from
   load paths or from a user-painted field. Real structural optimisation needs
   an actual solver, and the sensible route there is an external one (CalculiX,
   or scikit-fem for small linear problems) rather than writing one here.

6. Robust mesh booleans

   VTK's boolean filter fails on the dirty, self-intersecting meshes people
   actually import. Now that `bvh.mesh_sdf` exposes any mesh as a field, a
   voxel-remeshing fallback is straightforward: convert both operands to fields,
   combine them with `min`/`max`, and re-extract with Surface Nets. It is not
   exact and it resamples the surface, but it never fails, which is the right
   trade for a fallback. The exact path - a proper mesh arrangement with
   coplanar handling - is a much larger piece of work.
