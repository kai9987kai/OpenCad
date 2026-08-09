"""Tests for isosurface extraction.

The properties that matter for a mesher are topological (is the result closed
and consistently wound?) and metric (does the volume converge to the analytic
answer as the grid refines?). Both are checked here against shapes whose
volume and genus are known exactly.

Resolutions are kept modest so the suite stays fast; the convergence tests care
about the *trend*, not the absolute error.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.kernel import sdf
from src.kernel.mesh import Mesh
from src.kernel.meshing import grid_to_mesh, surface_nets, voxelize


class TestTopology:
    def test_a_meshed_sphere_is_closed_and_consistently_wound(self):
        mesh = surface_nets(sdf.sphere(1.0), resolution=40)
        assert not mesh.is_empty
        assert mesh.is_watertight
        assert mesh.is_edge_manifold
        assert mesh.is_oriented
        assert len(mesh.boundary_edges()) == 0
        assert mesh.n_components() == 1
        assert mesh.euler_characteristic == 2
        assert mesh.genus == 0

    def test_normals_face_outward(self):
        """A sign error here inverts every mesh the app produces."""
        mesh = surface_nets(sdf.sphere(1.0), resolution=32)
        assert mesh.volume > 0
        outward = np.einsum("ij,ij->i", mesh.vertex_normals(), mesh.vertices)
        assert np.all(outward > 0)

    def test_a_torus_comes_out_with_genus_one(self):
        mesh = surface_nets(sdf.torus(2.0, 0.7), resolution=56)
        assert mesh.is_watertight
        assert mesh.genus == 1

    def test_disjoint_solids_produce_separate_components(self):
        field = sdf.sphere(1.0, center=(-3, 0, 0)) | sdf.sphere(1.0, center=(3, 0, 0))
        mesh = surface_nets(field, resolution=56)
        assert mesh.n_components() == 2
        assert mesh.is_watertight

    def test_a_field_with_no_surface_returns_an_empty_mesh(self):
        # A sphere far outside the sampled box never crosses zero.
        far = sdf.sphere(1.0, center=(100.0, 0.0, 0.0))
        mesh = surface_nets(far, bounds=([-2, -2, -2], [2, 2, 2]), resolution=16)
        assert mesh.is_empty


class TestConvergence:
    def test_sphere_volume_converges_from_below(self):
        exact = 4.0 / 3.0 * np.pi
        errors = []
        for resolution in (24, 48, 96):
            mesh = surface_nets(sdf.sphere(1.0), resolution=resolution)
            assert mesh.volume < exact  # a faceted sphere is always inscribed
            errors.append(abs(mesh.volume - exact))
        assert errors[0] > errors[1] > errors[2]
        assert errors[-1] / exact < 0.01

    def test_sphere_area_converges(self):
        exact = 4.0 * np.pi
        mesh = surface_nets(sdf.sphere(1.0), resolution=96)
        assert mesh.area == pytest.approx(exact, rel=0.01)

    @pytest.mark.parametrize("radius", [0.5, 1.0, 3.0])
    def test_volume_scales_with_the_cube_of_the_radius(self, radius):
        mesh = surface_nets(sdf.sphere(radius), resolution=64)
        assert mesh.volume == pytest.approx(4.0 / 3.0 * np.pi * radius**3, rel=0.01)

    def test_a_box_is_reproduced_essentially_exactly(self):
        """A box is grid-aligned, so a dual mesher should nail it."""
        mesh = surface_nets(sdf.box((2.0, 2.0, 2.0)), resolution=64)
        assert mesh.volume == pytest.approx(8.0, rel=1e-6)
        assert mesh.is_watertight

    def test_torus_volume_matches_the_closed_form(self):
        major, minor = 2.0, 0.6
        mesh = surface_nets(sdf.torus(major, minor), resolution=80)
        exact = 2.0 * np.pi**2 * major * minor**2
        assert mesh.volume == pytest.approx(exact, rel=0.01)

    def test_a_csg_difference_has_the_right_volume(self):
        # The sphere sits entirely inside the box, so volumes simply subtract.
        field = sdf.box((3.0, 3.0, 3.0)) - sdf.sphere(1.0)
        mesh = surface_nets(field, resolution=96)
        exact = 27.0 - 4.0 / 3.0 * np.pi
        assert mesh.volume == pytest.approx(exact, rel=0.01)
        assert mesh.is_watertight

    def test_a_shell_has_the_volume_of_its_wall(self):
        outer, wall = 2.0, 0.3
        mesh = surface_nets(sdf.sphere(outer).shell(wall), resolution=96)
        inner_r, outer_r = outer - wall / 2, outer + wall / 2
        exact = 4.0 / 3.0 * np.pi * (outer_r**3 - inner_r**3)
        assert mesh.volume == pytest.approx(exact, rel=0.02)


class TestSharpen:
    def test_projection_puts_vertices_on_the_true_surface(self):
        field = sdf.sphere(1.0)
        sharp = surface_nets(field, resolution=32, sharpen=True)
        blunt = surface_nets(field, resolution=32, sharpen=False)

        sharp_error = np.abs(field(sharp.vertices)).mean()
        blunt_error = np.abs(field(blunt.vertices)).mean()
        assert sharp_error < blunt_error
        assert sharp_error < 1e-6

    def test_sharpening_improves_the_volume(self):
        exact = 4.0 / 3.0 * np.pi
        sharp = surface_nets(sdf.sphere(1.0), resolution=32, sharpen=True)
        blunt = surface_nets(sdf.sphere(1.0), resolution=32, sharpen=False)
        assert abs(sharp.volume - exact) <= abs(blunt.volume - exact)

    def test_sharpening_keeps_the_mesh_closed(self):
        mesh = surface_nets(sdf.torus(2.0, 0.6), resolution=48, sharpen=True)
        assert mesh.is_watertight


class TestGridToMesh:
    def test_accepts_a_raw_numpy_grid(self):
        """A field is not required - any sampled scalar volume works, which is
        what lets imported voxel data be meshed."""
        axis = np.linspace(-2, 2, 40)
        x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
        values = np.sqrt(x**2 + y**2 + z**2) - 1.0
        spacing = (axis[1] - axis[0],) * 3
        mesh = grid_to_mesh(values, origin=(-2, -2, -2), spacing=spacing)
        assert mesh.is_watertight
        assert mesh.volume == pytest.approx(4.0 / 3.0 * np.pi, rel=0.02)

    def test_level_selects_a_different_isosurface(self):
        axis = np.linspace(-3, 3, 48)
        x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
        values = np.sqrt(x**2 + y**2 + z**2)  # unsigned radius
        spacing = (axis[1] - axis[0],) * 3
        mesh = grid_to_mesh(values, origin=(-3, -3, -3), spacing=spacing, level=2.0)
        assert mesh.volume == pytest.approx(4.0 / 3.0 * np.pi * 8.0, rel=0.02)

    def test_origin_and_spacing_place_the_result_correctly(self):
        axis = np.linspace(-2, 2, 32)
        x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
        values = np.sqrt(x**2 + y**2 + z**2) - 1.0
        spacing = (axis[1] - axis[0],) * 3
        mesh = grid_to_mesh(values, origin=(8.0, -2.0, -2.0), spacing=spacing)
        assert mesh.center == pytest.approx([10.0, 0.0, 0.0], abs=0.05)

    def test_anisotropic_spacing_is_respected(self):
        axis = np.linspace(-2, 2, 40)
        x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
        values = np.sqrt(x**2 + y**2 + z**2) - 1.0
        step = axis[1] - axis[0]
        mesh = grid_to_mesh(values, origin=(-2, -2, -2), spacing=(step, step, 2 * step))
        # Doubling the Z spacing stretches the sphere into an ellipsoid.
        extents = mesh.extents
        assert extents[2] == pytest.approx(2.0 * extents[0], rel=0.05)

    def test_an_all_inside_grid_has_no_surface(self):
        assert grid_to_mesh(np.full((8, 8, 8), -1.0)).is_empty

    def test_an_all_outside_grid_has_no_surface(self):
        assert grid_to_mesh(np.full((8, 8, 8), 1.0)).is_empty

    @pytest.mark.parametrize(
        "values", [np.zeros((2, 2)), np.zeros(8), np.zeros((1, 4, 4))]
    )
    def test_malformed_grids_are_rejected(self, values):
        with pytest.raises(ValueError):
            grid_to_mesh(values)

    def test_non_positive_spacing_is_rejected(self):
        with pytest.raises(ValueError):
            grid_to_mesh(np.zeros((4, 4, 4)), spacing=(1.0, 0.0, 1.0))


class TestLattices:
    def test_a_trimmed_lattice_is_closed_even_where_it_pinches(self):
        """Surface Nets can leave non-manifold edges where two sheets share a
        cell, but the result must still have no holes - otherwise it will not
        slice or export."""
        field = sdf.sphere(6.0) & sdf.tpms_sheet("gyroid", period=5.0, thickness=0.9)
        mesh = surface_nets(field, bounds=([-6.5] * 3, [6.5] * 3), resolution=72)
        assert not mesh.is_empty
        assert len(mesh.boundary_edges()) == 0

    def test_a_thicker_wall_produces_more_material(self):
        box = ([-6.5] * 3, [6.5] * 3)
        solid = sdf.sphere(6.0)
        thin = surface_nets(
            solid & sdf.tpms_sheet("gyroid", period=5.0, thickness=0.6), bounds=box, resolution=64
        )
        thick = surface_nets(
            solid & sdf.tpms_sheet("gyroid", period=5.0, thickness=1.2), bounds=box, resolution=64
        )
        assert thick.volume > thin.volume

    def test_progress_is_reported_monotonically(self):
        seen = []
        surface_nets(
            sdf.sphere(1.0), resolution=24, progress=lambda f, m: seen.append(f)
        )
        assert seen == sorted(seen)
        assert seen[-1] == pytest.approx(1.0)


class TestVoxelize:
    def test_occupancy_fraction_approximates_the_volume(self):
        occupied, _, spacing = voxelize(
            sdf.sphere(1.0), bounds=([-1.5] * 3, [1.5] * 3), resolution=64
        )
        cell_volume = float(np.prod(spacing))
        assert occupied.sum() * cell_volume == pytest.approx(4.0 / 3.0 * np.pi, rel=0.02)

    def test_returns_a_boolean_grid_with_the_requested_shape(self):
        occupied, origin, _ = voxelize(sdf.sphere(1.0), resolution=(8, 10, 12))
        assert occupied.dtype == bool
        assert occupied.shape == (8, 10, 12)
        assert len(origin) == 3


class TestRoundTrip:
    def test_a_meshed_field_survives_a_mesh_cleanup(self):
        mesh = surface_nets(sdf.sphere(1.0), resolution=40)
        cleaned = mesh.cleaned()
        assert cleaned.is_watertight
        assert cleaned.volume == pytest.approx(mesh.volume, rel=1e-9)

    def test_the_result_is_a_kernel_mesh(self):
        assert isinstance(surface_nets(sdf.sphere(1.0), resolution=16), Mesh)
