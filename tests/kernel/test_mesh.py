"""Tests for the numpy mesh type.

Every assertion here is against a closed-form value (cube volume, tetrahedron
inertia, Euler characteristic) rather than a recorded output, so the suite stays
meaningful if the implementation is rewritten.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.kernel.mesh import Mesh, axis_index, rotation_matrix, transform_matrix
from tests.conftest import build_cube, build_tetrahedron


class TestConstruction:
    def test_empty_mesh_is_empty(self):
        mesh = Mesh.empty()
        assert mesh.is_empty
        assert mesh.n_vertices == 0
        assert mesh.n_faces == 0
        assert mesh.volume == 0.0
        assert mesh.area == 0.0
        assert mesh.bounds == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def test_two_dimensional_vertices_are_lifted_to_z0(self):
        mesh = Mesh([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [[0, 1, 2]])
        assert mesh.vertices.shape == (3, 3)
        assert np.allclose(mesh.vertices[:, 2], 0.0)

    def test_non_triangular_faces_are_rejected(self):
        with pytest.raises(ValueError):
            Mesh(np.zeros((4, 3)), [[0, 1, 2, 3]])

    def test_from_polygons_fan_triangulates(self):
        quad = Mesh.from_polygons(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [[0, 1, 2, 3]]
        )
        assert quad.n_faces == 2
        assert quad.area == pytest.approx(1.0)

    def test_copy_is_independent(self, cube):
        clone = cube.copy()
        clone.vertices[0] += 5.0
        assert not np.allclose(clone.vertices[0], cube.vertices[0])


class TestMeasurement:
    def test_cube_volume_and_area(self):
        cube = build_cube(2.0)
        assert cube.volume == pytest.approx(8.0)
        assert cube.area == pytest.approx(24.0)

    @pytest.mark.parametrize("size", [0.5, 1.0, 3.0, 7.25])
    def test_volume_scales_cubically(self, size):
        assert build_cube(size).volume == pytest.approx(size**3)

    def test_tetrahedron_volume(self):
        assert build_tetrahedron(1.0).volume == pytest.approx(1.0 / 6.0)

    def test_flipped_winding_negates_volume(self, cube):
        assert cube.flipped().volume == pytest.approx(-cube.volume)

    def test_open_surface_has_zero_volume(self, grid_patch):
        assert grid_patch.volume == pytest.approx(0.0)
        assert grid_patch.area == pytest.approx(1.0)

    def test_bounds_and_extents(self):
        cube = build_cube(4.0, center=(1.0, 2.0, 3.0))
        assert cube.bounds == pytest.approx((-1.0, 3.0, 0.0, 4.0, 1.0, 5.0))
        assert cube.extents == pytest.approx([4.0, 4.0, 4.0])
        assert cube.center == pytest.approx([1.0, 2.0, 3.0])

    def test_face_normals_are_unit_length(self, cube):
        lengths = np.linalg.norm(cube.face_normals(), axis=1)
        assert np.allclose(lengths, 1.0)

    def test_vertex_normals_point_outward_on_a_convex_solid(self, cube):
        normals = cube.vertex_normals()
        radial = cube.vertices - cube.center
        assert np.all(np.einsum("ij,ij->i", normals, radial) > 0)


class TestMassProperties:
    def test_cube_inertia_matches_closed_form(self):
        cube = build_cube(2.0)
        properties = cube.mass_properties()
        # I = m (h^2 + d^2) / 12 with unit density: 8 * (4 + 4) / 12.
        expected = 8.0 * (4.0 + 4.0) / 12.0
        assert properties.volume == pytest.approx(8.0)
        assert properties.center_of_mass == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
        assert np.allclose(properties.inertia_tensor, np.eye(3) * expected)

    def test_rectangular_box_inertia_axes_differ(self):
        box = build_cube(2.0).scaled([1.0, 2.0, 3.0])
        # Half-extents 1, 2, 3 -> full sides 2, 4, 6; volume 48.
        properties = box.mass_properties()
        assert properties.volume == pytest.approx(48.0)
        ixx = 48.0 * (4.0**2 + 6.0**2) / 12.0
        iyy = 48.0 * (2.0**2 + 6.0**2) / 12.0
        izz = 48.0 * (2.0**2 + 4.0**2) / 12.0
        assert np.diag(properties.inertia_tensor) == pytest.approx([ixx, iyy, izz])

    def test_inertia_is_translation_invariant(self, cube):
        here = cube.mass_properties()
        there = cube.translated([12.0, -4.0, 7.5]).mass_properties()
        assert there.center_of_mass == pytest.approx([12.0, -4.0, 7.5])
        assert np.allclose(here.inertia_tensor, there.inertia_tensor)

    def test_inertia_tensor_is_symmetric_positive_definite(self, cube):
        properties = build_cube(2.0).scaled([1.0, 1.7, 0.4]).mass_properties()
        assert np.allclose(properties.inertia_tensor, properties.inertia_tensor.T)
        assert np.all(properties.principal_moments > 0)

    def test_as_dict_round_trips_to_plain_python(self, cube):
        payload = cube.mass_properties().as_dict()
        assert isinstance(payload["center_of_mass"], list)
        assert payload["volume"] == pytest.approx(8.0)


class TestTopology:
    def test_closed_solid_is_watertight_and_oriented(self, cube):
        assert cube.is_watertight
        assert cube.is_edge_manifold
        assert cube.is_oriented
        assert cube.euler_characteristic == 2
        assert cube.genus == 0

    def test_open_patch_reports_boundary(self, grid_patch):
        assert not grid_patch.is_watertight
        assert len(grid_patch.boundary_edges()) == 12
        assert grid_patch.genus is None

    def test_inconsistent_winding_is_detected(self, cube):
        faces = cube.faces.copy()
        faces[0] = faces[0][::-1]
        assert not Mesh(cube.vertices, faces).is_oriented

    def test_non_manifold_edge_is_detected(self, cube):
        # Glue an extra fin onto one existing edge.
        vertices = np.vstack([cube.vertices, [[5.0, 5.0, 5.0]]])
        faces = np.vstack([cube.faces, [[cube.faces[0][0], cube.faces[0][1], 8]]])
        mesh = Mesh(vertices, faces)
        assert not mesh.is_edge_manifold
        assert len(mesh.non_manifold_edges()) == 1

    def test_components_and_split(self, cube):
        pair = Mesh.concatenate([cube, cube.translated([10.0, 0.0, 0.0])])
        assert pair.n_components() == 2
        assert pair.volume == pytest.approx(16.0)
        pieces = pair.split()
        assert [piece.n_faces for piece in pieces] == [12, 12]
        assert pieces[0].n_vertices == 8
        assert pair.largest_component().n_faces == 12


class TestCleanup:
    def test_weld_merges_duplicated_geometry(self, cube):
        doubled = Mesh(
            np.vstack([cube.vertices, cube.vertices]),
            np.vstack([cube.faces, cube.faces + cube.n_vertices]),
        )
        cleaned = doubled.cleaned()
        assert cleaned.n_vertices == 8
        assert cleaned.n_faces == 12
        assert cleaned.volume == pytest.approx(8.0)
        assert cleaned.is_watertight

    def test_degenerate_faces_are_dropped(self, cube):
        faces = np.vstack([cube.faces, [[0, 0, 1]]])
        mesh = Mesh(cube.vertices, faces)
        assert mesh.remove_degenerate_faces().n_faces == 12

    def test_unreferenced_vertices_are_compacted(self, cube):
        mesh = Mesh(np.vstack([cube.vertices, [[9.0, 9.0, 9.0]]]), cube.faces)
        assert mesh.remove_unreferenced_vertices().n_vertices == 8

    def test_validate_reports_out_of_range_indices(self):
        mesh = Mesh(np.zeros((3, 3)), [[0, 1, 5]])
        assert any("do not exist" in problem for problem in mesh.validate())

    def test_validate_is_clean_for_a_good_mesh(self, cube):
        assert cube.validate() == []


class TestTransforms:
    def test_translation_moves_bounds(self, cube):
        moved = cube.translated([1.0, 2.0, 3.0])
        assert moved.center == pytest.approx([1.0, 2.0, 3.0])
        assert moved.volume == pytest.approx(cube.volume)

    def test_uniform_scale_cubes_the_volume(self, cube):
        assert cube.scaled(2.0).volume == pytest.approx(8.0 * 8.0)

    def test_rotation_preserves_volume_and_area(self, cube):
        turned = cube.rotated(37.0, "y")
        assert turned.volume == pytest.approx(cube.volume)
        assert turned.area == pytest.approx(cube.area)

    def test_mirror_keeps_volume_positive(self, cube):
        mirrored = cube.translated([3.0, 0.0, 0.0]).mirrored("x")
        assert mirrored.volume == pytest.approx(8.0)
        assert mirrored.center == pytest.approx([-3.0, 0.0, 0.0])

    def test_negative_determinant_transform_flips_winding(self, cube):
        flipped = cube.transform(np.diag([-1.0, 1.0, 1.0, 1.0]))
        assert flipped.volume == pytest.approx(8.0)

    def test_transform_matrix_applies_scale_then_rotate_then_translate(self):
        matrix = transform_matrix(translate=[1, 2, 3], rotate=[0, 0, 90], scale=2.0)
        point = matrix @ np.array([1.0, 0.0, 0.0, 1.0])
        assert point[:3] == pytest.approx([1.0, 4.0, 3.0])

    def test_rotation_matrix_is_orthonormal(self):
        matrix = rotation_matrix(31.0, [1.0, 2.0, -0.5])
        assert np.allclose(matrix @ matrix.T, np.eye(3))
        assert np.linalg.det(matrix) == pytest.approx(1.0)

    def test_rotation_about_z_maps_x_to_y(self):
        assert rotation_matrix(90, "z") @ np.array([1.0, 0, 0]) == pytest.approx(
            [0.0, 1.0, 0.0], abs=1e-12
        )

    @pytest.mark.parametrize(
        ("axis", "expected"), [("x", 0), ("Y", 1), ("z", 2), (0, 0), (2, 2)]
    )
    def test_axis_index_accepts_names_and_numbers(self, axis, expected):
        assert axis_index(axis) == expected

    def test_axis_index_rejects_nonsense(self):
        with pytest.raises(ValueError):
            axis_index("w")


class TestRefinement:
    def test_subdivision_preserves_a_flat_solid(self, cube):
        refined = cube.subdivide(2)
        assert refined.n_faces == 12 * 16
        assert refined.volume == pytest.approx(8.0)
        assert refined.is_watertight

    def test_taubin_smoothing_resists_shrinkage(self):
        sphere_like = build_cube(2.0).subdivide(3)
        laplacian = sphere_like.smooth_laplacian(iterations=20, factor=0.5)
        taubin = sphere_like.smooth_taubin(iterations=20)
        assert taubin.volume > laplacian.volume

    def test_smoothing_can_pin_the_boundary(self, grid_patch):
        smoothed = grid_patch.smooth_laplacian(iterations=25, preserve_boundary=True)
        boundary = np.unique(grid_patch.boundary_edges())
        assert np.allclose(smoothed.vertices[boundary], grid_patch.vertices[boundary])

    def test_vertex_clustering_reduces_face_count(self, cube):
        dense = cube.subdivide(3)
        reduced = dense.decimate_vertex_cluster(divisions=4)
        assert reduced.n_faces < dense.n_faces
        assert reduced.n_faces > 0

    def test_surface_sampling_stays_on_the_solid(self, cube):
        points = cube.sample_surface(400, seed=7)
        assert points.shape == (400, 3)
        # Every sample must sit on one of the six faces of the cube.
        assert np.all(np.isclose(np.abs(points).max(axis=1), 1.0))


class TestSummary:
    def test_summary_exposes_json_safe_values(self, cube):
        summary = cube.summary()
        assert summary["vertices"] == 8
        assert summary["faces"] == 12
        assert summary["watertight"] is True
        assert summary["volume"] == pytest.approx(8.0)
        assert isinstance(summary["bounds_min"], list)

    def test_repr_is_informative(self, cube):
        assert "vertices=8" in repr(cube)

    def test_equality_compares_geometry(self, cube):
        assert cube == cube.copy()
        assert cube != cube.translated([1.0, 0.0, 0.0])
