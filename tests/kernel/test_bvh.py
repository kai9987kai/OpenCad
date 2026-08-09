"""Tests for the bounding volume hierarchy and its spatial queries.

A BVH that returns *nearly* the right answer is worse than one that fails
loudly, because everything downstream - containment, thickness, boolean
classification - silently inherits the error. So these assert exact analytic
values wherever the geometry is exact (rays against a cube, closest points in
each Voronoi region) and convergence where it is not (a faceted sphere).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.kernel import primitives
from src.kernel.bvh import BVH, mesh_sdf
from src.kernel.mesh import Mesh
from tests.conftest import build_cube


@pytest.fixture(scope="module")
def cube_bvh():
    return BVH(primitives.cube(2.0))


@pytest.fixture(scope="module")
def sphere_mesh():
    return primitives.icosphere(radius=2.0, subdivisions=4)


@pytest.fixture(scope="module")
def sphere_bvh(sphere_mesh):
    return BVH(sphere_mesh)


class TestConstruction:
    def test_covers_every_triangle(self):
        mesh = primitives.icosphere(radius=1.0, subdivisions=2)
        bvh = BVH(mesh)
        assert bvh.n_faces == mesh.n_faces
        assert not bvh.is_empty
        assert bvh.n_nodes >= 1

    def test_root_bounds_match_the_mesh(self):
        mesh = primitives.cube(2.0)
        low, high = BVH(mesh).bounds
        assert low == pytest.approx([-1.0, -1.0, -1.0])
        assert high == pytest.approx([1.0, 1.0, 1.0])

    def test_an_empty_mesh_does_not_crash(self):
        bvh = BVH(Mesh.empty())
        assert bvh.is_empty
        assert bvh.n_faces == 0
        assert len(bvh.contains(np.zeros((3, 3)))) == 3

    def test_from_mesh_matches_the_constructor(self):
        mesh = primitives.cube(1.0)
        assert BVH.from_mesh(mesh).n_faces == BVH(mesh).n_faces

    def test_a_single_triangle_is_valid(self):
        mesh = Mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
        bvh = BVH(mesh)
        assert bvh.n_faces == 1
        _, distance, _ = bvh.closest_point(np.array([[0.0, 0.0, 3.0]]))
        assert distance == pytest.approx([3.0])


class TestRayCasting:
    def test_axis_aligned_rays_hit_a_cube_at_exact_distances(self, cube_bvh):
        origins = np.array(
            [[-5.0, 0, 0], [0, -5.0, 0], [0, 0, 5.0], [5.0, 0, 0]], dtype=float
        )
        directions = np.array(
            [[1.0, 0, 0], [0, 1.0, 0], [0, 0, -1.0], [-1.0, 0, 0]], dtype=float
        )
        distances, faces = cube_bvh.first_hit(origins, directions)
        assert distances == pytest.approx([4.0, 4.0, 4.0, 4.0])
        assert np.all(faces >= 0)

    def test_a_ray_that_misses_reports_infinity(self, cube_bvh):
        distances, faces = cube_bvh.first_hit(
            np.array([[-5.0, 3.0, 0.0]]), np.array([[1.0, 0.0, 0.0]])
        )
        assert np.isinf(distances[0])
        assert faces[0] == -1

    def test_a_ray_through_a_solid_records_entry_and_exit(self, cube_bvh):
        # Offset off the face diagonal so the ray passes through triangle
        # interiors; see the degenerate case below for why that matters.
        hits = cube_bvh.ray_intersections(
            np.array([[-5.0, 0.31, 0.17]]), np.array([[1.0, 0.0, 0.0]])
        )
        assert len(hits) == 1
        assert list(hits.counts) == [2]
        distances, faces = hits[0]
        assert distances == pytest.approx([4.0, 6.0])  # sorted near to far
        assert len(set(faces.tolist())) == 2

    def test_a_ray_along_a_shared_edge_hits_both_triangles(self, cube_bvh):
        """Documented degenerate case, not a defect.

        Each cube face is two triangles sharing a diagonal. A ray down the
        centre line hits that shared edge, so it legitimately intersects both
        triangles of both faces - four hits, not two. This is exactly why
        containment uses the winding number rather than counting crossings.
        """
        hits = cube_bvh.ray_intersections(
            np.array([[-5.0, 0.0, 0.0]]), np.array([[1.0, 0.0, 0.0]])
        )
        distances, _ = hits[0]
        assert list(hits.counts) == [4]
        assert distances == pytest.approx([4.0, 4.0, 6.0, 6.0])
        # Containment must still be right despite the odd crossing count.
        assert cube_bvh.contains(np.array([[0.0, 0.0, 0.0]]))[0]

    def test_hits_are_sorted_and_first_matches_first_hit(self, cube_bvh):
        origins = np.array([[-5.0, 0.31, 0.17], [-5.0, 3.0, 0.0]])
        directions = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        hits = cube_bvh.ray_intersections(origins, directions)
        nearest, faces = hits.first()
        expected_distance, expected_faces = cube_bvh.first_hit(origins, directions)
        assert nearest == pytest.approx(expected_distance)
        assert list(faces) == list(expected_faces)
        for index in range(len(hits)):
            distances, _ = hits[index]
            assert list(distances) == sorted(distances)

    def test_ray_distance_scales_with_the_sphere_radius(self):
        for radius in (0.5, 1.0, 3.0):
            bvh = BVH(primitives.icosphere(radius=radius, subdivisions=4))
            distances, _ = bvh.first_hit(
                np.array([[-10.0, 0.0, 0.0]]), np.array([[1.0, 0.0, 0.0]])
            )
            # A faceted sphere is inscribed, so the hit is never early.
            assert distances[0] >= 10.0 - radius - 1e-9
            assert distances[0] == pytest.approx(10.0 - radius, rel=0.01)

    def test_the_result_is_a_distance_not_a_ray_parameter(self, cube_bvh):
        """Direction magnitude must not scale the answer: the hierarchy
        normalises, so the result is millimetres either way."""
        one, _ = cube_bvh.first_hit(
            np.array([[-5.0, 0.0, 0.0]]), np.array([[1.0, 0.0, 0.0]])
        )
        two, _ = cube_bvh.first_hit(
            np.array([[-5.0, 0.0, 0.0]]), np.array([[7.0, 0.0, 0.0]])
        )
        assert one == pytest.approx([4.0])
        assert two == pytest.approx([4.0])


class TestClosestPoint:
    def test_lands_in_the_right_voronoi_region_of_a_cube(self, cube_bvh):
        """Face, edge, and corner regions each need a different branch of the
        point-to-triangle test; a projection-only implementation fails the
        edge and corner cases."""
        query = np.array(
            [[5.0, 0, 0], [5.0, 5.0, 0], [5.0, 5.0, 5.0], [0.0, 0, 0]], dtype=float
        )
        closest, distance, faces = cube_bvh.closest_point(query)
        assert closest[0] == pytest.approx([1.0, 0.0, 0.0])
        assert closest[1] == pytest.approx([1.0, 1.0, 0.0])
        assert closest[2] == pytest.approx([1.0, 1.0, 1.0])
        assert distance == pytest.approx(
            [4.0, np.sqrt(32.0), np.sqrt(48.0), 1.0]
        )
        assert np.all(faces >= 0)

    def test_a_point_on_the_surface_has_zero_distance(self, cube_bvh):
        _, distance, _ = cube_bvh.closest_point(np.array([[1.0, 0.3, -0.2]]))
        assert distance == pytest.approx([0.0], abs=1e-12)

    def test_closest_point_lies_on_the_reported_face(self, sphere_mesh, sphere_bvh, rng):
        query = rng.normal(size=(200, 3)) * 3.0
        closest, _, faces = sphere_bvh.closest_point(query)
        triangles = sphere_mesh.triangles()[faces]
        # Barycentric reconstruction: the point must be inside its triangle.
        for point, triangle in zip(closest[:25], triangles[:25], strict=False):
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            assert abs(np.dot(point - triangle[0], normal)) < 1e-9 * max(
                1.0, np.linalg.norm(normal)
            )

    def test_distance_matches_closest_point(self, cube_bvh, rng):
        query = rng.normal(size=(100, 3)) * 2.0
        _, distance, _ = cube_bvh.closest_point(query)
        assert cube_bvh.distance(query) == pytest.approx(distance)


class TestContainment:
    def test_a_cube_classifies_inside_and_outside(self, cube_bvh):
        query = np.array(
            [[0, 0, 0], [0.9, 0.9, 0.9], [1.1, 0, 0], [5, 5, 5]], dtype=float
        )
        assert list(cube_bvh.contains(query)) == [True, True, False, False]

    def test_a_sphere_agrees_with_its_analytic_definition(self, sphere_bvh, rng):
        query = rng.normal(size=(2000, 3)) * 2.5
        radius = np.linalg.norm(query, axis=1)
        # Skip points within the faceting error of the surface.
        away = np.abs(radius - 2.0) > 0.02
        assert np.array_equal(sphere_bvh.contains(query[away]), radius[away] < 2.0)

    def test_the_hole_of_a_torus_is_outside(self):
        """The case a sloppy parity ray cast gets wrong: the axis of a torus is
        outside the solid, but a ray along it grazes the surface."""
        torus = primitives.torus(
            major_radius=3.0, minor_radius=1.0, major_resolution=96, minor_resolution=48
        )
        bvh = BVH(torus)
        query = np.array(
            [[0, 0, 0], [3.0, 0, 0], [3.0, 0, 0.5], [10.0, 0, 0], [0, 0, 5.0]],
            dtype=float,
        )
        assert list(bvh.contains(query)) == [False, True, True, False, False]

    def test_winding_number_is_one_inside_and_zero_outside(self, cube_bvh):
        winding = cube_bvh.winding_number(np.array([[0.0, 0, 0], [9.0, 0, 0]]))
        assert winding[0] == pytest.approx(1.0, abs=1e-3)
        assert winding[1] == pytest.approx(0.0, abs=1e-3)


class TestSignedDistance:
    def test_approximates_the_analytic_sphere_distance(self, sphere_bvh, rng):
        query = rng.normal(size=(1500, 3)) * 2.5
        exact = np.linalg.norm(query, axis=1) - 2.0
        assert sphere_bvh.signed_distance(query) == pytest.approx(exact, abs=0.01)

    def test_error_shrinks_as_the_mesh_refines(self, rng):
        query = rng.normal(size=(400, 3)) * 2.5
        exact = np.linalg.norm(query, axis=1) - 2.0
        errors = []
        for subdivisions in (2, 3, 4):
            bvh = BVH(primitives.icosphere(radius=2.0, subdivisions=subdivisions))
            errors.append(np.abs(bvh.signed_distance(query) - exact).mean())
        assert errors[0] > errors[1] > errors[2]

    def test_sign_is_negative_inside(self, cube_bvh):
        values = cube_bvh.signed_distance(np.array([[0.0, 0, 0], [3.0, 0, 0]]))
        assert values[0] < 0
        assert values[1] > 0
        assert values == pytest.approx([-1.0, 2.0])

    def test_mesh_sdf_bridges_a_mesh_into_the_field_world(self):
        """This is what lets an imported STL take part in implicit modelling."""
        field = mesh_sdf(primitives.cube(2.0))
        assert field(np.array([[0.0, 0, 0]])) == pytest.approx([-1.0])
        assert field(np.array([[3.0, 0, 0]])) == pytest.approx([2.0])

    def test_a_mesh_field_can_be_offset_like_any_other(self):
        from src.kernel.meshing import surface_nets

        field = mesh_sdf(primitives.cube(10.0)).offset(1.0)
        mesh = surface_nets(field, resolution=48)
        # Growing a 10 mm cube by 1 mm gives a rounded 12 mm box, so its
        # volume sits between the cube and its bounding box.
        assert 12.0**3 > mesh.volume > 10.0**3
        assert mesh.is_watertight


class TestIntersection:
    def test_overlapping_solids_report_intersecting_triangles(self):
        a = BVH(primitives.cube(2.0))
        b = BVH(primitives.cube(2.0).translated([1.0, 0.0, 0.0]))
        assert a.intersects(b)
        assert len(a.intersection_pairs(b)) > 0

    def test_separated_solids_do_not_intersect(self):
        a = BVH(primitives.cube(2.0))
        b = BVH(primitives.cube(2.0).translated([10.0, 0.0, 0.0]))
        assert not a.intersects(b)
        assert len(a.intersection_pairs(b)) == 0

    def test_a_clean_solid_has_no_self_intersections(self, cube_bvh):
        assert len(cube_bvh.self_intersections()) == 0

    def test_crossing_sheets_are_detected(self):
        horizontal = Mesh(
            [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [[0, 1, 2], [0, 2, 3]]
        )
        vertical = Mesh(
            [[0, -1, -1], [0, 1, -1], [0, 1, 1], [0, -1, 1]], [[0, 1, 2], [0, 2, 3]]
        )
        crossed = Mesh.concatenate([horizontal, vertical])
        assert len(BVH(crossed).self_intersections()) > 0

    def test_box_query_finds_overlapping_faces(self, cube_bvh):
        faces = cube_bvh.box_query(np.array([0.9, -2.0, -2.0]), np.array([2.0, 2.0, 2.0]))
        assert len(faces) > 0
        assert len(faces) < cube_bvh.n_faces

    def test_box_query_outside_the_mesh_returns_nothing(self, cube_bvh):
        faces = cube_bvh.box_query(np.array([10.0, 10.0, 10.0]), np.array([11.0, 11.0, 11.0]))
        assert len(faces) == 0


class TestAgainstTheMeshType:
    def test_works_on_a_mesh_built_by_the_test_fixtures(self):
        bvh = BVH(build_cube(2.0))
        assert bvh.signed_distance(np.array([[0.0, 0, 0]])) == pytest.approx([-1.0])

    def test_transformed_meshes_are_handled(self):
        moved = primitives.cube(2.0).translated([10.0, 0.0, 0.0])
        bvh = BVH(moved)
        assert bvh.contains(np.array([[10.0, 0.0, 0.0]]))[0]
        assert not bvh.contains(np.array([[0.0, 0.0, 0.0]]))[0]
