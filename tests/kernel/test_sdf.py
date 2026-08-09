"""Tests for the signed distance field layer.

Distance fields are easy to get subtly wrong in ways that still look plausible
when meshed - a sign flipped, a field that is a bound rather than a distance, a
rotation applied forwards instead of inverted. These tests check the actual
numbers: hand-computed distances at chosen points, gradient magnitude (which is
exactly 1 for a true distance field and nothing else), and algebraic identities
between the CSG operators.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.kernel import sdf


@pytest.fixture
def points(rng):
    return rng.normal(size=(400, 3)) * 3.0


class TestExactPrimitives:
    def test_sphere_distance_is_exact(self):
        field = sdf.sphere(2.0)
        query = np.array([[0, 0, 0], [2, 0, 0], [4, 0, 0], [0, 3, 0]], dtype=float)
        assert field(query) == pytest.approx([-2.0, 0.0, 2.0, 1.0])

    def test_sphere_honours_its_centre(self):
        field = sdf.sphere(1.0, center=(5.0, 0.0, 0.0))
        assert field(np.array([[5.0, 0.0, 0.0]])) == pytest.approx([-1.0])
        assert field(np.array([[7.0, 0.0, 0.0]])) == pytest.approx([1.0])

    def test_box_distance_is_exact_inside_and_out(self):
        field = sdf.box((2.0, 2.0, 2.0))
        query = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 2, 2], [0.5, 0, 0]], dtype=float)
        # Inside, the distance is to the nearest face; outside a corner it is the
        # true diagonal distance, which a naive max-of-slabs gets wrong.
        assert field(query) == pytest.approx([-1.0, 0.0, 1.0, np.sqrt(3.0), -0.5])

    def test_plane_is_a_signed_height(self):
        field = sdf.plane((0.0, 0.0, 1.0), 0.0)
        assert field(np.array([[0, 0, -3], [0, 0, 0], [0, 0, 5]], dtype=float)) == pytest.approx(
            [-3.0, 0.0, 5.0]
        )

    def test_plane_normal_is_normalised(self):
        field = sdf.plane((0.0, 0.0, 7.0), 0.0)
        assert field(np.array([[0.0, 0.0, 2.0]])) == pytest.approx([2.0])

    def test_cylinder_distance_is_exact(self):
        field = sdf.cylinder(1.0, 2.0)
        query = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 0, 1], [0, 0, 2]], dtype=float)
        assert field(query) == pytest.approx([-1.0, 0.0, 1.0, 0.0, 1.0])

    def test_torus_distance_is_exact(self):
        field = sdf.torus(2.0, 0.5)
        query = np.array([[2, 0, 0], [2.5, 0, 0], [0, 0, 0], [3, 0, 0]], dtype=float)
        assert field(query) == pytest.approx([-0.5, 0.0, 1.5, 0.5])

    def test_capsule_reduces_to_a_sphere_for_a_zero_length_segment(self):
        field = sdf.capsule((0, 0, 0), (0, 0, 0), 1.5)
        assert field(np.array([[3.0, 0.0, 0.0]])) == pytest.approx([1.5])

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: sdf.sphere(2.0),
            lambda: sdf.box((2.0, 3.0, 1.5)),
            lambda: sdf.cylinder(1.0, 2.0),
            lambda: sdf.torus(2.0, 0.5),
            lambda: sdf.plane((1.0, 1.0, 1.0), 0.5),
            lambda: sdf.capsule((-1, 0, 0), (1, 0, 0), 0.5),
            lambda: sdf.cone(1.0, 2.0),
            lambda: sdf.hex_prism(1.0, 2.0),
        ],
    )
    def test_exact_fields_have_unit_gradient(self, factory, points):
        """|grad f| == 1 everywhere is the definition of a true distance field."""
        magnitude = np.linalg.norm(factory().gradient(points), axis=1)
        assert magnitude == pytest.approx(1.0, abs=2e-3)

    def test_ellipsoid_is_honest_about_being_approximate(self):
        field = sdf.ellipsoid((3.0, 2.0, 1.0))
        assert field.exact is False
        # The zero level set is still correct even though distances are not.
        assert field(np.array([[3.0, 0.0, 0.0]])) == pytest.approx(0.0, abs=1e-9)
        assert field(np.array([[0.0, 2.0, 0.0]])) == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize(
        ("factory", "kwargs"),
        [
            (sdf.sphere, {"radius": 0}),
            (sdf.sphere, {"radius": -1}),
            (sdf.box, {"size": (0, 1, 1)}),
            (sdf.cylinder, {"radius": -1}),
            (sdf.cone, {"height": 0}),
            (sdf.torus, {"minor_radius": 0}),
            (sdf.hex_prism, {"radius": 0}),
        ],
    )
    def test_degenerate_parameters_are_rejected(self, factory, kwargs):
        with pytest.raises(ValueError):
            factory(**kwargs)


class TestBounds:
    def test_primitive_bounds_are_tight(self):
        low, high = sdf.sphere(2.0, center=(1.0, 0.0, 0.0)).bounds
        assert low == pytest.approx([-1.0, -2.0, -2.0])
        assert high == pytest.approx([3.0, 2.0, 2.0])

    def test_union_bounds_span_both_operands(self):
        field = sdf.sphere(1.0) | sdf.sphere(1.0, center=(10.0, 0.0, 0.0))
        low, high = field.bounds
        assert low == pytest.approx([-1.0, -1.0, -1.0])
        assert high == pytest.approx([11.0, 1.0, 1.0])

    def test_difference_keeps_the_left_bounds(self):
        field = sdf.box((4.0, 4.0, 4.0)) - sdf.sphere(1.0)
        low, _high = field.bounds
        assert low == pytest.approx([-2.0, -2.0, -2.0])

    def test_a_half_space_has_no_bounds(self):
        assert sdf.plane().bounds is None

    def test_meshing_an_unbounded_field_raises_a_helpful_error(self):
        with pytest.raises(ValueError, match="no known extent"):
            sdf.tpms_sheet("gyroid").sample_bounds()

    def test_bounded_attaches_an_extent(self):
        field = sdf.tpms_sheet("gyroid").bounded([-5, -5, -5], [5, 5, 5])
        low, _ = field.sample_bounds()
        assert low[0] < -5.0


class TestBooleans:
    def test_operators_match_their_definitions(self, points):
        a, b = sdf.sphere(1.5), sdf.box((2.0, 2.0, 2.0))
        assert (a | b)(points) == pytest.approx(np.minimum(a(points), b(points)))
        assert (a & b)(points) == pytest.approx(np.maximum(a(points), b(points)))
        assert (a - b)(points) == pytest.approx(np.maximum(a(points), -b(points)))

    def test_complement_swaps_inside_and_outside(self, points):
        field = sdf.sphere(1.0)
        assert (~field)(points) == pytest.approx(-field(points))

    def test_difference_actually_removes_material(self):
        field = sdf.box((4.0, 4.0, 4.0)) - sdf.sphere(1.0)
        assert field(np.array([[0.0, 0.0, 0.0]]))[0] > 0  # the hole is empty
        assert field(np.array([[1.8, 0.0, 0.0]]))[0] < 0  # the shell is solid

    def test_union_accepts_several_operands(self, points):
        parts = [sdf.sphere(1.0, center=(x, 0, 0)) for x in (-3, 0, 3)]
        combined = parts[0].union(parts[1], parts[2])
        expected = np.minimum.reduce([part(points) for part in parts])
        assert combined(points) == pytest.approx(expected)


class TestSmoothBooleans:
    def test_zero_blend_is_exactly_the_sharp_operator(self, points):
        a, b = sdf.sphere(1.5), sdf.box((2.0, 2.0, 2.0))
        assert a.smooth_union(b, 0.0)(points) == pytest.approx(np.minimum(a(points), b(points)))
        assert a.smooth_intersection(b, 0.0)(points) == pytest.approx(
            np.maximum(a(points), b(points))
        )

    def test_smooth_union_only_ever_adds_material(self, points):
        a, b = sdf.sphere(1.5), sdf.box((2.0, 2.0, 2.0))
        blended = a.smooth_union(b, 0.5)(points)
        assert np.all(blended <= np.minimum(a(points), b(points)) + 1e-12)

    def test_smooth_intersection_only_ever_removes_material(self, points):
        a, b = sdf.sphere(1.5), sdf.box((2.0, 2.0, 2.0))
        blended = a.smooth_intersection(b, 0.5)(points)
        assert np.all(blended >= np.maximum(a(points), b(points)) - 1e-12)

    def test_a_larger_blend_radius_adds_more_material(self, points):
        a = sdf.sphere(1.0, center=(-0.8, 0, 0))
        b = sdf.sphere(1.0, center=(0.8, 0, 0))
        small = a.smooth_union(b, 0.2)(points)
        large = a.smooth_union(b, 0.8)(points)
        assert np.all(large <= small + 1e-12)
        assert np.any(large < small - 1e-6)

    def test_blend_interpolates_between_fields(self, points):
        a, b = sdf.sphere(1.0), sdf.box((2.0, 2.0, 2.0))
        assert a.blend(b, 0.0)(points) == pytest.approx(a(points))
        assert a.blend(b, 1.0)(points) == pytest.approx(b(points))
        assert a.blend(b, 0.5)(points) == pytest.approx((a(points) + b(points)) / 2)


class TestOffsetAndShell:
    def test_offset_moves_the_surface_by_a_real_distance(self):
        grown = sdf.sphere(1.0).offset(0.5)
        assert grown(np.array([[1.5, 0.0, 0.0]])) == pytest.approx([0.0])

    def test_negative_offset_shrinks(self):
        shrunk = sdf.sphere(1.0).offset(-0.25)
        assert shrunk(np.array([[0.75, 0.0, 0.0]])) == pytest.approx([0.0])

    def test_shell_produces_a_wall_of_the_requested_thickness(self):
        shell = sdf.sphere(2.0).shell(0.4)
        # The wall is centred on r=2, so it spans 1.8 to 2.2.
        assert shell(np.array([[2.0, 0, 0]])) == pytest.approx([-0.2])
        assert shell(np.array([[1.8, 0, 0]])) == pytest.approx([0.0], abs=1e-12)
        assert shell(np.array([[2.2, 0, 0]])) == pytest.approx([0.0], abs=1e-12)
        assert shell(np.array([[0.0, 0, 0]]))[0] > 0  # hollow inside


class TestTransforms:
    def test_translate_moves_the_solid(self):
        field = sdf.sphere(1.0).translate((5.0, 0.0, 0.0))
        assert field(np.array([[5.0, 0.0, 0.0]])) == pytest.approx([-1.0])

    def test_rotation_is_applied_as_an_inverse_to_the_query_point(self):
        # A long box on X, turned onto Y, must now be long in Y.
        field = sdf.box((6.0, 1.0, 1.0)).rotate(90.0, "z")
        assert field(np.array([[0.0, 2.5, 0.0]]))[0] < 0
        assert field(np.array([[2.5, 0.0, 0.0]]))[0] > 0

    def test_rotating_back_restores_the_original(self, points):
        original = sdf.box((2.0, 3.0, 1.0))
        round_trip = original.rotate(37.0, "y").rotate(-37.0, "y")
        assert round_trip(points) == pytest.approx(original(points), abs=1e-9)

    def test_uniform_scale_keeps_the_field_metric(self):
        field = sdf.sphere(1.0).scale(3.0)
        assert field(np.array([[3.0, 0.0, 0.0]])) == pytest.approx([0.0])
        assert field(np.array([[0.0, 0.0, 0.0]])) == pytest.approx([-3.0])

    def test_zero_scale_is_rejected(self):
        with pytest.raises(ValueError):
            sdf.sphere(1.0).scale(0.0)

    def test_transform_marks_non_similarities_as_approximate(self):
        stretched = sdf.sphere(1.0).transform(np.diag([2.0, 1.0, 1.0, 1.0]))
        assert stretched.exact is False
        # The zero level set is still right even though distances are not.
        assert stretched(np.array([[2.0, 0.0, 0.0]])) == pytest.approx([0.0], abs=1e-9)

    def test_elongate_stretches_without_distorting_the_ends(self):
        field = sdf.sphere(1.0).elongate((2.0, 0.0, 0.0))
        assert field(np.array([[3.0, 0.0, 0.0]])) == pytest.approx([0.0])
        assert field(np.array([[0.0, 1.0, 0.0]])) == pytest.approx([0.0])

    def test_repeat_tiles_the_field(self):
        field = sdf.sphere(0.4).repeat((2.0, 2.0, 2.0))
        for centre in ([0, 0, 0], [2, 0, 0], [-4, 2, 6]):
            assert field(np.array([centre], dtype=float)) == pytest.approx([-0.4])

    def test_bounded_repeat_stops_after_the_requested_count(self):
        field = sdf.sphere(0.4).repeat((2.0, 2.0, 2.0), count=(1, 0, 0))
        assert field(np.array([[2.0, 0.0, 0.0]])) == pytest.approx([-0.4])
        # Beyond the count the last instance is reused, not a new one.
        assert field(np.array([[6.0, 0.0, 0.0]]))[0] > 0

    def test_repeat_rejects_a_non_positive_spacing(self):
        with pytest.raises(ValueError):
            sdf.sphere(1.0).repeat((0.0, 1.0, 1.0))


class TestQueries:
    def test_contains_matches_the_sign(self):
        field = sdf.sphere(1.0)
        query = np.array([[0, 0, 0], [2, 0, 0]], dtype=float)
        assert list(field.contains(query)) == [True, False]

    def test_normals_point_outward_and_are_unit_length(self, rng):
        field = sdf.sphere(2.0)
        query = rng.normal(size=(50, 3)) * 4.0
        normals = field.normal(query)
        assert np.linalg.norm(normals, axis=1) == pytest.approx(1.0, abs=1e-3)
        # For a sphere at the origin the normal is the radial direction.
        radial = query / np.linalg.norm(query, axis=1)[:, None]
        assert np.einsum("ij,ij->i", normals, radial) == pytest.approx(1.0, abs=1e-3)

    def test_project_lands_points_on_the_surface(self, rng):
        field = sdf.sphere(2.0)
        start = rng.normal(size=(100, 3)) * 3.0 + 0.5
        landed = field.project(start, iterations=12)
        assert field(landed) == pytest.approx(0.0, abs=1e-9)

    def test_single_point_returns_a_scalar(self):
        assert isinstance(sdf.sphere(1.0)([0.0, 0.0, 0.0]), float)

    def test_empty_query_returns_an_empty_array(self):
        assert sdf.sphere(1.0)(np.zeros((0, 3))).shape == (0,)


class TestProfileSolids:
    def test_extrude_gives_the_right_volume_signs(self):
        square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
        field = sdf.extrude(square, height=2.0)
        assert field(np.array([[0.0, 0.0, 0.0]])) == pytest.approx([-1.0])
        assert field(np.array([[0.0, 0.0, 2.0]])) == pytest.approx([1.0])
        assert field(np.array([[2.0, 0.0, 0.0]])) == pytest.approx([1.0])

    def test_extrude_needs_a_real_profile(self):
        with pytest.raises(ValueError):
            sdf.extrude(np.array([[0.0, 0.0], [1.0, 0.0]]), 1.0)

    def test_revolve_reproduces_a_cylinder(self):
        profile = np.array([[0.0, -1.0], [1.0, -1.0], [1.0, 1.0], [0.0, 1.0]])
        field = sdf.revolve(profile)
        reference = sdf.cylinder(1.0, 2.0)
        query = np.array([[0, 0, 0], [0.5, 0, 0], [1.5, 0, 0], [0, 0, 1.5]], dtype=float)
        assert field(query) == pytest.approx(reference(query), abs=1e-9)

    def test_cone_apex_and_base(self):
        field = sdf.cone(1.0, 2.0)
        assert field(np.array([[0.0, 0.0, 1.0]])) == pytest.approx([0.0], abs=1e-9)
        assert field(np.array([[1.0, 0.0, -1.0]])) == pytest.approx([0.0], abs=1e-9)
        assert field(np.array([[0.0, 0.0, -2.0]])) == pytest.approx([1.0])


class TestTPMS:
    @pytest.mark.parametrize("kind", list(sdf.TPMS_KINDS))
    def test_fields_are_periodic(self, kind, rng):
        period = 8.0
        field = sdf.tpms(kind, period=period)
        query = rng.normal(size=(50, 3)) * 5.0
        shifted = query + np.array([period, -period, 2 * period])
        assert field(shifted) == pytest.approx(field(query), abs=1e-9)

    @pytest.mark.parametrize(
        "alias", ["Schwarz P", "schwarz-p", "primitive", "SCHWARZ_P", " p "]
    )
    def test_kind_aliases_resolve(self, alias):
        assert sdf.normalize_tpms_kind(alias) == "schwarz_p"

    def test_unknown_kind_lists_the_supported_ones(self):
        with pytest.raises(ValueError, match="Supported"):
            sdf.tpms("hyperboloid")

    @pytest.mark.parametrize("kind", ["gyroid", "diamond", "iwp"])
    def test_sheet_wall_thickness_is_close_to_the_request(self, kind):
        """Land on the minimal surface, step half a wall along its normal, and
        the sheet boundary should be there. Measuring along an arbitrary ray
        would measure a chord instead."""
        period, wall = 10.0, 1.0
        surface = sdf.tpms_solid(kind, period=period, level=0.0)
        sheet = sdf.tpms_sheet(kind, period=period, thickness=wall)

        generator = np.random.default_rng(4)
        start = generator.uniform(-period, period, size=(600, 3))
        landed = surface.project(start, iterations=40)
        landed = landed[np.abs(surface(landed)) < 1e-6]
        assert len(landed) > 100

        probe = landed + surface.normal(landed) * (wall / 2.0)
        error = np.abs(sheet(probe))
        # Documented accuracy for these three kinds is a few percent.
        assert float(np.median(error)) < 0.05 * wall

    def test_sheet_is_solid_on_the_surface_and_hollow_between(self):
        sheet = sdf.tpms_sheet("gyroid", period=10.0, thickness=1.0)
        assert sheet(np.array([[0.0, 0.0, 0.0]])) == pytest.approx([-0.5], abs=1e-9)

    def test_graded_thickness_really_varies_through_space(self):
        graded = sdf.tpms_sheet(
            "gyroid", period=10.0, thickness=lambda p: 0.4 + 0.15 * (p[:, 0] + 10.0)
        )
        thin = graded(np.array([[-10.0, 0.0, 0.0]]))[0]
        thick = graded(np.array([[10.0, 0.0, 0.0]]))[0]
        # Deeper inside the wall means a more negative field value.
        assert thick < thin
        assert thin == pytest.approx(-0.2, abs=1e-9)
        assert thick == pytest.approx(-1.7, abs=1e-9)

    def test_graded_period_is_accepted(self):
        field = sdf.tpms_sheet("gyroid", period=lambda p: 5.0 + 0.1 * p[:, 2], thickness=0.5)
        assert np.isfinite(field(np.array([[1.0, 2.0, 3.0]]))).all()

    def test_solid_level_trades_porosity_for_strut_thickness(self, rng):
        query = rng.uniform(-5, 5, size=(2000, 3))
        thin = sdf.tpms_solid("gyroid", period=10.0, level=0.0)
        thick = sdf.tpms_solid("gyroid", period=10.0, level=1.0)
        assert thick.contains(query).sum() > thin.contains(query).sum()

    def test_a_lattice_can_be_trimmed_to_a_solid(self):
        part = sdf.sphere(10.0) & sdf.tpms_sheet("gyroid", period=5.0, thickness=0.6)
        assert part(np.array([[30.0, 0.0, 0.0]]))[0] > 0  # nothing outside the sphere
        assert part.bounds is not None  # intersection recovers a finite extent


class TestSampleGrid:
    def test_shape_origin_and_spacing(self):
        values, origin, spacing = sdf.sample_grid(
            sdf.sphere(1.0), bounds=([-1, -1, -1], [1, 1, 1]), resolution=5
        )
        assert values.shape == (5, 5, 5)
        assert origin == pytest.approx([-1.0, -1.0, -1.0])
        assert spacing == pytest.approx([0.5, 0.5, 0.5])

    def test_values_match_direct_evaluation(self):
        field = sdf.sphere(1.0)
        values, origin, spacing = sdf.sample_grid(
            field, bounds=([-2, -2, -2], [2, 2, 2]), resolution=9
        )
        for index in [(0, 0, 0), (4, 4, 4), (8, 3, 1)]:
            point = origin + np.array(index) * spacing
            assert values[index] == pytest.approx(field(point[None, :])[0])

    def test_indexing_is_x_y_z_in_that_order(self):
        """values[i, j, k] must vary with x along i - getting this backwards
        mirrors every extracted mesh."""
        field = sdf.plane((1.0, 0.0, 0.0), 0.0)  # value == x
        values, _origin, _spacing = sdf.sample_grid(
            field, bounds=([-1, -1, -1], [1, 1, 1]), resolution=5
        )
        assert values[0, 2, 2] == pytest.approx(-1.0)
        assert values[4, 2, 2] == pytest.approx(1.0)
        assert values[2, 0, 4] == pytest.approx(0.0)

    def test_anisotropic_resolution(self):
        values, _, _ = sdf.sample_grid(
            sdf.sphere(1.0), bounds=([-1, -1, -1], [1, 1, 1]), resolution=(4, 6, 8)
        )
        assert values.shape == (4, 6, 8)

    def test_chunking_does_not_change_the_result(self):
        field = sdf.sphere(1.0)
        box = ([-2, -2, -2], [2, 2, 2])
        whole, _, _ = sdf.sample_grid(field, box, 12, chunk=1 << 20)
        chunked, _, _ = sdf.sample_grid(field, box, 12, chunk=1024)
        assert whole == pytest.approx(chunked)

    def test_degenerate_bounds_and_resolution_are_rejected(self):
        with pytest.raises(ValueError):
            sdf.sample_grid(sdf.sphere(1.0), ([0, 0, 0], [0, 1, 1]), 8)
        with pytest.raises(ValueError):
            sdf.sample_grid(sdf.sphere(1.0), ([-1, -1, -1], [1, 1, 1]), 1)
