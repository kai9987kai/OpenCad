"""Tests for the analytic mesh primitives.

Solids are checked against their closed-form volume and area, and - just as
importantly - against the topological properties everything downstream assumes:
watertight, consistently wound, positive volume. A primitive that looks right
but is inside out breaks booleans, exports, and printability all at once.

Tessellated primitives are inscribed in the true surface, so their volume
approaches the analytic value *from below*. That direction is asserted too: it
catches a normal or winding error that a symmetric tolerance would let through.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from src.kernel import primitives
from src.kernel.mesh import Mesh

SOLIDS = [
    ("cube", {"size": 2.0}),
    ("box", {"size": (2.0, 3.0, 4.0)}),
    ("sphere", {"radius": 1.0, "theta_resolution": 32, "phi_resolution": 16}),
    ("icosphere", {"radius": 1.0, "subdivisions": 2}),
    ("cylinder", {"radius": 1.0, "height": 2.0, "resolution": 32}),
    ("cone", {"radius": 1.0, "height": 2.0, "resolution": 32}),
    ("torus", {"major_radius": 2.0, "minor_radius": 0.5}),
    ("capsule", {"radius": 0.5, "height": 2.0}),
    ("tube", {"outer_radius": 1.0, "inner_radius": 0.6, "height": 2.0}),
    ("prism", {"sides": 6, "radius": 1.0, "height": 2.0}),
    ("pyramid", {"base": (2.0, 2.0), "height": 2.0}),
    ("wedge", {"size": (2.0, 2.0, 2.0)}),
    ("tetrahedron", {"size": 1.0}),
    ("octahedron", {}),
    ("icosahedron", {}),
]


class TestSolidInvariants:
    @pytest.mark.parametrize(("name", "kwargs"), SOLIDS, ids=[n for n, _ in SOLIDS])
    def test_every_solid_is_closed_and_correctly_wound(self, name, kwargs):
        mesh = getattr(primitives, name)(**kwargs)
        assert isinstance(mesh, Mesh)
        assert not mesh.is_empty
        assert mesh.is_watertight, f"{name} is not watertight"
        assert mesh.is_edge_manifold, f"{name} has non-manifold edges"
        assert mesh.is_oriented, f"{name} has inconsistent winding"
        assert mesh.volume > 0, f"{name} is inside out"
        assert mesh.n_components() == 1
        assert mesh.validate() == []

    @pytest.mark.parametrize(("name", "kwargs"), SOLIDS, ids=[n for n, _ in SOLIDS])
    def test_outward_normals(self, name, kwargs):
        mesh = getattr(primitives, name)(**kwargs)
        # For these convex-ish solids the outward normal has a positive
        # component along the direction from the centroid.
        centroid = mesh.centroid
        radial = mesh.face_centroids() - centroid
        alignment = np.einsum("ij,ij->i", mesh.face_normals(), radial)
        assert alignment.mean() > 0, f"{name} normals point inward on average"

    @pytest.mark.parametrize(("name", "kwargs"), SOLIDS, ids=[n for n, _ in SOLIDS])
    def test_centering_is_honoured(self, name, kwargs):
        if name in ("octahedron", "icosahedron"):
            pytest.skip("platonic helpers take no centre argument in this API")
        offset = (10.0, -5.0, 2.5)
        try:
            mesh = getattr(primitives, name)(center=offset, **kwargs)
        except TypeError:
            pytest.skip(f"{name} does not accept a centre")
        assert mesh.center == pytest.approx(offset, abs=1e-6)


class TestExactVolumes:
    """Primitives with flat faces must be exact, not merely close."""

    def test_cube(self):
        assert primitives.cube(3.0).volume == pytest.approx(27.0)
        assert primitives.cube(3.0).area == pytest.approx(54.0)

    def test_box(self):
        mesh = primitives.box((2.0, 3.0, 4.0))
        assert mesh.volume == pytest.approx(24.0)
        assert mesh.area == pytest.approx(2 * (6 + 8 + 12))

    def test_tetrahedron_is_regular(self):
        mesh = primitives.tetrahedron(1.0)
        edges = mesh.edges()
        lengths = np.linalg.norm(
            mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1
        )
        assert lengths == pytest.approx(lengths[0])
        assert mesh.n_faces == 4

    def test_regular_prism_matches_the_polygon_formula(self):
        sides, radius, height = 6, 1.0, 2.0
        mesh = primitives.prism(sides=sides, radius=radius, height=height)
        # Area of a regular n-gon: (n/2) r^2 sin(2 pi / n).
        base = 0.5 * sides * radius**2 * np.sin(2 * np.pi / sides)
        assert mesh.volume == pytest.approx(base * height)

    def test_pyramid_is_a_third_of_its_prism(self):
        mesh = primitives.pyramid(base=(2.0, 3.0), height=4.0)
        assert mesh.volume == pytest.approx(2.0 * 3.0 * 4.0 / 3.0)

    def test_wedge_is_half_a_box(self):
        mesh = primitives.wedge(size=(2.0, 3.0, 4.0))
        assert mesh.volume == pytest.approx(2.0 * 3.0 * 4.0 / 2.0)

    def test_octahedron_and_icosahedron_are_closed_platonics(self):
        assert primitives.octahedron().n_faces == 8
        assert primitives.icosahedron().n_faces == 20


class TestConvergence:
    @pytest.mark.parametrize(
        ("subdivisions", "tolerance"), [(1, 0.13), (2, 0.04), (3, 0.01), (4, 0.003)]
    )
    def test_icosphere_approaches_the_true_volume_from_below(self, subdivisions, tolerance):
        radius = 2.0
        mesh = primitives.icosphere(radius=radius, subdivisions=subdivisions)
        exact = 4.0 / 3.0 * np.pi * radius**3
        assert mesh.volume < exact  # inscribed, so never over
        assert mesh.volume == pytest.approx(exact, rel=tolerance)

    def test_icosphere_converges_at_second_order(self):
        """Each subdivision halves the edge length, so the volume error should
        fall by roughly four. Asserting the *rate* catches a systematic bias
        that a loose absolute tolerance would happily accept."""
        exact = 4.0 / 3.0 * np.pi
        errors = [
            abs(primitives.icosphere(radius=1.0, subdivisions=n).volume - exact)
            for n in (1, 2, 3, 4)
        ]
        assert errors == sorted(errors, reverse=True)
        for coarse, fine in itertools.pairwise(errors):
            assert 3.0 < coarse / fine < 5.0

    def test_uv_sphere_converges_too(self):
        exact = 4.0 / 3.0 * np.pi
        coarse = primitives.sphere(radius=1.0, theta_resolution=12, phi_resolution=6)
        fine = primitives.sphere(radius=1.0, theta_resolution=96, phi_resolution=48)
        assert abs(fine.volume - exact) < abs(coarse.volume - exact)
        assert fine.volume == pytest.approx(exact, rel=0.01)

    def test_sphere_has_no_degenerate_triangles_at_the_poles(self):
        """A naive UV sphere leaves zero-area quads at the poles."""
        mesh = primitives.sphere(radius=1.0, theta_resolution=24, phi_resolution=12)
        assert np.all(mesh.face_areas() > 1e-12)

    def test_cylinder_volume_converges(self):
        radius, height = 1.0, 2.0
        exact = np.pi * radius**2 * height
        mesh = primitives.cylinder(radius=radius, height=height, resolution=256)
        assert mesh.volume < exact
        assert mesh.volume == pytest.approx(exact, rel=0.001)

    def test_cone_volume_converges(self):
        radius, height = 1.0, 2.0
        exact = np.pi * radius**2 * height / 3.0
        mesh = primitives.cone(radius=radius, height=height, resolution=256)
        assert mesh.volume == pytest.approx(exact, rel=0.001)

    def test_torus_volume_converges(self):
        major, minor = 2.0, 0.5
        exact = 2.0 * np.pi**2 * major * minor**2
        mesh = primitives.torus(
            major_radius=major, minor_radius=minor,
            major_resolution=200, minor_resolution=100,
        )
        assert mesh.volume == pytest.approx(exact, rel=0.005)

    def test_capsule_volume_is_cylinder_plus_sphere(self):
        radius, height = 0.5, 2.0
        exact = np.pi * radius**2 * height + 4.0 / 3.0 * np.pi * radius**3
        mesh = primitives.capsule(radius=radius, height=height, resolution=96)
        assert mesh.volume == pytest.approx(exact, rel=0.02)

    def test_tube_volume_is_the_difference_of_two_cylinders(self):
        outer, inner, height = 1.0, 0.6, 2.0
        exact = np.pi * (outer**2 - inner**2) * height
        mesh = primitives.tube(
            outer_radius=outer, inner_radius=inner, height=height, resolution=200
        )
        assert mesh.volume == pytest.approx(exact, rel=0.005)


class TestTopologyOfHoledSolids:
    def test_a_torus_has_genus_one(self):
        mesh = primitives.torus(major_radius=2.0, minor_radius=0.5)
        assert mesh.is_watertight
        assert mesh.genus == 1

    def test_a_tube_has_genus_one(self):
        mesh = primitives.tube(outer_radius=1.0, inner_radius=0.6, height=2.0)
        assert mesh.is_watertight
        assert mesh.genus == 1


class TestOpenSurfaces:
    def test_a_plane_is_open(self):
        mesh = primitives.plane(size=(2.0, 2.0), resolution=(1, 1))
        assert not mesh.is_watertight
        assert mesh.area == pytest.approx(4.0)
        assert len(mesh.boundary_edges()) == 4

    def test_plane_resolution_subdivides(self):
        mesh = primitives.plane(size=(2.0, 2.0), resolution=(4, 4))
        assert mesh.n_faces == 4 * 4 * 2
        assert mesh.area == pytest.approx(4.0)

    def test_a_disc_is_open_and_has_the_right_area(self):
        mesh = primitives.disc(radius=1.0, resolution=256)
        assert not mesh.is_watertight
        assert mesh.area == pytest.approx(np.pi, rel=0.001)

    def test_an_annulus_subtracts_the_inner_area(self):
        mesh = primitives.disc(radius=1.0, inner_radius=0.5, resolution=256)
        assert mesh.area == pytest.approx(np.pi * (1.0 - 0.25), rel=0.001)


class TestOrientation:
    def test_cylinder_direction_reorients_the_solid(self):
        along_x = primitives.cylinder(radius=0.5, height=4.0, direction=(1, 0, 0))
        extents = along_x.extents
        assert extents[0] == pytest.approx(4.0, rel=1e-6)
        assert extents[1] == pytest.approx(1.0, rel=0.01)
        assert along_x.is_watertight
        assert along_x.volume > 0

    def test_an_antiparallel_direction_does_not_produce_nan(self):
        """Rotating +Z onto -Z is the case a naive cross product cannot handle."""
        mesh = primitives.cylinder(radius=0.5, height=2.0, direction=(0, 0, -1))
        assert np.all(np.isfinite(mesh.vertices))
        assert mesh.is_watertight
        assert mesh.volume == pytest.approx(np.pi * 0.25 * 2.0, rel=0.01)

    def test_direction_preserves_volume(self):
        base = primitives.cone(radius=1.0, height=2.0, direction=(0, 0, 1))
        tilted = primitives.cone(radius=1.0, height=2.0, direction=(1, 1, 1))
        assert tilted.volume == pytest.approx(base.volume, rel=1e-6)


class TestRegistry:
    def test_every_registry_entry_builds(self):
        for name, factory in primitives.PRIMITIVES.items():
            mesh = factory()
            assert not mesh.is_empty, f"{name} produced nothing"

    def test_create_dispatches_by_name(self):
        mesh = primitives.create("cube", size=2.0)
        assert mesh.volume == pytest.approx(8.0)

    def test_create_is_case_insensitive(self):
        assert primitives.create("CUBE", size=1.0).volume == pytest.approx(1.0)

    def test_an_unknown_name_lists_the_supported_ones(self):
        with pytest.raises(ValueError) as info:
            primitives.create("dodecahedron")
        assert "cube" in str(info.value)


class TestValidation:
    @pytest.mark.parametrize(
        ("name", "kwargs"),
        [
            ("sphere", {"radius": 0}),
            ("sphere", {"radius": -1}),
            ("cylinder", {"radius": 0}),
            ("cylinder", {"height": -1}),
            ("cylinder", {"resolution": 2}),
            ("cone", {"radius": 0}),
            ("torus", {"minor_radius": 0}),
            ("tube", {"outer_radius": 1.0, "inner_radius": 1.0}),
            ("tube", {"outer_radius": 0.5, "inner_radius": 1.0}),
            ("prism", {"sides": 2}),
            ("box", {"size": (0, 1, 1)}),
        ],
    )
    def test_invalid_parameters_raise(self, name, kwargs):
        with pytest.raises(ValueError):
            getattr(primitives, name)(**kwargs)
