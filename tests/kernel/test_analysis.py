"""Tests for the analysis and printability checks.

An analysis tool that reports a problem which is not there is worse than one
that reports nothing, so these tests pin down both directions: a clean solid
must come back clean, and a deliberately broken one must be caught.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.kernel import sdf
from src.kernel.analysis import (
    ERROR,
    WARNING,
    mesh_report,
    oriented_bounding_box,
    overhangs,
    printability,
    triangle_quality,
    volume_fraction,
    wall_thickness,
)
from src.kernel.mesh import Mesh
from src.kernel.meshing import surface_nets
from src.kernel.units import UnitSystem
from tests.conftest import build_cube, build_grid_patch


def titles(findings):
    return [finding.title for finding in findings]


class TestOverhangs:
    def test_a_cube_overhangs_exactly_its_underside(self):
        cube = build_cube(2.0)
        mask, area, fraction = overhangs(cube, (0, 0, 1), max_angle=45.0)
        # One of six faces points straight down: 4 of 24 square units.
        assert area == pytest.approx(4.0)
        assert fraction == pytest.approx(1.0 / 6.0)
        assert mask.sum() == 2  # the two triangles of the bottom face

    def test_rotating_the_build_direction_moves_the_overhang(self):
        cube = build_cube(2.0)
        _, area, _ = overhangs(cube, (1, 0, 0), max_angle=45.0)
        assert area == pytest.approx(4.0)

    def test_a_45_degree_face_sits_exactly_on_the_threshold(self):
        """45 degrees is the printable/unprintable boundary, so pin it down.

        The convention is that exactly 45 degrees still prints; only steeper
        faces need support.
        """
        # A triangle whose normal is (0, 1, -1)/sqrt(2): 45 degrees below level.
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]], dtype=float)
        face = Mesh(vertices, [[0, 2, 1]])
        assert face.face_normals()[0] == pytest.approx([0.0, 1 / np.sqrt(2), -1 / np.sqrt(2)])

        assert overhangs(face, (0, 0, 1), max_angle=45.0)[0].sum() == 0
        assert overhangs(face, (0, 0, 1), max_angle=44.0)[0].sum() == 1
        assert overhangs(face, (0, 0, 1), max_angle=46.0)[0].sum() == 0

    def test_a_steeper_threshold_never_flags_fewer_faces(self):
        sphere = surface_nets(sdf.sphere(1.0), resolution=32)
        counts = [overhangs(sphere, (0, 0, 1), max_angle=a)[1] for a in (10, 30, 50, 80)]
        assert counts == sorted(counts, reverse=True)

    def test_a_sphere_overhangs_less_than_half_its_surface(self):
        mesh = surface_nets(sdf.sphere(1.0), resolution=48)
        _, _, fraction = overhangs(mesh, (0, 0, 1), max_angle=45.0)
        assert 0.1 < fraction < 0.5

    def test_empty_mesh_reports_nothing(self):
        mask, area, fraction = overhangs(Mesh.empty())
        assert len(mask) == 0
        assert area == 0.0
        assert fraction == 0.0

    def test_zero_build_direction_is_rejected(self):
        with pytest.raises(ValueError):
            overhangs(build_cube(2.0), (0, 0, 0))


class TestOrientedBoundingBox:
    def test_an_axis_aligned_box_keeps_its_extents(self):
        cube = build_cube(2.0).scaled([1.0, 2.0, 3.0])
        extents, _, _ = oriented_bounding_box(cube)
        assert sorted(extents) == pytest.approx([2.0, 4.0, 6.0])

    def test_a_rotated_box_recovers_its_true_size(self):
        """This is the whole point: the axis-aligned box of a diagonal part
        badly overstates how much build plate it needs."""
        rotated = build_cube(2.0).scaled([1.0, 3.0, 1.0]).rotated(45.0, "z")
        assert rotated.extents[0] > 3.0  # the naive box is inflated
        extents, _, _ = oriented_bounding_box(rotated)
        assert sorted(extents) == pytest.approx([2.0, 2.0, 6.0], rel=1e-6)

    def test_empty_mesh_is_handled(self):
        extents, axes, center = oriented_bounding_box(Mesh.empty())
        assert extents == pytest.approx([0.0, 0.0, 0.0])
        assert np.allclose(axes, np.eye(3))
        assert center == pytest.approx([0.0, 0.0, 0.0])


class TestTriangleQuality:
    def test_a_clean_cube_has_no_slivers(self):
        quality = triangle_quality(build_cube(2.0))
        assert quality["count"] == 12
        assert quality["slivers"] == 0
        assert quality["degenerate"] == 0
        assert quality["min_angle_deg"] == pytest.approx(45.0)

    def test_a_sliver_is_detected(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1e-6, 0]], dtype=float)
        quality = triangle_quality(Mesh(vertices, [[0, 1, 2]]))
        assert quality["slivers"] == 1
        assert quality["min_angle_deg"] < 1.0
        assert quality["max_aspect_ratio"] > 100

    def test_an_equilateral_triangle_has_aspect_ratio_one(self):
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0]]
        )
        quality = triangle_quality(Mesh(vertices, [[0, 1, 2]]))
        assert quality["max_aspect_ratio"] == pytest.approx(1.0, rel=1e-6)
        assert quality["min_angle_deg"] == pytest.approx(60.0)

    def test_empty_mesh_is_handled(self):
        assert triangle_quality(Mesh.empty())["count"] == 0


class TestMeshReport:
    def test_reports_geometry_and_topology_for_a_cube(self):
        report = mesh_report(build_cube(2.0))
        assert report["geometry"]["volume"] == pytest.approx(8.0)
        assert report["geometry"]["area"] == pytest.approx(24.0)
        assert report["geometry"]["triangles"] == 12
        assert report["topology"]["watertight"] is True
        assert report["topology"]["genus"] == 0
        assert report["topology"]["components"] == 1

    def test_open_surfaces_are_reported_as_such(self):
        report = mesh_report(build_grid_patch(3, 1.0))
        assert report["topology"]["watertight"] is False
        assert report["topology"]["boundary_edges"] == 12

    def test_unit_system_drives_the_formatted_text(self):
        report = mesh_report(build_cube(25.4), UnitSystem.inches())
        assert report["geometry"]["volume_text"] == "1 in^3"

    def test_rejects_non_meshes(self):
        with pytest.raises(TypeError):
            mesh_report("not a mesh")


class TestPrintability:
    def test_a_clean_cube_reports_no_problems(self):
        findings = printability(build_cube(20.0))
        assert not any(finding.is_problem for finding in findings)
        assert "Material estimate" in titles(findings)

    def test_an_open_surface_is_an_error(self):
        findings = printability(build_grid_patch(3, 20.0))
        assert "Model is not closed" in titles(findings)
        assert findings[0].severity == ERROR

    def test_findings_are_sorted_most_severe_first(self):
        findings = printability(build_grid_patch(3, 0.1))
        severities = [finding.severity for finding in findings]
        assert severities == sorted(severities, key=lambda s: {"error": 0, "warning": 1, "info": 2}[s])

    def test_a_part_larger_than_the_printer_is_an_error(self):
        findings = printability(build_cube(300.0), build_volume=(220, 220, 250))
        assert "Larger than the build volume" in titles(findings)

    def test_a_part_that_fits_is_not_flagged(self):
        findings = printability(build_cube(50.0), build_volume=(220, 220, 250))
        assert "Larger than the build volume" not in titles(findings)

    def test_a_very_thin_part_is_flagged(self):
        thin = build_cube(20.0).scaled([1.0, 1.0, 0.01])  # 0.2 mm tall
        findings = printability(thin, min_feature_size=0.8)
        assert "Very thin part" in titles(findings)
        assert any(f.severity == WARNING for f in findings if f.title == "Very thin part")

    def test_separate_bodies_are_reported(self):
        pair = Mesh.concatenate([build_cube(10.0), build_cube(10.0).translated([30, 0, 0])])
        findings = printability(pair)
        assert "Multiple bodies" in titles(findings)

    def test_inconsistent_winding_is_reported(self):
        cube = build_cube(20.0)
        faces = cube.faces.copy()
        faces[0] = faces[0][::-1]
        findings = printability(Mesh(cube.vertices, faces))
        assert "Inconsistent face winding" in titles(findings)

    def test_an_empty_model_is_a_single_error(self):
        findings = printability(Mesh.empty())
        assert len(findings) == 1
        assert findings[0].severity == ERROR

    def test_findings_serialise_for_the_ui(self):
        finding = printability(build_cube(20.0))[0]
        payload = finding.as_dict()
        assert set(payload) == {"severity", "title", "detail", "value"}
        assert isinstance(payload["title"], str)


class TestFieldAnalysis:
    def test_volume_fraction_of_a_solid_box_is_one(self):
        result = volume_fraction(
            sdf.box((2.0, 2.0, 2.0)), bounds=([-1, -1, -1], [1, 1, 1]), resolution=16
        )
        assert result["fraction"] == pytest.approx(1.0)

    def test_volume_fraction_of_a_sphere_in_its_box(self):
        result = volume_fraction(
            sdf.sphere(1.0), bounds=([-1, -1, -1], [1, 1, 1]), resolution=64
        )
        # A unit sphere fills pi/6 of its circumscribing cube.
        assert result["fraction"] == pytest.approx(np.pi / 6.0, rel=0.02)

    def test_a_lattice_reports_a_low_relative_density(self):
        """The headline number for a lattice: a light one is a small fraction."""
        lattice = sdf.tpms_sheet("gyroid", period=5.0, thickness=0.5)
        result = volume_fraction(lattice, bounds=([-5] * 3, [5] * 3), resolution=64)
        assert 0.02 < result["fraction"] < 0.35

    def test_a_thicker_wall_raises_the_density(self):
        box = ([-5] * 3, [5] * 3)
        thin = volume_fraction(
            sdf.tpms_sheet("gyroid", period=5.0, thickness=0.4), box, 48
        )["fraction"]
        thick = volume_fraction(
            sdf.tpms_sheet("gyroid", period=5.0, thickness=1.2), box, 48
        )["fraction"]
        assert thick > thin

    def test_wall_thickness_of_a_sphere_matches_its_diameter(self):
        result = wall_thickness(
            sdf.sphere(2.0), bounds=([-2.5] * 3, [2.5] * 3), resolution=64
        )
        # The deepest interior point is the centre, at distance r, so 2|f| = 2r.
        assert result["median"] > 0
        assert max(2.0 * 2.0 - result["median"], 0) >= 0

    def test_wall_thickness_of_a_shell_is_near_the_requested_wall(self):
        wall = 0.6
        result = wall_thickness(
            sdf.sphere(3.0).shell(wall), bounds=([-4] * 3, [4] * 3), resolution=96
        )
        # No interior point can be more than half a wall from the surface.
        assert result["median"] <= wall + 1e-6
        assert result["median"] > wall * 0.2

    def test_thickness_reports_its_own_resolution_limit(self):
        result = wall_thickness(
            sdf.sphere(1.0), bounds=([-1.5] * 3, [1.5] * 3), resolution=32
        )
        assert result["resolution_limit"] == pytest.approx(3.0 / 31.0)

    def test_a_field_with_no_solid_returns_zeros(self):
        far = sdf.sphere(0.5, center=(100, 0, 0))
        result = wall_thickness(far, bounds=([-1] * 3, [1] * 3), resolution=8)
        assert result["min"] == 0.0
