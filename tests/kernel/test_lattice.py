"""Tests for building a finite lattice solid from a specification.

The defect these exist to prevent is subtle and expensive: a lattice that looks
right on screen but is not actually a closed solid, or one whose mesh quietly
contains less material than was asked for because the sampling grid could not
resolve the wall. Both produce a part that prints wrong rather than failing.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.kernel.analysis import volume_fraction
from src.kernel.lattice import (
    DEFAULT_SPEC,
    MIN_CELLS_PER_WALL,
    build_lattice_field,
    cells_per_wall,
)
from src.kernel.meshing import surface_nets
from src.kernel.sdf import TPMS_KINDS


@pytest.fixture
def spec():
    return dict(DEFAULT_SPEC, size=20.0, resolution=64)


class TestFieldConstruction:
    @pytest.mark.parametrize("kind", list(TPMS_KINDS))
    @pytest.mark.parametrize("mode", ["sheet", "solid"])
    def test_every_family_and_mode_yields_a_finite_field(self, spec, kind, mode):
        field, sampling, region = build_lattice_field(dict(spec, kind=kind, mode=mode))
        values = field(np.array([[0.1, 0.2, 0.3], [5.0, -2.0, 1.0]]))
        assert np.all(np.isfinite(values))
        assert np.all(sampling[0] <= region[0])
        assert np.all(sampling[1] >= region[1])

    def test_defaults_fill_in_for_a_partial_spec(self):
        field, _, region = build_lattice_field({"kind": "gyroid"})
        assert np.isfinite(field(np.array([[0.0, 0.0, 0.0]]))).all()
        assert region[1][0] - region[0][0] == pytest.approx(DEFAULT_SPEC["size"])

    def test_an_explicit_region_is_honoured(self, spec):
        bounds = (np.array([-5.0, -2.0, 0.0]), np.array([5.0, 2.0, 8.0]))
        _, _, region = build_lattice_field(spec, bounds)
        assert region[0] == pytest.approx(bounds[0])
        assert region[1] == pytest.approx(bounds[1])

    def test_nothing_survives_outside_the_region(self, spec):
        field, _, _ = build_lattice_field(spec)
        outside = np.array([[100.0, 0.0, 0.0], [0.0, -50.0, 0.0]])
        assert np.all(field(outside) > 0)

    def test_a_degenerate_region_is_rejected(self, spec):
        with pytest.raises(ValueError):
            build_lattice_field(spec, (np.zeros(3), np.array([1.0, 0.0, 1.0])))


class TestClosedSolid:
    def test_the_meshed_lattice_has_no_holes(self, spec):
        """An uncapped TPMS field meshes to an open surface. It must be capped."""
        field, sampling, _ = build_lattice_field(spec)
        mesh = surface_nets(field, bounds=sampling, resolution=spec["resolution"])
        assert not mesh.is_empty
        assert len(mesh.boundary_edges()) == 0
        assert mesh.volume > 0

    def test_the_lattice_stays_inside_its_region(self, spec):
        field, sampling, region = build_lattice_field(spec)
        mesh = surface_nets(field, bounds=sampling, resolution=spec["resolution"])
        low, high = mesh.bounding_box
        assert np.all(low >= region[0] - 1e-6)
        assert np.all(high <= region[1] + 1e-6)

    def test_a_network_lattice_is_also_closed(self, spec):
        field, sampling, _ = build_lattice_field(dict(spec, mode="solid", level=0.0))
        mesh = surface_nets(field, bounds=sampling, resolution=spec["resolution"])
        assert len(mesh.boundary_edges()) == 0
        assert mesh.volume > 0


class TestDensity:
    def test_relative_density_matches_the_analytic_estimate(self, spec):
        """A gyroid's area is about 3.091 L^2 per L^3 cell, so a sheet of
        thickness t fills roughly t * 3.091 / L of the volume."""
        field, _, region = build_lattice_field(spec)
        expected = spec["thickness"] * 3.091 / spec["period"]
        measured = volume_fraction(field, bounds=region, resolution=96)["fraction"]
        assert measured == pytest.approx(expected, rel=0.12)

    def test_a_thicker_wall_gives_a_denser_lattice(self, spec):
        def density(thickness):
            field, _, region = build_lattice_field(dict(spec, thickness=thickness))
            return volume_fraction(field, bounds=region, resolution=64)["fraction"]

        assert density(1.4) > density(0.9) > density(0.5)

    def test_the_meshed_volume_agrees_with_the_sampled_density(self, spec):
        """Mesh and voxel measurements must tell the same story; when they
        disagree the grid is too coarse to resolve the wall."""
        fine = dict(spec, resolution=112)
        field, sampling, region = build_lattice_field(fine)
        mesh = surface_nets(field, bounds=sampling, resolution=fine["resolution"])
        region_volume = float(np.prod(region[1] - region[0]))
        sampled = volume_fraction(field, bounds=region, resolution=96)["fraction"]
        assert mesh.volume / region_volume == pytest.approx(sampled, rel=0.08)


class TestGrading:
    @pytest.mark.parametrize("axis", ["x", "y", "z", "radial"])
    def test_graded_thickness_varies_along_the_axis(self, spec, axis):
        field, _, _ = build_lattice_field(
            dict(spec, grade_target="thickness", grade_axis=axis, grade_amount=0.6)
        )
        uniform, _, _ = build_lattice_field(dict(spec, grade_target="none"))

        # Sample many points and compare how much of each is solid: a graded
        # lattice must not be identical to a uniform one.
        generator = np.random.default_rng(11)
        points = generator.uniform(-9, 9, size=(4000, 3))
        assert not np.allclose(field(points), uniform(points))

    def test_grading_thickness_makes_one_end_denser(self, spec):
        field, _, _ = build_lattice_field(
            dict(spec, grade_target="thickness", grade_axis="z", grade_amount=1.0)
        )
        generator = np.random.default_rng(5)
        low_half = generator.uniform([-9, -9, -9], [9, 9, 0], size=(6000, 3))
        high_half = generator.uniform([-9, -9, 0], [9, 9, 9], size=(6000, 3))
        assert field.contains(high_half).mean() > field.contains(low_half).mean()

    def test_grading_the_cell_size_changes_the_field(self, spec):
        graded, _, _ = build_lattice_field(
            dict(spec, grade_target="period", grade_axis="z", grade_amount=3.0)
        )
        uniform, _, _ = build_lattice_field(spec)
        generator = np.random.default_rng(7)
        points = generator.uniform(-9, 9, size=(2000, 3))
        assert not np.allclose(graded(points), uniform(points))

    def test_an_extreme_gradient_never_produces_a_negative_wall(self, spec):
        """Clamping matters: a wall driven through zero inverts the sheet."""
        field, _, _ = build_lattice_field(
            dict(spec, thickness=0.4, grade_target="thickness", grade_axis="z", grade_amount=5.0)
        )
        generator = np.random.default_rng(13)
        points = generator.uniform(-9, 9, size=(3000, 3))
        assert np.all(np.isfinite(field(points)))
        # Some material must survive at the thin end rather than the field
        # turning inside out and filling everything.
        thin_end = generator.uniform([-9, -9, -9.5], [9, 9, -8], size=(3000, 3))
        occupancy = field.contains(thin_end).mean()
        assert 0.0 <= occupancy < 0.9

    def test_zero_amount_is_the_same_as_no_grading(self, spec):
        graded, _, _ = build_lattice_field(
            dict(spec, grade_target="thickness", grade_amount=0.0)
        )
        uniform, _, _ = build_lattice_field(dict(spec, grade_target="none"))
        points = np.random.default_rng(3).uniform(-9, 9, size=(500, 3))
        assert graded(points) == pytest.approx(uniform(points))


class TestResolutionGuidance:
    def test_cells_per_wall_scales_with_resolution(self, spec):
        coarse = cells_per_wall(spec, 48)
        fine = cells_per_wall(spec, 192)
        assert fine > coarse
        assert fine == pytest.approx(coarse * 191 / 47, rel=1e-6)

    def test_a_known_under_resolved_case_is_flagged(self):
        """A 0.8 mm wall in a 30 mm cube at resolution 48 measures ~11% light;
        the guidance must call that out."""
        spec = dict(DEFAULT_SPEC, size=30.0, thickness=0.8)
        assert cells_per_wall(spec, 48) < MIN_CELLS_PER_WALL
        assert cells_per_wall(spec, 128) > MIN_CELLS_PER_WALL

    def test_under_resolving_really_does_lose_material(self):
        """The warning is not cosmetic - verify the loss it warns about."""
        spec = dict(DEFAULT_SPEC, size=30.0, thickness=0.8)
        field, sampling, region = build_lattice_field(spec)
        region_volume = float(np.prod(region[1] - region[0]))

        coarse = surface_nets(field, bounds=sampling, resolution=48).volume / region_volume
        fine = surface_nets(field, bounds=sampling, resolution=128).volume / region_volume
        assert coarse < fine
        assert fine - coarse > 0.02

    def test_degenerate_inputs_return_zero(self, spec):
        assert cells_per_wall(spec, 1) == 0.0
        assert cells_per_wall(spec, 64, span=0.0) == 0.0
