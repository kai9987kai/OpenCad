"""Tests for the kernel/PyVista bridge.

Only the parts that do not need VTK are exercised here - matrix composition,
decomposition, and colour normalisation - which is deliberate: those are the
pieces that silently corrupt a saved project when they are wrong, and they are
the pieces that can be checked in headless CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.bridge import (
    compose_matrix,
    decompose_matrix,
    normalize_color,
    world_mesh_from_parts,
)
from tests.conftest import build_cube


class TestComposeMatrix:
    def test_identity_by_default(self):
        assert np.allclose(compose_matrix(), np.eye(4))

    def test_translation_lands_in_the_last_column(self):
        matrix = compose_matrix(position=(1.0, 2.0, 3.0))
        assert matrix[:3, 3] == pytest.approx([1.0, 2.0, 3.0])

    def test_scale_then_rotate_then_translate(self):
        matrix = compose_matrix(
            position=(1.0, 2.0, 3.0), orientation=(0.0, 0.0, 90.0), scale=(2.0, 2.0, 2.0)
        )
        point = matrix @ np.array([1.0, 0.0, 0.0, 1.0])
        # Scale x2 -> (2,0,0); rotate 90 about Z -> (0,2,0); translate -> (1,4,3).
        assert point[:3] == pytest.approx([1.0, 4.0, 3.0])

    def test_orientation_uses_vtk_z_x_y_order(self):
        # Applying X and Z together must match Rz @ Rx, not Rx @ Rz.
        matrix = compose_matrix(orientation=(90.0, 0.0, 90.0))
        point = matrix @ np.array([0.0, 1.0, 0.0, 1.0])
        # Rx(90): (0,1,0) -> (0,0,1).  Rz(90) leaves (0,0,1) alone.
        assert point[:3] == pytest.approx([0.0, 0.0, 1.0], abs=1e-12)

    def test_user_matrix_pre_multiplies(self):
        user = np.eye(4)
        user[:3, 3] = [10.0, 0.0, 0.0]
        matrix = compose_matrix(position=(1.0, 0.0, 0.0), user_matrix=user)
        assert matrix[:3, 3] == pytest.approx([11.0, 0.0, 0.0])

    def test_rotation_block_stays_orthonormal_without_scale(self):
        matrix = compose_matrix(orientation=(17.0, -34.0, 61.0))
        block = matrix[:3, :3]
        assert np.allclose(block @ block.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(block) == pytest.approx(1.0)


class TestDecomposeMatrix:
    @pytest.mark.parametrize(
        "orientation",
        [(0.0, 0.0, 0.0), (30.0, 0.0, 0.0), (0.0, 45.0, 0.0), (0.0, 0.0, 60.0), (12.0, -25.0, 80.0)],
    )
    def test_round_trip_through_compose(self, orientation):
        position = (1.5, -2.5, 3.5)
        scale = (2.0, 3.0, 0.5)
        matrix = compose_matrix(position, orientation, scale)

        got_position, got_orientation, got_scale = decompose_matrix(matrix)
        assert got_position == pytest.approx(position)
        assert got_scale == pytest.approx(scale)
        # Compare the rebuilt matrix rather than the angles: Euler triples are
        # not unique, but the transform they describe is.
        assert np.allclose(compose_matrix(got_position, got_orientation, got_scale), matrix)

    def test_gimbal_lock_still_produces_a_valid_transform(self):
        matrix = compose_matrix(orientation=(90.0, 40.0, 25.0))
        position, orientation, scale = decompose_matrix(matrix)
        assert np.allclose(compose_matrix(position, orientation, scale), matrix, atol=1e-9)

    def test_mirror_is_folded_into_the_scale(self):
        matrix = np.diag([-1.0, 1.0, 1.0, 1.0])
        _, _, scale = decompose_matrix(matrix)
        assert float(np.prod(scale)) < 0

    def test_identity_decomposes_to_nothing(self):
        position, orientation, scale = decompose_matrix(np.eye(4))
        assert position == pytest.approx([0.0, 0.0, 0.0])
        assert orientation == pytest.approx([0.0, 0.0, 0.0])
        assert scale == pytest.approx([1.0, 1.0, 1.0])


class TestWorldMesh:
    def test_transform_is_baked_into_the_vertices(self):
        cube = build_cube(2.0)
        world = world_mesh_from_parts(cube, position=(5.0, 0.0, 0.0))
        assert world.center == pytest.approx([5.0, 0.0, 0.0])
        assert world.volume == pytest.approx(8.0)

    def test_non_uniform_scale_changes_the_volume(self):
        cube = build_cube(2.0)
        world = world_mesh_from_parts(cube, scale=(2.0, 3.0, 4.0))
        assert world.volume == pytest.approx(8.0 * 24.0)

    def test_rotation_preserves_volume(self):
        cube = build_cube(2.0)
        world = world_mesh_from_parts(cube, orientation=(15.0, 25.0, 35.0))
        assert world.volume == pytest.approx(8.0)
        assert world.is_watertight


class TestColor:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("#ff0000", "#ff0000"),
            ("#FF0000", "#ff0000"),
            ("#abc", "#aabbcc"),
            ("cyan", "#00ffff"),
            ("Grey", "#808080"),
            ((1.0, 0.0, 0.0), "#ff0000"),
            ([0.0, 1.0, 0.0], "#00ff00"),
            ((255, 255, 0), "#ffff00"),
            (np.array([0.5, 0.5, 0.5]), "#808080"),
        ],
    )
    def test_known_conversions(self, value, expected):
        assert normalize_color(value) == expected

    def test_unknown_input_falls_back(self):
        assert normalize_color(None) == "#89b4fa"
        assert normalize_color("not a colour") == "#89b4fa"
        assert normalize_color(object(), default="#123456") == "#123456"

    def test_out_of_range_floats_are_clamped(self):
        assert normalize_color((1.4, -0.2, 0.5)) == "#ff0080"

    def test_vtk_float_round_trip_is_stable(self):
        """A colour read from VTK and saved must reload as the same colour."""
        original = "#89b4fa"
        as_floats = [int(original[i : i + 2], 16) / 255.0 for i in (1, 3, 5)]
        assert normalize_color(as_floats) == original
