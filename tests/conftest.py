"""Shared pytest fixtures for the OpenCad kernel test suite.

The kernel is numpy-only by design, so these fixtures build exact analytic
solids whose area, volume, and inertia are known in closed form.  Tests assert
against those values rather than golden files, which keeps the suite honest
when algorithms are rewritten.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kernel.mesh import Mesh


def build_cube(size=2.0, center=(0.0, 0.0, 0.0)):
    """An axis-aligned cube with outward-facing (CCW) winding.

    Volume is ``size ** 3`` and surface area ``6 * size ** 2``.
    """
    half = float(size) / 2.0
    center = np.asarray(center, dtype=float)
    vertices = (
        np.array(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ],
            dtype=float,
        )
        * half
        + center
    )
    faces = np.array(
        [
            [0, 3, 2], [0, 2, 1],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [2, 3, 7], [2, 7, 6],
            [1, 2, 6], [1, 6, 5],
            [0, 4, 7], [0, 7, 3],
        ],
        dtype=np.int64,
    )
    return Mesh(vertices, faces)


def build_tetrahedron(scale=1.0):
    """A regular-ish tetrahedron with outward winding and volume ``scale**3 / 6``."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    ) * float(scale)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
    return Mesh(vertices, faces)


def build_grid_patch(divisions=3, size=1.0):
    """An open, planar triangle patch in Z=0 - useful for boundary-edge tests."""
    steps = int(divisions) + 1
    coords = np.linspace(0.0, float(size), steps)
    xs, ys = np.meshgrid(coords, coords, indexing="xy")
    vertices = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])
    faces = []
    for row in range(divisions):
        for col in range(divisions):
            a = row * steps + col
            faces.append([a, a + 1, a + steps + 1])
            faces.append([a, a + steps + 1, a + steps])
    return Mesh(vertices, faces)


@pytest.fixture
def cube():
    return build_cube(2.0)


@pytest.fixture
def unit_cube():
    return build_cube(1.0)


@pytest.fixture
def tetrahedron():
    return build_tetrahedron(1.0)


@pytest.fixture
def grid_patch():
    return build_grid_patch(3, 1.0)


@pytest.fixture
def rng():
    return np.random.default_rng(20260809)
