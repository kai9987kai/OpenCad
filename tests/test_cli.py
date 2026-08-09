"""Tests for the headless command line.

The CLI is the proof that the kernel really is display-free: if these pass in
CI, geometry can be built and checked on a machine with no graphics stack.
Exit codes matter as much as output - ``check`` is meant to gate a build.
"""

from __future__ import annotations

import json

import pytest

from src.cli import EXIT_OK, EXIT_PROBLEMS, EXIT_USAGE, main
from src.kernel import io_mesh, primitives
from tests.conftest import build_grid_patch


@pytest.fixture
def solid(tmp_path):
    path = tmp_path / "solid.stl"
    io_mesh.write_mesh(path, primitives.cube(20.0))
    return path


class TestInfo:
    def test_reports_the_measurements(self, solid, capsys):
        assert main(["info", str(solid)]) == EXIT_OK
        out = capsys.readouterr().out
        assert "triangles" in out
        assert "watertight     yes" in out
        assert "8000" in out  # 20 mm cube

    def test_json_output_is_machine_readable(self, solid, capsys):
        assert main(["info", str(solid), "--json"]) == EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        assert payload["geometry"]["volume"] == pytest.approx(8000.0)
        assert payload["topology"]["watertight"] is True

    def test_units_change_the_display(self, solid, capsys):
        main(["info", str(solid), "--units", "in", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["geometry"]["volume_text"].endswith("in^3")

    def test_a_missing_file_exits_cleanly(self, tmp_path):
        with pytest.raises(SystemExit) as info:
            main(["info", str(tmp_path / "nope.stl")])
        assert "cannot read" in str(info.value)


class TestCheck:
    def test_a_clean_solid_exits_zero(self, solid):
        assert main(["check", str(solid)]) == EXIT_OK

    def test_an_open_surface_exits_non_zero(self, tmp_path):
        path = tmp_path / "open.stl"
        io_mesh.write_mesh(path, build_grid_patch(3, 20.0))
        assert main(["check", str(path)]) == EXIT_PROBLEMS

    def test_oversize_part_is_flagged(self, tmp_path, capsys):
        path = tmp_path / "big.stl"
        io_mesh.write_mesh(path, primitives.cube(400.0))
        assert main(["check", str(path), "--build", "220x220x250"]) == EXIT_PROBLEMS
        assert "build volume" in capsys.readouterr().out

    def test_build_volume_must_have_three_dimensions(self, solid):
        with pytest.raises(SystemExit, match="THREE"):
            main(["check", str(solid), "--build", "220x220"])

    def test_json_findings(self, solid, capsys):
        main(["check", str(solid), "--json"])
        findings = json.loads(capsys.readouterr().out)
        assert isinstance(findings, list)
        assert {"severity", "title", "detail", "value"} <= set(findings[0])


class TestConvert:
    @pytest.mark.parametrize("suffix", [".stl", ".obj", ".ply", ".off", ".3mf"])
    def test_round_trips_through_every_format(self, solid, tmp_path, suffix):
        target = tmp_path / f"out{suffix}"
        assert main(["convert", str(solid), str(target)]) == EXIT_OK
        assert io_mesh.read_mesh(target).volume == pytest.approx(8000.0)

    def test_clean_and_largest_are_applied(self, tmp_path):
        from src.kernel.mesh import Mesh

        pair = Mesh.concatenate(
            [primitives.cube(20.0), primitives.cube(4.0).translated([60, 0, 0])]
        )
        source = tmp_path / "pair.stl"
        io_mesh.write_mesh(source, pair)

        target = tmp_path / "one.stl"
        assert main(["convert", str(source), str(target), "--clean", "--largest"]) == EXIT_OK
        assert io_mesh.read_mesh(target).volume == pytest.approx(8000.0)

    def test_an_unknown_extension_is_refused(self, solid, tmp_path):
        with pytest.raises(SystemExit, match="cannot write"):
            main(["convert", str(solid), str(tmp_path / "out.xyz")])


class TestPrimitive:
    def test_generates_a_solid(self, tmp_path):
        target = tmp_path / "ball.stl"
        assert main(["primitive", "icosphere", "--radius", "10", "-o", str(target)]) == EXIT_OK
        mesh = io_mesh.read_mesh(target)
        assert mesh.is_watertight
        assert mesh.volume == pytest.approx(4 / 3 * 3.14159265 * 1000, rel=0.05)

    def test_options_are_passed_through(self, tmp_path):
        target = tmp_path / "cyl.stl"
        main(["primitive", "cylinder", "--radius", "5", "--height", "20", "-o", str(target)])
        mesh = io_mesh.read_mesh(target)
        assert mesh.extents[2] == pytest.approx(20.0, rel=1e-6)

    def test_irrelevant_options_are_ignored_not_fatal(self, tmp_path):
        """A cube has no --radius; passing one should not crash the tool."""
        target = tmp_path / "cube.stl"
        assert main(["primitive", "cube", "--size", "10", "--radius", "3", "-o", str(target)]) == EXIT_OK

    def test_an_unknown_primitive_lists_the_options(self, tmp_path):
        with pytest.raises(SystemExit, match="Supported"):
            main(["primitive", "dodecahedron", "-o", str(tmp_path / "x.stl")])


class TestLattice:
    def test_generates_a_closed_lattice(self, tmp_path, capsys):
        target = tmp_path / "lat.stl"
        code = main(
            [
                "lattice", "--size", "20", "--period", "6", "--thickness", "1.0",
                "--resolution", "64", "--quiet", "-o", str(target),
            ]
        )
        assert code == EXIT_OK
        assert "Relative density" in capsys.readouterr().out
        mesh = io_mesh.read_mesh(target)
        assert mesh.n_faces > 1000
        assert mesh.volume > 0

    def test_warns_when_the_grid_cannot_resolve_the_wall(self, tmp_path, capsys):
        main(
            [
                "lattice", "--size", "30", "--thickness", "0.8",
                "--resolution", "40", "--quiet", "-o", str(tmp_path / "coarse.stl"),
            ]
        )
        assert "Raise --resolution" in capsys.readouterr().err

    def test_can_fill_the_bounds_of_an_existing_mesh(self, solid, tmp_path):
        target = tmp_path / "filled.stl"
        code = main(
            [
                "lattice", "--fill", str(solid), "--period", "6", "--thickness", "1.0",
                "--resolution", "56", "--quiet", "-o", str(target),
            ]
        )
        assert code == EXIT_OK
        mesh = io_mesh.read_mesh(target)
        # The lattice must stay within the host part's bounding box.
        assert mesh.extents[0] <= 20.0 + 1e-6

    def test_grading_flags_reach_the_geometry(self, tmp_path):
        """The CLI's job is to pass the grading through; that it produces a
        genuinely different solid is what proves the flags are wired up.

        Which end ends up denser is asserted against the *field* in
        ``tests/kernel/test_lattice.py``, where occupancy can be measured
        directly. It cannot be read off the mesh centroid here: a lattice is
        not watertight, so ``Mesh.centroid`` returns the area centroid, and
        surface area barely moves when a wall thickens.
        """
        common = [
            "lattice", "--size", "24", "--period", "6", "--thickness", "0.9",
            "--resolution", "72", "--quiet", "-o",
        ]
        plain = tmp_path / "plain.stl"
        graded = tmp_path / "graded.stl"

        assert main([*common, str(plain)]) == EXIT_OK
        assert main(
            [*common, str(graded), "--grade", "thickness",
             "--grade-axis", "z", "--grade-amount", "1.0"]
        ) == EXIT_OK

        plain_mesh = io_mesh.read_mesh(plain)
        graded_mesh = io_mesh.read_mesh(graded)
        assert graded_mesh.n_faces != plain_mesh.n_faces
        assert graded_mesh.volume != pytest.approx(plain_mesh.volume, rel=1e-3)

    def test_every_family_is_selectable(self, tmp_path):
        from src.kernel.sdf import TPMS_KINDS

        for kind in TPMS_KINDS:
            target = tmp_path / f"{kind}.stl"
            code = main(
                [
                    "lattice", "--kind", kind, "--size", "12", "--period", "6",
                    "--thickness", "1.0", "--resolution", "40", "--quiet", "-o", str(target),
                ]
            )
            assert code == EXIT_OK, kind


class TestUsage:
    def test_no_command_prints_help(self, capsys):
        assert main([]) == EXIT_USAGE
        assert "COMMAND" in capsys.readouterr().out

    def test_version(self, capsys):
        with pytest.raises(SystemExit) as info:
            main(["--version"])
        assert info.value.code == 0
        assert "OpenCad" in capsys.readouterr().out
