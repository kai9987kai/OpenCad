"""Tests for pure-Python mesh file I/O.

The assertions here are anchored to closed-form geometry - a cube of side 2 has
volume 8 and area 24, a fan-triangulated unit square has area 1 - so a format
bug shows up as a wrong number rather than as a silently different file.  Every
format is exercised both by round-tripping through the public API and by parsing
hand-written files whose exact bytes are known.
"""

from __future__ import annotations

import io
import struct
import zipfile

import numpy as np
import pytest

from src.kernel.io_mesh import (
    SUPPORTED_READ_FORMATS,
    SUPPORTED_WRITE_FORMATS,
    MeshIOError,
    detect_format,
    file_filter,
    read_3mf,
    read_mesh,
    read_obj,
    read_off,
    read_ply,
    read_stl,
    write_3mf,
    write_mesh,
    write_obj,
    write_off,
    write_ply,
    write_stl,
)
from src.kernel.mesh import Mesh
from tests.conftest import build_cube, build_grid_patch, build_tetrahedron

FORMATS = ("stl", "obj", "ply", "off", "3mf")

#: Binary STL stores coordinates as float32, so it is the one lossy path.
LOSSY_FORMATS = ("stl",)


def build_sample():
    """Three disjoint closed solids with awkward coordinates.

    Volume is ``2**3 + 1**3 + 3**3 / 6 = 8 + 1 + 4.5 = 13.5`` and every corner
    coordinate is a dyadic rational, so float32 stores them exactly and an STL
    round-trip is limited only by the format's precision, not by rounding noise.
    """
    return Mesh.concatenate(
        [
            build_cube(2.0),
            build_cube(1.0, center=(5.5, -0.25, 1.75)),
            build_tetrahedron(3.0).translated([-6.125, 0.5, -0.25]),
        ]
    )


def tolerance_for(fmt):
    return 1e-6 if fmt in LOSSY_FORMATS else 1e-12


def zip_package(path, members):
    """Write a zip archive from ``{name: bytes}`` verbatim, including bad names."""
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return path


def model_xml(vertices, triangles, item_attributes="", unit="millimeter"):
    """Build a 3MF model part by hand so tests control the exact XML."""
    body = ["<vertices>"]
    body += [f'<vertex x="{x}" y="{y}" z="{z}"/>' for x, y, z in vertices]
    body.append("</vertices>")
    body.append("<triangles>")
    body += [f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in triangles]
    body.append("</triangles>")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<model unit="{unit}" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
        '<resources><object id="1" type="model"><mesh>'
        + "".join(body)
        + "</mesh></object></resources>"
        f'<build><item objectid="1" {item_attributes}/></build>'
        "</model>"
    ).encode("utf-8")


def minimal_3mf(path, model_part):
    return zip_package(
        path,
        {
            "[Content_Types].xml": (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>'
            ).encode(),
            "_rels/.rels": (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>'
            ).encode(),
            "3D/3dmodel.model": model_part,
        },
    )


CUBE_VERTICES = [
    (-1, -1, -1),
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, 1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, 1, 1),
]
CUBE_TRIANGLES = [
    (0, 3, 2), (0, 2, 1),
    (4, 5, 6), (4, 6, 7),
    (0, 1, 5), (0, 5, 4),
    (2, 3, 7), (2, 7, 6),
    (1, 2, 6), (1, 6, 5),
    (0, 4, 7), (0, 7, 3),
]


class TestRoundTrip:
    @pytest.mark.parametrize("fmt", FORMATS)
    def test_cube_survives_every_format(self, tmp_path, fmt):
        original = build_cube(2.0)
        path = tmp_path / f"cube.{fmt}"
        write_mesh(path, original)
        restored = read_mesh(path)
        rel = tolerance_for(fmt)

        assert restored.n_vertices == 8
        assert restored.n_faces == 12
        assert restored.volume == pytest.approx(8.0, rel=rel)
        assert restored.area == pytest.approx(24.0, rel=rel)
        assert restored.bounds == pytest.approx(original.bounds, rel=rel, abs=1e-9)
        assert restored.is_watertight
        assert restored.is_oriented

    @pytest.mark.parametrize("fmt", FORMATS)
    def test_multi_solid_sample_survives_every_format(self, tmp_path, fmt):
        original = build_sample()
        # Ground truth first: 8 + 1 + 27/6.
        assert original.volume == pytest.approx(13.5)
        assert original.n_vertices == 20
        assert original.n_faces == 28

        path = tmp_path / f"sample.{fmt}"
        write_mesh(path, original)
        restored = read_mesh(path)
        rel = tolerance_for(fmt)

        assert restored.n_vertices == 20
        assert restored.n_faces == 28
        assert restored.volume == pytest.approx(13.5, rel=rel)
        assert restored.area == pytest.approx(original.area, rel=rel)
        assert restored.bounds == pytest.approx(original.bounds, rel=rel, abs=1e-9)
        assert restored.n_components() == 3

    @pytest.mark.parametrize("fmt", ("obj", "ply", "off", "3mf"))
    def test_lossless_formats_reproduce_coordinates_exactly(self, tmp_path, fmt):
        # repr() is the shortest decimal string that round-trips a float64, so
        # the ascii formats must return bit-identical coordinates.
        original = build_cube(2.0).rotated(37.0, [1.0, 2.0, -0.5]).translated([0.1, 0.2, 0.3])
        path = tmp_path / f"exact.{fmt}"
        write_mesh(path, original)
        restored = read_mesh(path)
        assert np.array_equal(restored.vertices, original.vertices)
        assert np.array_equal(restored.faces, original.faces)

    @pytest.mark.parametrize("fmt", FORMATS)
    def test_open_surface_round_trips(self, tmp_path, fmt):
        original = build_grid_patch(3, 1.0)
        path = tmp_path / f"patch.{fmt}"
        write_mesh(path, original)
        restored = read_mesh(path)
        assert restored.n_faces == original.n_faces
        assert restored.area == pytest.approx(1.0, rel=tolerance_for(fmt))
        assert len(restored.boundary_edges()) == 12

    @pytest.mark.parametrize("fmt", FORMATS)
    def test_extreme_aspect_ratio_survives(self, tmp_path, fmt):
        # A 200 x 2 x 0.02 slab: a 10000:1 aspect ratio, volume exactly 8.
        original = build_cube(2.0).scaled([100.0, 1.0, 0.01])
        path = tmp_path / f"slab.{fmt}"
        write_mesh(path, original)
        restored = read_mesh(path)
        assert restored.n_vertices == 8
        assert restored.volume == pytest.approx(8.0, rel=tolerance_for(fmt))
        assert restored.extents == pytest.approx([200.0, 2.0, 0.02], rel=1e-6)

    @pytest.mark.parametrize("fmt", FORMATS)
    def test_single_triangle_survives(self, tmp_path, fmt):
        original = Mesh([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]], [[0, 1, 2]])
        path = tmp_path / f"one.{fmt}"
        write_mesh(path, original)
        restored = read_mesh(path)
        assert restored.n_vertices == 3
        assert restored.n_faces == 1
        assert restored.area == pytest.approx(6.0, rel=tolerance_for(fmt))

    @pytest.mark.parametrize("fmt", FORMATS)
    def test_empty_mesh_round_trips(self, tmp_path, fmt):
        path = tmp_path / f"empty.{fmt}"
        write_mesh(path, Mesh.empty())
        assert read_mesh(path).is_empty


class TestStl:
    def test_binary_and_ascii_agree(self, tmp_path):
        original = build_sample()
        binary_path = tmp_path / "binary.stl"
        ascii_path = tmp_path / "ascii.stl"
        write_stl(binary_path, original, binary=True)
        write_stl(ascii_path, original, binary=False)

        binary = read_stl(binary_path)
        text = read_stl(ascii_path)
        assert binary.n_vertices == text.n_vertices == 20
        assert binary.n_faces == text.n_faces == 28
        assert binary.volume == pytest.approx(text.volume, rel=1e-6)
        assert np.allclose(binary.vertices, text.vertices, atol=1e-5)
        # Ascii output keeps full float64 precision, so it is exact.
        assert text.volume == pytest.approx(13.5)

    def test_ascii_file_starting_with_solid_is_not_misdetected(self, tmp_path):
        path = tmp_path / "ascii.stl"
        write_stl(path, build_sample(), binary=False, name="solid_looking_name")
        raw = path.read_bytes()
        assert raw[:5] == b"solid"
        assert len(raw) > 84  # long enough that the naive size test could fire
        mesh = read_stl(path)
        assert mesh.n_faces == 28
        assert mesh.volume == pytest.approx(13.5)

    def test_binary_file_whose_header_starts_with_solid_is_detected(self, tmp_path):
        path = tmp_path / "trap.stl"
        write_stl(path, build_sample(), binary=True, name="solid trap header")
        raw = path.read_bytes()
        assert raw[:5] == b"solid"
        # The size arithmetic is what identifies it: 84 + 50 * n.
        assert len(raw) == 84 + 50 * 28
        mesh = read_stl(path)
        assert mesh.n_faces == 28
        assert mesh.n_vertices == 20
        assert mesh.volume == pytest.approx(13.5, rel=1e-6)

    def test_binary_writes_correct_normals_and_zero_attributes(self, tmp_path):
        original = build_cube(2.0)
        path = tmp_path / "cube.stl"
        write_stl(path, original)
        raw = path.read_bytes()

        assert struct.unpack_from("<I", raw, 80)[0] == 12
        dtype = np.dtype(
            [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attributes", "<u2")]
        )
        records = np.frombuffer(raw, dtype=dtype, count=12, offset=84)
        assert np.all(records["attributes"] == 0)
        assert np.allclose(records["normal"], original.face_normals(), atol=1e-6)
        # Cube facet normals are axis aligned, so each is a signed unit basis vector.
        assert np.allclose(np.abs(records["normal"]).sum(axis=1), 1.0, atol=1e-6)

    def test_reading_welds_the_triangle_soup(self, tmp_path):
        path = tmp_path / "cube.stl"
        write_stl(path, build_cube(2.0))
        assert read_stl(path, weld=False).n_vertices == 36
        welded = read_stl(path)
        assert welded.n_vertices == 8
        assert welded.is_watertight

    def test_duplicated_input_vertices_collapse(self, tmp_path):
        # Two copies of the same cube stacked into one vertex-soup file.
        cube = build_cube(2.0)
        doubled = Mesh(
            np.vstack([cube.vertices, cube.vertices]),
            np.vstack([cube.faces, cube.faces + cube.n_vertices]),
        )
        path = tmp_path / "doubled.stl"
        write_stl(path, doubled)
        restored = read_stl(path)
        assert restored.n_vertices == 8
        assert restored.n_faces == 24  # duplicate facets are kept, vertices are not

    def test_handwritten_ascii_file(self, tmp_path):
        path = tmp_path / "tri.stl"
        path.write_text(
            "solid demo\n"
            "  facet normal 0 0 1\n"
            "    outer loop\n"
            "      vertex 0.0 0.0 0.0\n"
            "      vertex 3.0 0.0 0.0\n"
            "      vertex 0.0 4.0 0.0\n"
            "    endloop\n"
            "  endfacet\n"
            "endsolid demo\n"
        )
        mesh = read_stl(path)
        assert mesh.n_faces == 1
        assert mesh.area == pytest.approx(6.0)

    def test_truncated_binary_raises_mesh_io_error(self, tmp_path):
        path = tmp_path / "cut.stl"
        write_stl(path, build_sample())
        raw = path.read_bytes()
        path.write_bytes(raw[:-25])
        with pytest.raises(MeshIOError, match="Truncated binary STL"):
            read_stl(path)

    def test_ascii_with_incomplete_triangle_raises(self, tmp_path):
        path = tmp_path / "bad.stl"
        path.write_text(
            "solid demo\n"
            "facet normal 0 0 1\n"
            "outer loop\n"
            "vertex 0 0 0\n"
            "vertex 1 0 0\n"
            "endloop\n"
            "endfacet\n"
            "endsolid demo\n"
        )
        with pytest.raises(MeshIOError, match="whole number"):
            read_stl(path)

    def test_ascii_with_non_numeric_vertex_raises(self, tmp_path):
        path = tmp_path / "bad.stl"
        path.write_text("solid d\nvertex a b c\nendsolid d\n")
        with pytest.raises(MeshIOError, match="coordinates"):
            read_stl(path)

    def test_empty_file_raises(self, tmp_path):
        path = tmp_path / "nothing.stl"
        path.write_bytes(b"")
        with pytest.raises(MeshIOError, match="Empty STL"):
            read_stl(path)


class TestObj:
    def test_negative_indices_are_relative(self, tmp_path):
        path = tmp_path / "negative.obj"
        path.write_text(
            "v 0 0 0\n"
            "v 3 0 0\n"
            "v 0 4 0\n"
            "f -3 -2 -1\n"
        )
        mesh = read_obj(path)
        assert mesh.n_faces == 1
        assert np.array_equal(mesh.faces, [[0, 1, 2]])
        assert mesh.area == pytest.approx(6.0)

    def test_negative_indices_track_the_running_vertex_count(self, tmp_path):
        # The second face's -1 must mean vertex 6, not vertex 3.
        path = tmp_path / "running.obj"
        path.write_text(
            "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
            "f -3 -2 -1\n"
            "v 0 0 5\nv 1 0 5\nv 0 1 5\n"
            "f -3 -2 -1\n"
        )
        mesh = read_obj(path)
        assert np.array_equal(mesh.faces, [[0, 1, 2], [3, 4, 5]])
        assert mesh.area == pytest.approx(1.0)

    def test_polygon_faces_are_fan_triangulated(self, tmp_path):
        path = tmp_path / "quad.obj"
        path.write_text(
            "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 0.5 2 0\n"
            "f 1 2 3 4\n"
            "f 1 2 3 4 5\n"
        )
        mesh = read_obj(path)
        assert mesh.n_faces == 2 + 3
        # The quad alone has area 1; the pentagon adds the two extra fans.
        assert mesh.area > 1.0

    @pytest.mark.parametrize(
        "face_line",
        ["f 1 2 3", "f 1/1 2/2 3/3", "f 1//1 2//2 3//3", "f 1/1/1 2/2/2 3/3/3"],
    )
    def test_all_face_reference_forms(self, tmp_path, face_line):
        path = tmp_path / "forms.obj"
        path.write_text(
            "vt 0 0\nvt 1 0\nvt 0 1\n"
            "vn 0 0 1\nvn 0 0 1\nvn 0 0 1\n"
            "v 0 0 0\nv 3 0 0\nv 0 4 0\n"
            f"{face_line}\n"
        )
        mesh = read_obj(path)
        assert np.array_equal(mesh.faces, [[0, 1, 2]])
        assert mesh.area == pytest.approx(6.0)

    def test_comments_blank_lines_and_material_statements_are_ignored(self, tmp_path):
        path = tmp_path / "noisy.obj"
        path.write_text(
            "# a comment\n"
            "mtllib scene.mtl\n"
            "o my_object\n"
            "g group_one\n"
            "s off\n"
            "\n"
            "v 0 0 0  # trailing comment\n"
            "v 3 0 0\n"
            "v 0 4 0\n"
            "usemtl steel\n"
            "\n"
            "f 1 2 3\n"
        )
        mesh = read_obj(path)
        assert mesh.n_vertices == 3
        assert mesh.area == pytest.approx(6.0)

    def test_write_emits_one_based_indices(self, tmp_path):
        path = tmp_path / "cube.obj"
        write_obj(path, build_cube(2.0))
        lines = path.read_text().splitlines()
        faces = [line for line in lines if line.startswith("f ")]
        assert len(faces) == 12
        indices = [int(token) for line in faces for token in line.split()[1:]]
        assert min(indices) == 1
        assert max(indices) == 8

    def test_write_with_normals(self, tmp_path):
        path = tmp_path / "cube.obj"
        write_obj(path, build_cube(2.0), name="cube", normals=True)
        text = path.read_text()
        assert text.count("\nvn ") == 8
        assert "o cube" in text
        assert "f 1//1" in text
        assert read_obj(path).volume == pytest.approx(8.0)

    def test_index_zero_is_rejected(self, tmp_path):
        path = tmp_path / "zero.obj"
        path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 1 2\n")
        with pytest.raises(MeshIOError, match="start at 1"):
            read_obj(path)

    def test_out_of_range_index_is_rejected(self, tmp_path):
        path = tmp_path / "range.obj"
        path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 9\n")
        with pytest.raises(MeshIOError, match="outside the range"):
            read_obj(path)

    def test_garbage_face_token_is_rejected(self, tmp_path):
        path = tmp_path / "garbage.obj"
        path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 three\n")
        with pytest.raises(MeshIOError, match="valid face reference"):
            read_obj(path)

    def test_short_vertex_line_is_rejected(self, tmp_path):
        path = tmp_path / "short.obj"
        path.write_text("v 0 0\n")
        with pytest.raises(MeshIOError, match="three coordinates"):
            read_obj(path)


class TestPly:
    def test_ascii_with_extra_colour_properties(self, tmp_path):
        path = tmp_path / "coloured.ply"
        path.write_text(
            "ply\n"
            "format ascii 1.0\n"
            "comment made by hand\n"
            "element vertex 4\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "property float nx\n"
            "element face 2\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
            "0 0 0 255 0 0 1.0\n"
            "2 0 0 0 255 0 1.0\n"
            "2 3 0 0 0 255 1.0\n"
            "0 3 0 255 255 0 1.0\n"
            "3 0 1 2\n"
            "3 0 2 3\n"
        )
        mesh = read_ply(path)
        assert mesh.n_vertices == 4
        assert mesh.n_faces == 2
        assert mesh.area == pytest.approx(6.0)
        assert mesh.vertices[2] == pytest.approx([2.0, 3.0, 0.0])

    def test_binary_with_extra_colour_properties(self, tmp_path):
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            "element vertex 3\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "element face 1\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
        ).encode("ascii")
        vertex_dtype = np.dtype(
            [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")]
        )
        rows = np.zeros(3, dtype=vertex_dtype)
        rows["x"] = [0.0, 3.0, 0.0]
        rows["y"] = [0.0, 0.0, 4.0]
        rows["r"] = [255, 0, 0]
        body = rows.tobytes() + struct.pack("<B3i", 3, 0, 1, 2)
        path = tmp_path / "coloured_binary.ply"
        path.write_bytes(header + body)

        mesh = read_ply(path)
        assert mesh.n_vertices == 3
        assert mesh.n_faces == 1
        assert mesh.area == pytest.approx(6.0)

    def test_binary_big_endian_is_read(self, tmp_path):
        header = (
            "ply\n"
            "format binary_big_endian 1.0\n"
            "element vertex 3\n"
            "property double x\n"
            "property double y\n"
            "property double z\n"
            "element face 1\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
        ).encode("ascii")
        points = np.array([[0, 0, 0], [3, 0, 0], [0, 4, 0]], dtype=">f8")
        body = points.tobytes() + struct.pack(">B3i", 3, 0, 1, 2)
        path = tmp_path / "big_endian.ply"
        path.write_bytes(header + body)

        mesh = read_ply(path)
        assert mesh.area == pytest.approx(6.0)
        assert mesh.vertices[1] == pytest.approx([3.0, 0.0, 0.0])

    def test_unknown_element_between_vertex_and_face_is_skipped(self, tmp_path):
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            "element vertex 3\n"
            "property double x\n"
            "property double y\n"
            "property double z\n"
            "element edge 2\n"
            "property int vertex1\n"
            "property int vertex2\n"
            "property list uchar float weights\n"
            "element face 1\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
        ).encode("ascii")
        points = np.array([[0, 0, 0], [3, 0, 0], [0, 4, 0]], dtype="<f8")
        edges = struct.pack("<iiB2f", 0, 1, 2, 0.5, 0.5) + struct.pack("<iiB1f", 1, 2, 1, 1.0)
        body = points.tobytes() + edges + struct.pack("<B3i", 3, 0, 1, 2)
        path = tmp_path / "edges.ply"
        path.write_bytes(header + body)

        mesh = read_ply(path)
        assert mesh.n_faces == 1
        assert mesh.area == pytest.approx(6.0)

    def test_polygon_faces_are_fan_triangulated(self, tmp_path):
        path = tmp_path / "quad.ply"
        path.write_text(
            "ply\n"
            "format ascii 1.0\n"
            "element vertex 4\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "element face 1\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
            "0 0 0\n1 0 0\n1 1 0\n0 1 0\n"
            "4 0 1 2 3\n"
        )
        mesh = read_ply(path)
        assert mesh.n_faces == 2
        assert mesh.area == pytest.approx(1.0)

    def test_ascii_and_binary_writers_agree(self, tmp_path):
        original = build_sample()
        binary_path = tmp_path / "binary.ply"
        ascii_path = tmp_path / "ascii.ply"
        write_ply(binary_path, original, binary=True)
        write_ply(ascii_path, original, binary=False)
        assert read_ply(binary_path) == read_ply(ascii_path)
        assert read_ply(binary_path).volume == pytest.approx(13.5)

    def test_single_precision_option(self, tmp_path):
        path = tmp_path / "float.ply"
        write_ply(path, build_cube(2.0), binary=True, double=False)
        assert b"property float x" in path.read_bytes()
        assert read_ply(path).volume == pytest.approx(8.0, rel=1e-6)

    def test_missing_end_header_raises(self, tmp_path):
        path = tmp_path / "bad.ply"
        path.write_bytes(b"ply\nformat ascii 1.0\nelement vertex 1\n")
        with pytest.raises(MeshIOError, match="end_header"):
            read_ply(path)

    def test_missing_magic_raises(self, tmp_path):
        path = tmp_path / "bad.ply"
        path.write_bytes(b"nope\nend_header\n")
        with pytest.raises(MeshIOError, match="magic word"):
            read_ply(path)

    def test_unsupported_format_raises(self, tmp_path):
        path = tmp_path / "bad.ply"
        path.write_bytes(b"ply\nformat binary_middle_endian 1.0\nend_header\n")
        with pytest.raises(MeshIOError, match="Unsupported PLY format"):
            read_ply(path)

    def test_unknown_property_type_raises(self, tmp_path):
        path = tmp_path / "bad.ply"
        path.write_bytes(
            b"ply\nformat ascii 1.0\nelement vertex 1\nproperty quadruple x\nend_header\n"
        )
        with pytest.raises(MeshIOError, match="Unknown PLY property type"):
            read_ply(path)

    def test_truncated_binary_body_raises(self, tmp_path):
        path = tmp_path / "cut.ply"
        write_ply(path, build_sample(), binary=True)
        raw = path.read_bytes()
        path.write_bytes(raw[: len(raw) - 200])
        with pytest.raises(MeshIOError, match="Truncated PLY"):
            read_ply(path)

    def test_truncated_ascii_body_raises(self, tmp_path):
        path = tmp_path / "cut.ply"
        path.write_text(
            "ply\n"
            "format ascii 1.0\n"
            "element vertex 4\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "element face 0\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
            "0 0 0\n1 0 0\n"
        )
        with pytest.raises(MeshIOError, match="Truncated ascii PLY"):
            read_ply(path)

    def test_vertex_without_xyz_raises(self, tmp_path):
        path = tmp_path / "bad.ply"
        path.write_text(
            "ply\n"
            "format ascii 1.0\n"
            "element vertex 1\n"
            "property float u\n"
            "element face 0\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
            "0\n"
        )
        with pytest.raises(MeshIOError, match="x, y, and z"):
            read_ply(path)


class TestOff:
    def test_round_trip_and_header(self, tmp_path):
        path = tmp_path / "cube.off"
        write_off(path, build_cube(2.0))
        lines = path.read_text().splitlines()
        assert lines[0] == "OFF"
        assert lines[1] == "8 12 0"
        assert read_off(path).volume == pytest.approx(8.0)

    def test_polygon_faces_are_fan_triangulated(self, tmp_path):
        path = tmp_path / "quad.off"
        path.write_text("OFF\n4 1 0\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n4 0 1 2 3\n")
        mesh = read_off(path)
        assert mesh.n_faces == 2
        assert mesh.area == pytest.approx(1.0)

    def test_counts_on_the_magic_line_are_accepted(self, tmp_path):
        path = tmp_path / "inline.off"
        path.write_text("OFF 3 1 0\n0 0 0\n3 0 0\n0 4 0\n3 0 1 2\n")
        assert read_off(path).area == pytest.approx(6.0)

    def test_colour_variant_ignores_extra_columns(self, tmp_path):
        path = tmp_path / "coff.off"
        path.write_text(
            "COFF\n"
            "# a comment line\n"
            "3 1 0\n"
            "0 0 0 255 0 0 255\n"
            "3 0 0 0 255 0 255\n"
            "0 4 0 0 0 255 255\n"
            "3 0 1 2 128 128 128\n"
        )
        mesh = read_off(path)
        assert mesh.n_vertices == 3
        assert mesh.area == pytest.approx(6.0)

    def test_bad_magic_raises(self, tmp_path):
        path = tmp_path / "bad.off"
        path.write_text("NOTOFFATALL\n1 0 0\n")
        with pytest.raises(MeshIOError, match="Not an OFF file"):
            read_off(path)

    def test_truncated_body_raises(self, tmp_path):
        path = tmp_path / "cut.off"
        path.write_text("OFF\n8 12 0\n0 0 0\n1 0 0\n")
        with pytest.raises(MeshIOError, match="Truncated OFF"):
            read_off(path)

    def test_empty_file_raises(self, tmp_path):
        path = tmp_path / "empty.off"
        path.write_text("\n# nothing but a comment\n")
        with pytest.raises(MeshIOError, match="Empty OFF"):
            read_off(path)

    def test_non_numeric_vertex_raises(self, tmp_path):
        path = tmp_path / "bad.off"
        path.write_text("OFF\n1 0 0\nx y z\n")
        with pytest.raises(MeshIOError, match="non-numeric"):
            read_off(path)


class Test3mf:
    def test_package_contains_the_required_parts(self, tmp_path):
        path = tmp_path / "cube.3mf"
        write_3mf(path, build_cube(2.0))
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            model = archive.read("3D/3dmodel.model").decode("utf-8")
        assert {"[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model"} <= names
        assert "http://schemas.microsoft.com/3dmanufacturing/core/2015/02" in model
        assert model.count("<vertex ") == 8
        assert model.count("<triangle ") == 12

    def test_unit_is_preserved_as_an_attribute(self, tmp_path):
        path = tmp_path / "cube.3mf"
        write_3mf(path, build_cube(2.0), unit="inch")
        mesh = read_3mf(path)
        assert mesh.attributes["unit"] == "inch"
        assert mesh.volume == pytest.approx(8.0)

    def test_invalid_unit_is_rejected(self, tmp_path):
        with pytest.raises(MeshIOError, match="not a 3MF unit"):
            write_3mf(tmp_path / "bad.3mf", build_cube(2.0), unit="furlong")

    def test_build_item_transform_is_applied(self, tmp_path):
        # Row-major 4x3: rows 0-2 are the basis, row 3 is the translation.
        transform = 'transform="2 0 0 0 2 0 0 0 2 1 2 3"'
        path = minimal_3mf(
            tmp_path / "scaled.3mf",
            model_xml(CUBE_VERTICES, CUBE_TRIANGLES, item_attributes=transform),
        )
        mesh = read_3mf(path)
        assert mesh.volume == pytest.approx(64.0)
        assert mesh.bounds == pytest.approx((-1.0, 3.0, 0.0, 4.0, 1.0, 5.0))

    def test_mirroring_build_transform_keeps_the_volume_positive(self, tmp_path):
        transform = 'transform="-1 0 0 0 1 0 0 0 1 0 0 0"'
        path = minimal_3mf(
            tmp_path / "mirror.3mf",
            model_xml(CUBE_VERTICES, CUBE_TRIANGLES, item_attributes=transform),
        )
        mesh = read_3mf(path)
        assert mesh.volume == pytest.approx(8.0)

    def test_malformed_transform_is_rejected(self, tmp_path):
        path = minimal_3mf(
            tmp_path / "bad.3mf",
            model_xml(CUBE_VERTICES, CUBE_TRIANGLES, item_attributes='transform="1 2 3"'),
        )
        with pytest.raises(MeshIOError, match="expected 12 numbers"):
            read_3mf(path)

    def test_components_are_resolved_with_their_transforms(self, tmp_path):
        vertices = "".join(
            f'<vertex x="{x}" y="{y}" z="{z}"/>' for x, y, z in CUBE_VERTICES
        )
        triangles = "".join(
            f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in CUBE_TRIANGLES
        )
        model = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<model unit="millimeter" '
            'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
            "<resources>"
            f'<object id="1" type="model"><mesh><vertices>{vertices}</vertices>'
            f"<triangles>{triangles}</triangles></mesh></object>"
            '<object id="2" type="model"><components>'
            '<component objectid="1"/>'
            '<component objectid="1" transform="1 0 0 0 1 0 0 0 1 10 0 0"/>'
            "</components></object>"
            "</resources>"
            '<build><item objectid="2"/></build>'
            "</model>"
        ).encode("utf-8")
        path = minimal_3mf(tmp_path / "components.3mf", model)
        mesh = read_3mf(path)
        assert mesh.n_faces == 24
        assert mesh.volume == pytest.approx(16.0)
        assert mesh.n_components() == 2

    def test_zip_path_traversal_member_is_rejected(self, tmp_path):
        path = zip_package(
            tmp_path / "evil.3mf",
            {
                "3D/3dmodel.model": model_xml(CUBE_VERTICES, CUBE_TRIANGLES),
                "../../etc/evil.model": b"pwned",
            },
        )
        with pytest.raises(MeshIOError, match="escapes the archive root"):
            read_3mf(path)

    def test_absolute_zip_member_is_rejected(self, tmp_path):
        path = zip_package(
            tmp_path / "absolute.3mf",
            {
                "3D/3dmodel.model": model_xml(CUBE_VERTICES, CUBE_TRIANGLES),
                "/tmp/evil.model": b"pwned",
            },
        )
        with pytest.raises(MeshIOError, match="absolute path"):
            read_3mf(path)

    def test_windows_drive_zip_member_is_rejected(self, tmp_path):
        path = zip_package(
            tmp_path / "drive.3mf",
            {
                "3D/3dmodel.model": model_xml(CUBE_VERTICES, CUBE_TRIANGLES),
                "C:/Windows/evil.model": b"pwned",
            },
        )
        with pytest.raises(MeshIOError, match="absolute path"):
            read_3mf(path)

    def test_entity_expansion_is_refused(self, tmp_path):
        bomb = (
            '<?xml version="1.0"?>'
            "<!DOCTYPE model ["
            '<!ENTITY a "aaaaaaaaaa">'
            '<!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">'
            '<!ENTITY c "&b;&b;&b;&b;&b;&b;&b;&b;&b;&b;">'
            "]>"
            '<model unit="millimeter" '
            'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
            "<resources><object id=\"1\"><mesh><vertices/><triangles/></mesh></object>"
            "</resources><build><item objectid=\"1\"/></build><metadata>&c;</metadata>"
            "</model>"
        ).encode("utf-8")
        path = minimal_3mf(tmp_path / "bomb.3mf", bomb)
        with pytest.raises(MeshIOError, match="document type declaration"):
            read_3mf(path)

    def test_oversized_part_is_refused(self, tmp_path):
        path = minimal_3mf(
            tmp_path / "big.3mf", model_xml(CUBE_VERTICES, CUBE_TRIANGLES)
        )
        with pytest.raises(MeshIOError, match="safety limit"):
            read_3mf(path, max_part_bytes=32)

    def test_not_a_zip_raises(self, tmp_path):
        path = tmp_path / "plain.3mf"
        path.write_bytes(b"this is not a zip archive")
        with pytest.raises(MeshIOError, match="not a valid zip"):
            read_3mf(path)

    def test_archive_without_a_model_part_raises(self, tmp_path):
        path = zip_package(tmp_path / "empty.3mf", {"readme.txt": b"nothing here"})
        with pytest.raises(MeshIOError, match="no 3D model part"):
            read_3mf(path)

    def test_model_without_objects_raises(self, tmp_path):
        model = (
            '<?xml version="1.0"?>'
            '<model unit="millimeter" '
            'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
            "<resources/><build/></model>"
        ).encode("utf-8")
        path = minimal_3mf(tmp_path / "bare.3mf", model)
        with pytest.raises(MeshIOError, match="declares no objects"):
            read_3mf(path)

    def test_dangling_object_reference_raises(self, tmp_path):
        model = model_xml(CUBE_VERTICES, CUBE_TRIANGLES).replace(
            b'<item objectid="1"', b'<item objectid="99"'
        )
        path = minimal_3mf(tmp_path / "dangling.3mf", model)
        with pytest.raises(MeshIOError, match="does not exist"):
            read_3mf(path)

    def test_broken_xml_raises(self, tmp_path):
        path = minimal_3mf(tmp_path / "broken.3mf", b"<model><resources>")
        with pytest.raises(MeshIOError, match="not valid XML"):
            read_3mf(path)

    def test_relationship_target_is_followed(self, tmp_path):
        path = zip_package(
            tmp_path / "custom.3mf",
            {
                "[Content_Types].xml": b"<Types/>",
                "_rels/.rels": (
                    '<?xml version="1.0"?>'
                    '<Relationships xmlns="http://schemas.openxmlformats.org/'
                    'package/2006/relationships">'
                    '<Relationship Id="rel0" '
                    'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel" '
                    'Target="/models/part.model"/>'
                    "</Relationships>"
                ).encode(),
                "models/part.model": model_xml(CUBE_VERTICES, CUBE_TRIANGLES),
            },
        )
        assert read_3mf(path).volume == pytest.approx(8.0)


class TestDispatch:
    def test_supported_format_tuples(self):
        assert SUPPORTED_READ_FORMATS == ("stl", "obj", "ply", "off", "3mf")
        assert SUPPORTED_WRITE_FORMATS == ("stl", "obj", "ply", "off", "3mf")

    @pytest.mark.parametrize("suffix", [".STL", ".Obj", ".PLY", ".oFF", ".3MF"])
    def test_extension_matching_is_case_insensitive(self, tmp_path, suffix):
        path = tmp_path / f"model{suffix}"
        write_mesh(path, build_cube(2.0))
        assert read_mesh(path).volume == pytest.approx(8.0, rel=1e-6)

    def test_detect_format_reports_the_key(self):
        assert detect_format("part.STL") == "stl"
        assert detect_format("/some/dir/part.3mf") == "3mf"

    def test_unsupported_extension_raises_on_read(self, tmp_path):
        path = tmp_path / "model.step"
        path.write_bytes(b"")
        with pytest.raises(MeshIOError, match="Unsupported mesh format"):
            read_mesh(path)

    def test_unsupported_extension_raises_on_write(self, tmp_path):
        with pytest.raises(MeshIOError, match="Unsupported mesh format"):
            write_mesh(tmp_path / "model.iges", build_cube(2.0))

    def test_missing_extension_raises(self, tmp_path):
        with pytest.raises(MeshIOError, match="no file extension"):
            write_mesh(tmp_path / "model", build_cube(2.0))

    def test_explicit_format_overrides_the_extension(self, tmp_path):
        path = tmp_path / "model.dat"
        write_mesh(path, build_cube(2.0), fmt="obj")
        assert path.read_text().startswith("# Written by OpenCad")
        assert read_mesh(path, fmt="obj").volume == pytest.approx(8.0)

    def test_write_options_reach_the_format_writer(self, tmp_path):
        path = tmp_path / "ascii.stl"
        write_mesh(path, build_cube(2.0), binary=False)
        assert path.read_bytes().startswith(b"solid")

    def test_bad_option_raises_mesh_io_error(self, tmp_path):
        with pytest.raises(MeshIOError, match="Unsupported option"):
            write_mesh(tmp_path / "cube.off", build_cube(2.0), binary=True)

    def test_file_filter_shape(self):
        text = file_filter()
        assert text.startswith("Mesh Files (*.stl *.obj *.ply *.off *.3mf)")
        assert "STL (*.stl)" in text
        assert "All Files (*)" in text
        assert len(text.split(";;")) == 1 + len(SUPPORTED_READ_FORMATS) + 1
        assert ";;" in file_filter("write", include_all=False)
        assert "All Files" not in file_filter("write", include_all=False)

    def test_file_filter_rejects_a_bad_mode(self):
        with pytest.raises(MeshIOError, match="'read' or 'write'"):
            file_filter("append")


class TestFileObjectsAndGuards:
    @pytest.mark.parametrize("fmt", FORMATS)
    def test_binary_file_objects_are_accepted(self, fmt):
        buffer = io.BytesIO()
        write_mesh(buffer, build_cube(2.0), fmt=fmt)
        buffer.seek(0)
        mesh = read_mesh(buffer, fmt=fmt)
        assert mesh.volume == pytest.approx(8.0, rel=tolerance_for(fmt))

    def test_text_mode_file_object_is_rejected(self, tmp_path):
        path = tmp_path / "cube.obj"
        write_obj(path, build_cube(2.0))
        with open(path) as handle, pytest.raises(MeshIOError, match="binary file object"):
            read_obj(handle)

    def test_format_cannot_be_inferred_from_a_nameless_buffer(self):
        with pytest.raises(MeshIOError, match="Cannot infer a mesh format"):
            read_mesh(io.BytesIO(b""))

    def test_missing_file_raises_mesh_io_error(self, tmp_path):
        with pytest.raises(MeshIOError, match="Could not read"):
            read_mesh(tmp_path / "absent.stl")

    def test_mesh_io_error_is_an_os_error(self, tmp_path):
        with pytest.raises(OSError):
            read_mesh(tmp_path / "absent.ply")

    @pytest.mark.parametrize("fmt", FORMATS)
    def test_non_finite_coordinates_are_refused(self, tmp_path, fmt):
        broken = build_cube(2.0)
        vertices = broken.vertices.copy()
        vertices[0, 0] = np.nan
        with pytest.raises(MeshIOError, match="NaN or infinity"):
            write_mesh(tmp_path / f"nan.{fmt}", Mesh(vertices, broken.faces))

    @pytest.mark.parametrize("fmt", FORMATS)
    def test_dangling_face_indices_are_refused(self, tmp_path, fmt):
        broken = Mesh(np.zeros((3, 3)), np.array([[0, 1, 2]], dtype=np.int64))
        broken.faces[0, 2] = 7
        with pytest.raises(MeshIOError, match="do not exist"):
            write_mesh(tmp_path / f"bad.{fmt}", broken)

    def test_writing_none_is_refused(self, tmp_path):
        with pytest.raises(MeshIOError, match="no mesh was supplied"):
            write_mesh(tmp_path / "none.stl", None)
