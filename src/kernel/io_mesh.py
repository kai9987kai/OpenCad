"""Pure-Python mesh file I/O for the OpenCad geometry kernel.

OpenCad's desktop layer reads and writes geometry through PyVista, so every
import/export path drags in VTK.  That makes headless scripting impossible and
leaves the file formats completely untested in CI - the one place where format
bugs are cheapest to catch.  This module implements the exchange formats the
application actually cares about (STL, OBJ, PLY, OFF, 3MF) directly on top of
numpy and the standard library, so the kernel can load and save meshes alone.

Conventions
-----------
- Every reader returns a :class:`~src.kernel.mesh.Mesh` with the winding exactly
  as stored in the file; no reader tries to re-orient a surface.  Use
  ``Mesh.flipped`` or the repair helpers if an import comes in inside-out.
- Polygons with more than three corners are fan-triangulated.  That is exact for
  convex polygons and matches what every mesh viewer does; a concave polygon
  will produce triangles outside its outline, which is a property of the format
  rather than of this code.
- Readers and writers accept ``str``, ``pathlib.Path``, or an already-open
  **binary** file object.  Text-mode file objects are rejected with a clear
  message rather than failing deep inside a decoder.
- Ascii output (ascii STL, OBJ, ascii PLY, OFF, 3MF) is written with ``repr``,
  which is the shortest decimal string that round-trips a float64 exactly.  Those
  formats therefore lose no precision.  Binary STL is float32 by definition, so
  it is the only lossy path here.
- Anything wrong with a file - unknown extension, truncated payload, malformed
  header, hostile archive - raises :class:`MeshIOError`.  A caller never sees a
  bare ``struct.error``, ``IndexError``, or ``KeyError`` from this module.
"""

from __future__ import annotations

import os
import struct
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

from src.kernel.mesh import EPS, Mesh

__all__ = [
    "MAX_3MF_PART_BYTES",
    "SUPPORTED_READ_FORMATS",
    "SUPPORTED_WRITE_FORMATS",
    "MeshIOError",
    "detect_format",
    "file_filter",
    "read_3mf",
    "read_mesh",
    "read_obj",
    "read_off",
    "read_ply",
    "read_stl",
    "write_3mf",
    "write_mesh",
    "write_obj",
    "write_off",
    "write_ply",
    "write_stl",
]


class MeshIOError(OSError):
    """Raised for any unreadable, unwritable, or unsupported mesh file.

    It derives from :class:`OSError` so application code can keep a single
    ``except OSError`` around a load/save action and still catch malformed-file
    problems alongside missing-file and permission problems.
    """


SUPPORTED_READ_FORMATS = ("stl", "obj", "ply", "off", "3mf")
SUPPORTED_WRITE_FORMATS = ("stl", "obj", "ply", "off", "3mf")

#: Cap on the uncompressed size of a single part read out of a 3MF archive.
#: A zip can claim a tiny compressed size while expanding to gigabytes, so the
#: reader refuses anything larger unless the caller raises the limit.
MAX_3MF_PART_BYTES = 256 * 1024 * 1024

_FORMAT_LABELS = {
    "stl": "STL",
    "obj": "Wavefront OBJ",
    "ply": "Stanford PLY",
    "off": "OFF",
    "3mf": "3D Manufacturing Format",
}


# ----------------------------------------------------------------------
# Shared plumbing
# ----------------------------------------------------------------------
def _as_path(value):
    try:
        return Path(os.fspath(value))
    except TypeError as exc:
        raise MeshIOError(
            f"Expected a filesystem path or a binary file object, got {type(value).__name__}."
        ) from exc


def _read_all_bytes(source):
    """Slurp a path or binary file object, mapping OS failures to MeshIOError."""
    if hasattr(source, "read"):
        data = source.read()
        if isinstance(data, str):
            raise MeshIOError(
                "Mesh files must be read from a binary file object; open it with mode 'rb'."
            )
        return bytes(data)
    path = _as_path(source)
    try:
        return path.read_bytes()
    except MeshIOError:
        raise
    except OSError as exc:
        raise MeshIOError(f"Could not read '{path}': {exc}") from exc


def _write_all_bytes(target, data):
    """Write bytes to a path or binary file object."""
    if hasattr(target, "write"):
        try:
            target.write(data)
        except TypeError as exc:
            raise MeshIOError(
                "Mesh files must be written to a binary file object; open it with mode 'wb'."
            ) from exc
        return
    path = _as_path(target)
    try:
        path.write_bytes(data)
    except MeshIOError:
        raise
    except OSError as exc:
        raise MeshIOError(f"Could not write '{path}': {exc}") from exc


def _decode(data):
    """Decode text payloads leniently - CAD exporters are careless about encoding."""
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _faces_from_polygons(polygons):
    """Fan-triangulate a ragged list of index lists into an ``(M, 3)`` array."""
    if not polygons:
        return np.zeros((0, 3), dtype=np.int64)
    if all(len(polygon) == 3 for polygon in polygons):
        return np.asarray(polygons, dtype=np.int64)
    triangles = []
    for polygon in polygons:
        if len(polygon) < 3:
            continue
        anchor = polygon[0]
        for corner in range(1, len(polygon) - 1):
            triangles.append((anchor, polygon[corner], polygon[corner + 1]))
    if not triangles:
        return np.zeros((0, 3), dtype=np.int64)
    return np.asarray(triangles, dtype=np.int64)


def _mesh_from(vertices, faces, label, attributes=None):
    """Build a Mesh from parsed arrays, rejecting out-of-range indices."""
    vertices = np.asarray(vertices, dtype=float).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(faces):
        if faces.min() < 0 or faces.max() >= len(vertices):
            raise MeshIOError(
                f"{label} references vertex indices outside the range "
                f"0..{max(len(vertices) - 1, 0)}; the file is inconsistent."
            )
    return Mesh(vertices, faces, attributes)


def _export_arrays(mesh, label):
    """Validate a mesh on the way out and return plain ``(vertices, faces)``."""
    if mesh is None:
        raise MeshIOError(f"Cannot write {label}: no mesh was supplied.")
    try:
        vertices = np.asarray(mesh.vertices, dtype=float).reshape(-1, 3)
        faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)
    except (AttributeError, ValueError) as exc:
        raise MeshIOError(f"Cannot write {label}: the object is not a triangle mesh.") from exc
    if len(vertices) and not np.all(np.isfinite(vertices)):
        raise MeshIOError(f"Cannot write {label}: vertex coordinates contain NaN or infinity.")
    if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise MeshIOError(
            f"Cannot write {label}: face indices reference vertices that do not exist."
        )
    return vertices, faces


def _vertex_lines(vertices, prefix):
    """``repr``-formatted coordinate lines - shortest text that round-trips."""
    return [f"{prefix}{x!r} {y!r} {z!r}" for x, y, z in vertices.tolist()]


# ----------------------------------------------------------------------
# STL
# ----------------------------------------------------------------------
_STL_FACET_DTYPE = np.dtype(
    [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attributes", "<u2")]
)


def _stl_looks_binary(data):
    """Decide binary vs ascii from the declared triangle count, not the magic word.

    A binary STL's 80-byte header is free-form and plenty of exporters start it
    with the word ``solid``, which defeats the naive sniff.  The reliable test is
    arithmetic: a binary file is exactly ``84 + 50 * n`` bytes long for the ``n``
    it declares at offset 80.  Only when that check fails do we fall back to the
    keyword, and even then we require the payload to actually look like text.
    """
    if len(data) < 84:
        return False
    count = int(struct.unpack_from("<I", data, 80)[0])
    if 84 + 50 * count == len(data):
        return True
    if data[:5].lower() != b"solid":
        return True
    sample = data[:4096]
    textual = sum(1 for byte in sample if 32 <= byte < 127 or byte in (9, 10, 13))
    return textual < len(sample)


def _read_stl_binary(data):
    count = int(struct.unpack_from("<I", data, 80)[0])
    payload = data[84:]
    if count * 50 > len(payload):
        available = len(payload) // 50
        raise MeshIOError(
            f"Truncated binary STL: the header declares {count} triangles but only "
            f"{available} are present in the file."
        )
    if count == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    records = np.frombuffer(payload, dtype=_STL_FACET_DTYPE, count=count)
    vertices = records["vertices"].reshape(-1, 3).astype(np.float64)
    faces = np.arange(3 * count, dtype=np.int64).reshape(count, 3)
    return vertices, faces


def _read_stl_ascii(data):
    text = _decode(data)
    coordinates = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or not line[:1].lower() == "v":
            continue
        parts = line.split()
        if parts[0].lower() != "vertex":
            continue
        if len(parts) < 4:
            raise MeshIOError(f"Ascii STL line {number}: 'vertex' needs three coordinates.")
        try:
            coordinates.append((float(parts[1]), float(parts[2]), float(parts[3])))
        except ValueError as exc:
            raise MeshIOError(
                f"Ascii STL line {number}: could not parse the vertex coordinates."
            ) from exc
    if len(coordinates) % 3 != 0:
        raise MeshIOError(
            f"Truncated ascii STL: {len(coordinates)} vertices is not a whole number "
            "of triangles."
        )
    if not coordinates:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    vertices = np.asarray(coordinates, dtype=float)
    faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    return vertices, faces


def _auto_weld_tolerance(vertices):
    """A weld tolerance of roughly one float32 quantum at the size of the model.

    STL stores coordinates in single precision, so vertices the format made
    indistinguishable should collapse while genuinely distinct ones survive.
    Scaling by the bounding-box diagonal keeps that true for a 2 mm screw and a
    2 m chassis alike.
    """
    if len(vertices) == 0:
        return EPS
    diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    return max(diagonal * 1e-7, EPS)


def read_stl(source, weld=True, tolerance=None):
    """Read a binary or ascii STL, welding the triangle soup into an indexed mesh.

    STL has no vertex table - every triangle repeats its three corners - so a
    literal read produces ``3 * n`` vertices and a mesh with no connectivity.
    Welding restores the shared vertices, which is what makes the result usable
    for topology queries, booleans, and smoothing.  Pass ``weld=False`` to keep
    the raw soup, or ``tolerance`` to override the automatic size-scaled value.
    """
    data = _read_all_bytes(source)
    if len(data) == 0:
        raise MeshIOError("Empty STL file: there is nothing to read.")
    if _stl_looks_binary(data):
        vertices, faces = _read_stl_binary(data)
    else:
        vertices, faces = _read_stl_ascii(data)
    mesh = _mesh_from(vertices, faces, "STL file")
    if weld and not mesh.is_empty:
        limit = _auto_weld_tolerance(mesh.vertices) if tolerance is None else float(tolerance)
        mesh = mesh.weld(limit).remove_unreferenced_vertices()
    return mesh


def write_stl(target, mesh, binary=True, name="opencad"):
    """Write an STL file.

    Binary output carries a correct outward facet normal per triangle and a zero
    attribute byte count, which is what slicers expect.  The 80-byte header is
    free-form; note that starting it with the word ``solid`` makes naive readers
    misclassify the file (this module's reader is immune - see
    :func:`_stl_looks_binary`).  Ascii output is written at full float64
    precision, so only the binary path is lossy.
    """
    vertices, faces = _export_arrays(mesh, "an STL file")
    triangles = vertices[faces] if len(faces) else np.zeros((0, 3, 3))
    if len(triangles):
        normals = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        lengths = np.linalg.norm(normals, axis=1)
        normals = normals / np.where(lengths > EPS, lengths, 1.0)[:, None]
    else:
        normals = np.zeros((0, 3))

    if binary:
        header = str(name).encode("ascii", "replace")[:80].ljust(80, b"\0")
        records = np.zeros(len(triangles), dtype=_STL_FACET_DTYPE)
        records["normal"] = normals
        records["vertices"] = triangles
        records["attributes"] = 0
        payload = header + struct.pack("<I", len(triangles)) + records.tobytes()
        _write_all_bytes(target, payload)
        return

    solid = str(name).replace("\n", " ").strip() or "opencad"
    lines = [f"solid {solid}"]
    for normal, triangle in zip(normals.tolist(), triangles.tolist(), strict=True):
        lines.append(f"  facet normal {normal[0]!r} {normal[1]!r} {normal[2]!r}")
        lines.append("    outer loop")
        for corner in triangle:
            lines.append(f"      vertex {corner[0]!r} {corner[1]!r} {corner[2]!r}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {solid}")
    _write_all_bytes(target, ("\n".join(lines) + "\n").encode("ascii"))


# ----------------------------------------------------------------------
# OBJ
# ----------------------------------------------------------------------
def read_obj(source):
    """Read a Wavefront OBJ, keeping only geometry.

    Handles the four face-reference spellings (``v``, ``v/vt``, ``v//vn``,
    ``v/vt/vn``), negative indices - which OBJ defines as relative to the number
    of vertices seen *so far* - polygons of any size, comments, and blank lines.
    Material, group, object, and smoothing statements are ignored rather than
    rejected, so a textured export still imports as clean geometry.
    """
    text = _decode(_read_all_bytes(source))
    positions = []
    polygons = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        keyword = parts[0]
        if keyword == "v":
            if len(parts) < 4:
                raise MeshIOError(f"OBJ line {number}: 'v' needs at least three coordinates.")
            try:
                positions.append((float(parts[1]), float(parts[2]), float(parts[3])))
            except ValueError as exc:
                raise MeshIOError(
                    f"OBJ line {number}: could not parse the vertex coordinates."
                ) from exc
        elif keyword == "f":
            polygon = []
            for token in parts[1:]:
                reference = token.split("/", 1)[0]
                try:
                    index = int(reference)
                except ValueError as exc:
                    raise MeshIOError(
                        f"OBJ line {number}: '{token}' is not a valid face reference."
                    ) from exc
                if index > 0:
                    polygon.append(index - 1)
                elif index < 0:
                    polygon.append(len(positions) + index)
                else:
                    raise MeshIOError(
                        f"OBJ line {number}: index 0 is not valid, OBJ indices start at 1."
                    )
            if len(polygon) >= 3:
                polygons.append(polygon)
    vertices = np.asarray(positions, dtype=float) if positions else np.zeros((0, 3))
    return _mesh_from(vertices, _faces_from_polygons(polygons), "OBJ file")


def write_obj(target, mesh, name=None, normals=False):
    """Write a Wavefront OBJ with 1-based indices.

    With ``normals=True`` the file also carries one area-weighted vertex normal
    per vertex and faces are written in ``v//vn`` form, which is what viewers
    need to shade a smooth import without recomputing anything.
    """
    vertices, faces = _export_arrays(mesh, "an OBJ file")
    lines = ["# Written by OpenCad"]
    if name:
        lines.append(f"o {str(name).splitlines()[0]}")
    lines.extend(_vertex_lines(vertices, "v "))
    if normals:
        lines.extend(_vertex_lines(mesh.vertex_normals(), "vn "))
        lines.extend(
            f"f {a + 1}//{a + 1} {b + 1}//{b + 1} {c + 1}//{c + 1}"
            for a, b, c in faces.tolist()
        )
    else:
        lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces.tolist())
    _write_all_bytes(target, ("\n".join(lines) + "\n").encode("utf-8"))


# ----------------------------------------------------------------------
# PLY
# ----------------------------------------------------------------------
_PLY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}

_STRUCT_CODES = {
    "i1": "b",
    "u1": "B",
    "i2": "h",
    "u2": "H",
    "i4": "i",
    "u4": "I",
    "f4": "f",
    "f8": "d",
}

_PLY_INDEX_NAMES = ("vertex_indices", "vertex_index")


class _PlyProperty:
    """One declared PLY property, scalar or list."""

    __slots__ = ("name", "value_type", "is_list", "count_type")

    def __init__(self, name, value_type, is_list=False, count_type=None):
        self.name = name
        self.value_type = value_type
        self.is_list = is_list
        self.count_type = count_type


class _PlyElement:
    """One declared PLY element and the properties of each of its records."""

    __slots__ = ("name", "count", "properties")

    def __init__(self, name, count):
        self.name = name
        self.count = count
        self.properties = []


def _ply_numpy_type(name):
    try:
        return _PLY_TYPES[name]
    except KeyError as exc:
        raise MeshIOError(f"Unknown PLY property type '{name}'.") from exc


def _parse_ply_header(data):
    end = data.find(b"end_header")
    if end < 0:
        raise MeshIOError("Malformed PLY file: no 'end_header' line was found.")
    line_end = data.find(b"\n", end)
    body_offset = len(data) if line_end < 0 else line_end + 1
    header = data[:end].decode("ascii", "replace")

    lines = [line.strip() for line in header.splitlines()]
    if not lines or lines[0].lower() != "ply":
        raise MeshIOError("Malformed PLY file: the first line must be 'ply'.")

    order = None
    elements = []
    for line in lines[1:]:
        if not line:
            continue
        parts = line.split()
        keyword = parts[0].lower()
        if keyword in ("comment", "obj_info"):
            continue
        if keyword == "format":
            if len(parts) < 2:
                raise MeshIOError("Malformed PLY header: 'format' line is incomplete.")
            encoding = parts[1].lower()
            if encoding == "ascii":
                order = "ascii"
            elif encoding == "binary_little_endian":
                order = "<"
            elif encoding == "binary_big_endian":
                order = ">"
            else:
                raise MeshIOError(f"Unsupported PLY format '{parts[1]}'.")
        elif keyword == "element":
            if len(parts) < 3:
                raise MeshIOError("Malformed PLY header: 'element' line is incomplete.")
            try:
                count = int(parts[2])
            except ValueError as exc:
                raise MeshIOError(
                    f"Malformed PLY header: '{parts[2]}' is not an element count."
                ) from exc
            if count < 0:
                raise MeshIOError("Malformed PLY header: negative element count.")
            elements.append(_PlyElement(parts[1], count))
        elif keyword == "property":
            if not elements:
                raise MeshIOError("Malformed PLY header: a property precedes any element.")
            if len(parts) >= 2 and parts[1].lower() == "list":
                if len(parts) < 5:
                    raise MeshIOError("Malformed PLY header: 'property list' is incomplete.")
                elements[-1].properties.append(
                    _PlyProperty(
                        parts[4],
                        _ply_numpy_type(parts[3].lower()),
                        True,
                        _ply_numpy_type(parts[2].lower()),
                    )
                )
            else:
                if len(parts) < 3:
                    raise MeshIOError("Malformed PLY header: 'property' line is incomplete.")
                elements[-1].properties.append(
                    _PlyProperty(parts[2], _ply_numpy_type(parts[1].lower()))
                )
    if order is None:
        raise MeshIOError("Malformed PLY header: no 'format' line.")
    return elements, order, body_offset


def _ply_xyz_columns(element):
    names = [prop.name.lower() for prop in element.properties]
    try:
        return [names.index("x"), names.index("y"), names.index("z")]
    except ValueError as exc:
        raise MeshIOError(
            "PLY vertex element does not declare x, y, and z properties."
        ) from exc


def _ply_scalar_dtype(element, order):
    if any(prop.is_list for prop in element.properties):
        return None
    return np.dtype(
        [
            (f"column{index}", order + prop.value_type)
            for index, prop in enumerate(element.properties)
        ]
    )


def _unpack_scalar(data, offset, code):
    try:
        (value,) = struct.unpack_from(code, data, offset)
    except struct.error as exc:
        raise MeshIOError(
            "Truncated PLY file: the data ends in the middle of an element record."
        ) from exc
    return value, offset + struct.calcsize(code)


def _read_ply_binary_vertices(data, offset, element, order):
    dtype = _ply_scalar_dtype(element, order)
    if dtype is None:
        raise MeshIOError("PLY vertex elements with list properties are not supported.")
    columns = _ply_xyz_columns(element)
    if element.count == 0:
        return np.zeros((0, 3)), offset
    needed = element.count * dtype.itemsize
    if len(data) - offset < needed:
        raise MeshIOError(
            f"Truncated PLY file: {element.count} vertices need {needed} bytes but only "
            f"{len(data) - offset} remain."
        )
    records = np.frombuffer(data, dtype=dtype, count=element.count, offset=offset)
    vertices = np.column_stack(
        [records[f"column{index}"].astype(np.float64) for index in columns]
    )
    return vertices, offset + needed


def _try_fast_ply_faces(data, offset, element, order):
    """Vectorised path for the overwhelmingly common all-triangle face element."""
    if len(element.properties) != 1 or not element.properties[0].is_list:
        return None
    prop = element.properties[0]
    count_dtype = np.dtype(order + prop.count_type)
    index_dtype = np.dtype(order + prop.value_type)
    record = count_dtype.itemsize + 3 * index_dtype.itemsize
    needed = element.count * record
    if element.count == 0 or len(data) - offset < needed:
        return None
    block = np.frombuffer(data, dtype=np.uint8, count=needed, offset=offset)
    block = block.reshape(element.count, record)
    lengths = np.ascontiguousarray(block[:, : count_dtype.itemsize]).view(count_dtype)
    if not np.all(lengths.ravel() == 3):
        return None
    indices = np.ascontiguousarray(block[:, count_dtype.itemsize :]).view(index_dtype)
    return indices.reshape(element.count, 3).astype(np.int64), offset + needed


def _read_ply_binary_faces(data, offset, element, order):
    fast = _try_fast_ply_faces(data, offset, element, order)
    if fast is not None:
        return fast
    polygons = []
    for _ in range(element.count):
        polygon = None
        for prop in element.properties:
            if prop.is_list:
                length, offset = _unpack_scalar(
                    data, offset, order + _STRUCT_CODES[prop.count_type]
                )
                length = int(length)
                if length < 0:
                    raise MeshIOError("Malformed PLY file: negative list length in a face.")
                code = f"{order}{length}{_STRUCT_CODES[prop.value_type]}"
                try:
                    values = struct.unpack_from(code, data, offset)
                except struct.error as exc:
                    raise MeshIOError(
                        "Truncated PLY file: the data ends in the middle of a face record."
                    ) from exc
                offset += struct.calcsize(code)
                if polygon is None or prop.name.lower() in _PLY_INDEX_NAMES:
                    polygon = [int(value) for value in values]
            else:
                _, offset = _unpack_scalar(data, offset, order + _STRUCT_CODES[prop.value_type])
        if polygon:
            polygons.append(polygon)
    return _faces_from_polygons(polygons), offset


def _skip_ply_binary_element(data, offset, element, order):
    dtype = _ply_scalar_dtype(element, order)
    if dtype is not None:
        needed = element.count * dtype.itemsize
        if len(data) - offset < needed:
            raise MeshIOError(
                f"Truncated PLY file: element '{element.name}' needs {needed} more bytes."
            )
        return offset + needed
    for _ in range(element.count):
        for prop in element.properties:
            if prop.is_list:
                length, offset = _unpack_scalar(
                    data, offset, order + _STRUCT_CODES[prop.count_type]
                )
                code = f"{order}{int(length)}{_STRUCT_CODES[prop.value_type]}"
                if len(data) - offset < struct.calcsize(code):
                    raise MeshIOError("Truncated PLY file: a list property is incomplete.")
                offset += struct.calcsize(code)
            else:
                _, offset = _unpack_scalar(data, offset, order + _STRUCT_CODES[prop.value_type])
    return offset


def _read_ply_binary_body(data, offset, elements, order):
    vertices = np.zeros((0, 3))
    faces = np.zeros((0, 3), dtype=np.int64)
    for element in elements:
        name = element.name.lower()
        if name == "vertex":
            vertices, offset = _read_ply_binary_vertices(data, offset, element, order)
        elif name == "face":
            faces, offset = _read_ply_binary_faces(data, offset, element, order)
        else:
            offset = _skip_ply_binary_element(data, offset, element, order)
    return vertices, faces


def _read_ply_ascii_body(payload, elements):
    lines = [line for line in _decode(payload).splitlines() if line.strip()]
    cursor = 0
    vertices = np.zeros((0, 3))
    faces = np.zeros((0, 3), dtype=np.int64)
    for element in elements:
        if cursor + element.count > len(lines):
            raise MeshIOError(
                f"Truncated ascii PLY file: element '{element.name}' declares "
                f"{element.count} records but only {len(lines) - cursor} lines remain."
            )
        rows = lines[cursor : cursor + element.count]
        cursor += element.count
        name = element.name.lower()
        if name == "vertex":
            columns = _ply_xyz_columns(element)
            if element.count == 0:
                continue
            try:
                table = np.array(
                    [[float(token) for token in row.split()] for row in rows], dtype=float
                )
            except ValueError as exc:
                raise MeshIOError(
                    "Malformed ascii PLY file: a vertex line has non-numeric or missing values."
                ) from exc
            if table.ndim != 2 or table.shape[1] <= max(columns):
                raise MeshIOError(
                    "Malformed ascii PLY file: a vertex line has fewer values than the "
                    "header declares."
                )
            vertices = table[:, columns].astype(np.float64)
        elif name == "face":
            polygons = []
            for row in rows:
                tokens = row.split()
                try:
                    length = int(tokens[0])
                    polygon = [int(token) for token in tokens[1 : 1 + length]]
                except (ValueError, IndexError) as exc:
                    raise MeshIOError(
                        "Malformed ascii PLY file: a face line is not a list of indices."
                    ) from exc
                if len(polygon) != length:
                    raise MeshIOError(
                        f"Malformed ascii PLY file: a face declares {length} indices but "
                        f"supplies {len(polygon)}."
                    )
                polygons.append(polygon)
            faces = _faces_from_polygons(polygons)
    return vertices, faces


def read_ply(source):
    """Read a Stanford PLY in ascii, binary little-endian, or binary big-endian.

    Extra vertex properties - colours, normals, confidence, texture coordinates -
    are skipped by stride rather than treated as an error, and elements the file
    declares but this reader does not understand are stepped over correctly so
    the face element that follows them still lines up.
    """
    data = _read_all_bytes(source)
    if data[:3].lower() != b"ply":
        raise MeshIOError("Not a PLY file: the magic word 'ply' is missing.")
    elements, order, body_offset = _parse_ply_header(data)
    if order == "ascii":
        vertices, faces = _read_ply_ascii_body(data[body_offset:], elements)
    else:
        vertices, faces = _read_ply_binary_body(data, body_offset, elements, order)
    return _mesh_from(vertices, faces, "PLY file")


def write_ply(target, mesh, binary=True, double=True):
    """Write a Stanford PLY with the standard vertex/face layout.

    Faces use ``property list uchar int vertex_indices``, which every consumer
    understands.  Coordinates default to ``double`` so a PLY round-trip through
    OpenCad is bit-exact; pass ``double=False`` for the more common ``float``
    layout when file size or a picky legacy reader matters.
    """
    vertices, faces = _export_arrays(mesh, "a PLY file")
    scalar = "double" if double else "float"
    encoding = "binary_little_endian 1.0" if binary else "ascii 1.0"
    header = "\n".join(
        [
            "ply",
            f"format {encoding}",
            "comment Written by OpenCad",
            f"element vertex {len(vertices)}",
            f"property {scalar} x",
            f"property {scalar} y",
            f"property {scalar} z",
            f"element face {len(faces)}",
            "property list uchar int vertex_indices",
            "end_header",
            "",
        ]
    ).encode("ascii")

    if binary:
        point_dtype = np.dtype("<f8" if double else "<f4")
        face_dtype = np.dtype([("count", "u1"), ("indices", "<i4", (3,))])
        records = np.zeros(len(faces), dtype=face_dtype)
        records["count"] = 3
        records["indices"] = faces
        body = vertices.astype(point_dtype).tobytes() + records.tobytes()
        _write_all_bytes(target, header + body)
        return

    lines = _vertex_lines(vertices, "")
    lines.extend(f"3 {a} {b} {c}" for a, b, c in faces.tolist())
    payload = ("\n".join(lines) + "\n").encode("ascii") if lines else b""
    _write_all_bytes(target, header + payload)


# ----------------------------------------------------------------------
# OFF
# ----------------------------------------------------------------------
def read_off(source):
    """Read an OFF file, including the ``COFF``/``NOFF`` colour and normal variants.

    Per-vertex colours or normals trailing the coordinates, and per-face colours
    trailing the indices, are ignored - only the geometry is imported.  The
    header counts may sit on the magic line or on the line after it, as both
    spellings occur in the wild.
    """
    text = _decode(_read_all_bytes(source))
    lines = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    if not lines:
        raise MeshIOError("Empty OFF file: there is nothing to read.")

    first = lines[0].split()
    if not first[0].upper().endswith("OFF"):
        raise MeshIOError(f"Not an OFF file: the header starts with '{first[0]}'.")
    if len(first) > 1:
        counts_tokens, cursor = first[1:], 1
    else:
        if len(lines) < 2:
            raise MeshIOError("Truncated OFF file: the vertex/face count line is missing.")
        counts_tokens, cursor = lines[1].split(), 2
    if len(counts_tokens) < 2:
        raise MeshIOError("Malformed OFF file: the count line needs vertex and face counts.")
    try:
        n_vertices = int(counts_tokens[0])
        n_faces = int(counts_tokens[1])
    except ValueError as exc:
        raise MeshIOError("Malformed OFF file: the counts are not integers.") from exc
    if n_vertices < 0 or n_faces < 0:
        raise MeshIOError("Malformed OFF file: negative vertex or face count.")
    if cursor + n_vertices + n_faces > len(lines):
        raise MeshIOError(
            f"Truncated OFF file: it declares {n_vertices} vertices and {n_faces} faces "
            f"but holds only {len(lines) - cursor} data lines."
        )

    positions = []
    for line in lines[cursor : cursor + n_vertices]:
        parts = line.split()
        if len(parts) < 3:
            raise MeshIOError("Malformed OFF file: a vertex line has fewer than 3 coordinates.")
        try:
            positions.append((float(parts[0]), float(parts[1]), float(parts[2])))
        except ValueError as exc:
            raise MeshIOError("Malformed OFF file: non-numeric vertex coordinates.") from exc
    cursor += n_vertices

    polygons = []
    for line in lines[cursor : cursor + n_faces]:
        parts = line.split()
        try:
            length = int(parts[0])
            polygon = [int(token) for token in parts[1 : 1 + length]]
        except (ValueError, IndexError) as exc:
            raise MeshIOError("Malformed OFF file: a face line is not a list of indices.") from exc
        if len(polygon) != length:
            raise MeshIOError(
                f"Malformed OFF file: a face declares {length} indices but supplies "
                f"{len(polygon)}."
            )
        polygons.append(polygon)

    vertices = np.asarray(positions, dtype=float) if positions else np.zeros((0, 3))
    return _mesh_from(vertices, _faces_from_polygons(polygons), "OFF file")


def write_off(target, mesh):
    """Write an OFF file.

    The edge count in the header is written as 0: it is optional information
    that every reader ignores, and computing it would cost a full edge pass.
    """
    vertices, faces = _export_arrays(mesh, "an OFF file")
    lines = ["OFF", f"{len(vertices)} {len(faces)} 0"]
    lines.extend(_vertex_lines(vertices, ""))
    lines.extend(f"3 {a} {b} {c}" for a, b, c in faces.tolist())
    _write_all_bytes(target, ("\n".join(lines) + "\n").encode("ascii"))


# ----------------------------------------------------------------------
# 3MF
# ----------------------------------------------------------------------
_3MF_CORE_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
_3MF_REL_TYPE = "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"
_OPC_CONTENT_TYPES_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
_OPC_RELATIONSHIPS_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_3MF_MODEL_PART = "3D/3dmodel.model"
_3MF_UNITS = ("micron", "millimeter", "centimeter", "inch", "foot", "meter")
_3MF_MAX_DEPTH = 12


def _localname(tag):
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _children(element, name):
    return [child for child in element if _localname(child.tag) == name]


def _first_child(element, name):
    for child in element:
        if _localname(child.tag) == name:
            return child
    return None


def _check_zip_member_names(names):
    """Reject archive members that would escape the package root.

    This module never extracts a 3MF to disk, but a name like ``../../evil`` or
    ``/etc/passwd`` is a reliable sign of a hostile archive and would become a
    real traversal the moment any caller did extract it.  Refusing the archive
    outright is cheap and keeps the invariant local.
    """
    for name in names:
        normalised = name.replace("\\", "/")
        head = normalised.split("/", 1)[0]
        if normalised.startswith("/") or (len(head) >= 2 and head[1] == ":"):
            raise MeshIOError(
                f"Refusing to read this 3MF: member '{name}' uses an absolute path."
            )
        if any(part == ".." for part in normalised.split("/")):
            raise MeshIOError(
                f"Refusing to read this 3MF: member '{name}' escapes the archive root."
            )


def _read_zip_member(archive, name, limit):
    try:
        info = archive.getinfo(name)
    except KeyError as exc:
        raise MeshIOError(f"3MF package has no part named '{name}'.") from exc
    if info.file_size > limit:
        raise MeshIOError(
            f"3MF part '{name}' declares {info.file_size} uncompressed bytes, above the "
            f"{limit} byte safety limit; raise max_part_bytes if this is genuine."
        )
    try:
        with archive.open(info) as handle:
            data = handle.read(limit + 1)
    except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
        raise MeshIOError(f"Could not read 3MF part '{name}': {exc}") from exc
    if len(data) > limit:
        raise MeshIOError(
            f"3MF part '{name}' expands past the {limit} byte safety limit; the archive "
            "understates its size."
        )
    return data


def _parse_xml_safely(data, label):
    """Parse XML while refusing any document type declaration.

    A 3MF part has no legitimate use for a DTD, and refusing one closes the whole
    entity-expansion family of attacks - billion laughs, external entity
    inclusion, parameter entities - without relying on parser internals: entities
    cannot be declared without a DTD, and the five predefined entities plus
    numeric character references each expand to a single character.  The scan is
    a deliberate over-approximation; the string ``<!DOCTYPE`` cannot appear in
    legitimate XML content, where ``<`` must be escaped.
    """
    if b"<!doctype" in data.lower():
        raise MeshIOError(
            f"Refusing to parse {label}: it contains a document type declaration, which "
            "3MF does not use and which enables entity-expansion attacks."
        )
    try:
        return ET.fromstring(data)
    except ET.ParseError as exc:
        raise MeshIOError(f"{label} is not valid XML: {exc}") from exc


def _parse_3mf_matrix(text):
    """Convert a 3MF row-vector 4x3 transform into a column-vector 4x4 matrix."""
    try:
        values = [float(token) for token in text.split()]
    except ValueError as exc:
        raise MeshIOError(f"Malformed 3MF transform '{text}'.") from exc
    if len(values) != 12:
        raise MeshIOError(
            f"Malformed 3MF transform: expected 12 numbers, got {len(values)}."
        )
    rows = np.asarray(values, dtype=float).reshape(4, 3)
    matrix = np.eye(4)
    matrix[:3, :3] = rows[:3].T
    matrix[:3, 3] = rows[3]
    return matrix


def _3mf_object_mesh(objects, object_id, depth=0):
    """Build the mesh of one object, resolving ``<components>`` recursively."""
    if depth > _3MF_MAX_DEPTH:
        raise MeshIOError("3MF object graph is nested too deeply or contains a cycle.")
    node = objects.get(object_id)
    if node is None:
        raise MeshIOError(f"3MF file references object id '{object_id}', which does not exist.")

    mesh_node = _first_child(node, "mesh")
    if mesh_node is not None:
        return _3mf_mesh_element(mesh_node)

    components = _first_child(node, "components")
    if components is None:
        raise MeshIOError(f"3MF object id '{object_id}' has neither a mesh nor components.")
    pieces = []
    for component in _children(components, "component"):
        child_id = component.get("objectid")
        if child_id is None:
            raise MeshIOError("3MF component is missing its 'objectid' attribute.")
        piece = _3mf_object_mesh(objects, child_id, depth + 1)
        transform = component.get("transform")
        if transform:
            piece = piece.transform(_parse_3mf_matrix(transform))
        pieces.append(piece)
    return Mesh.concatenate(pieces)


def _3mf_mesh_element(mesh_node):
    vertices_node = _first_child(mesh_node, "vertices")
    triangles_node = _first_child(mesh_node, "triangles")
    positions = []
    if vertices_node is not None:
        for vertex in _children(vertices_node, "vertex"):
            try:
                positions.append(
                    (float(vertex.get("x")), float(vertex.get("y")), float(vertex.get("z")))
                )
            except (TypeError, ValueError) as exc:
                raise MeshIOError("Malformed 3MF vertex: x, y, and z must all be numbers.") from exc
    triangles = []
    if triangles_node is not None:
        for triangle in _children(triangles_node, "triangle"):
            try:
                triangles.append(
                    (
                        int(triangle.get("v1")),
                        int(triangle.get("v2")),
                        int(triangle.get("v3")),
                    )
                )
            except (TypeError, ValueError) as exc:
                raise MeshIOError(
                    "Malformed 3MF triangle: v1, v2, and v3 must all be integers."
                ) from exc
    vertices = np.asarray(positions, dtype=float) if positions else np.zeros((0, 3))
    faces = np.asarray(triangles, dtype=np.int64) if triangles else np.zeros((0, 3), np.int64)
    return _mesh_from(vertices, faces, "3MF model")


def _locate_3mf_model_part(archive, limit):
    names = archive.namelist()
    _check_zip_member_names(names)
    lookup = {name.replace("\\", "/"): name for name in names}

    if "_rels/.rels" in lookup:
        root = _parse_xml_safely(
            _read_zip_member(archive, lookup["_rels/.rels"], limit), "3MF relationship part"
        )
        for relationship in _children(root, "Relationship"):
            if relationship.get("Type") != _3MF_REL_TYPE:
                continue
            target = (relationship.get("Target") or "").lstrip("/")
            if target in lookup:
                return lookup[target]
    if _3MF_MODEL_PART in lookup:
        return lookup[_3MF_MODEL_PART]
    for key, name in lookup.items():
        if key.lower().endswith(".model"):
            return name
    raise MeshIOError("This 3MF package contains no 3D model part ('3D/3dmodel.model').")


def read_3mf(source, max_part_bytes=MAX_3MF_PART_BYTES):
    """Read the first build item of a 3MF package.

    3MF is the additive-manufacturing exchange format: a zip whose
    ``3D/3dmodel.model`` part holds the geometry as XML.  This reader resolves
    the model part through the package relationships, follows the first
    ``<build><item>`` to its object, expands ``<components>`` references, and
    applies both the item and component transforms.  The document's ``unit``
    attribute is preserved in ``mesh.attributes["unit"]`` - coordinates are not
    rescaled, because the kernel is unitless by design.

    Security: member names that escape the archive root are rejected, parts are
    capped at ``max_part_bytes`` uncompressed, and any document type declaration
    aborts the parse so entity expansion cannot be triggered.
    """
    limit = int(max_part_bytes)
    if hasattr(source, "read"):
        handle = source
    else:
        handle = _as_path(source)
    try:
        archive = zipfile.ZipFile(handle)
    except zipfile.BadZipFile as exc:
        raise MeshIOError("Not a 3MF package: the file is not a valid zip archive.") from exc
    except MeshIOError:
        raise
    except OSError as exc:
        raise MeshIOError(f"Could not open the 3MF package: {exc}") from exc

    with archive:
        part = _locate_3mf_model_part(archive, limit)
        root = _parse_xml_safely(_read_zip_member(archive, part, limit), f"3MF part '{part}'")

    if _localname(root.tag) != "model":
        raise MeshIOError("Malformed 3MF: the model part's root element is not <model>.")

    resources = _first_child(root, "resources")
    objects = {}
    if resources is not None:
        for node in _children(resources, "object"):
            identifier = node.get("id")
            if identifier is not None:
                objects.setdefault(identifier, node)
    if not objects:
        raise MeshIOError("Malformed 3MF: the model declares no objects.")

    build = _first_child(root, "build")
    items = _children(build, "item") if build is not None else []
    if items:
        item = items[0]
        object_id = item.get("objectid")
        if object_id is None:
            raise MeshIOError("Malformed 3MF: the first build item has no 'objectid'.")
        mesh = _3mf_object_mesh(objects, object_id)
        transform = item.get("transform")
        if transform:
            mesh = mesh.transform(_parse_3mf_matrix(transform))
    else:
        mesh = _3mf_object_mesh(objects, next(iter(objects)))

    unit = (root.get("unit") or "millimeter").lower()
    return mesh.with_attributes(unit=unit)


def write_3mf(target, mesh, unit="millimeter", name="opencad"):
    """Write a minimal but valid 3MF package.

    The archive holds the three parts a consumer needs to open it -
    ``[Content_Types].xml``, ``_rels/.rels``, and ``3D/3dmodel.model`` - with the
    OPC and 3MF core namespaces.  Triangle winding is written unchanged, which is
    correct because both OpenCad and 3MF define outward-facing counter-clockwise
    triangles.
    """
    vertices, faces = _export_arrays(mesh, "a 3MF file")
    unit = str(unit).lower()
    if unit not in _3MF_UNITS:
        raise MeshIOError(
            f"'{unit}' is not a 3MF unit; use one of {', '.join(_3MF_UNITS)}."
        )
    label = "".join(character for character in str(name) if character.isalnum() or character in "-_ ")

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<model unit="{unit}" xml:lang="en-US" xmlns="{_3MF_CORE_NS}">',
        '<metadata name="Application">OpenCad</metadata>',
        f'<metadata name="Title">{label.strip() or "opencad"}</metadata>',
        "<resources>",
        '<object id="1" type="model">',
        "<mesh>",
        "<vertices>",
    ]
    parts.extend(
        f'<vertex x="{x!r}" y="{y!r}" z="{z!r}"/>' for x, y, z in vertices.tolist()
    )
    parts.append("</vertices>")
    parts.append("<triangles>")
    parts.extend(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in faces.tolist())
    parts.extend(
        [
            "</triangles>",
            "</mesh>",
            "</object>",
            "</resources>",
            "<build>",
            '<item objectid="1"/>',
            "</build>",
            "</model>",
        ]
    )
    model = ("\n".join(parts) + "\n").encode("utf-8")

    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<Types xmlns="{_OPC_CONTENT_TYPES_NS}">'
        '<Default Extension="rels" '
        'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="model" '
        'ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
        "</Types>\n"
    ).encode("utf-8")
    relationships = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<Relationships xmlns="{_OPC_RELATIONSHIPS_NS}">'
        f'<Relationship Id="rel0" Type="{_3MF_REL_TYPE}" Target="/{_3MF_MODEL_PART}"/>'
        "</Relationships>\n"
    ).encode("utf-8")

    handle = target if hasattr(target, "write") else _as_path(target)
    try:
        with zipfile.ZipFile(handle, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("[Content_Types].xml", content_types)
            archive.writestr("_rels/.rels", relationships)
            archive.writestr(_3MF_MODEL_PART, model)
    except MeshIOError:
        raise
    except OSError as exc:
        raise MeshIOError(f"Could not write the 3MF package: {exc}") from exc


# ----------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------
_READERS = {
    "stl": read_stl,
    "obj": read_obj,
    "ply": read_ply,
    "off": read_off,
    "3mf": read_3mf,
}

_WRITERS = {
    "stl": write_stl,
    "obj": write_obj,
    "ply": write_ply,
    "off": write_off,
    "3mf": write_3mf,
}


def detect_format(target, formats=SUPPORTED_READ_FORMATS):
    """Return the lowercase format key implied by a path's extension.

    File objects are accepted when they expose a usable ``name``; otherwise the
    caller must say which format it means.
    """
    name = target
    if not isinstance(name, (str, os.PathLike)):
        name = getattr(target, "name", None)
        if not isinstance(name, (str, os.PathLike)):
            raise MeshIOError(
                "Cannot infer a mesh format from this object; pass fmt='stl' (or another "
                "supported format) explicitly."
            )
    suffix = Path(os.fspath(name)).suffix.lower().lstrip(".")
    if not suffix:
        raise MeshIOError(f"'{name}' has no file extension, so the mesh format is unknown.")
    if suffix not in formats:
        supported = ", ".join(f".{key}" for key in formats)
        raise MeshIOError(f"Unsupported mesh format '.{suffix}'. Supported formats: {supported}.")
    return suffix


def read_mesh(source, fmt=None, **options):
    """Read any supported mesh file, choosing the reader by extension.

    ``fmt`` overrides the extension, which is what you want for an in-memory
    buffer or a temporary file with the wrong suffix.  Remaining keyword
    arguments go straight to the format-specific reader.
    """
    key = str(fmt).lower().lstrip(".") if fmt else detect_format(source, SUPPORTED_READ_FORMATS)
    if key not in _READERS:
        supported = ", ".join(f".{name}" for name in SUPPORTED_READ_FORMATS)
        raise MeshIOError(f"Unsupported mesh format '{key}'. Supported formats: {supported}.")
    return _READERS[key](source, **options)


def write_mesh(target, mesh, fmt=None, **options):
    """Write any supported mesh file, choosing the writer by extension.

    Options are passed through to the format-specific writer, so
    ``write_mesh(path, mesh, binary=False)`` produces an ascii STL or PLY
    depending on the extension.
    """
    key = str(fmt).lower().lstrip(".") if fmt else detect_format(target, SUPPORTED_WRITE_FORMATS)
    if key not in _WRITERS:
        supported = ", ".join(f".{name}" for name in SUPPORTED_WRITE_FORMATS)
        raise MeshIOError(f"Unsupported mesh format '{key}'. Supported formats: {supported}.")
    try:
        return _WRITERS[key](target, mesh, **options)
    except TypeError as exc:
        raise MeshIOError(f"Unsupported option for the .{key} writer: {exc}") from exc


def file_filter(mode="read", include_all=True):
    """Build a Qt-style file dialog filter string for the supported formats.

    Returns something like ``"Mesh Files (*.stl *.obj *.ply *.off *.3mf);;STL
    (*.stl);;..."`` so the UI never has to hard-code a format list that can drift
    away from what the kernel actually implements.
    """
    normalized = str(mode).lower()
    if normalized == "read":
        formats = SUPPORTED_READ_FORMATS
    elif normalized == "write":
        formats = SUPPORTED_WRITE_FORMATS
    else:
        raise MeshIOError("file_filter mode must be 'read' or 'write'.")
    patterns = " ".join(f"*.{key}" for key in formats)
    entries = [f"Mesh Files ({patterns})"]
    entries.extend(f"{_FORMAT_LABELS[key]} (*.{key})" for key in formats)
    if include_all:
        entries.append("All Files (*)")
    return ";;".join(entries)
