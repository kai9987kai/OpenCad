"""Generate the OpenCad application icon.

An isometric cube: unambiguous for a CAD tool and still legible at 16 px, which
a more literal picture of the app's output is not - a gyroid slice degenerates
into three unrelated blobs once it is small enough to sit in a taskbar.

Everything is drawn from signed distance functions and supersampled, the same
approach the geometry kernel uses, so the artwork is resolution independent.
Keeping it generated rather than checked in as an opaque binary means it can be
re-rendered at any size or recoloured with the theme.

Writes ``assets/opencad.ico`` with every size Windows asks for. PNG-compressed
entries are used throughout, which Windows has understood since Vista.

Run with::

    python packaging/make_icon.py
"""

from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIZES = (16, 24, 32, 48, 64, 128, 256)
SUPERSAMPLE = 4

# The application palette, from src/ui/styles.py.
BACKGROUND = np.array([30, 30, 46], dtype=float)      # #1e1e2e
ACCENT_A = np.array([137, 180, 250], dtype=float)     # #89b4fa
ACCENT_B = np.array([139, 213, 202], dtype=float)     # #8bd5ca
EDGE = np.array([69, 71, 90], dtype=float)            # #45475a


def _write_png(rgba):
    """Encode an (H, W, 4) uint8 array as PNG bytes."""
    height, width = rgba.shape[:2]

    raw = bytearray()
    for row in rgba:
        raw.append(0)  # filter type 0 (None) for each scanline
        raw.extend(row.tobytes())

    def chunk(tag, payload):
        data = tag + payload
        return struct.pack(">I", len(payload)) + data + struct.pack(">I", zlib.crc32(data))

    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )


def _rounded_mask(coords, half, radius):
    """Signed-distance mask for a rounded square, in the same style as the kernel."""
    q = np.abs(coords) - (half - radius)
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.maximum(q[..., 0], q[..., 1]), 0.0)
    return outside + inside - radius


def _convex_polygon_mask(coords, polygon):
    """Signed distance to a convex polygon given counter-clockwise in screen space.

    Negative inside. Taking the maximum over each edge's outward half-plane is
    exact in the interior and slightly conservative near a corner, which is more
    than good enough for anti-aliasing an icon.
    """
    polygon = np.asarray(polygon, dtype=float)
    following = np.roll(polygon, -1, axis=0)
    edge = following - polygon

    # Screen space has +y downward, so the outward normal of a visually
    # counter-clockwise loop is (dy, -dx) rather than (-dy, dx).
    normals = np.stack([edge[:, 1], -edge[:, 0]], axis=1)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(lengths > 0, lengths, 1.0)

    offsets = coords[..., None, :] - polygon
    return np.max(np.einsum("...cj,cj->...c", offsets, normals), axis=-1)


def _hexagon_points(radius):
    """The six corners of a flat isometric cube silhouette, plus its centre."""
    angles = np.deg2rad(np.array([90.0, 30.0, 330.0, 270.0, 210.0, 150.0]))
    # Negate y so the first point is visually at the top of the image.
    return {
        round(float(np.rad2deg(a))) % 360: np.array(
            [radius * np.cos(a), -radius * np.sin(a)]
        )
        for a in angles
    }


def render(size):
    """Render one icon size as an (size, size, 4) uint8 array."""
    n = size * SUPERSAMPLE
    axis = (np.arange(n) + 0.5) / n * 2.0 - 1.0  # -1 .. 1
    x, y = np.meshgrid(axis, axis, indexing="xy")
    coords = np.stack([x, y], axis=-1)
    pixel = 2.0 / n  # one output pixel in coordinate units

    def coverage(distance, softness=1.0):
        """Antialiased inside-ness from a signed distance."""
        return np.clip(-distance / (pixel * SUPERSAMPLE * softness), 0.0, 1.0)

    # --- the tile --------------------------------------------------------
    plate = _rounded_mask(coords, half=0.94, radius=0.30)
    plate_alpha = coverage(plate)

    # --- an isometric cube ----------------------------------------------
    v = _hexagon_points(radius=0.62)
    centre = np.zeros(2)

    # Three rhombi, each listed counter-clockwise as seen on screen.
    faces = {
        "top": ([v[150], v[90], v[30], centre], 1.00),
        "right": ([centre, v[30], v[330], v[270]], 0.62),
        "left": ([v[210], v[150], centre, v[270]], 0.80),
    }

    rgb = np.broadcast_to(BACKGROUND, (*x.shape, 3)).copy()

    # A hairline border lifts the tile off a dark taskbar.
    border = coverage(plate + 0.055) * (1.0 - coverage(plate + 0.02))
    rgb = rgb * (1.0 - border[..., None]) + EDGE * border[..., None]

    # Shade the three visible faces so the solid reads as a solid.
    ramp = np.clip((x + y + 2.0) / 4.0, 0.0, 1.0)[..., None]
    base = ACCENT_A * (1.0 - ramp) + ACCENT_B * ramp

    for polygon, brightness in faces.values():
        mask = coverage(_convex_polygon_mask(coords, polygon))[..., None]
        colour = base * brightness + BACKGROUND * (1.0 - brightness) * 0.35
        rgb = rgb * (1.0 - mask) + colour * mask

    # Dark seams between the faces keep the three planes distinct when small.
    seam_width = max(pixel * SUPERSAMPLE * 0.9, 0.012)
    seams = np.zeros_like(x)
    for corner in (v[90], v[210], v[330]):
        direction = corner - centre
        length = np.linalg.norm(direction)
        unit = direction / length
        offset = coords - centre
        along = np.clip(offset @ unit, 0.0, length)
        perpendicular = offset - along[..., None] * unit
        seams = np.maximum(
            seams, coverage(np.linalg.norm(perpendicular, axis=-1) - seam_width)
        )
    silhouette = coverage(_convex_polygon_mask(coords, [v[90], v[30], v[330], v[270], v[210], v[150]]))
    seams = seams * silhouette
    rgb = rgb * (1.0 - seams[..., None]) + BACKGROUND * seams[..., None]

    rgba = np.concatenate([rgb, plate_alpha[..., None] * 255.0], axis=-1)

    # Box-filter the supersampled buffer down to the requested size.
    rgba = rgba.reshape(size, SUPERSAMPLE, size, SUPERSAMPLE, 4).mean(axis=(1, 3))
    return np.clip(np.round(rgba), 0, 255).astype(np.uint8)


def build_ico(path):
    images = [_write_png(render(size)) for size in SIZES]

    entries = bytearray()
    offset = 6 + 16 * len(images)
    for size, payload in zip(SIZES, images, strict=True):
        entries += struct.pack(
            "<BBBBHHII",
            size if size < 256 else 0,  # 0 means 256 in the ICO header
            size if size < 256 else 0,
            0,  # palette size, 0 for true colour
            0,  # reserved
            1,  # colour planes
            32,  # bits per pixel
            len(payload),
            offset,
        )
        offset += len(payload)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        struct.pack("<HHH", 0, 1, len(images)) + bytes(entries) + b"".join(images)
    )
    return path


def main():
    target = ROOT / "assets" / "opencad.ico"
    build_ico(target)
    print(f"Wrote {target} ({target.stat().st_size:,} bytes, sizes {', '.join(map(str, SIZES))})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
