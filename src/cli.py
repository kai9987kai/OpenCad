"""Headless command line for OpenCad.

The geometry kernel needs no display, no OpenGL, and no VTK, so everything it
can do is available from a terminal, a build script, or CI:

.. code-block:: bash

    opencad info part.stl                     # measure it
    opencad check part.stl --build 220x220x250  # will it print?
    opencad convert part.stl part.3mf         # change format
    opencad primitive icosphere --radius 20 -o ball.stl
    opencad lattice --kind gyroid --size 40 --thickness 0.8 -o lattice.stl

Run as ``python -m src.cli`` from a source checkout, or as ``opencad`` once the
package is installed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.kernel import io_mesh, primitives
from src.kernel.analysis import mesh_report, printability, volume_fraction
from src.kernel.lattice import DEFAULT_SPEC, build_lattice_field, cells_per_wall
from src.kernel.meshing import surface_nets
from src.kernel.sdf import TPMS_KINDS
from src.kernel.units import UnitSystem

__all__ = ["main"]

EXIT_OK = 0
EXIT_PROBLEMS = 1
EXIT_USAGE = 2


def _load(path):
    try:
        return io_mesh.read_mesh(path)
    except Exception as error:
        raise SystemExit(f"opencad: cannot read {path}: {error}") from error


def _save(mesh, path):
    try:
        io_mesh.write_mesh(path, mesh)
    except Exception as error:
        raise SystemExit(f"opencad: cannot write {path}: {error}") from error
    size = Path(path).stat().st_size
    print(f"Wrote {path} - {mesh.n_faces:,} triangles, {size:,} bytes")


def _parse_build_volume(text):
    if not text:
        return None
    parts = text.lower().replace(",", "x").split("x")
    if len(parts) != 3:
        raise SystemExit("opencad: --build expects THREE dimensions, e.g. 220x220x250")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as error:
        raise SystemExit(f"opencad: could not read --build {text!r}") from error


def _progress(fraction, message):
    sys.stderr.write(f"\r  {message} ({fraction * 100:3.0f}%)   ")
    sys.stderr.flush()
    if fraction >= 1.0:
        sys.stderr.write("\n")


# ----------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------
def command_info(args):
    mesh = _load(args.path)
    units = UnitSystem(length_unit=args.units)
    report = mesh_report(mesh, units)

    if args.json:
        print(json.dumps(report, indent=2))
        return EXIT_OK

    geometry, topology, quality = report["geometry"], report["topology"], report["quality"]
    print(f"{args.path}")
    print(f"  triangles      {geometry['triangles']:,}   vertices {geometry['vertices']:,}")
    print(f"  size           {geometry['size_text']}")
    print(f"  volume         {geometry['volume_text']}")
    print(f"  surface area   {geometry['area_text']}")
    print(f"  centre of mass {geometry['center_of_mass_text']}")
    print(f"  watertight     {'yes' if topology['watertight'] else 'no'}")
    print(f"  manifold       {'yes' if topology['edge_manifold'] else 'no'}")
    print(f"  bodies         {topology['components']}   genus {topology['genus']}")
    print(f"  smallest angle {quality['min_angle_deg']:.2f} deg   slivers {quality['slivers']}")
    return EXIT_OK


def command_check(args):
    mesh = _load(args.path)
    units = UnitSystem(length_unit=args.units)
    findings = printability(
        mesh,
        units=units,
        max_overhang_angle=args.overhang,
        min_feature_size=args.min_feature,
        build_volume=_parse_build_volume(args.build),
    )

    if args.json:
        print(json.dumps([finding.as_dict() for finding in findings], indent=2))
    else:
        print(f"{args.path}")
        marker = {"error": "ERROR  ", "warning": "WARNING", "info": "note   "}
        for finding in findings:
            print(f"  {marker.get(finding.severity, '       ')} {finding.title}")
            if finding.detail:
                print(f"          {finding.detail}")

    # A non-zero exit lets this gate a build pipeline.
    return EXIT_PROBLEMS if any(finding.is_problem for finding in findings) else EXIT_OK


def command_convert(args):
    mesh = _load(args.source)
    if args.clean:
        mesh = mesh.cleaned()
    if args.largest:
        mesh = mesh.largest_component()
    _save(mesh, args.destination)
    return EXIT_OK


def command_primitive(args):
    try:
        mesh = primitives.create(args.name, **_primitive_kwargs(args))
    except (ValueError, TypeError) as error:
        raise SystemExit(f"opencad: {error}") from error
    _save(mesh, args.output)
    return EXIT_OK


def _primitive_kwargs(args):
    """Pass through only the options the requested primitive accepts."""
    import inspect

    factory = primitives.PRIMITIVES.get(str(args.name).lower())
    if factory is None:
        supported = ", ".join(sorted(primitives.PRIMITIVES))
        raise SystemExit(f"opencad: unknown primitive {args.name!r}. Supported: {supported}")

    accepted = inspect.signature(factory).parameters
    candidates = {
        "radius": args.radius,
        "height": args.height,
        "size": args.size,
        "resolution": args.resolution,
        "subdivisions": args.subdivisions,
    }
    return {key: value for key, value in candidates.items() if value is not None and key in accepted}


def command_lattice(args):
    spec = dict(DEFAULT_SPEC)
    spec.update(
        {
            "kind": args.kind,
            "mode": args.mode,
            "period": args.period,
            "thickness": args.thickness,
            "level": args.level,
            "size": args.size,
            "grade_target": args.grade,
            "grade_axis": args.grade_axis,
            "grade_amount": args.grade_amount,
        }
    )

    bounds = None
    if args.fill:
        host = _load(args.fill)
        bounds = host.bounding_box

    per_wall = cells_per_wall(spec, args.resolution, None if bounds is not None else args.size)
    if args.mode == "sheet" and 0 < per_wall < 2.0:
        print(
            f"warning: only {per_wall:.1f} grid cells across a {args.thickness} mm wall; "
            f"the result will be lighter than specified. Raise --resolution.",
            file=sys.stderr,
        )

    field, sampling, region = build_lattice_field(spec, bounds)
    mesh = surface_nets(
        field, bounds=sampling, resolution=args.resolution, progress=None if args.quiet else _progress
    )
    if mesh.is_empty:
        raise SystemExit(
            "opencad: those settings produced no geometry. Try a thicker wall, "
            "a smaller cell size, or a higher resolution."
        )

    density = volume_fraction(field, bounds=region, resolution=min(args.resolution, 96))
    print(
        f"Relative density {density['fraction'] * 100:.1f}% "
        f"({density['solid_volume']:.0f} mm3 of {density['sampled_volume']:.0f} mm3)"
    )
    _save(mesh, args.output)
    return EXIT_OK


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(
        prog="opencad",
        description="Headless geometry tools for OpenCad.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  opencad info part.stl\n"
            "  opencad check part.stl --build 220x220x250\n"
            "  opencad convert part.stl part.3mf\n"
            "  opencad primitive icosphere --radius 20 -o ball.stl\n"
            "  opencad lattice --kind gyroid --size 40 --thickness 0.8 -o lattice.stl\n"
        ),
    )
    parser.add_argument("--version", action="version", version="OpenCad 0.3.0")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    info = sub.add_parser("info", help="measure a mesh file")
    info.add_argument("path")
    info.add_argument("--units", default="mm", help="display units (default: mm)")
    info.add_argument("--json", action="store_true", help="machine-readable output")
    info.set_defaults(handler=command_info)

    check = sub.add_parser("check", help="printability report; exits non-zero on problems")
    check.add_argument("path")
    check.add_argument("--units", default="mm")
    check.add_argument("--overhang", type=float, default=45.0, help="max unsupported angle")
    check.add_argument("--min-feature", type=float, default=0.8, dest="min_feature")
    check.add_argument("--build", default=None, metavar="XxYxZ", help="build volume in mm")
    check.add_argument("--json", action="store_true")
    check.set_defaults(handler=command_check)

    convert = sub.add_parser("convert", help="convert between mesh formats")
    convert.add_argument("source")
    convert.add_argument("destination")
    convert.add_argument("--clean", action="store_true", help="weld and drop degenerate faces")
    convert.add_argument("--largest", action="store_true", help="keep only the largest body")
    convert.set_defaults(handler=command_convert)

    primitive = sub.add_parser("primitive", help="generate an analytic solid")
    primitive.add_argument("name", help=", ".join(sorted(primitives.PRIMITIVES)))
    primitive.add_argument("-o", "--output", required=True)
    primitive.add_argument("--radius", type=float, default=None)
    primitive.add_argument("--height", type=float, default=None)
    primitive.add_argument("--size", type=float, default=None)
    primitive.add_argument("--resolution", type=int, default=None)
    primitive.add_argument("--subdivisions", type=int, default=None)
    primitive.set_defaults(handler=command_primitive)

    lattice = sub.add_parser("lattice", help="generate a TPMS lattice solid")
    lattice.add_argument("-o", "--output", required=True)
    lattice.add_argument("--kind", default="gyroid", choices=sorted(TPMS_KINDS))
    lattice.add_argument("--mode", default="sheet", choices=["sheet", "solid"])
    lattice.add_argument("--size", type=float, default=30.0, help="cube size in mm")
    lattice.add_argument("--fill", default=None, metavar="MESH", help="fill this mesh's bounds")
    lattice.add_argument("--period", type=float, default=6.0, help="unit cell size in mm")
    lattice.add_argument("--thickness", type=float, default=0.8, help="wall thickness in mm")
    lattice.add_argument("--level", type=float, default=0.0, help="surface offset (network mode)")
    lattice.add_argument("--resolution", type=int, default=128)
    lattice.add_argument(
        "--grade", default="none", choices=["none", "thickness", "period"],
        help="vary wall thickness or cell size through the part",
    )
    lattice.add_argument("--grade-axis", default="z", choices=["x", "y", "z", "radial"], dest="grade_axis")
    lattice.add_argument("--grade-amount", type=float, default=0.0, dest="grade_amount")
    lattice.add_argument("--quiet", action="store_true", help="no progress output")
    lattice.set_defaults(handler=command_lattice)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "handler", None):
        parser.print_help()
        return EXIT_USAGE
    return args.handler(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
