"""Length and angle units.

OpenCad had no unit concept at all: a cube of "size 2" was two of nothing.  That
is fine while everything stays inside the viewport, but it breaks down the
moment geometry meets the physical world - STL files carry no units, 3MF is
defined in millimetres, printers think in millimetres, and machinists think in
inches.  Wall-thickness and feature-size warnings are meaningless without a
scale to compare against.

The kernel stores every coordinate in **millimetres**.  That choice follows STEP,
3MF, and essentially every slicer; imperial workflows convert at the boundary
rather than carrying a second internal representation around.
"""

from __future__ import annotations

import re

__all__ = [
    "ANGLE_UNITS",
    "BASE_LENGTH_UNIT",
    "LENGTH_UNITS",
    "UnitSystem",
    "convert_angle",
    "convert_length",
    "format_length",
    "parse_length",
]

BASE_LENGTH_UNIT = "mm"

#: Multipliers converting each unit into the millimetre base.
LENGTH_UNITS = {
    "nm": 1e-6,
    "um": 1e-3,
    "µm": 1e-3,
    "micron": 1e-3,
    "micrometre": 1e-3,
    "micrometer": 1e-3,
    "mm": 1.0,
    "millimetre": 1.0,
    "millimeter": 1.0,
    "cm": 10.0,
    "centimetre": 10.0,
    "centimeter": 10.0,
    "dm": 100.0,
    "m": 1000.0,
    "metre": 1000.0,
    "meter": 1000.0,
    "km": 1e6,
    "thou": 0.0254,
    "mil": 0.0254,
    "in": 25.4,
    '"': 25.4,
    "inch": 25.4,
    "inches": 25.4,
    "ft": 304.8,
    "foot": 304.8,
    "feet": 304.8,
    "yd": 914.4,
    "yard": 914.4,
}

#: Multipliers converting each unit into degrees.
ANGLE_UNITS = {
    "deg": 1.0,
    "degree": 1.0,
    "°": 1.0,
    "rad": 57.29577951308232,
    "radian": 57.29577951308232,
    "grad": 0.9,
    "turn": 360.0,
    "rev": 360.0,
}

# Accepts "12", "12mm", "-3.5 in", "1e3 um", '2"'.
_LENGTH_PATTERN = re.compile(
    r"""^\s*
    (?P<value>[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)
    \s*
    (?P<unit>[a-zA-Zµ°"]*)
    \s*$""",
    re.VERBOSE,
)


def _normalize_unit(unit, table, kind):
    key = str(unit).strip()
    if key in table:
        return key
    lowered = key.lower()
    if lowered in table:
        return lowered
    # Tolerate a regular plural; irregular ones ("inches", "feet") are listed
    # explicitly in the tables because English morphology is not worth guessing.
    if lowered.endswith("s") and lowered[:-1] in table:
        return lowered[:-1]
    supported = ", ".join(sorted({key for key in table if key.isalpha()}))
    raise ValueError(f"Unknown {kind} unit {unit!r}. Supported: {supported}.")


def convert_length(value, from_unit=BASE_LENGTH_UNIT, to_unit=BASE_LENGTH_UNIT):
    """Convert a length between units. Works on scalars and numpy arrays."""
    source = LENGTH_UNITS[_normalize_unit(from_unit, LENGTH_UNITS, "length")]
    target = LENGTH_UNITS[_normalize_unit(to_unit, LENGTH_UNITS, "length")]
    return value * (source / target)


def convert_angle(value, from_unit="deg", to_unit="deg"):
    """Convert an angle between units. Works on scalars and numpy arrays."""
    source = ANGLE_UNITS[_normalize_unit(from_unit, ANGLE_UNITS, "angle")]
    target = ANGLE_UNITS[_normalize_unit(to_unit, ANGLE_UNITS, "angle")]
    return value * (source / target)


def parse_length(text, default_unit=BASE_LENGTH_UNIT, to_unit=BASE_LENGTH_UNIT):
    """Parse ``"12.5mm"`` or ``'2"'`` into a number in ``to_unit``.

    A bare number is interpreted as ``default_unit``, which is what lets a
    dimension field accept both ``25`` and ``1in`` without a mode switch.
    """
    if isinstance(text, (int, float)):
        return convert_length(float(text), default_unit, to_unit)

    match = _LENGTH_PATTERN.match(str(text))
    if not match:
        raise ValueError(f"Could not read a length from {text!r}.")

    value = float(match.group("value"))
    unit = match.group("unit") or default_unit
    return convert_length(value, unit, to_unit)


def format_length(value, unit=BASE_LENGTH_UNIT, precision=3, with_unit=True, from_unit=BASE_LENGTH_UNIT):
    """Render a length for display, trimming trailing zeros.

    ``25.400`` reads worse than ``25.4`` in a properties panel, and ``0.000``
    actively misleads, so very small non-zero values fall back to scientific
    notation rather than rounding to nothing.
    """
    converted = float(convert_length(float(value), from_unit, unit))
    if converted != 0.0 and abs(converted) < 10.0 ** (-int(precision)):
        text = f"{converted:.{max(int(precision), 1)}e}"
    else:
        text = f"{converted:.{int(precision)}f}".rstrip("0").rstrip(".")
        if text in ("", "-"):
            text = "0"
    return f"{text} {unit}" if with_unit else text


class UnitSystem:
    """The display units and precision for a document.

    Geometry stays in millimetres; this only affects what the user reads and
    types.  Area and volume are derived from the length unit so a document in
    inches reports ``in^2`` and ``in^3`` without a second setting to forget.
    """

    __slots__ = ("angle_unit", "length_unit", "name", "precision")

    def __init__(self, name="Millimetres", length_unit="mm", angle_unit="deg", precision=3):
        self.name = str(name)
        self.length_unit = _normalize_unit(length_unit, LENGTH_UNITS, "length")
        self.angle_unit = _normalize_unit(angle_unit, ANGLE_UNITS, "angle")
        self.precision = int(precision)

    @classmethod
    def millimetres(cls):
        return cls("Millimetres", "mm", "deg", 3)

    @classmethod
    def centimetres(cls):
        return cls("Centimetres", "cm", "deg", 4)

    @classmethod
    def metres(cls):
        return cls("Metres", "m", "deg", 4)

    @classmethod
    def inches(cls):
        return cls("Inches", "in", "deg", 4)

    @property
    def scale_to_base(self):
        """How many millimetres one display unit represents."""
        return LENGTH_UNITS[self.length_unit]

    def to_display(self, value_mm):
        return convert_length(value_mm, BASE_LENGTH_UNIT, self.length_unit)

    def to_base(self, value_display):
        return convert_length(value_display, self.length_unit, BASE_LENGTH_UNIT)

    def parse(self, text):
        """Read user input into millimetres, defaulting to the document unit."""
        return parse_length(text, default_unit=self.length_unit, to_unit=BASE_LENGTH_UNIT)

    def format_length(self, value_mm, with_unit=True):
        return format_length(
            value_mm, self.length_unit, self.precision, with_unit, BASE_LENGTH_UNIT
        )

    def format_area(self, value_mm2, with_unit=True):
        converted = float(value_mm2) / (self.scale_to_base**2)
        text = format_length(converted, self.length_unit, self.precision, False, self.length_unit)
        return f"{text} {self.length_unit}^2" if with_unit else text

    def format_volume(self, value_mm3, with_unit=True):
        converted = float(value_mm3) / (self.scale_to_base**3)
        text = format_length(converted, self.length_unit, self.precision, False, self.length_unit)
        return f"{text} {self.length_unit}^3" if with_unit else text

    def format_angle(self, value_degrees, with_unit=True):
        converted = float(convert_angle(value_degrees, "deg", self.angle_unit))
        text = f"{converted:.{self.precision}f}".rstrip("0").rstrip(".") or "0"
        suffix = "°" if self.angle_unit == "deg" else f" {self.angle_unit}"
        return f"{text}{suffix}" if with_unit else text

    def format_point(self, point_mm, with_unit=False):
        """Format an XYZ triple the way a coordinate readout should look."""
        parts = [self.format_length(float(value), with_unit=False) for value in point_mm]
        joined = ", ".join(parts)
        return f"{joined} {self.length_unit}" if with_unit else joined

    def to_dict(self):
        return {
            "name": self.name,
            "length_unit": self.length_unit,
            "angle_unit": self.angle_unit,
            "precision": self.precision,
        }

    @classmethod
    def from_dict(cls, data):
        data = data or {}
        return cls(
            data.get("name", "Millimetres"),
            data.get("length_unit", "mm"),
            data.get("angle_unit", "deg"),
            data.get("precision", 3),
        )

    def __eq__(self, other):
        if not isinstance(other, UnitSystem):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __repr__(self):
        return f"UnitSystem({self.name!r}, {self.length_unit!r}, precision={self.precision})"


#: The presets a document-settings dialog should offer.
PRESETS = {
    "Millimetres": UnitSystem.millimetres,
    "Centimetres": UnitSystem.centimetres,
    "Metres": UnitSystem.metres,
    "Inches": UnitSystem.inches,
}
